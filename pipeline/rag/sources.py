"""여러 종류의 입력을 '조각(chunk)'으로 바꾼다. 조각마다 원본 위치(파일·쪽·시각·주소)를 기억한다.

  이미지   → 사진 조각 1개 (VLM 설명문 + 원본 사진 = 영상 첫 프레임 후보)
  동영상   → 장면 전환마다 구간 조각 (시작 프레임 + VLM 설명 + 움직임·카메라) + 말소리 조각 (Whisper)
  PDF      → 쪽별 글 조각 + 사진 조각 (홍보 모드의 브로셔 읽기 재사용)
  문서     → .txt .md .docx .pptx .xlsx .hwpx .json .csv 글 조각 (.url·주소만 적은 txt는 웹 주소로)
  웹사이트 → 본문 글 조각 + 큰 사진 조각 (사용자 PC에서 직접 가져옴)

조각 형식: {id, modality, source, loc, text, file, meta}
  modality: text | image | video_segment | speech
  text    : 검색에 쓰는 글 (설명문·본문·받아쓰기)
  file    : 저장소 안의 이미지 경로 (image, video_segment만) — Wan의 첫 프레임으로 쓸 수 있다
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from .. import cvio, prompts as P, videoio
from ..agents import _ask_json
from ..hub import Hub, HubError
from ..utils import log

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
TXT_EXT = {".txt", ".md", ".csv", ".json"}
MAX_SEGMENTS = int(__import__("os").getenv("VS_RAG_MAX_SEGMENTS", "40"))   # 영상 하나에서 뽑는 최대 구간 수


def _id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def split_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    """문단을 살려 약 size자 단위로 자른다 (앞뒤 overlap자 겹침)."""
    text = re.sub(r"[ \t ]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[•\-·▶■□○●◆※])", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = (cur + "\n" + p).strip()
            continue
        if cur:
            out.append(cur)
        while len(p) > size:
            out.append(p[:size])
            p = p[size - overlap:]
        cur = p
    if cur:
        out.append(cur)
    return [c for c in out if len(re.sub(r"\W", "", c)) >= 8]


def _detail() -> str:
    """max 모드: 검색에 걸릴 단서를 늘리도록 긴 상세 설명을 함께 받는다."""
    from ..budget import BUDGET
    if not BUDGET.mode["detailed_captions"]:
        return ""
    return ('\nAlso add "details" (English, 120-200 words: every visible object, material, colour, light, layout, '
            'readable signs, atmosphere, what could move) and "details_ko" (같은 내용을 한국어로 5~8문장).')


# ------------------------------------------------------------------ 이미지
def image_chunk(hub: Hub, src: Path, store_files: Path, source: str, loc: str = "") -> dict:
    from PIL import Image
    cid = _id(source, loc, src.stat().st_size)
    dst = store_files / f"{cid}.jpg"
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((1920, 1920))
        im.save(dst, quality=92)
        w, h = im.size
    try:
        d = _ask_json(hub, P.RAG_IMAGE, "Describe this image." + _detail(), images=[("", dst)], temperature=0.1)
    except (HubError, ValueError) as e:
        d = {"caption": src.stem, "notes": f"설명 실패: {e}"}
    ocr = ""
    if d.get("has_text") or d.get("kind") in ("graphic_text", "screenshot"):
        try:                                             # 도표·안내판·문서 사진은 글자를 OCR로 뽑아 검색에 포함
            ocr = hub.ocr(dst)[:2000]
        except HubError as e:
            log(f"    (OCR 실패: {e})")
    text = " ".join(str(x) for x in (d.get("caption"), d.get("caption_ko"), d.get("setting"),
                                     ", ".join(map(str, d.get("subjects") or [])), d.get("readable_text"),
                                     d.get("details"), d.get("details_ko"), ocr) if x)
    return {"id": cid, "modality": "image", "source": source, "loc": loc, "text": text, "file": dst.name,
            "meta": {"kind": d.get("kind", ""), "people": int(d.get("people") or 0), "has_text": bool(d.get("has_text")),
                     "good_first_frame": bool(d.get("good_first_frame", True)), "size": [w, h],
                     "caption": d.get("caption", ""), "caption_ko": d.get("caption_ko", ""), "ocr": ocr[:500]}}


# ------------------------------------------------------------------ 동영상
def _segments(video: Path, max_seg: float = 8.0, min_seg: float = 1.5) -> tuple[list[tuple[float, float]], float]:
    """HSV 히스토그램 상관으로 장면 전환을 찾고, 긴 장면은 max_seg초로 나눈다."""
    import cv2
    cap = cvio.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(fps / 4)))                  # 초당 4장만 본다
    prev, cuts, i, last_t = None, [0.0], 0, 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            t = i / fps
            hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
            hist = cv2.normalize(cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]), None).flatten()
            if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < 0.55 and t - cuts[-1] >= min_seg:
                cuts.append(t)
            prev, last_t = hist, t
        i += 1
    cap.release()
    dur = max(last_t, i / fps)
    bounds = cuts + [dur]
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        n = max(1, int((b - a) // max_seg) + (1 if (b - a) % max_seg > min_seg else 0))
        for k in range(n):
            s0 = a + (b - a) * k / n
            s1 = a + (b - a) * (k + 1) / n
            if s1 - s0 >= min(min_seg, b - a):
                segs.append((round(s0, 2), round(s1, 2)))
    if len(segs) > MAX_SEGMENTS:                         # 너무 길면 고르게 골라낸다
        idx = [round(i * (len(segs) - 1) / (MAX_SEGMENTS - 1)) for i in range(MAX_SEGMENTS)]
        segs = [segs[i] for i in sorted(set(idx))]
    return segs, dur


def _frame_at(video: Path, t: float, out: Path) -> Path | None:
    import cv2
    cap = cvio.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    cvio.imwrite(str(out), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def video_chunks(hub: Hub, video: Path, store_files: Path, source: str, speech: bool = True) -> list[dict]:
    segs, dur = _segments(video)
    log(f"    영상 {video.name}: {dur:.0f}초, 구간 {len(segs)}개")
    out = []
    for s0, s1 in segs:
        cid = _id(source, s0, s1)
        frames = []
        for j, t in enumerate((s0 + 0.05, (s0 + s1) / 2, max(s0, s1 - 0.1))):
            p = _frame_at(video, t, store_files / f"{cid}_{j}.jpg")
            if p:
                frames.append((t, p))
        if not frames:
            continue
        d, how = None, "frames"
        if hub.video_ok:                                 # 구간 영상을 직접 보여주면 움직임·카메라를 더 정확히 설명
            clip = videoio.small_clip(video, store_files / f"{cid}_clip.mp4", s0, s1)
            if clip:
                try:
                    d = _ask_json(hub, P.RAG_SEGMENT, "Describe this video segment (attached as VIDEO)." + _detail(), video=clip,
                                  temperature=0.1)
                    how = "video"
                except (HubError, ValueError) as e:
                    log(f"    (영상 직접 입력 실패 → 프레임으로: {str(e)[:100]})")
                    hub.video_ok = False
                finally:
                    clip.unlink(missing_ok=True)
        if d is None:
            try:
                d = _ask_json(hub, P.RAG_SEGMENT, "Describe this video segment." + _detail(),
                              images=[(f"t={t:.1f}s:", p) for t, p in frames], temperature=0.1)
            except (HubError, ValueError) as e:
                d = {"caption": f"{video.stem} {s0:.0f}s", "notes": f"설명 실패: {e}"}
        start = frames[0][1]
        for _, p in frames[1:]:                         # 시작 프레임만 남긴다 (Wan 첫 프레임 후보)
            p.unlink(missing_ok=True)
        final = store_files / f"{cid}.jpg"
        start.replace(final)
        text = " ".join(str(x) for x in (d.get("caption"), d.get("caption_ko"), d.get("motion"), d.get("camera"),
                                         d.get("details"), d.get("details_ko")) if x)
        out.append({"id": cid, "modality": "video_segment", "source": source, "loc": f"{s0:.1f}-{s1:.1f}s",
                    "text": text, "file": final.name,
                    "meta": {"t0": s0, "t1": s1, "motion": d.get("motion", ""), "camera": d.get("camera", ""), "described_from": how,
                             "people": int(d.get("people") or 0), "has_text": bool(d.get("has_text")),
                             "good_first_frame": bool(d.get("good_first_frame", True)),
                             "caption": d.get("caption", ""), "caption_ko": d.get("caption_ko", "")}})
    if speech and shutil.which("ffmpeg"):
        wav = store_files / f"{_id(source, 'audio')}.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                           capture_output=True)
        if r.returncode == 0 and wav.exists() and wav.stat().st_size > 32000:
            res = hub.transcribe(wav)
            wav.unlink(missing_ok=True)
            if res and res.get("text"):
                segs_t = res.get("segments") or [{"start": 0, "end": dur, "text": res["text"]}]
                buf, b0 = "", None
                for sg in segs_t:                       # 약 30초 단위로 묶는다
                    b0 = sg.get("start", 0) if b0 is None else b0
                    buf += " " + str(sg.get("text", "")).strip()
                    if sg.get("end", 0) - b0 >= 30 or sg is segs_t[-1]:
                        if buf.strip():
                            out.append({"id": _id(source, "speech", b0), "modality": "speech", "source": source,
                                        "loc": f"{b0:.0f}-{sg.get('end', 0):.0f}s", "text": buf.strip(), "file": "",
                                        "meta": {"t0": b0, "t1": sg.get("end", 0)}})
                        buf, b0 = "", None
        else:
            wav.unlink(missing_ok=True)
    return out


# ------------------------------------------------------------------ 문서 · PDF
def text_chunks(text: str, source: str, loc: str = "") -> list[dict]:
    return [{"id": _id(source, loc, i), "modality": "text", "source": source, "loc": f"{loc}#{i + 1}" if loc else f"#{i + 1}",
             "text": c, "file": "", "meta": {}} for i, c in enumerate(split_text(text))]


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n\n", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def pdf_chunks(hub: Hub, pdf: Path, store_files: Path, source: str) -> list[dict]:
    from ..promo.ingest import read_pdf
    tmp = store_files / f"_pdf_{_id(source)}"
    pages, imgs = read_pdf(pdf, tmp)
    out = []
    for i, t in enumerate(pages, 1):
        out += text_chunks(t, source, f"p.{i}")
    for a in imgs:
        out.append(image_chunk(hub, Path(a["file"]), store_files, source, a["source"]))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# ------------------------------------------------------------------ 웹사이트
class _Page(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form", "aside"}

    def __init__(self):
        super().__init__()
        self.depth, self.title, self.texts, self.imgs, self._in_title = 0, "", [], [], False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "img" and a.get("src"):
            self.imgs.append((a["src"], a.get("alt", "")))
        if tag == "meta" and a.get("property") in ("og:image", "twitter:image") and a.get("content"):
            self.imgs.insert(0, (a["content"], "og:image"))
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "br", "section", "article", "tr"):
            self.texts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.depth and data.strip():
            self.texts.append(data.strip() + " ")


def web_chunks(hub: Hub, url: str, store_files: Path, max_images: int = 8) -> list[dict]:
    import requests
    from urllib.parse import urljoin
    hdr = {"User-Agent": "Mozilla/5.0 (VIDEO_Studio RAG)"}
    r = requests.get(url, headers=hdr, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    pg = _Page()
    pg.feed(r.text)
    text = re.sub(r"\n\s*\n+", "\n\n", "".join(pg.texts))
    out = text_chunks((pg.title.strip() + "\n\n" + text).strip(), url, "web")
    seen, n = set(), 0
    for src, alt in pg.imgs:
        full = urljoin(url, src)
        if full in seen or full.startswith("data:") or re.search(r"(logo|icon|sprite|blank|pixel)", full, re.I):
            continue
        seen.add(full)
        try:
            ir = requests.get(full, headers=hdr, timeout=20)
            if ir.status_code != 200 or len(ir.content) < 30_000:
                continue
            tmp = store_files / f"_web_{_id(full)}"
            tmp.write_bytes(ir.content)
            from PIL import Image
            with Image.open(tmp) as im:
                big = min(im.size) >= 360
            if big:
                out.append(image_chunk(hub, tmp, store_files, url, full))
                n += 1
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 — 사진 하나 실패는 건너뛴다
            continue
        if n >= max_images:
            break
    log(f"    웹 {url}: 글 조각 {sum(1 for c in out if c['modality'] == 'text')}개, 사진 {n}장")
    return out


# ------------------------------------------------------------------ 종류 판별
def collect(hub: Hub, item: str, store_files: Path, speech: bool = True) -> list[dict]:
    """파일·폴더·URL 하나를 조각 목록으로."""
    if re.match(r"^https?://", item):
        return web_chunks(hub, item, store_files)
    p = Path(item)
    if p.is_dir():
        out = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and not f.name.startswith(("~$", ".")) and \
                    f.suffix.lower() in IMG_EXT | VID_EXT | TXT_EXT | {".pdf", ".docx", ".pptx", ".xlsx", ".hwpx", ".url"}:
                out += collect(hub, str(f), store_files, speech)
        return out
    ext = p.suffix.lower()
    src = p.name
    if ext in IMG_EXT:
        return [image_chunk(hub, p, store_files, src)]
    if ext in VID_EXT:
        return video_chunks(hub, p, store_files, src, speech)
    if ext == ".pdf":
        return pdf_chunks(hub, p, store_files, src)
    from .. import inputs
    if ext in (".docx", ".pptx", ".xlsx", ".hwpx"):
        out = []
        for loc, t in inputs.doc_pages(p):
            out += text_chunks(t, src, loc)
        return out
    if ext == ".url" or (ext == ".txt" and inputs.is_url_list(p)):
        out = []
        for u in inputs.urls_in(p):
            try:
                out += web_chunks(hub, u, store_files)
            except Exception as e:  # noqa: BLE001
                log(f"    (웹 {u} 가져오기 실패: {e})")
        return out
    if ext in TXT_EXT:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if ext == ".json":
            try:
                raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=1)
            except ValueError:
                pass
        return text_chunks(raw, src)
    log(f"    (지원하지 않는 형식이라 건너뜀: {p.name})")
    return []
