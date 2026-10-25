"""브로셔(PDF) + 사진 폴더 → 홍보 프로젝트(brief.json).
   또는 입력 폴더 하나(--input): PDF 여러 개·사진·동영상·문서(docx/pptx/xlsx/hwpx/txt)·웹 주소를 함께 읽는다.

1. 페이지 텍스트·사진 추출 (PyMuPDF — Adobe CMYK 사진도 색 반전 없이 RGB로)
2. LLM이 숫자(facts)·문구(messages)·문제점(issues)을 뽑고 → 코드가 '원문에 그대로 있는지' 검증
   원문에 없는 숫자·문장은 영상에 쓸 수 없게 표시한다 (홍보영상이 사실과 다르면 안 되므로)
3. 연락처(주소·전화·메일)는 LLM 없이 정규식으로
4. 규칙 검사: 단위가 이름과 안 맞는 숫자, 서로 다른 항목에 같은 숫자, '로고예정' 같은 자리표시
5. VLM이 사진을 분류(실사진 건물/실내/사람, 개념 이미지, 글자 그래픽…) → 움직임 레시피 연결
6. 페이지에서 브랜드 색 추출
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .. import prompts as P
from ..agents import _ask_json
from ..hub import Hub, HubError
from ..utils import log
from . import kb

IMG_MIN_W, IMG_MIN_H = 900, 500
PLACEHOLDERS = ("로고예정", "로고 예정", "QR", "TBD", "OOO", "000-0000")


_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d"} |
                       {"\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u3000": " "})


_CTRL = {c: None for c in list(range(0, 9)) + [11, 12] + list(range(14, 32)) + [127]}


def clean(t: str) -> str:
    """PDF 글자 정리: 줄바꿈 안 되는 하이픈·공백 등을 일반 문자로 (전화번호·주소 인식용),
    눈에 안 보이는 제어 문자(\x07 등)는 지운다 — 모델이 그대로 옮겨 적으면 JSON이 깨진다."""
    return t.translate(_DASHES).translate(_CTRL)


def norm(s: str) -> str:
    return re.sub(r"[\s,·]", "", str(s)).lower()


# ------------------------------------------------------------------ PDF
def _pymupdf():
    """PyMuPDF (예전 버전은 모듈 이름이 fitz). 없으면 설치 방법을 알려 준다.

    PyMuPDF는 pymupdf-layout·pymupdf4llm이 설치돼 있으면 시작할 때 레이아웃 분석(onnxruntime)을 자동으로 불러온다.
    Windows에서 onnxruntime DLL이 충돌하면 오류 메시지도 없이 프로그램이 꺼지므로(access violation),
    쓰지 않는 그 기능은 불러오지 않게 막는다. 글·사진 추출에는 필요 없다.
    """
    import os
    import sys
    if os.getenv("VS_PYMUPDF_LAYOUT", "0") != "1" and "pymupdf" not in sys.modules:
        for opt in ("pymupdf.layout", "pymupdf4llm"):        # None으로 두면 import가 ImportError → PyMuPDF가 건너뜀
            sys.modules.setdefault(opt, None)
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            raise SystemExit("PDF를 읽으려면 PyMuPDF가 필요합니다. 지금 쓰는 가상환경에서 설치해 주세요:\n"
                             "    pip install -r requirements.txt   (또는 pip install pymupdf)") from None


def open_pdf(pdf: Path):
    """파일 경로 대신 바이트로 연다 (Windows에서 한글·공백·괄호가 든 경로 문제를 피함)."""
    pymupdf = _pymupdf()
    log(f"    PDF 여는 중: {Path(pdf).name} ({Path(pdf).stat().st_size / 1e6:.1f}MB, PyMuPDF {getattr(pymupdf, 'VersionBind', '?')})")
    return pymupdf.open(stream=Path(pdf).read_bytes(), filetype="pdf")


def read_pdf(pdf: Path, img_dir: Path) -> tuple[list[str], list[dict]]:
    pymupdf = _pymupdf()
    from PIL import Image
    doc = open_pdf(pdf)
    pages = [clean(p.get_text()) for p in doc]
    img_dir.mkdir(parents=True, exist_ok=True)
    seen, assets = set(), []
    for pno, page in enumerate(doc, 1):
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.width < IMG_MIN_W or pix.height < IMG_MIN_H or (pix.colorspace and pix.colorspace.n == 1):
                    continue
                if pix.alpha:
                    pix = pymupdf.Pixmap(pix, 0)
                if pix.colorspace is None or pix.colorspace.n != 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            except Exception as e:  # noqa: BLE001
                log(f"    (이미지 {xref} 건너뜀: {e})")
                continue
            im.thumbnail((1920, 1920))
            out = img_dir / f"pdf_p{pno:02d}_{xref}.jpg"
            im.save(out, quality=92)
            assets.append({"file": out, "source": f"PDF {pno}쪽"})
    return pages, assets


def brand_colors(pdf: Path, n_pages: int = 6) -> dict:
    """페이지를 작게 렌더링해 채도 있는 색을 k-means로 묶어 브랜드 색을 고른다."""
    import numpy as np
    pymupdf = _pymupdf()
    doc = open_pdf(pdf)
    px = []
    for page in list(doc)[:n_pages]:
        pm = page.get_pixmap(dpi=24)
        a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[..., :3].reshape(-1, 3)
        px.append(a)
    a = np.concatenate(px).astype("float32")
    mx, mn = a.max(1), a.min(1)
    sat = (mx - mn) / (mx + 1)
    a = a[(sat > 0.35) & (mx > 40)]
    if len(a) < 50:
        return dict(kb.DEFAULT_BRAND)
    rng = np.random.default_rng(0)
    c = a[rng.choice(len(a), 5, replace=False)]
    for _ in range(12):
        lab = ((a[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        c = np.array([a[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(5)])
    cnt = np.bincount(lab, minlength=5)
    order = cnt.argsort()[::-1]
    cols = [c[i] for i in order]
    hexs = ["#%02x%02x%02x" % tuple(int(v) for v in col) for col in cols]
    lum = [0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2] for col in cols]
    dark = hexs[int(np.argmin(lum))] if min(lum) < 70 else kb.DEFAULT_BRAND["dark"]   # 충분히 어두운 색이 없으면 기본 남색
    return {**kb.DEFAULT_BRAND, "primary": hexs[0], "secondary": hexs[1] if len(hexs) > 1 else hexs[0],
            "accent": hexs[2] if len(hexs) > 2 else hexs[0], "dark": dark}


# ------------------------------------------------------------------ 연락처·규칙 검사
URL_RE = re.compile(r"\b(?:https?://)?(?:[a-z0-9-]+\.)+(?:ai|kr|com|net|org|io)(?:/[^\s]*)?\b", re.I)
PHONE_RE = re.compile(r"\b0\d{1,2}[.\-\s]{1,2}\d{3,4}[.\-\s]{1,2}\d{4}\b")
MAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def find_contacts(pages: list[str]) -> dict:
    """주소·전화·메일. 안내 페이지(뒤쪽)에 나오고 자주 나오는 것을 앞에. 하위 도메인 주소(hafa.…)를 우선."""
    from collections import Counter
    text = "\n".join(pages)
    mails = sorted(set(MAIL_RE.findall(text)))
    seen_u: Counter = Counter()
    last: dict[str, int] = {}
    for i, t in enumerate(pages):
        for u in URL_RE.findall(t):
            u = u.lower()
            if "@" in u or any(u in m.lower() for m in mails):
                continue
            seen_u[u] += 1
            last[u] = i
    urls = sorted(seen_u, key=lambda u: (-(u.count(".") >= 2 and not u.startswith("x.")), -last[u], -seen_u[u]))
    seen_p: Counter = Counter()
    for t in pages:
        for p in PHONE_RE.findall(t):
            seen_p[re.sub(r"[\s.\-]+", "-", p.strip())] += 1
    phones = [p for p, _ in seen_p.most_common()]
    return {"urls": urls, "phones": phones, "emails": mails}


STORAGE_WORDS = ("storage", "스토리지", "hdd", "ssd", "저장", "ram", "메모리", "memory")
COMPUTE_UNITS = ("core", "코어")
SIZE_UNITS = ("tb", "gb", "pb")


def rule_issues(facts: list[dict], pages: list[str]) -> list[dict]:
    out = []
    for f in facts:
        lab, unit = f.get("label", "").lower(), f.get("unit", "").lower()
        if any(w in lab for w in STORAGE_WORDS) and any(u in unit for u in COMPUTE_UNITS):
            out.append({"page": f.get("page"), "text": f"'{f['label']}' 값이 {f['display']} {f['unit']}입니다. 저장장치 항목에 코어(Core) 단위가 붙어 있어 오기로 보입니다."})
        item = re.sub(r"\b(cpu|gpu|npu)\s*zone\b|(cpu|gpu|npu)\s*존", "", lab)   # 'CPU ZONE 메모리'의 영역 이름은 빼고 본다
        if ("cpu" in item or "코어" in item) and any(u == unit for u in SIZE_UNITS):
            out.append({"page": f.get("page"), "text": f"'{f['label']}'에 용량 단위({f['unit']})가 붙어 있습니다. 이름이나 단위를 확인하세요."})
    by_val: dict[str, list[dict]] = {}
    for f in facts:
        by_val.setdefault(norm(f["display"]), []).append(f)
    for v, fs in by_val.items():
        labels = {x["label"] for x in fs}
        if len(labels) > 1 and len(v) >= 3:
            out.append({"page": fs[0].get("page"), "text": f"같은 숫자 {fs[0]['display']}가 서로 다른 항목({', '.join(sorted(labels))})에 쓰였습니다."})
    ph_pages = [i for i, t in enumerate(pages, 1) if any(norm(ph) in norm(t) for ph in PLACEHOLDERS[:2])]
    if ph_pages:
        out.append({"page": ph_pages[0], "text": f"'로고예정' 자리표시가 {', '.join(map(str, ph_pages))}쪽에 남아 있습니다. 영상 엔딩 로고가 정해지지 않았습니다."})
    return out


# ------------------------------------------------------------------ LLM 추출 + 검증
def salvage(raw: str) -> dict | None:
    """JSON이 깨졌거나 중간에 잘린 답에서 완성된 항목만 건진다 (항목은 중괄호가 한 겹인 평평한 객체)."""
    import re as _re
    text = _re.sub(r"<think>.*?</think>", "", raw or "", flags=_re.S)
    out: dict[str, list] = {"facts": [], "messages": [], "issues": []}
    for m in _re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0), strict=False)
        except ValueError:
            continue
        if not isinstance(o, dict):
            continue
        if "display" in o:
            out["facts"].append(o)
        elif "tag" in o and "text" in o:
            out["messages"].append(o)
        elif "text" in o:
            out["issues"].append(o)
    return out if any(out.values()) else None


def _read_part(hub: Hub, part: str) -> tuple[dict | None, bool, str]:
    """한 묶음 읽기 → (결과, 잘림/깨짐 여부, 원문 답)."""
    from ..utils import extract_json
    from datetime import date
    head = f"Today's date: {date.today().isoformat()} (dates up to today are past or current, not future).\n"
    raw = hub.chat([{"role": "system", "content": P.PROMO_FACTS}, {"role": "user", "content": head + part}], temperature=0.1)
    cut = getattr(hub, "last_finish", "") == "length"
    try:
        d = extract_json(raw)
        if isinstance(d, dict) and not cut:
            return d, False, raw
    except ValueError:
        pass
    return salvage(raw), True, raw


def extract(hub: Hub, pages: list[str], labels: list[str] | None = None, debug_dir: Path | None = None) -> dict:
    """쪽을 자르지 않고, 글 모델 최대 길이 × 모드 비율 안에 들어가는 만큼 여러 쪽을 한 번에 읽힌다.
    답이 깨지거나 최대 길이에서 잘리면 완성된 항목은 건지고, 그 묶음을 반으로 나눠 다시 읽힌다 (한 쪽까지)."""
    from ..budget import BUDGET, est_tokens
    budget = BUDGET.fill_tokens(hub, hub.models().text)
    groups, cur, n = [], [], 0
    for i, t in enumerate(pages):
        k = est_tokens(t) + 10
        if cur and n + k > budget:
            groups.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += k
    if cur:
        groups.append(cur)
    facts, msgs, issues = [], [], []
    failed: list[int] = []

    def run(g: list[int]) -> None:
        part = "".join(f"\n=== page {i + 1}{f' ({labels[i]})' if labels else ''} ===\n{pages[i]}" for i in g)
        log(f"[promo] 자료 {g[0] + 1}~{g[-1] + 1}쪽 한 번에 읽기 (약 {est_tokens(part):,}토큰)")
        try:
            d, bad, raw = _read_part(hub, part)
        except HubError as e:
            d, bad, raw = None, True, str(e)
        if bad and debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"facts_raw_p{g[0] + 1}-{g[-1] + 1}.txt").write_text(
                f"finish_reason={getattr(hub, 'last_finish', '')}\n\n{raw}", encoding="utf-8")
        if d:
            facts.extend(d.get("facts") or [])
            msgs.extend(d.get("messages") or [])
            issues.extend(d.get("issues") or [])
        if not bad:
            return
        why = "최대 길이에서 잘림" if getattr(hub, "last_finish", "") == "length" else "형식이 깨짐"
        got = f"완성된 항목 {sum(len(d.get(k) or []) for k in ('facts', 'messages', 'issues'))}개는 건짐" if d else "건진 항목 없음"
        if len(g) > 1:
            log(f"    (답이 {why} — {got}, 쪽을 반으로 나눠 다시 읽기)")
            run(g[: len(g) // 2])
            run(g[len(g) // 2:])
        else:
            log(f"    ({g[0] + 1}쪽 답이 {why} — {got})")
            if not d:
                failed.append(g[0] + 1)

    for g in groups:
        run(g)
    if failed:
        log(f"    (끝내 못 읽은 쪽: {failed} — 원문 답은 {debug_dir} 의 facts_raw_*.txt)")
    return {"facts": facts, "messages": msgs, "issues": issues}


def verify(raw: dict, pages: list[str]) -> dict:
    """원문에 그대로 있는 것만 'ok'. 숫자가 인용문 안에, 인용문이 해당 쪽(또는 전체) 원문 안에 있어야 한다."""
    allt = norm("\n".join(pages))

    def page_text(p) -> str:
        try:
            return norm(pages[int(p) - 1])
        except (TypeError, ValueError, IndexError):
            return allt

    facts, seen = [], set()
    for f in raw["facts"]:
        disp, quote = str(f.get("display", "")).strip(), str(f.get("quote", "")).strip()
        if not disp or not re.search(r"\d", disp):
            continue
        ok = norm(disp) in norm(quote) and (norm(quote) in page_text(f.get("page")) or norm(quote) in allt)
        key = (norm(f.get("label", "")), norm(disp), norm(f.get("unit", "")))
        if key in seen:
            continue
        seen.add(key)
        facts.append({"id": f"f{len(facts) + 1}", "label": f.get("label", ""), "display": disp,
                      "unit": f.get("unit", ""), "tags": f.get("tags") or ["other"], "page": f.get("page"),
                      "quote": quote, "status": "ok" if ok else "unverified"})
    msgs, seen = [], set()
    for m in raw["messages"]:
        text = re.sub(r"\s+", " ", str(m.get("text", ""))).strip()
        if not text or norm(text) in seen:
            continue
        seen.add(norm(text))
        ok = norm(text) in page_text(m.get("page")) or norm(text) in allt
        short = str(m.get("short") or "").strip()
        if not short or not all(norm(w) in norm(text) for w in short.split()):
            short = text if len(text) <= 18 else ""
        msgs.append({"id": f"m{len(msgs) + 1}", "tag": m.get("tag", "other"), "text": text, "short": short,
                     "page": m.get("page"), "status": "ok" if ok else "unverified"})
    return {"facts": facts, "messages": msgs}


# ------------------------------------------------------------------ 사진 분류
def classify_assets(hub: Hub | None, assets: list[dict]) -> list[dict]:
    import numpy as np
    from PIL import Image
    out = []
    for i, a in enumerate(assets, 1):
        f = Path(a["file"])
        with Image.open(f) as im:
            arr = np.asarray(im.convert("L").resize((64, 64)), dtype="float32")
        rec = {"id": f"a{i:02d}", "file": f.name, "source": a["source"]}
        if arr.std() < 4:
            rec.update(kind="blank", subject="plain background", has_text=False, people=0, usable=False)
        elif hub is None:
            rec.update(kind=a.get("kind", "concept_tech"), subject=a.get("subject", f.stem), has_text=False,
                       people=0, usable=True)
        else:
            try:
                d = _ask_json(hub, P.PROMO_ASSET, "Classify this image.", images=[("", f)], temperature=0.1)
            except (HubError, ValueError) as e:
                d = {"kind": "concept_tech", "subject": f.stem, "notes_ko": f"분류 실패: {e}"}
            kind = d.get("kind") if d.get("kind") in kb.ASSET_KINDS else "concept_tech"
            rec.update(kind=kind, subject=d.get("subject", ""), has_text=bool(d.get("has_text")),
                       people=int(d.get("people") or 0), usable=bool(d.get("usable", True)),
                       notes_ko=d.get("notes_ko", ""))
        if rec["kind"] in ("graphic_text", "logo", "blank"):
            rec["usable"] = False
        rec["recipes"] = kb.recipes_for(rec["kind"], rec["subject"])
        out.append(rec)
        log(f"    {rec['id']} {rec['kind']:<14} {rec['subject'][:60]}  → {rec['recipes'][:2]}")
    return out


# ------------------------------------------------------------------ 입력 폴더 (여러 형식)
def _video_material(hub: Hub, video: Path, adir: Path, max_frames: int = 8) -> tuple[list[dict], str]:
    """동영상 → 장면마다 대표 프레임(사진 후보) + 말소리 받아쓰기."""
    import shutil as _sh
    import subprocess
    from ..rag.sources import _frame_at, _segments
    segs, dur = _segments(video)
    if len(segs) > max_frames:
        idx = sorted({round(i * (len(segs) - 1) / (max_frames - 1)) for i in range(max_frames)})
        segs = [segs[i] for i in idx]
    assets = []
    adir.mkdir(parents=True, exist_ok=True)
    for s0, s1 in segs:
        t = (s0 + s1) / 2
        out = adir / f"vid_{video.stem[:30]}_{t:06.1f}.jpg"
        if _frame_at(video, t, out):
            assets.append({"file": out, "source": f"{video.name} {t:.1f}초"})
    speech = ""
    if _sh.which("ffmpeg"):
        wav = adir / f"_{video.stem}.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                           capture_output=True)
        if r.returncode == 0 and wav.exists() and wav.stat().st_size > 32000:
            res = hub.transcribe(wav)
            speech = (res or {}).get("text", "") if res else ""
        wav.unlink(missing_ok=True)
    log(f"    동영상 {video.name}: {dur:.0f}초 → 프레임 {len(assets)}장, 말소리 {len(speech)}자")
    return assets, speech


def _web_material(url: str, adir: Path, max_images: int = 6) -> tuple[str, list[dict]]:
    import requests
    from urllib.parse import urljoin
    from PIL import Image
    from ..rag.sources import _Page
    hdr = {"User-Agent": "Mozilla/5.0 (VIDEO_Studio)"}
    r = requests.get(url, headers=hdr, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    pg = _Page()
    pg.feed(r.text)
    text = (pg.title.strip() + "\n\n" + re.sub(r"\n\s*\n+", "\n\n", "".join(pg.texts))).strip()
    assets, seen = [], set()
    adir.mkdir(parents=True, exist_ok=True)
    for src, _alt in pg.imgs:
        full = urljoin(url, src)
        if full in seen or full.startswith("data:") or re.search(r"(logo|icon|sprite|blank|pixel)", full, re.I):
            continue
        seen.add(full)
        try:
            ir = requests.get(full, headers=hdr, timeout=20)
            if ir.status_code != 200 or len(ir.content) < 30_000:
                continue
            out = adir / f"web_{len(assets) + 1:02d}_{re.sub(r'[^A-Za-z0-9]', '', url)[-20:]}.jpg"
            tmp = out.with_suffix(".bin")
            tmp.write_bytes(ir.content)
            with Image.open(tmp) as im:
                if im.width < IMG_MIN_W or im.height < IMG_MIN_H * 0.7:
                    tmp.unlink(missing_ok=True)
                    continue
                im.convert("RGB").save(out, quality=92)
            tmp.unlink(missing_ok=True)
            assets.append({"file": out, "source": f"웹 {url}"})
        except Exception:  # noqa: BLE001 — 사진 하나 실패는 건너뛴다
            continue
        if len(assets) >= max_images:
            break
    log(f"    웹 {url}: 글 {len(text):,}자, 사진 {len(assets)}장")
    return text, assets


def gather(hub: Hub, items: list[str], adir: Path) -> tuple[list[str], list[str], list[dict], list[Path], dict]:
    """입력 폴더·파일·주소 → (쪽 글, 쪽 위치 표시, 사진 후보, PDF 목록, 찾은 목록)."""
    from PIL import Image
    from .. import inputs
    found = inputs.scan(items)
    inputs.log_found(found)
    pages, labels, photos, others = [], [], [], []
    for pdf in found["pdf"]:
        ps, imgs = read_pdf(pdf, adir)
        for i, t in enumerate(ps, 1):
            pages.append(t)
            labels.append(f"{pdf.name} {i}쪽")
        for a in imgs:
            a["file"] = Path(a["file"]).replace(adir / f"{pdf.stem[:30]}_{Path(a['file']).name}")   # replace: 다시 실행해도 덮어씀 (Windows rename은 있으면 실패)
            a["source"] = f"{pdf.name} {a['source'].replace('PDF ', '')}"
        others += imgs
        log(f"    PDF {pdf.name}: {len(ps)}쪽, 사진 {len(imgs)}장")
    for doc in found["doc"]:
        try:
            for loc, t in inputs.doc_pages(doc):
                if t.strip():
                    pages.append(clean(t))
                    labels.append(f"{doc.name} {loc}".strip())
        except Exception as e:  # noqa: BLE001
            log(f"    (문서 {doc.name} 읽기 실패: {e})")
    for img in found["image"]:
        adir.mkdir(parents=True, exist_ok=True)
        dst = adir / f"photo_{img.stem}.jpg"
        with Image.open(img) as im:
            im = im.convert("RGB")
            im.thumbnail((2560, 2560))
            im.save(dst, quality=93)
        photos.append({"file": dst, "source": f"사진 {img.name}"})
    for v in found["video"]:
        a, speech = _video_material(hub, v, adir)
        photos += a                                      # 실제 촬영 영상의 프레임은 실사진과 같은 우선순위
        if speech.strip():
            pages.append(speech)
            labels.append(f"{v.name} 말소리")
    for u in found["url"]:
        try:
            t, a = _web_material(u, adir)
        except Exception as e:  # noqa: BLE001
            log(f"    (웹 {u} 가져오기 실패: {e})")
            continue
        if t:
            pages.append(clean(t))
            labels.append(f"웹 {u}")
        others += a
    return pages, labels, photos + others, found["pdf"], found


# ------------------------------------------------------------------ 전체
def _where(x: dict, labels: list[str]) -> dict:
    try:
        x["where"] = labels[int(x.get("page")) - 1]
    except (TypeError, ValueError, IndexError):
        pass
    return x


def ingest(pdf: Path | None, photos: Path | None, out_dir: Path, hub: Hub | None = None,
           org: str | None = None, inputs: list[str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    adir = out_dir / "assets"
    hub = hub or Hub()
    pages, labels, assets, pdfs, found = [], [], [], [], None
    if inputs:
        pages, labels, assets, pdfs, found = gather(hub, inputs, adir)
    if pdf:
        shutil.copy(pdf, out_dir / "brochure.pdf")
        ps, imgs = read_pdf(pdf, adir)
        pages += ps
        labels += [f"{pdf.name} {i}쪽" for i in range(1, len(ps) + 1)]
        assets += imgs
        pdfs.append(pdf)
        log(f"[promo] PDF {len(ps)}쪽, 사진 {len(imgs)}장")
    if photos:
        for p in sorted(Path(photos).iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                dst = adir / f"photo_{p.stem}.jpg"
                adir.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((2560, 2560))
                    im.save(dst, quality=93)
                assets.insert(0, {"file": dst, "source": f"사진 {p.name}"})   # 직접 준 실사진을 우선
    if not pages and not assets:
        raise SystemExit("읽을 수 있는 자료가 없습니다 (PDF·사진·동영상·문서·웹 주소).")
    (out_dir / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "pages_index.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[promo] 글 {len(pages)}쪽(문서·슬라이드·말소리·웹 포함), 사진 후보 {len(assets)}장")
    raw = extract(hub, pages, labels, out_dir / "debug") if pages else {"facts": [], "messages": [], "issues": []}
    ver = verify(raw, pages)
    issues = [{"page": i.get("page"), "text": i.get("text", ""), "by": "LLM"} for i in raw["issues"]]
    issues += [{**i, "by": "규칙"} for i in rule_issues([f for f in ver["facts"] if f["status"] == "ok"], pages)]
    for x in ver["facts"] + ver["messages"] + issues:
        _where(x, labels)
    log("[promo] 사진 분류")
    catalog = classify_assets(hub, assets)
    contacts = find_contacts(pages)
    tagline = next((m for m in ver["messages"] if m["tag"] == "tagline" and m["status"] == "ok"), None)
    org_name = org or next((m["text"] for m in ver["messages"] if m["tag"] == "org" and m["status"] == "ok"), "")
    main_pdf = max(pdfs, key=lambda q: q.stat().st_size) if pdfs else None     # 색은 가장 큰 PDF(본 브로셔)에서
    brief = {
        "org": {"name": org_name, "tagline": tagline["text"] if tagline else ""},
        "brand": brand_colors(main_pdf) if main_pdf else dict(kb.DEFAULT_BRAND),
        "facts": ver["facts"], "messages": ver["messages"],
        "cta": {"url": contacts["urls"][0] if contacts["urls"] else "", "phone": contacts["phones"][0] if contacts["phones"] else "",
                "email": contacts["emails"][0] if contacts["emails"] else "", "found": contacts},
        "made_with": "",
        "assets": catalog, "issues": issues,
        "inputs": ({k: [str(x) for x in v] for k, v in found.items() if v} if found else
                   {"pdf": [str(pdf)] if pdf else [], "photos": [str(photos)] if photos else []}),
    }
    (out_dir / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for f in ver["facts"] if f["status"] == "ok")
    log(f"[promo] 사실 {n_ok}/{len(ver['facts'])}개 원문 확인, 문구 {sum(1 for m in ver['messages'] if m['status'] == 'ok')}개, "
        f"사진 {sum(1 for a in catalog if a['usable'])}/{len(catalog)}장 사용 가능, 확인할 점 {len(issues)}개")
    log(f"[promo] → {out_dir / 'brief.json'}  (확인 후 promo plan)")
    return out_dir
