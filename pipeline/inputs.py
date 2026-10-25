"""입력 폴더 하나(또는 파일·주소 여러 개)를 받아, 안에 있는 여러 형식의 파일을 종류별로 읽는다.

  PDF                    쪽별 글 + 큰 사진 (브로셔·리플렛 여러 개 가능)
  사진 .jpg .png .webp   실사진으로 우선 사용
  동영상 .mp4 .mov …     장면 전환마다 대표 프레임(사진 후보) + 말소리 받아쓰기(글)
  문서 .docx .pptx .xlsx .hwpx .txt .md .csv .json   글
  웹 주소                .url(인터넷 바로가기)·urls.txt·links.txt 안의 주소 → 본문 글 + 큰 사진
  .hwp(구형 한글)        읽을 수 없어 건너뜀 → 한글에서 .hwpx 또는 PDF로 저장해서 넣어 주세요

홍보 모드(promo ingest --input)와 장편 모드(long --input → 지식 저장소)가 함께 쓴다.
"""

from __future__ import annotations

import html
import json
import re
import zipfile
from pathlib import Path

from .utils import log

PDF_EXT = {".pdf"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
DOC_EXT = {".docx", ".pptx", ".xlsx", ".hwpx"}
TXT_EXT = {".txt", ".md", ".csv", ".json"}
URL_LIST_NAMES = {"urls.txt", "links.txt", "url.txt", "링크.txt", "주소.txt"}
URL_LINE = re.compile(r"^\s*(https?://\S+)\s*$")


# ------------------------------------------------------------------ 문서 글 읽기
def _xml_text(xml: str, para_tags: tuple[str, ...]) -> str:
    for t in para_tags:
        xml = re.sub(rf"</{t}>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return re.sub(r"\n\s*\n+", "\n\n", html.unescape(xml)).strip()


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return _xml_text(z.read("word/document.xml").decode("utf-8", "ignore"), ("w:p",))


def pptx_text(path: Path) -> list[str]:
    """슬라이드마다 글 하나 (슬라이드 번호순)."""
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                       key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return [_xml_text(z.read(n).decode("utf-8", "ignore"), ("a:p",)) for n in names]


def xlsx_text(path: Path) -> str:
    """시트의 글자(공유 문자열)와 숫자를 행 단위로."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            xml = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
            shared = [html.unescape(re.sub(r"<[^>]+>", "", s)) for s in re.findall(r"<si>(.*?)</si>", xml, re.S)]
        rows = []
        for n in sorted(x for x in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", x)):
            xml = z.read(n).decode("utf-8", "ignore")
            for row in re.findall(r"<row[^>]*>(.*?)</row>", xml, re.S):
                cells = []
                for attrs, val in re.findall(r"<c([^>]*)>.*?<v>(.*?)</v>", row, re.S):
                    if 't="s"' in attrs and val.isdigit() and int(val) < len(shared):
                        cells.append(shared[int(val)])
                    else:
                        cells.append(val)
                if cells:
                    rows.append(" | ".join(cells))
        return "\n".join(rows)


def hwpx_text(path: Path) -> list[str]:
    """한글 .hwpx (구역마다 글 하나)."""
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n)),
                       key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return [_xml_text(z.read(n).decode("utf-8", "ignore"), ("hp:p",)) for n in names]


def doc_pages(path: Path) -> list[tuple[str, str]]:
    """문서 하나 → [(위치 표시, 글)]. 슬라이드·구역은 따로 나눈다."""
    ext = path.suffix.lower()
    if ext == ".docx":
        return [("", docx_text(path))]
    if ext == ".pptx":
        return [(f"슬라이드 {i}", t) for i, t in enumerate(pptx_text(path), 1)]
    if ext == ".xlsx":
        return [("", xlsx_text(path))]
    if ext == ".hwpx":
        return [(f"구역 {i}", t) for i, t in enumerate(hwpx_text(path), 1)]
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".json":
        try:
            raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=1)
        except ValueError:
            pass
    return [("", raw)]


# ------------------------------------------------------------------ 웹 주소
def urls_in(path: Path) -> list[str]:
    """.url 바로가기, 또는 urls.txt처럼 주소만 한 줄씩 적은 파일."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".url":
        return re.findall(r"^URL=(https?://\S+)", txt, re.M)
    return [m.group(1) for ln in txt.splitlines() if (m := URL_LINE.match(ln))]


def is_url_list(path: Path) -> bool:
    if path.suffix.lower() == ".url":
        return True
    if path.name.lower() in URL_LIST_NAMES:
        return True
    if path.suffix.lower() == ".txt":                  # 주소만 적힌 txt도 주소 목록으로 본다
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        return bool(lines) and all(URL_LINE.match(ln) for ln in lines)
    return False


# ------------------------------------------------------------------ 폴더 훑기
def scan(items: list[str | Path]) -> dict[str, list]:
    """파일·폴더·주소 → 종류별 목록. 폴더는 하위 폴더까지."""
    out: dict[str, list] = {"pdf": [], "image": [], "video": [], "doc": [], "url": [], "skipped": []}

    def one(p: Path) -> None:
        ext = p.suffix.lower()
        if p.name.startswith(("~$", ".")):
            return
        if ext in PDF_EXT:
            out["pdf"].append(p)
        elif ext in IMG_EXT:
            out["image"].append(p)
        elif ext in VID_EXT:
            out["video"].append(p)
        elif ext == ".url" or (ext == ".txt" and is_url_list(p)):
            out["url"] += urls_in(p)
        elif ext in DOC_EXT | TXT_EXT:
            out["doc"].append(p)
        else:
            out["skipped"].append(p)

    for it in items:
        s = str(it)
        if re.match(r"^https?://", s):
            out["url"].append(s)
            continue
        p = Path(s)
        if not p.exists():
            raise SystemExit(f"입력이 없습니다: {p}")
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    one(f)
        else:
            one(p)
    out["url"] = list(dict.fromkeys(out["url"]))
    return out


def describe(found: dict[str, list]) -> str:
    names = {"pdf": "PDF", "image": "사진", "video": "동영상", "doc": "문서", "url": "웹 주소"}
    s = ", ".join(f"{names[k]} {len(v)}개" for k, v in found.items() if k in names and v)
    if found["skipped"]:
        hwp = [p for p in found["skipped"] if p.suffix.lower() == ".hwp"]
        other = [p for p in found["skipped"] if p.suffix.lower() != ".hwp"]
        if hwp:
            s += f"  (.hwp {len(hwp)}개는 읽을 수 없어 건너뜀 → 한글에서 .hwpx나 PDF로 저장해서 넣어 주세요)"
        if other:
            s += f"  (지원하지 않는 형식 {len(other)}개 건너뜀: {', '.join(p.name for p in other[:5])})"
    return s or "읽을 수 있는 파일이 없습니다"


def log_found(found: dict[str, list]) -> None:
    log(f"[입력] {describe(found)}")
    for k in ("pdf", "image", "video", "doc"):
        for p in found[k]:
            log(f"    - {p.name}")
    for u in found["url"]:
        log(f"    - {u}")
