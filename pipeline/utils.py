from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def image_to_data_url(path: Path, max_side: int = 1024) -> str:
    """VLM·I2V 입력용 data URL. 긴 변을 max_side로 줄이고 JPEG로 보낸다."""
    raw = Path(path).read_bytes()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        raw, mime = buf.getvalue(), "image/jpeg"
    except Exception:
        pass
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def extract_json(text: str):
    """LLM 응답에서 JSON을 꺼낸다 (```json 블록, 앞뒤 설명문, <think> 블록 허용)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1], strict=False)   # 문자열 안의 제어 문자(PDF에서 옮겨 온 \x07 등) 허용
            except json.JSONDecodeError:
                continue
    raise ValueError(f"JSON 파싱 실패: {text[:300]}")


def korean_font(size: int):
    from PIL import ImageFont
    for fp in ("C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
               "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


if sys.platform == "win32":  # 윈도우 콘솔 한글 출력
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
