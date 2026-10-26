from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


VLM_UNIT = 32                 # Qwen3-VL: 패치 16px × 병합 2 → 32×32px가 토큰 1개
VLM_MIN_PIXELS = 65_536       # Qwen3-VL 전처리 shortest_edge (이보다 작으면 키운다)
VLM_MAX_PIXELS = 16_777_216   # Qwen3-VL 전처리 longest_edge (사진 1장 최대 약 1,670만 픽셀 = 1만 6천 토큰)


def fit_pixels(w: int, h: int, max_pixels: int, min_pixels: int = VLM_MIN_PIXELS, unit: int = VLM_UNIT) -> tuple[int, int]:
    """비율을 지키면서 픽셀 수를 [min_pixels, max_pixels] 안으로, 두 변을 unit의 배수로 (Qwen-VL smart_resize와 같은 방식).
    원본보다 크게 키우지는 않는다(최소 픽셀보다 작을 때만 예외)."""
    import math
    max_pixels = min(max_pixels, VLM_MAX_PIXELS)
    area = max(1, w * h)
    s = 1.0
    if area > max_pixels:
        s = math.sqrt(max_pixels / area)
    elif area < min_pixels:
        s = math.sqrt(min_pixels / area)
    nw = max(unit, int(w * s // unit) * unit)
    nh = max(unit, int(h * s // unit) * unit)
    return nw, nh


def vlm_tokens(w: int, h: int) -> int:
    return (w // VLM_UNIT) * (h // VLM_UNIT) + 2


def image_to_data_url(path: Path, max_side: int | None = None, max_pixels: int | None = None) -> str:
    """VLM·I2V 입력용 data URL. max_pixels가 있으면 비율을 지키며 픽셀 수로 맞추고(세로 사진도 손해 없이),
    없으면 긴 변을 max_side로 줄인다. JPEG로 보낸다."""
    raw = Path(path).read_bytes()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max_pixels:
            nw, nh = fit_pixels(im.width, im.height, max_pixels)
            if (nw, nh) != im.size:
                im = im.resize((nw, nh), Image.LANCZOS)
        elif max(im.size) > (max_side or 1024):
            im.thumbnail((max_side or 1024, max_side or 1024))
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
