"""홍보 그래픽 레이어 (Pillow). 숫자·글자·로고·QR은 생성 모델이 아니라 여기서 그린다.

draw(canvas, beat, ctx, t, sec) → RGBA 오버레이 한 장. t는 비트 안의 경과 초.
종류: title · bullets · lower_third · counter(숫자 카운트업) · steps · end_card · 자막 · 'AI 생성 장면' 표시
가로(16:9)·세로(9:16) 모두 같은 함수가 비율에 맞춰 배치한다.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = {
    True: [("C:/Windows/Fonts/malgunbd.ttf", 0), ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 6),
           ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1), ("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 0)],
    False: [("C:/Windows/Fonts/malgun.ttf", 0), ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1), ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 0)],
}


@lru_cache(None)
def font(size: int, bold: bool = False, custom: str = "") -> ImageFont.FreeTypeFont:
    if custom and Path(custom).exists():
        return ImageFont.truetype(custom, size)
    for path, idx in FONT_CANDIDATES[bold] + FONT_CANDIDATES[not bold]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=idx)
            except OSError:
                continue
    return ImageFont.load_default()


def rgb(h: str, a: int = 255) -> tuple:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def appear(t: float, start: float, dur: float = 0.6) -> float:
    return ease((t - start) / dur)


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if d.textlength(nxt, font=f) <= width or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:                                   # 한 단어가 너무 길면 글자 단위로 자른다
        while d.textlength(ln, font=f) > width and len(ln) > 1:
            k = len(ln)
            while k > 1 and d.textlength(ln[:k], font=f) > width:
                k -= 1
            out.append(ln[:k])
            ln = ln[k:]
        out.append(ln)
    return out


class Ctx:
    """브랜드·사실·문구·연락처를 그리기 함수에 넘기는 묶음."""

    def __init__(self, brief: dict, canvas: tuple[int, int], project: Path):
        self.b = brief["brand"]
        self.org = brief["org"]
        self.cta = brief["cta"]
        self.made_with = brief.get("made_with", "")
        self.ai_label = brief["brand"].get("ai_label", "AI 생성 장면")
        self.items = {m["id"]: m for m in brief["messages"]} | {f["id"]: f for f in brief["facts"]}
        self.W, self.H = canvas
        self.vertical = self.H > self.W
        self.u = min(self.W, self.H) / 720            # 기준 단위 (720p 기준 1.0)
        self.project = project
        self.logo = None
        lp = brief["brand"].get("logo")
        if lp and (project / lp).exists():
            self.logo = Image.open(project / lp).convert("RGBA")

    def F(self, px: float, bold: bool = False):
        return font(max(10, int(px * self.u)), bold, self.b.get("font_bold" if bold else "font", ""))


# ------------------------------------------------------------------ 배경 (그래픽 비트)
def gradient_bg(ctx: Ctx, t: float) -> Image.Image:
    import numpy as np
    W, H = ctx.W, ctx.H
    c1, c2, c3 = (np.array(rgb(ctx.b[k])[:3], "float32") for k in ("dark", "primary", "secondary"))
    y, x = np.mgrid[0:H, 0:W].astype("float32")
    g = (x / W * 0.6 + y / H * 0.4)
    img = c1 * (1 - g[..., None]) + c2 * g[..., None]
    cx, cy = W * (0.7 + 0.08 * math.sin(t * 0.6)), H * (0.35 + 0.06 * math.cos(t * 0.5))   # 천천히 움직이는 빛
    glow = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * (0.35 * max(W, H)) ** 2)))
    img = img * (1 - 0.45 * glow[..., None]) + c3 * 0.45 * glow[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"), "RGB")


# ------------------------------------------------------------------ 요소
def _panel(d, box, ctx, alpha=150, r=None):
    d.rounded_rectangle(box, radius=int((r or 14) * ctx.u), fill=rgb(ctx.b["dark"], alpha))


def _text(item: dict, short: bool = False) -> str:
    if item.get("type") == "fact" or "display" in item:
        return f"{item.get('label', '')} {item['display']}{item.get('unit', '')}"
    return (item.get("short") if short and item.get("short") else item.get("text", "")) or ""


def title(d, ctx: Ctx, items: list[dict], t: float):
    if not items:
        return
    a = appear(t, 0.2)
    txt = _text(items[0])
    f = ctx.F(46 if not ctx.vertical else 44, True)
    lines = wrap(d, txt, f, int(ctx.W * (0.8 if ctx.vertical else 0.62)))
    lh = f.size * 1.3
    x = int(ctx.W * 0.07)
    y0 = int(ctx.H * (0.38 if ctx.vertical else 0.62) - len(lines) * lh / 2 + (1 - a) * 30 * ctx.u)
    d.rectangle([x, y0 - 18 * ctx.u, x + int(90 * ctx.u * a), y0 - 12 * ctx.u], fill=rgb(ctx.b["accent"]))
    for i, ln in enumerate(lines):
        d.text((x + 2, y0 + i * lh + 2), ln, font=f, fill=(0, 0, 0, int(120 * a)))
        d.text((x, y0 + i * lh), ln, font=f, fill=(255, 255, 255, int(255 * a)))


def bullets(d, ctx: Ctx, items: list[dict], t: float, sec: float):
    if not items:
        return
    f = ctx.F(30 if not ctx.vertical else 32, True)
    w = int(ctx.W * (0.86 if ctx.vertical else 0.5))
    x = int(ctx.W * 0.07)
    lh = f.size * 1.35
    blocks = [wrap(d, _text(it), f, w - int(70 * ctx.u)) for it in items]
    total = sum(len(b) * lh + 24 * ctx.u for b in blocks)
    y = int(ctx.H * (0.30 if ctx.vertical else 0.5) - total / 2)
    step = max(0.6, (sec - 1.5) / max(1, len(items)))
    for i, lines in enumerate(blocks):
        a = appear(t, 0.3 + i * step)
        if a <= 0:
            break
        h = len(lines) * lh + 18 * ctx.u
        dx = int((1 - a) * -40 * ctx.u)
        _panel(d, [x + dx, y, x + dx + w, y + h], ctx, int(165 * a))
        d.ellipse([x + dx + 20 * ctx.u, y + h / 2 - 7 * ctx.u, x + dx + 34 * ctx.u, y + h / 2 + 7 * ctx.u], fill=rgb(ctx.b["accent"], int(255 * a)))
        for k, ln in enumerate(lines):
            d.text((x + dx + 52 * ctx.u, y + 9 * ctx.u + k * lh), ln, font=f, fill=(255, 255, 255, int(255 * a)))
        y += h + 14 * ctx.u


def lower_third(d, ctx: Ctx, items: list[dict], t: float):
    a = appear(t, 0.4)
    l1 = ctx.org.get("name") or (_text(items[0]) if items else "")
    l2 = _text(items[1]) if len(items) > 1 else (_text(items[0]) if items and ctx.org.get("name") else "")
    f1, f2 = ctx.F(34, True), ctx.F(22)
    w = int(max(d.textlength(l1, font=f1), d.textlength(l2, font=f2)) + 70 * ctx.u)
    w = min(w, int(ctx.W * 0.9))
    x = int(ctx.W * 0.06 - (1 - a) * w)
    y = int(ctx.H * (0.62 if ctx.vertical else 0.6))
    h = int(110 * ctx.u if l2 else 70 * ctx.u)
    _panel(d, [x, y, x + w, y + h], ctx, int(200 * a), r=8)
    d.rectangle([x, y, x + int(8 * ctx.u), y + h], fill=rgb(ctx.b["accent"]))
    d.text((x + 28 * ctx.u, y + 14 * ctx.u), l1, font=f1, fill=(255, 255, 255, 255))
    if l2:
        d.text((x + 28 * ctx.u, y + 64 * ctx.u), l2, font=f2, fill=(220, 230, 255, 255))


_NUM = re.compile(r"^(\D*)([\d,]+(?:\.\d+)?)(.*)$")


def counting(display: str, p: float) -> str:
    """'2,204' → 진행률 p만큼 센 숫자 (쉼표·접미사 '+' 유지)."""
    m = _NUM.match(display.strip())
    if not m:
        return display
    pre, num, post = m.groups()
    val = float(num.replace(",", ""))
    dec = len(num.split(".")[1]) if "." in num else 0
    cur = val * ease(p)
    s = f"{cur:,.{dec}f}" if "," in num else f"{cur:.{dec}f}"
    return pre + s + post


def counter(d, ctx: Ctx, items: list[dict], t: float, sec: float):
    facts = [it for it in items if "display" in it][:4]
    if not facts:
        return
    n = len(facts)
    cols = 2 if (ctx.vertical and n > 1) else n
    rows = math.ceil(n / cols)
    cw = int(ctx.W * (0.42 if ctx.vertical else min(0.21, 0.86 / n)))
    ch = int(150 * ctx.u)
    gap = int(16 * ctx.u)
    tw = cols * cw + (cols - 1) * gap
    x0 = (ctx.W - tw) // 2
    y0 = int(ctx.H * (0.36 if ctx.vertical else 0.56) - (rows * ch + (rows - 1) * gap) / 2)
    fn, fu, fl = ctx.F(52, True), ctx.F(22, True), ctx.F(20)
    for i, f in enumerate(facts):
        r, c = divmod(i, cols)
        x, y = x0 + c * (cw + gap), y0 + r * (ch + gap)
        a = appear(t, 0.2 + i * 0.25, 0.5)
        if a <= 0:
            continue
        _panel(d, [x, y, x + cw, y + ch], ctx, int(175 * a))
        d.rectangle([x, y, x + cw, y + int(5 * ctx.u)], fill=rgb(ctx.b["accent"], int(255 * a)))
        num = counting(f["display"], (t - 0.2 - i * 0.25) / 1.4)
        nw = d.textlength(num, font=fn)
        uw = d.textlength(f.get("unit", ""), font=fu)
        sx = x + (cw - nw - uw - 6 * ctx.u) / 2
        d.text((sx, y + 22 * ctx.u), num, font=fn, fill=(255, 255, 255, int(255 * a)))
        d.text((sx + nw + 6 * ctx.u, y + 22 * ctx.u + fn.size - fu.size - 4 * ctx.u), f.get("unit", ""), font=fu, fill=rgb(ctx.b["accent"], int(255 * a)))
        lab = wrap(d, f.get("label", ""), fl, cw - int(20 * ctx.u))[:2]
        for k, ln in enumerate(lab):
            lw = d.textlength(ln, font=fl)
            d.text((x + (cw - lw) / 2, y + 95 * ctx.u + k * fl.size * 1.2), ln, font=fl, fill=(215, 225, 255, int(255 * a)))


def steps(d, ctx: Ctx, items: list[dict], t: float, sec: float):
    labels = [(_text(it, short=True) or _text(it)) for it in items][:7]
    if not labels:
        return
    f = ctx.F(22, True)
    per_row = 2 if ctx.vertical else min(len(labels), 4)
    rows = math.ceil(len(labels) / per_row)
    gap = int(14 * ctx.u)
    cw = int((ctx.W * 0.88 - gap * (per_row - 1)) / per_row)
    chh = int(64 * ctx.u)
    x0 = int(ctx.W * 0.06)
    y0 = int(ctx.H * (0.30 if ctx.vertical else 0.56) - (rows * chh + (rows - 1) * gap) / 2)
    active = min(len(labels) - 1, int(max(0.0, t - 0.5) / max(0.4, (sec - 1.0) / len(labels))))
    for i, lb in enumerate(labels):
        r, c = divmod(i, per_row)
        x, y = x0 + c * (cw + gap), y0 + r * (chh + gap)
        a = appear(t, 0.15 * i, 0.4)
        on = i <= active
        fill = rgb(ctx.b["primary"] if on else ctx.b["dark"], int((230 if on else 150) * a))
        d.rounded_rectangle([x, y, x + cw, y + chh], radius=int(chh / 2), fill=fill,
                            outline=rgb(ctx.b["accent"], int(255 * a)) if i == active else None, width=max(1, int(3 * ctx.u)))
        num = f"{i + 1}"
        d.ellipse([x + 10 * ctx.u, y + 12 * ctx.u, x + 50 * ctx.u, y + 52 * ctx.u], fill=rgb(ctx.b["accent"], int(255 * a)))
        d.text((x + 30 * ctx.u - d.textlength(num, font=f) / 2, y + 17 * ctx.u), num, font=f, fill=rgb(ctx.b["dark"], int(255 * a)))
        ln = wrap(d, lb, f, cw - int(70 * ctx.u))[0]
        d.text((x + 62 * ctx.u, y + 17 * ctx.u), ln, font=f, fill=(255, 255, 255, int(255 * a)))


def _qr(url: str, size: int) -> Image.Image | None:
    try:
        import qrcode
    except ImportError:
        return None
    q = qrcode.QRCode(border=2, box_size=10)
    q.add_data(url if url.startswith("http") else "https://" + url)
    q.make(fit=True)
    return q.make_image(fill_color="black", back_color="white").convert("RGBA").resize((size, size), Image.NEAREST)


def end_card(img: Image.Image, d, ctx: Ctx, t: float):
    a = appear(t, 0.1, 0.8)
    W, H, u = ctx.W, ctx.H, ctx.u
    x = int(W * 0.08)
    y = int(H * (0.18 if ctx.vertical else 0.2))
    if ctx.logo:
        lg = ctx.logo.copy()
        lg.thumbnail((int(W * 0.35), int(90 * u)))
        img.alpha_composite(lg, (x, y))
        y += lg.height + int(24 * u)
    name = ctx.org.get("name", "")
    fN = ctx.F(48 if not ctx.vertical else 44, True)
    for ln in wrap(d, name, fN, int(W * 0.84 if ctx.vertical else W * 0.55)):
        d.text((x, y), ln, font=fN, fill=(255, 255, 255, int(255 * a)))
        y += int(fN.size * 1.25)
    tag = ctx.org.get("tagline", "")
    if tag:
        fT = ctx.F(26)
        for ln in wrap(d, tag, fT, int(W * 0.84 if ctx.vertical else W * 0.55)):
            d.text((x, y + 6 * u), ln, font=fT, fill=(220, 230, 255, int(255 * a)))
            y += int(fT.size * 1.35)
    y += int(30 * u)
    a2 = appear(t, 0.8, 0.6)
    fU, fP = ctx.F(36, True), ctx.F(24)
    if ctx.cta.get("url"):
        d.text((x, y), ctx.cta["url"], font=fU, fill=rgb(ctx.b["accent"], int(255 * a2)))
        y += int(fU.size * 1.5)
    for k in ("phone", "email"):
        if ctx.cta.get(k):
            d.text((x, y), ctx.cta[k], font=fP, fill=(255, 255, 255, int(255 * a2)))
            y += int(fP.size * 1.5)
    if ctx.cta.get("url"):
        qs = int(170 * u)
        q = _qr(ctx.cta["url"], qs)
        if q:
            qx = int(W * 0.08) if ctx.vertical else int(W * 0.78)
            qy = int(y + 20 * u) if ctx.vertical else int(H * 0.5 - qs / 2)
            q.putalpha(int(255 * a2))
            img.alpha_composite(q, (qx, qy))
    if ctx.made_with:
        fM = ctx.F(18)
        mw = d.textlength(ctx.made_with, font=fM)
        d.text(((W - mw) / 2, H - 46 * u), ctx.made_with, font=fM, fill=(200, 210, 240, int(230 * appear(t, 1.4))))


def subtitle(d, ctx: Ctx, text: str, t: float, sec: float):
    if not text:
        return
    f = ctx.F(26 if not ctx.vertical else 28, True)
    lines = wrap(d, text, f, int(ctx.W * 0.82))
    per = 2                                            # 두 줄씩 나눠 시간에 맞게 넘긴다
    pages = [lines[i:i + per] for i in range(0, len(lines), per)]
    k = min(len(pages) - 1, int(t / max(0.1, sec / len(pages))))
    pg = pages[k]
    lh = f.size * 1.3
    h = len(pg) * lh + 16 * ctx.u
    y = ctx.H - h - (110 if ctx.vertical else 34) * ctx.u
    for i, ln in enumerate(pg):
        w = d.textlength(ln, font=f)
        x = (ctx.W - w) / 2
        d.rounded_rectangle([x - 14 * ctx.u, y + i * lh, x + w + 14 * ctx.u, y + (i + 1) * lh + 6 * ctx.u], radius=int(6 * ctx.u), fill=(0, 0, 0, 150))
        d.text((x, y + i * lh + 4 * ctx.u), ln, font=f, fill=(255, 255, 255, 255))


def ai_badge(d, ctx: Ctx):
    if not ctx.ai_label:
        return
    f = ctx.F(15)
    w = d.textlength(ctx.ai_label, font=f)
    x, y = ctx.W - w - 34 * ctx.u, 22 * ctx.u
    d.rounded_rectangle([x - 10 * ctx.u, y - 5 * ctx.u, x + w + 10 * ctx.u, y + f.size + 7 * ctx.u], radius=int(10 * ctx.u), fill=(0, 0, 0, 110))
    d.text((x, y), ctx.ai_label, font=f, fill=(255, 255, 255, 220))


def draw(base: Image.Image, beat: dict, ctx: Ctx, t: float, subtitles: bool, ai_scene: bool) -> Image.Image:
    img = base.convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    sec = beat["seconds"]
    items = [ctx.items[i] | ({"type": "fact"} if i.startswith("f") else {}) for i in beat["overlay"]["items"] if i in ctx.items]
    kind = beat["overlay"]["kind"]
    if kind != "end_card":                           # 글자가 잘 읽히도록 아래쪽을 살짝 어둡게
        shade = Image.linear_gradient("L").resize(img.size).point(lambda v: int(v * 0.55))
        img.alpha_composite(Image.merge("RGBA", (*[Image.new("L", img.size, 0)] * 3, shade)))
    {"title": lambda: title(d, ctx, items, t), "bullets": lambda: bullets(d, ctx, items, t, sec),
     "lower_third": lambda: lower_third(d, ctx, items, t), "counter": lambda: counter(d, ctx, items, t, sec),
     "steps": lambda: steps(d, ctx, items, t, sec), "end_card": lambda: end_card(ov, d, ctx, t)}.get(kind, lambda: None)()
    if subtitles:
        subtitle(d, ctx, beat.get("subtitle", beat.get("narration", "")), t, sec)
    if ai_scene:
        ai_badge(d, ctx)
    img.alpha_composite(ov)
    return img.convert("RGB")
