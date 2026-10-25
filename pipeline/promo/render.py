"""storyboard_<형식>.json → 홍보영상.

비트마다
  화면  사진 비트: (본 렌더) Hub I2V로 레시피대로 움직임 → 홍보 검사(글자 생김·건물 변형·사람 추가) → 미달이면 1회 재생성
                    (초안 --draft) GPU 없이 레시피의 켄 번스(줌·이동)로 움직임 → 몇 분 안에 전체 미리보기
        그래픽 비트: 브랜드 색 그라데이션 배경
  그래픽 숫자 카운트업·문구·단계·엔딩카드(QR)·자막을 코드로 그림 (생성 모델은 글자를 그리지 않음)
  소리   --narration 이면 Hub TTS, --bgm 이면 배경음악
AI 영상은 cache/에 저장되므로 문구·숫자를 고쳐 다시 렌더해도 GPU를 다시 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image

from .. import cvio, prompts as P, videoio
from ..agents import _ask_json, _j, ask_clip
from ..hub import Hub, HubError, video_spec
from ..orchestrator import score, severe_artifacts as _severe
from ..utils import log
from . import graphics as G, kb

STRETCH = 1.5          # AI 클립을 최대 1.5배까지 느리게 늘려 샷 수를 줄인다
MIN_SCORE = 0.6


# ------------------------------------------------------------------ 프레임 소스
def _ease_io(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _fit(frame, W: int, H: int):
    """프레임(BGR)을 캔버스에 맞춘다. 비율 차이가 크면 흐린 배경 위에 원본 비율로 얹는다 (세로 숏폼용)."""
    import cv2
    h, w = frame.shape[:2]
    if abs(w / h - W / H) < 0.6:
        s = max(W / w, H / h)
        r = cv2.resize(frame, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        y, x = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
        return r[y:y + H, x:x + W]
    s = max(W / w, H / h)
    bg = cv2.resize(frame, (math.ceil(w * s), math.ceil(h * s)))
    y, x = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
    bg = cv2.GaussianBlur(bg[y:y + H, x:x + W], (0, 0), 25) * 0.55
    s2 = min(W / w, H / h)
    fg = cv2.resize(frame, (int(w * s2), int(h * s2)), interpolation=cv2.INTER_CUBIC)
    oy, ox = int(H * 0.42 - fg.shape[0] / 2), (W - fg.shape[1]) // 2
    oy = max(0, min(H - fg.shape[0], oy))
    bg[oy:oy + fg.shape[0], ox:ox + fg.shape[1]] = fg
    return bg.astype("uint8")


class KenBurns:
    """정지 사진을 레시피의 줌·이동 값대로 부드럽게 움직인다 (서브픽셀 warpAffine, 떨림 없음)."""

    def __init__(self, img: Path, kbp: dict, W: int, H: int):
        import cv2
        src = cvio.imread(str(img))
        self.W, self.H = W, H
        h, w = src.shape[:2]
        self.portrait_fill = abs(w / h - W / H) >= 0.6
        if self.portrait_fill:                                 # 세로 캔버스: 가로 사진 영역만 움직이고 나중에 _fit
            self.vw, self.vh = W, int(W * h / w)
        else:
            self.vw, self.vh = W, H
        s = 1.35 * max(self.vw / w, self.vh / h)
        self.src = cv2.resize(src, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_AREA)
        self.z0, self.z1 = kbp.get("zoom", [1.0, 1.1])
        self.px, self.py = kbp.get("pan", [0.0, 0.0])

    def frame(self, p: float):
        import cv2
        import numpy as np
        e = _ease_io(p)
        sh, sw = self.src.shape[:2]
        base = min(sw / self.vw, sh / self.vh) / 1.2          # 줌 1.0에서도 여유를 남겨 이동(pan)할 공간 확보
        z = self.z0 + (self.z1 - self.z0) * e
        cw, ch = self.vw * base / z, self.vh * base / z
        cx = sw / 2 + self.px * sw * (e - 0.5)
        cy = sh / 2 + self.py * sh * (e - 0.5)
        cx = min(max(cx, cw / 2), sw - cw / 2)
        cy = min(max(cy, ch / 2), sh - ch / 2)
        s = self.vw / cw
        M = np.float32([[s, 0, -(cx - cw / 2) * s], [0, s, -(cy - ch / 2) * s]])
        f = cv2.warpAffine(self.src, M, (self.vw, self.vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return _fit(f, self.W, self.H) if self.portrait_fill else f


class ClipSource:
    """AI 영상 클립을 비트 길이에 맞춰(필요하면 느리게) 캔버스 프레임으로."""

    def __init__(self, clip: Path, W: int, H: int):
        import cv2
        cap = cvio.VideoCapture(str(clip))
        self.frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            self.frames.append(f)
        cap.release()
        self.W, self.H = W, H

    def frame(self, p: float):
        i = min(len(self.frames) - 1, int(p * (len(self.frames) - 1) + 0.5))
        return _fit(self.frames[i], self.W, self.H)


# ------------------------------------------------------------------ AI 클립 (본 렌더)
def _sha(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:16]


def ai_clip(hub: Hub, project: Path, beat: dict, brief: dict) -> tuple[Path | None, dict]:
    rec = kb.recipes()["recipes"].get(beat["recipe"]) or kb.recipes()["recipes"][kb.recipes()["fallback"]]
    model = hub.models().video
    spec = video_spec(model)
    img = project / "assets" / beat["asset_file"]
    size = videoio.choose_size(img, spec["sizes"])
    n = max(1, math.ceil(beat["seconds"] / (spec["max_seconds"] * STRETCH)))
    shot_s = min(spec["max_seconds"], beat["seconds"] / n)
    prompt = rec["prompt"].replace("{subject}", beat.get("asset_subject") or "the scene")
    neg = kb.recipes()["common_negative"]
    key = _sha(img.read_bytes(), prompt, neg, n, shot_s, model, size)
    cdir = project / "cache"
    out = cdir / f"ai_{key}.mp4"
    meta_p = out.with_suffix(".json")
    if out.exists() and meta_p.exists():
        log(f"    (캐시 사용: {out.name})")
        return out, json.loads(meta_p.read_text(encoding="utf-8"))
    cdir.mkdir(parents=True, exist_ok=True)
    first = videoio.prepare_image(img, size, cdir / f"first_{key}_0.jpg", "crop")
    qs = []
    for j, q in enumerate(rec["checks"]):
        is_text = "text" in q.lower() or "letter" in q.lower()
        qs.append({"id": f"k{j + 1}", "kind": "preserve" if is_text else "goal", "weight": 2.0 if is_text else 1.5, "question": q})
    qs.append({"id": "kar", "kind": "quality", "weight": 1.0, "question": "Free of visible artifacts?"})
    parts, shots = [], []
    for k in range(n):
        best = None
        p_k = prompt + (" The motion continues smoothly from the previous moment." if k else "")
        for attempt in range(2):
            p_try = p_k + (" Keep every detail of the first frame unchanged; only the camera and the light move." if attempt else "")
            log(f"    [AI] {beat['role']} 샷 {k + 1}/{n} 시도 {attempt + 1} ({beat['recipe']}, {shot_s:.1f}초)")
            res = hub.generate_video(p_try, cdir / f"ai_{key}_s{k}_a{attempt}", first, shot_s, size, negative=neg)
            vid = Path(res["file"])
            frames = videoio.sample_frames(vid, cdir / "frames", k=4)
            metrics = videoio.motion_metrics(vid, first)
            try:
                clip = videoio.small_clip(vid, cdir / f"check_{vid.stem}.mp4") if hub.video_ok else None
                chk, _mode = ask_clip(hub, P.SHOT_CHECK,
                                      f"Shot event: {rec['label']}\nQuestions:\n{_j([{k2: q[k2] for k2 in ('id', 'question')} for q in qs[:-1]])}",
                                      ("ANCHOR:", img), frames, clip)
            except (HubError, ValueError) as e:
                chk = {"answers": {}, "notes_ko": f"검사 실패: {e}"}
            ans = {q["id"]: {"answer": "yes" if str((chk.get("answers") or {}).get(q["id"], {}).get("answer", "no")).lower().startswith("y") else "no"} for q in qs[:-1]}
            ans["kar"] = {"answer": "no" if _severe(chk) else "yes"}
            s, failed, pen = score(qs, {"answers": ans, "motion_quality": 3, "temporal_consistency": 3}, metrics,
                                   [{"intensity": "moderate", "category": "camera"}])
            att = {"file": str(vid), "score": s, "failed": [f["question"] for f in failed], "penalties": pen,
                   "notes_ko": chk.get("notes_ko", ""), "seed": res["seed"]}
            log(f"      점수 {s:.2f}" + (f" ({'; '.join(pen)})" if pen else ""))
            if best is None or s > best["score"]:
                best = att
            if s >= MIN_SCORE:
                break
        shots.append(best)
        seg = cdir / f"seg_{key}_{k}.mp4"
        videoio.normalize_segment(Path(best["file"]), seg, size, spec["fps"], drop_first=k > 0)
        parts.append(seg)
        if k + 1 < n:
            first = videoio.last_frame(Path(best["file"]), cdir / f"first_{key}_{k + 1}.jpg", size)
    videoio.concat_copy(parts, out)
    meta = {"score": round(min(s["score"] for s in shots), 3), "shots": shots, "prompt": prompt, "model": model}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, meta


# ------------------------------------------------------------------ 비트 렌더
def _writer(out: Path, W: int, H: int, fps: int):
    ff = videoio.need_ffmpeg()
    return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)


def render_beat(beat: dict, ctx: G.Ctx, src, out: Path, fps: int, subtitles: bool, ai_scene: bool,
                thumb: Path, draft: bool) -> Path:
    import cv2
    W, H = ctx.W, ctx.H
    n = int(round(beat["seconds"] * fps))
    pw = _writer(out, W, H, fps)
    for i in range(n):
        t = i / fps
        p = i / max(1, n - 1)
        if src is None:
            base = G.gradient_bg(ctx, t)
        else:
            base = Image.fromarray(cv2.cvtColor(src.frame(p), cv2.COLOR_BGR2RGB))
        fr = G.draw(base, beat, ctx, t, subtitles, ai_scene)
        if draft:
            from PIL import ImageDraw
            d = ImageDraw.Draw(fr)
            d.text((int(18 * ctx.u), int(14 * ctx.u)), "초안 · 카메라 움직임만 (AI 영상 아님)", font=ctx.F(16, True), fill=(255, 220, 120))
        if i == min(n - 1, int(n * 0.75)):
            fr.save(thumb, quality=85)
        pw.stdin.write(fr.tobytes())
    pw.stdin.close()
    if pw.wait() != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패: {out}")
    return out


# ------------------------------------------------------------------ 전체
def render(project: Path, fmt: str, draft: bool = False, narration: bool = False, subtitles: bool = True,
           bgm: Path | None = None, hub: Hub | None = None) -> Path:
    videoio.need_ffmpeg()
    sb_p = project / f"storyboard_{fmt}.json"
    if not sb_p.exists():
        raise SystemExit(f"{sb_p.name}이 없습니다. 먼저 promo plan {project} --format {fmt}")
    sb = json.loads(sb_p.read_text(encoding="utf-8"))
    from .planner import load_brief
    brief = load_brief(project)
    if draft:
        brief = {**brief, "made_with": ""}                          # 초안은 AI 생성이 아니므로 표시하지 않음
    W, H = (int(x) for x in sb["canvas"].split("x"))
    fps = sb["fps"]
    ctx = G.Ctx(brief, (W, H), project)
    rdir = project / "renders" / f"{fmt}{'_draft' if draft else ''}"
    rdir.mkdir(parents=True, exist_ok=True)
    if not draft:
        hub = hub or Hub()
    beats_out, parts = [], []
    log(f"\n=== 홍보영상 {sb['label']} {sb['seconds']:g}초 {'(초안: GPU 없이)' if draft else ''} → {rdir}")
    for b in sb["beats"]:
        log(f"[beat {b['index'] + 1}/{len(sb['beats'])}] {b['role']} {b['seconds']:g}초 {b['overlay']['kind']}")
        info = {"index": b["index"], "role": b["role"]}
        src, ai_scene = None, False
        if b["visual"] == "photo" and b.get("asset_file"):
            rec = kb.recipes()["recipes"].get(b["recipe"]) or {}
            if draft:
                src = KenBurns(project / "assets" / b["asset_file"], rec.get("kenburns", {}), W, H)
            else:
                try:
                    clip, meta = ai_clip(hub, project, b, brief)
                    src, ai_scene = ClipSource(clip, W, H), True
                    info["ai"] = {"clip": str(clip), "score": meta["score"]}
                except (HubError, RuntimeError) as e:
                    log(f"    (AI 생성 실패 → 켄 번스로 대체: {e})")
                    info["ai_error"] = str(e)
                    src = KenBurns(project / "assets" / b["asset_file"], rec.get("kenburns", {}), W, H)
        raw = render_beat(b, ctx, src, rdir / f"beat{b['index']:02d}_raw.mp4", fps, subtitles, ai_scene,
                          rdir / f"beat{b['index']:02d}.jpg", draft)
        audio = None
        if narration and b.get("narration") and not draft:
            wav = rdir / f"beat{b['index']:02d}.wav"
            audio = wav if wav.exists() else hub.tts(b["narration"], wav)
        last = b["index"] == len(sb["beats"]) - 1
        parts.append(videoio.finish_scene(raw, rdir / f"beat{b['index']:02d}.mp4", fade_in=b["index"] == 0,
                                          fade_out=last, audio=audio))
        beats_out.append(info)
    out = project / f"promo_{fmt}{'_draft' if draft else ''}.mp4"
    tmp = videoio.concat_copy(parts, rdir / "joined.mp4")
    if bgm and Path(bgm).exists():
        ff = videoio.need_ffmpeg()
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-stream_loop", "-1", "-i", str(bgm),
                            "-filter_complex", "[1:a]volume=0.18[b];[0:a][b]amix=inputs=2:duration=first:dropout_transition=0[a]",
                            "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log(f"    (배경음악 합성 실패, 음악 없이 저장: {r.stderr[-200:]})")
            shutil.copy(tmp, out)
    else:
        shutil.copy(tmp, out)
    srt, k = [], 1
    for b in sb["beats"]:
        if b.get("narration"):
            srt.append(f"{k}\n{_ts(b['start'] + 0.3)} --> {_ts(b['start'] + b['seconds'] - 0.2)}\n{b['narration']}\n")
            k += 1
    out.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    report(project, sb, brief, beats_out, out, rdir, draft)
    log(f"[promo] 완료 → {out}  (보고서 {project / ('report_' + fmt + ('_draft' if draft else '') + '.html')})")
    return out


def _ts(s: float) -> str:
    h, r = divmod(max(0.0, s), 3600)
    m, x = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{x:06.3f}".replace(".", ",")


# ------------------------------------------------------------------ 보고서
def report(project: Path, sb: dict, brief: dict, beats: list[dict], video: Path, rdir: Path, draft: bool) -> Path:
    e = html.escape
    items = {m["id"]: m for m in brief["messages"]} | {f["id"]: f for f in brief["facts"]}

    def item_txt(i):
        it = items.get(i)
        if not it:
            return e(i)
        s = f"{it['label']} {it['display']}{it.get('unit', '')}" if "display" in it else it["text"]
        return f"{e(s)} <small>({e(it.get('where') or 'p.' + str(it.get('page', '?')))})</small>"

    rows = []
    for b, info in zip(sb["beats"], beats):
        ai = info.get("ai")
        rows.append(
            f"<tr><td><img src='{e(Path(rdir.name).as_posix())}/beat{b['index']:02d}.jpg'></td>"
            f"<td><b>{e(b['role'])}</b><br>{b['start']:.0f}–{b['start'] + b['seconds']:.0f}초</td>"
            f"<td>{e(str(b.get('asset') or '그래픽'))}<br><small>{e(str(b.get('asset_subject') or ''))}</small><br>"
            f"<small>{e(str(b.get('recipe') or ''))}</small></td>"
            f"<td>{e(b['overlay']['kind'])}<br>{'<br>'.join(item_txt(i) for i in b['overlay']['items'])}</td>"
            f"<td>{e(b.get('narration') or '')}</td>"
            f"<td>{('AI ' + format(ai['score'], '.2f')) if ai else ('초안' if draft else ('대체: ' + e(info.get('ai_error', ''))[:60] if info.get('ai_error') else '그래픽'))}</td></tr>")
    facts = "".join(f"<tr><td>{e(f['label'])}</td><td><b>{e(f['display'])}{e(f.get('unit', ''))}</b></td><td>{e(f.get('where') or 'p.' + str(f.get('page')))}</td>"
                    f"<td><small>{e(f.get('quote', ''))}</small></td><td>{'확인' if f['status'] == 'ok' else '<b>원문 불일치 · 사용 안 함</b>'}</td></tr>"
                    for f in brief["facts"])
    issues = "".join(f"<li>{e(i.get('where') or 'p.' + str(i.get('page')))} {e(i['text'])} <small>({e(i.get('by', ''))})</small></li>" for i in brief.get("issues", []))
    css = """body{font:15px/1.5 -apple-system,'Malgun Gothic',sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px;background:#f6f7fb;color:#1a1d29}
@media (prefers-color-scheme:dark){body{background:#13151d;color:#e8eaf2}th{background:#1d2030!important}td,th{border-color:#2c3045!important}}
video{width:100%;max-height:70vh;background:#000;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:.9em;margin:10px 0}
td,th{border:1px solid #dde1ec;padding:6px 8px;vertical-align:top;text-align:left}th{background:#eef1f8}td img{width:200px;border-radius:6px}
small{color:#7a8094}.warn{border-left:4px solid #e08a1e;padding:6px 12px;background:rgba(224,138,30,.08)}"""
    page = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>홍보영상 보고서</title><style>{css}</style></head><body>"
            f"<h1>{e(brief['org'].get('name', ''))} · {e(sb['label'])}{' (초안)' if draft else ''}</h1>"
            f"<p>{sb['seconds']:g}초 · {e(sb['aspect'])} · 만든 시각 {datetime.now():%Y-%m-%d %H:%M}</p>"
            f"<video src='{e(video.name)}' controls playsinline></video>"
            + (f"<div class='warn'><b>영상 전에 확인할 점</b><ul>{issues}</ul></div>" if issues else "")
            + f"<h2>스토리보드</h2><table><tr><th>장면</th><th>역할</th><th>사진·움직임</th><th>그래픽 (출처 쪽)</th><th>내레이션·자막</th><th>생성</th></tr>{''.join(rows)}</table>"
            f"<h2>사실 시트 (영상에 쓸 수 있는 숫자)</h2><table><tr><th>항목</th><th>값</th><th>쪽</th><th>원문</th><th>상태</th></tr>{facts}</table>"
            f"</body></html>")
    out = project / f"report_{sb['format']}{'_draft' if draft else ''}.html"
    out.write_text(page, encoding="utf-8")
    return out
