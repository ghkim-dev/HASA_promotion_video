"""영상 입출력·측정 도구 (opencv + Pillow, ffmpeg는 있으면 사용).

- prepare_image : 입력 사진을 영상 모델 해상도에 맞게 자르거나(crop) 여백을 채워(pad) 첫 프레임으로 만든다
- sample_frames : 시각(초)을 새긴 프레임 K장 추출 → VLM 검증 입력 (Video-Bench식 프레임 질의)
- contact_sheet : 프레임을 가로로 이어 붙인 한 장 → 보고서·후보 비교(rank)용
- motion_metrics: 모델 없이 계산하는 보조 지표 (VBench의 dynamic degree·temporal flickering·
                  I2V 첫 프레임 충실도를 단순화한 근사치)
- last_frame    : 다음 샷의 첫 프레임 (Vlogger식 샷 이어 붙이기)
- concat_videos : 샷 영상을 하나로 합침 (ffmpeg 있으면 H.264 mp4, 없으면 opencv mp4v)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import cvio


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise RuntimeError("opencv-python-headless가 필요합니다: pip install opencv-python-headless") from e


# ---------------------------------------------------------------- 입력 이미지
def parse_size(size: str) -> tuple[int, int]:
    w, h = size.lower().split("x")
    return int(w), int(h)


def choose_size(image: Path, sizes: list[str]) -> str:
    """모델이 받는 해상도 중 입력 사진 비율과 가장 가까운 것."""
    from PIL import Image
    with Image.open(image) as im:
        w, h = im.size
    ar = w / h
    return min(sizes, key=lambda s: abs(parse_size(s)[0] / parse_size(s)[1] - ar))


CROP_MAX_MISMATCH = 1.3       # 사진과 영상 비율이 이보다 많이 다르면 crop 대신 pad


def prepare_image(src: Path, size: str, out: Path, fit: str = "crop") -> Path:
    """첫 프레임용 이미지. crop=가운데 기준 잘라 채움, pad=흐린 배경으로 여백 채움, none=그대로 크기만 맞춤."""
    from PIL import Image, ImageFilter
    tw, th = parse_size(size)
    im = Image.open(src).convert("RGB")
    if fit == "crop":                                   # 비율 차이가 크면(세로 사진 → 가로 영상 등) 자르면 대부분 잘려 나가므로 여백 채움
        ar_in, ar_out = im.width / im.height, tw / th
        if max(ar_in / ar_out, ar_out / ar_in) > CROP_MAX_MISMATCH:
            fit = "pad"
    if fit == "none":
        res = im.resize((tw, th), Image.LANCZOS)
    elif fit == "pad":
        bg = im.resize((tw, th), Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
        fg = im.copy()
        fg.thumbnail((tw, th), Image.LANCZOS)
        bg.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
        res = bg
    else:
        s = max(tw / im.width, th / im.height)
        r = im.resize((max(tw, round(im.width * s)), max(th, round(im.height * s))), Image.LANCZOS)
        x, y = (r.width - tw) // 2, (r.height - th) // 2
        res = r.crop((x, y, x + tw, y + th))
    out.parent.mkdir(parents=True, exist_ok=True)
    res.save(out, quality=95)
    return out


# ---------------------------------------------------------------- 프레임
def probe(video: Path) -> dict:
    cv2 = _cv2()
    cap = cvio.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:                                  # 일부 webm은 프레임 수를 안 알려줌 → 직접 센다
        n = 0
        while cap.read()[0]:
            n += 1
    cap.release()
    return {"fps": fps, "frames": n, "width": w, "height": h,
            "seconds": round(n / fps, 2) if fps else 0.0}


def _read_all(video: Path, max_width: int | None = None) -> tuple[list, float]:
    cv2 = _cv2()
    cap = cvio.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_width and f.shape[1] > max_width:
            s = max_width / f.shape[1]
            f = cv2.resize(f, (max_width, round(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, fps


def sample_frames(video: Path, out_dir: Path, k: int = 5, label: bool = True) -> list[dict]:
    """처음~끝을 고르게 k장. 각 프레임 왼쪽 위에 't=1.5s' 를 새겨 VLM이 시간 순서를 알게 한다."""
    cv2 = _cv2()
    frames, fps = _read_all(video)
    if not frames:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(frames)
    res = []
    for i in range(k):
        idx = round((total - 1) * i / max(k - 1, 1))
        f = frames[idx].copy()
        t = idx / fps
        if label:
            txt = f"t={t:.1f}s"
            sc = max(0.5, f.shape[1] / 900)
            cv2.rectangle(f, (0, 0), (int(150 * sc), int(36 * sc)), (0, 0, 0), -1)
            cv2.putText(f, txt, (int(8 * sc), int(26 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sc,
                        (255, 255, 255), max(1, int(2 * sc)), cv2.LINE_AA)
        p = out_dir / f"{video.stem}_f{i}.jpg"
        cvio.imwrite(str(p), f)
        res.append({"path": p, "t": round(t, 2), "index": idx,
                    "pos": "start" if i == 0 else ("end" if i == k - 1 else "middle")})
    return res


def contact_sheet(frames: list[Path], out: Path, height: int = 180) -> Path | None:
    from PIL import Image
    ims = []
    for p in frames:
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((round(im.width * height / im.height), height)))
    if not ims:
        return None
    gap = 4
    sheet = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), height), (20, 20, 20))
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def last_frame(video: Path, out: Path, size: str | None = None) -> Path:
    cv2 = _cv2()
    frames, _ = _read_all(video)
    if not frames:
        raise RuntimeError(f"프레임을 읽을 수 없습니다: {video}")
    f = frames[-1]
    if size:
        w, h = parse_size(size)
        if (f.shape[1], f.shape[0]) != (w, h):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
    out.parent.mkdir(parents=True, exist_ok=True)
    cvio.imwrite(str(out), f)
    return out


# ---------------------------------------------------------------- 보조 지표
STATIC_THRESHOLD = float(os.getenv("VS_STATIC_THRESHOLD", "0.35"))   # 움직임(%)이 이보다 작으면 '거의 정지'
JUMP_RATIO = 4.0                                                      # 프레임 변화가 중앙값의 4배↑이면 '끊김' 의심


def motion_metrics(video: Path, first_frame: Path | None = None) -> dict:
    """모델 없이 계산하는 근사 지표. 절대 기준이 아니라 '정지 영상·깜빡임·컷 튐·첫 프레임 이탈'을 거르는 용도.

    motion      : Farneback 광류 상위 10% 크기 평균 / 영상 폭 ×100 (VBench dynamic degree 근사, 클수록 많이 움직임)
    flicker     : 연속 프레임 평균 밝기 차의 평균 (VBench temporal flickering 근사, 0~255)
    jump        : 연속 프레임 차이가 중앙값보다 JUMP_RATIO배 이상 튀는 지점이 있는지 (컷 전환·순간 변형)
    fidelity    : 입력 이미지와 첫 프레임의 구조 유사도(-1~1, 64×64 흑백 상관) — I2V가 첫 프레임을 지켰는지
    """
    cv2 = _cv2()
    import numpy as np
    frames, fps = _read_all(video, max_width=256)
    if len(frames) < 2:
        return {"ok": False, "reason": "프레임 부족"}
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    step = max(1, len(gray) // 24)
    idx = list(range(0, len(gray), step))
    mags = []
    for a, b in zip(idx, idx[1:]):
        flow = cv2.calcOpticalFlowFarneback(gray[a], gray[b], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        m = np.linalg.norm(flow, axis=2).ravel()
        mags.append(float(np.mean(np.sort(m)[-max(1, len(m) // 10):])) / (b - a))
    w = gray[0].shape[1]
    motion = 100.0 * (sum(mags) / len(mags)) / w if mags else 0.0
    lum = [float(g.mean()) for g in gray]
    dl = [abs(x - y) for x, y in zip(lum, lum[1:])]
    flicker = sum(dl) / len(dl)
    diffs = [float(np.mean(cv2.absdiff(x, y))) for x, y in zip(gray, gray[1:])]
    med = float(np.median(diffs)) or 1e-3
    jump = bool(max(diffs) > max(JUMP_RATIO * med, 12.0))
    fidelity = None
    if first_frame and Path(first_frame).exists():
        ref = cvio.imread(str(first_frame), cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            a = cv2.resize(ref, (64, 64)).astype("float32").ravel()
            b = cv2.resize(gray[0], (64, 64)).astype("float32").ravel()
            a, b = a - a.mean(), b - b.mean()
            den = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            fidelity = round(float(a @ b) / den, 3)
    return {"ok": True, "frames": len(frames), "fps": round(fps, 2), "seconds": round(len(frames) / fps, 2),
            "motion": round(motion, 3), "static": motion < STATIC_THRESHOLD,
            "flicker": round(flicker, 2), "jump": jump, "fidelity": fidelity}


# ---------------------------------------------------------------- 합치기
def concat_videos(parts: list[Path], out: Path) -> Path:
    """샷 영상들을 순서대로 이어 붙인다. 하나뿐이면 mp4로 변환만(브라우저 재생 호환)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-y", "-loglevel", "error"]
        for p in parts:
            cmd += ["-i", str(p)]
        n = len(parts)
        # 해상도가 다를 수 있으므로 첫 샷 크기로 맞춘 뒤 concat
        info = probe(parts[0])
        w, h = info["width"] or 832, info["height"] or 480
        fps = info["fps"] or 16
        chains = "".join(f"[{i}:v]scale={w}:{h},fps={fps},setsar=1[v{i}];" for i in range(n))
        cmd += ["-filter_complex", chains + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out.exists():
            return out
    cv2 = _cv2()
    writer = None
    for p in parts:
        frames, fps = _read_all(p)
        for f in frames:
            if writer is None:
                h, w = f.shape[:2]
                writer = cvio.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps or 16, (w, h))
                size = (w, h)
            if (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size)
            writer.write(f)
    if writer is None:
        raise RuntimeError("합칠 프레임이 없습니다.")
    writer.release()
    return out


# ---------------------------------------------------------------- 장편: 색 보정·드리프트
def _lab_stats(img):
    cv2 = _cv2()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
    return lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-3


def color_distance(a: Path, b: Path) -> float:
    """두 이미지의 LAB 평균 거리 (ΔE 근사). 체인 생성 중 색·밝기가 기준 프레임에서 얼마나 벗어났는지."""
    cv2 = _cv2()
    import numpy as np
    ia, ib = cvio.imread(str(a)), cvio.imread(str(b))
    if ia is None or ib is None:
        return 0.0
    (ma, _), (mb, _) = _lab_stats(ia), _lab_stats(ib)
    return round(float(np.linalg.norm(ma - mb)), 2)


def color_match(src: Path, anchor: Path, out: Path, strength: float = 0.6) -> Path:
    """src의 LAB 평균·분산을 anchor 쪽으로 strength만큼 옮긴다 (StreamingT2V의 '첫 구간 기억'을 색 수준에서 흉내).
    샷을 이어 붙일수록 색·밝기가 조금씩 틀어지는 누적 오차를 다음 샷 시작 전에 되돌린다."""
    cv2 = _cv2()
    import numpy as np
    img, ref = cvio.imread(str(src)), cvio.imread(str(anchor))
    if img is None or ref is None:
        shutil.copy(src, out)
        return out
    ms, ss = _lab_stats(img)
    mr, sr = _lab_stats(ref)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
    fixed = (lab - ms) / ss * sr + mr
    lab = lab * (1 - strength) + fixed * strength
    res = cv2.cvtColor(np.clip(lab, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
    out.parent.mkdir(parents=True, exist_ok=True)
    cvio.imwrite(str(out), res)
    return out


# ---------------------------------------------------------------- 장편: 조립 (ffmpeg 필요)
def need_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("긴 영상 조립에는 ffmpeg가 필요합니다. (Windows: winget install Gyan.FFmpeg 후 새 터미널)")
    return ff


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패: {r.stderr[-800:]}")


def normalize_segment(src: Path, out: Path, size: str, fps: float, drop_first: bool) -> Path:
    """모든 샷을 같은 해상도·fps·코덱으로 맞춘다. 체인 샷은 첫 프레임(= 앞 샷 마지막 프레임)을 버려 이음매 정지를 없앤다."""
    ff = need_ffmpeg()
    w, h = parse_size(size)
    vf = (r"select=gte(n\,1),setpts=N/FRAME_RATE/TB," if drop_first else "") + \
        f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},setsar=1"
    _run([ff, "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(fps), str(out)])
    return out


def concat_copy(parts: list[Path], out: Path) -> Path:
    """같은 형식으로 맞춘 파일들을 재인코딩 없이 이어 붙인다 (수백 개도 빠름)."""
    ff = need_ffmpeg()
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    _run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    return out


def media_seconds(path: Path) -> float:
    ff = shutil.which("ffprobe")
    if ff:
        r = subprocess.run([ff, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    if str(path).endswith(".wav"):
        import wave
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    return probe(path)["seconds"]


def finish_scene(video: Path, out: Path, fade_in: bool, fade_out: bool, audio: Path | None) -> Path:
    """씬 영상에 (컷 경계면) 짧은 페이드와 오디오 트랙을 붙인다. 오디오가 없으면 무음 트랙을 넣어
    모든 씬의 스트림 구성을 같게 맞춘다 → 최종 concat을 재인코딩 없이 할 수 있다."""
    ff = need_ffmpeg()
    dur = media_seconds(video)
    vf = []
    if fade_in:
        vf.append("fade=t=in:st=0:d=0.4")
    if fade_out and dur > 1:
        vf.append(f"fade=t=out:st={dur - 0.4:.3f}:d=0.4")
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(video)]
    if audio:
        a_len = media_seconds(audio)
        tempo = min(1.5, a_len / max(dur - 0.3, 0.5)) if a_len > dur - 0.3 else 1.0
        af = (f"atempo={tempo:.3f}," if tempo > 1.0 else "") + f"adelay=300|300,apad,atrim=0:{dur:.3f}"
        cmd += ["-i", str(audio), "-filter_complex", f"[1:a]{af},aresample=44100,aformat=channel_layouts=stereo[a]",
                "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo", "-map", "0:v", "-map", "1:a"]
    if vf:
        cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest", str(out)]
    _run(cmd)
    return out


def small_clip(src: Path, out: Path, t0: float = 0.0, t1: float | None = None, width: int = 640) -> Path | None:
    """VLM에 보낼 작은 mp4 (H.264, 소리 없음). webm·긴 영상도 이걸로 잘라 보낸다."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", str(src)]
    if t1 is not None:
        cmd += ["-t", f"{max(0.5, t1 - t0):.2f}"]
    cmd += ["-an", "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    return out if r.returncode == 0 and out.exists() and out.stat().st_size > 0 else None
