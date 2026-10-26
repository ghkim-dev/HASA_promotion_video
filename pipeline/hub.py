"""Open AI Service Hub(open.hasa.re.kr) 클라이언트 — HAIR_Studio와 같은 규격.

- 채팅/비전:   POST {base}/chat/completions      (OpenAI 호환, 비전은 content에 image_url)
- 영상 생성:   POST {base}/videos/generations    JSON {model, prompt, size, length(프레임), steps, seed,
                                                    negative_prompt, image(I2V 첫 프레임 data URL)}
               → {job_id} → GET {base}/jobs/{job_id} 폴링 (QUEUED/RUNNING/COMPLETED/FAILED/CANCELLED)
               → artifact_url 다운로드 (상대 경로면 게이트웨이 루트 기준)
- 모델 목록:   GET {base}/models (공개)
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .utils import image_to_data_url, log


class HubError(RuntimeError):
    pass


def _img_side() -> int:
    from .budget import BUDGET
    return BUDGET.mode["img_side"]


MAX_CONCURRENCY = max(1, int(os.getenv("OASH_MAX_CONCURRENCY", "1")))
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)
_RETRIES = int(os.getenv("OASH_MAX_RETRIES", "6"))

# 역할별 자동 선택 (앞쪽일수록 우선)
TEXT_PREFS = ("qwen3-next", "gpt-oss-120b", "exaone", "qwen3", "deepseek", "llama", "gpt-oss", "qwen", "mistral")
VISION_PREFS = ("qwen3-vl", "qwen2.5-vl", "nemotron-omni", "internvl", "-vl")
SPECIALIST_KW = ("ocr", "coder", "code", "guard", "safety", "embed", "rerank", "pii", "groot", "cuquantum",
                 "whisper", "tts", "melotts", "cosyvoice", "bge")
IMAGE_KW = ("qwen-image", "flux", "sdxl", "diffusion")
VIDEO_KW = ("video", "wan", "hunyuan", "cogvideo", "ltx", "mochi", "t2v", "i2v")

# 영상 모델 규격 (포털 카탈로그 video_spec 값, HAIR_Studio에서 확인). 모르는 모델은 DEFAULT 사용
# Wan 2.2 I2V-A14B는 480P·720P를 지원하고 결과 비율은 입력 사진을 따른다 (가로 832×480·1280×720, 세로 480×832·720×1280).
# Hub가 처음 안내한 크기는 832×480·640×480이므로, 나머지는 '시도해 보고 거절되면 기억'한다 (output/video_sizes.json).
# OASH_VIDEO_QUALITY=720p 이면 720P 크기를 우선 (생성 시간이 약 2~4배).
WAN_480 = ["832x480", "480x832", "640x480", "480x640"]
WAN_720 = ["1280x720", "720x1280", "960x720", "720x960"]
HUB_KNOWN_SIZES = {"wan2.2-i2v": ["832x480", "640x480"]}
VIDEO_SPECS = {
    "wan2.2-i2v": {"fps": 16, "max_seconds": 4, "frame_align": 4, "sizes": WAN_480, "sizes_720": WAN_720 + WAN_480,
                   "family": "wan"},
    "wan2.2-t2v": {"fps": 16, "max_seconds": 10, "frame_align": 4,
                   "sizes": ["832x480", "640x480", "480x832", "512x512"], "family": "wan"},
    "ltx-2": {"fps": 24, "max_seconds": 10, "frame_align": 8, "sizes": ["1280x704", "1216x704", "768x512"],
              "family": "ltx"},
}
DEFAULT_VIDEO_SPEC = {"fps": 16, "max_seconds": 4, "frame_align": 4, "sizes": ["832x480"], "family": "generic"}
DEFAULT_NEGATIVE = ("blurry, low quality, flicker, jitter, morphing, distorted face, extra limbs, deformed hands, "
                    "different person, sudden scene change, static frozen video, watermark, subtitles, text")


def _sizes_path() -> Path:
    from .budget import STATE_DIR
    return STATE_DIR / "video_sizes.json"


def rejected_sizes(model: str) -> set[str]:
    try:
        return set(json.loads(_sizes_path().read_text(encoding="utf-8")).get(model.lower(), []))
    except (OSError, ValueError):
        return set()


def reject_size(model: str, size: str) -> None:
    p = _sizes_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        d = {}
    d.setdefault(model.lower(), [])
    if size not in d[model.lower()]:
        d[model.lower()].append(size)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def video_spec(model: str) -> dict:
    s = VIDEO_SPECS.get(model.lower())
    if not s:
        low = model.lower()
        fam = "wan" if "wan" in low else ("ltx" if "ltx" in low else "generic")
        return {**DEFAULT_VIDEO_SPEC, "family": fam}
    s = dict(s)
    if os.getenv("OASH_VIDEO_QUALITY", "480p").strip().lower() in ("720p", "720", "hd") and s.get("sizes_720"):
        s["sizes"] = list(s["sizes_720"])
    bad = rejected_sizes(model)
    known = HUB_KNOWN_SIZES.get(model.lower(), [])
    s["sizes"] = [z for z in s["sizes"] if z not in bad] or known or s["sizes"]
    s["known_sizes"] = known
    return s


def accepts_image(model: str) -> bool:
    """첫 프레임 이미지를 받는 모델인지. 이름에 i2v가 있거나 OASH_VIDEO_ACCEPTS_IMAGE=1(예: LTX-2 시험)이면 True."""
    return "i2v" in model.lower() or "image-to-video" in model.lower() or \
        os.getenv("OASH_VIDEO_ACCEPTS_IMAGE", "0").strip() in ("1", "true", "yes")


def _retry_after(r: requests.Response) -> int:
    try:
        v = int(float(r.headers.get("Retry-After", "0") or 0))
        if v:
            return v
        d = r.json().get("detail", {})
        return int(float(d.get("retry_after", 0))) if isinstance(d, dict) else 0
    except Exception:
        return 0


@dataclass
class Models:
    text: str = ""
    vision: str = ""
    video: str = ""
    image_edit: str = ""
    tts: str = ""
    available: list[str] = field(default_factory=list)


class Hub:
    def __init__(self) -> None:
        base = (os.getenv("OASH_BASE_URL") or "https://open.hasa.re.kr/v1").strip().strip('"\'').rstrip("/")
        for tail in ("/models", "/chat/completions", "/videos/generations", "/videos"):
            if base.endswith(tail):
                base = base[: -len(tail)]
        self.base = base
        u = urlsplit(base)
        self.root = f"{u.scheme}://{u.netloc}"
        self.key = os.getenv("OASH_API_KEY", "").strip().strip('"\'')
        self.timeout = int(os.getenv("OASH_TIMEOUT", "600"))
        self.video_steps = int(os.getenv("OASH_VIDEO_STEPS", "20"))
        if not self.key:
            raise HubError("OASH_API_KEY가 없습니다. .env에 개발키를 넣어 주세요.")
        self._models: Models | None = None
        self.calls = {"chat": 0, "vision": 0, "video": 0, "image_edit": 0, "tts": 0}
        self._tts_ok = True

    # ------------------------------------------------------------ 저수준
    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def _check(self, r: requests.Response, what: str) -> dict:
        if r.status_code >= 400:
            if "invalid_api_key" in r.text:
                raise HubError("개발키가 유효하지 않거나 만료/폐기되었습니다. 포털(account/dev-keys)에서 확인 후 "
                               ".env의 OASH_API_KEY를 고치세요.")
            hint = {401: " → API Key 없음/무효", 403: " → 이 키로 해당 모델 사용 불가",
                    404: " → 등록되지 않은 모델이거나 경로 오류", 429: " → 요청 한도 초과",
                    503: " → GPU 백엔드 준비 중"}.get(r.status_code, "")
            raise HubError(f"{what} 실패 (HTTP {r.status_code}): {r.text[:500]}{hint}")
        try:
            return r.json()
        except ValueError as e:
            raise HubError(f"{what} 응답이 JSON이 아닙니다: {r.text[:300]}") from e

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        """동시 요청 수를 키 한도 이하로 유지. 429·503은 서버가 알려준 시간만큼 기다렸다 재시도."""
        r = None
        for i in range(_RETRIES + 1):
            try:
                with _SLOTS:
                    r = requests.request(method, url, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                if i == _RETRIES:
                    raise HubError(f"Hub 연결이 계속 끊깁니다 ({type(e).__name__}): {str(e)[:200]}") from e
                wait = min(5 * (i + 1), 30)
                log(f"    (연결 끊김) {wait}초 후 재시도… [{i + 1}/{_RETRIES}]")
                time.sleep(wait)
                continue
            if r.status_code not in (429, 503) or i == _RETRIES:
                return r
            wait = _retry_after(r) or min(5 * (i + 1), 30)
            log(f"    (HTTP {r.status_code}) {wait}초 후 재시도… [{i + 1}/{_RETRIES}]")
            time.sleep(wait)
        return r  # type: ignore[return-value]

    def _post(self, path: str, body: dict, what: str, timeout: int | None = None) -> dict:
        r = self._req("POST", f"{self.base}{path}", headers=self._h(), json=body, timeout=timeout or self.timeout)
        return self._check(r, what)

    # ------------------------------------------------------------ 모델
    def list_models(self) -> list[str]:
        r = self._req("GET", f"{self.base}/models", headers=self._h(), timeout=30)
        return [m.get("id", "") for m in self._check(r, "모델 목록").get("data", [])]

    def models(self) -> Models:
        """역할별 모델: .env 값 우선, 없으면 /v1/models에서 자동 선택."""
        if self._models:
            return self._models
        avail = self.list_models()
        low = {m: m.lower() for m in avail}

        def pick(prefs, exclude=()):
            for kw in prefs:
                for m, l in low.items():
                    if kw in l and not any(x in l for x in exclude):
                        return m
            return ""

        vision = pick(VISION_PREFS, exclude=("ocr",) + IMAGE_KW + VIDEO_KW)
        text = pick(TEXT_PREFS, exclude=SPECIALIST_KW + VIDEO_KW + IMAGE_KW + ("-vl", "omni"))
        video = pick(("i2v", "image-to-video")) or pick(("ltx",))
        image_edit = pick(("image-edit", "edit", "kontext"), exclude=VIDEO_KW)
        tts = pick(("cosyvoice", "melotts", "tts"))       # 실제 Hub 점검에서 cosyvoice3 /audio/speech 동작 확인
        self._models = Models(
            text=os.getenv("OASH_TEXT_MODEL") or text,
            vision=os.getenv("OASH_VISION_MODEL") or vision or text,
            video=os.getenv("OASH_VIDEO_MODEL") or video,
            image_edit=os.getenv("OASH_IMAGE_EDIT_MODEL") or image_edit,
            tts=os.getenv("OASH_TTS_MODEL") or tts,
            available=avail,
        )
        return self._models

    # ------------------------------------------------------------ 채팅 / 비전
    @staticmethod
    def _input_tokens(messages: list[dict]) -> int:
        from .budget import est_tokens, image_tokens, BUDGET
        n = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                n += est_tokens(c)
                continue
            for part in c or []:
                t = part.get("type")
                if t == "text":
                    n += est_tokens(part.get("text", ""))
                elif t == "image_url":
                    n += part.get("_tokens") or image_tokens(BUDGET.mode["img_pixels"])
                elif t == "video_url":
                    n += 6000                                   # 짧은 클립 대략치
        return n

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.4,
             max_tokens: int | None = None) -> str:
        """max_tokens를 비워 두면 '모델 최대 길이 - 입력 길이'와 모드 상한 중 작은 값으로 자동 설정.
        길이 초과로 거절되면 서버가 알려준 최대 길이를 기억하고 한 번 줄여서 다시 보낸다."""
        from .budget import BUDGET
        model = model or self.models().text
        n_in = self._input_tokens(messages)
        mt = max_tokens or BUDGET.out_tokens(self, model, n_in)
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": mt}
        r = self._req("POST", f"{self.base}/chat/completions", headers=self._h(), json=body, timeout=self.timeout)
        if r.status_code == 400:
            lim = BUDGET.parse_limit(r.text)
            if lim:
                BUDGET.remember_limit(model, lim)
                body["max_tokens"] = max(256, min(mt, lim - n_in - 512))
                log(f"    ({model} 최대 길이 {lim:,}에 맞춰 답변 길이를 {body['max_tokens']:,}로 줄여 다시 요청)")
                r = self._req("POST", f"{self.base}/chat/completions", headers=self._h(), json=body, timeout=self.timeout)
        data = self._check(r, f"채팅({model})")
        self.calls["chat"] += 1
        ch = data["choices"][0]
        out = (ch["message"].get("content") or "").strip()
        self.last_finish = ch.get("finish_reason") or ""          # "length"면 답이 최대 길이에서 잘린 것
        self.last_raw = out
        BUDGET.add_usage(model, data.get("usage"), n_in, len(out) // 2)
        return out

    video_ok = os.getenv("VS_VLM_VIDEO", "1").strip() not in ("0", "false", "no")   # VLM에 mp4를 직접 보낼지

    def vision_video(self, prompt: str, video: Path, images: list[tuple[str, Path]] | None = None,
                     system: str | None = None, temperature: float = 0.1, max_tokens: int | None = None) -> str:
        """영상(mp4)을 VLM에 직접 보낸다 (실제 Hub 점검: qwen3-vl-32b가 video_url을 받아 움직임을 정확히 설명).
        images는 기준 사진 등 함께 볼 이미지. 실패하면 HubError → 호출하는 쪽이 프레임 방식으로 바꾼다."""
        import base64
        content: list[dict] = [{"type": "text", "text": prompt}]
        from .budget import vlm_pixels
        for label, p in images or []:
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(p, max_pixels=vlm_pixels(len(images or []) + 1))}})
        content.append({"type": "text", "text": "VIDEO:"})
        content.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(Path(video).read_bytes()).decode()}})
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        self.calls["vision"] += 1
        return self.chat(msgs, model=self.models().vision, temperature=temperature, max_tokens=max_tokens)

    def ocr(self, image: Path) -> str:
        """문서·도표 사진의 글자. 실제 Hub 점검에서 schift-ocr-1-beta는 정확, paddleocr-vl은 엉뚱한 글자였으므로 schift 우선."""
        import re as _re
        av = self.models().available
        m = os.getenv("OASH_OCR_MODEL") or next((x for x in av if "schift-ocr" in x.lower()), None) \
            or next((x for x in av if "ocr" in x.lower()), None)
        if not m:
            return ""
        content = [{"type": "text", "text": "이미지의 글자를 그대로 적어줘."},
                   {"type": "image_url", "image_url": {"url": image_to_data_url(image, max_side=1600)}}]
        raw = self.chat([{"role": "user", "content": content}], model=m, temperature=0.0)
        raw = _re.sub(r"\b(title|text|table|figure|header|footer|caption|list|paragraph)\s*\[[\d,\s]+\]", "\n", raw, flags=_re.I)
        raw = _re.sub(r"\[[\d,\s]+\]", " ", raw)                  # 남은 좌표 상자 제거
        return _re.sub(r"\n\s*\n+", "\n", raw).strip()

    def vision(self, prompt: str, images: list[tuple[str, Path]], system: str | None = None,
               temperature: float = 0.1, max_tokens: int | None = None, max_side: int | None = None,
               max_pixels: int | None = None) -> str:
        """images: [(라벨, 경로)] — 라벨을 이미지 바로 앞에 텍스트로 넣어 순서·시각을 알려준다.
        사진은 비율을 지키며 픽셀 수로 맞춘다 (Qwen3-VL은 32×32픽셀당 1토큰, 1장 최대 약 1,670만 픽셀).
        1장이면 크게(모드 기본 약 420만 픽셀), 여러 장이면 장당 작게. Hub가 너무 크다고 거절하면 1/4로 줄여 다시 보내고 기억한다."""
        from .budget import BUDGET, vlm_pixels
        from .utils import fit_pixels, vlm_tokens

        def build() -> list[dict]:
            px = max_pixels or vlm_pixels(len(images))
            content: list[dict] = [{"type": "text", "text": prompt}]
            for label, p in images:
                if label:
                    content.append({"type": "text", "text": label})
                url = image_to_data_url(p, max_side=max_side) if max_side else image_to_data_url(p, max_pixels=px)
                try:
                    from PIL import Image
                    with Image.open(p) as im:
                        tok = vlm_tokens(*fit_pixels(im.width, im.height, px))
                except Exception:  # noqa: BLE001
                    tok = None
                content.append({"type": "image_url", "image_url": {"url": url}, "_tokens": tok})
            return content

        def send(content: list[dict]) -> str:
            clean = [{k: v for k, v in c.items() if k != "_tokens"} for c in content]
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": clean}]
            self._last_tokens = sum(c.get("_tokens") or 0 for c in content)
            return self.chat(msgs, model=self.models().vision, temperature=temperature, max_tokens=max_tokens)

        self.calls["vision"] += 1
        try:
            return send(build())
        except HubError as e:
            big = any(w in str(e).lower() for w in ("image", "pixel", "too large", "too long", "exceed", "maximum", "413"))
            if max_side or max_pixels or not ("HTTP 400" in str(e) or "HTTP 413" in str(e) or "HTTP 422" in str(e)) or not big:
                raise
            sc = BUDGET.shrink_px()
            log(f"    (Hub가 사진 크기를 거절 → 사진 픽셀을 {sc:.0%}로 줄여 다시 보내고 기억합니다)")
            return send(build())

    # ------------------------------------------------------------ 영상
    @staticmethod
    def frames_for(seconds: float, spec: dict) -> int:
        s = max(0.5, min(float(seconds), spec["max_seconds"]))
        a = spec["frame_align"]
        return max(a + 1, round((s * spec["fps"] - 1) / a) * a + 1)

    def generate_video(self, prompt: str, out: Path, image: Path | None, seconds: float, size: str,
                       negative: str | None = None, seed: int | None = None,
                       interval: int = 5, max_wait: int = 3600, on_progress=None) -> dict:
        model = self.models().video
        if not model:
            raise HubError("영상 모델이 없습니다 (.env의 OASH_VIDEO_MODEL 지정)")
        spec = video_spec(model)
        if size in rejected_sizes(model) and spec.get("known_sizes"):     # 이미 거절된 크기는 바로 대체 크기로
            from .videoio import parse_size, prepare_image
            tw, th = parse_size(size)
            fb = min(spec["known_sizes"], key=lambda z: abs(parse_size(z)[0] / parse_size(z)[1] - tw / th))
            if image is not None:
                image = prepare_image(image, fb, Path(out).with_name(Path(out).name + f"_first_{fb}.jpg"), fit="pad")
            size = fb
        seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        body = {"model": model, "prompt": prompt, "size": size, "length": self.frames_for(seconds, spec),
                "steps": self.video_steps, "seed": seed, "negative_prompt": negative or DEFAULT_NEGATIVE}
        if image is not None:
            if not accepts_image(model):
                raise HubError(f"{model}은(는) 첫 프레임 이미지를 받지 않습니다. i2v 모델을 쓰거나, 받는 모델이면 "
                               f".env에 OASH_VIDEO_ACCEPTS_IMAGE=1을 넣으세요.")
            body["image"] = image_to_data_url(image, max_side=max(int(v) for v in size.split("x")))
        try:
            job = self._post("/videos/generations", body, f"영상 작업 생성({model})", timeout=120)
        except HubError as e:
            known = spec.get("known_sizes") or []
            if not known or size in known or not any(f"HTTP {c}" in str(e) for c in (400, 404, 422)):
                raise
            # 세로·720P 등 Hub가 받지 않는 크기 → 기억하고, 비율이 가장 가까운 기존 크기로 첫 프레임을 맞춰 다시 요청
            from .videoio import parse_size, prepare_image
            reject_size(model, size)
            tw, th = parse_size(size)
            fb = min(known, key=lambda z: abs(parse_size(z)[0] / parse_size(z)[1] - tw / th))
            log(f"    (Hub가 {size} 크기를 받지 않아 {fb}로 바꿔 다시 요청합니다 — 다음부터는 바로 {fb} 사용)")
            size = fb
            body["size"] = fb
            if image is not None:
                image = prepare_image(image, fb, Path(out).with_name(Path(out).name + f"_first_{fb}.jpg"), fit="pad")
                body["image"] = image_to_data_url(image, max_side=max(int(v) for v in fb.split("x")))
            job = self._post("/videos/generations", body, f"영상 작업 생성({model})", timeout=120)
        self.calls["video"] += 1
        job_id = job.get("job_id") or job.get("id")
        if not job_id:
            raise HubError(f"영상 작업 ID 없음: {json.dumps(job, ensure_ascii=False)[:400]}")
        log(f"    영상 작업 {job_id} ({model}, {seconds:g}초, {size}, seed={seed}) — 수 분 걸릴 수 있어요")
        start, last = time.time(), ""
        while True:
            r = self._req("GET", f"{self.base}/jobs/{job_id}", headers=self._h(), timeout=60)
            job = self._check(r, "영상 상태 조회")
            status = str(job.get("status", "")).upper()
            line = f"{status} {job.get('progress', 0)}%"
            if line != last:
                log(f"      {line}  [{int(time.time() - start)}초]")
                if on_progress:
                    on_progress(status, job.get("progress", 0))
                last = line
            if status in ("COMPLETED", "SUCCEEDED", "DONE"):
                break
            if status in ("FAILED", "CANCELLED", "CANCELED", "ERROR"):
                raise HubError(f"영상 생성 실패: {job.get('error') or status}")
            if time.time() - start > max_wait:
                raise HubError(f"영상 대기 시간 초과 (작업 ID {job_id})")
            time.sleep(interval)
        art = job.get("artifact_url") or job.get("url")
        if not art:
            raise HubError(f"완료됐지만 artifact_url이 없습니다: {json.dumps(job, ensure_ascii=False)[:300]}")
        ext = Path(urlsplit(art).path).suffix or ".mp4"
        out = out.with_suffix(ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self._download(art))
        return {"file": out, "model": model, "seed": seed, "size": size, "length": body["length"],
                "job_id": job_id, "elapsed": int(time.time() - start)}

    # ------------------------------------------------------------ 이미지 편집 (씬 키프레임)
    def edit_image(self, reference: Path, prompt: str, out: Path, size: str = "1024x1024",
                   negative: str | None = None) -> Path:
        """Qwen-Image-Edit: 원본 사진을 참조로 새 장면의 첫 프레임을 만든다 (HAIR_Studio와 같은 규격)."""
        import base64
        m = self.models().image_edit
        if not m:
            raise HubError("이미지 편집 모델이 없습니다 (.env의 OASH_IMAGE_EDIT_MODEL 지정)")
        body = {"model": m, "prompt": prompt, "reference": image_to_data_url(reference, max_side=1024),
                "size": size, "steps": int(os.getenv("OASH_IMAGE_STEPS", "20")), "response_format": "b64_json", "n": 1}
        if negative:
            body["negative_prompt"] = negative
        try:
            data = self._post("/images/generations", body, f"이미지 편집({m})")
        except HubError as e:
            if "negative_prompt" in body and ("HTTP 400" in str(e) or "HTTP 422" in str(e)):
                body.pop("negative_prompt")
                data = self._post("/images/generations", body, f"이미지 편집({m})")
            else:
                raise
        self.calls["image_edit"] += 1
        items = data.get("data") or []
        if not items:
            raise HubError(f"응답에 이미지가 없습니다: {json.dumps(data, ensure_ascii=False)[:300]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if items[0].get("b64_json"):
            out.write_bytes(base64.b64decode(items[0]["b64_json"]))
        elif items[0].get("url"):
            out.write_bytes(self._download(items[0]["url"]))
        else:
            raise HubError("이미지 응답에 b64_json/url이 없습니다.")
        return out

    # ------------------------------------------------------------ 음성 (내레이션, 선택)
    def tts(self, text: str, out: Path, voice: str | None = None) -> Path | None:
        """OpenAI 호환 /audio/speech 시도. Hub가 지원하지 않으면 None (한 번 실패하면 이후 시도 안 함)."""
        m = self.models().tts
        if not m or not self._tts_ok or not text.strip():
            return None
        body = {"model": m, "input": text, "response_format": "wav"}
        if voice:
            body["voice"] = voice
        r = self._req("POST", f"{self.base}/audio/speech", headers=self._h(), json=body, timeout=self.timeout)
        if r.status_code >= 400:
            self._tts_ok = False
            log(f"    (음성 합성 미지원 또는 오류 HTTP {r.status_code} — 내레이션은 자막으로만 넣습니다)")
            return None
        data = r.content
        if r.headers.get("Content-Type", "").startswith("application/json"):
            import base64
            j = r.json()
            b64 = j.get("audio") or j.get("b64_json") or (j.get("data") or [{}])[0].get("b64_json")
            if not b64:
                self._tts_ok = False
                return None
            data = base64.b64decode(b64)
        self.calls["tts"] += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    # ------------------------------------------------------------ 임베딩 · 리랭크 · 받아쓰기 (멀티모달 검색용)
    def embed_model(self) -> str:
        av = [m.lower() for m in self.models().available]
        env = os.getenv("OASH_EMBED_MODEL")
        if env:
            return env
        for kw in ("qwen3-embedding", "bge-m3", "nemotron-embed", "embed"):
            for m, l in zip(self.models().available, av):
                if kw in l:
                    return m
        raise HubError("임베딩 모델이 없습니다 (.env의 OASH_EMBED_MODEL 지정)")

    def embed_texts(self, texts: list[str], query: bool = False, model: str | None = None) -> list[list[float]]:
        """텍스트 임베딩. qwen3-embedding은 질의에만 작업 설명(instruction)을 붙인다 (모델 카드 권장 방식)."""
        model = model or self.embed_model()
        if query and "qwen3-embedding" in model.lower():
            texts = [f"Instruct: Given a description, retrieve the most relevant passage or image description\nQuery:{t}" for t in texts]
        out = []
        for i in range(0, len(texts), 32):
            batch = [t[:20000] or " " for t in texts[i:i + 32]]        # 임베딩 모델 32K 컨텍스트 안에서
            data = self._post("/embeddings", {"model": model, "input": batch}, f"임베딩({model})", timeout=180)
            items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
            if len(items) != len(batch):
                raise HubError(f"임베딩 응답 개수가 다릅니다 ({len(items)}/{len(batch)})")
            out += [d["embedding"] for d in items]
        self.calls["embed"] = self.calls.get("embed", 0) + 1
        return out

    _img_embed: dict | None = None      # {"model":..., "fmt":...} 또는 {} (지원 안 함)

    IMG_FORMATS = {
        "data_url": lambda u: u,
        "openai_parts": lambda u: [{"type": "image_url", "image_url": {"url": u}}],
        "nim_img_tag": lambda u: f'<img src="{u}" />',
    }

    def image_embed_support(self, probe_images: list[Path]) -> dict:
        """이미지를 직접 임베딩할 수 있는 모델·입력 형식을 한 번 찾아 기억한다.
        서로 다른 사진의 벡터가 거의 같으면 이미지를 못 읽고 문자열로 처리한 것이므로 '미지원'으로 본다."""
        if self._img_embed is not None:
            return self._img_embed
        env = os.getenv("OASH_IMAGE_EMBED_MODEL", "")
        if env.lower() in ("off", "none", "0"):
            self._img_embed = {}
            return {}
        cands = [env] if env else [m for m in self.models().available if "nemotron-embed" in m.lower() or "vl-embed" in m.lower()
                                   or "clip" in m.lower() or "siglip" in m.lower()]
        self._img_embed = {}
        if len(probe_images) < 2:
            return {}
        urls = [image_to_data_url(p, max_side=768) for p in probe_images[:2]]
        for m in cands:
            for name, fmt in self.IMG_FORMATS.items():
                r = self._req("POST", f"{self.base}/embeddings", headers=self._h(),
                              json={"model": m, "input": [fmt(u) for u in urls]}, timeout=120)
                if r.status_code != 200:
                    continue
                try:
                    v = [d["embedding"] for d in r.json()["data"]]
                except (KeyError, ValueError):
                    continue
                import math
                c = sum(a * b for a, b in zip(v[0], v[1])) / (math.sqrt(sum(a * a for a in v[0]) * sum(b * b for b in v[1])) or 1)
                if c < 0.98:
                    self._img_embed = {"model": m, "fmt": name}
                    log(f"    (이미지 직접 임베딩 사용: {m}, 형식 {name})")
                    return self._img_embed
        log("    (이미지 직접 임베딩 미지원 → 사진은 VLM 설명문으로 검색)")
        return {}

    def embed_images(self, paths: list[Path]) -> list[list[float]] | None:
        sup = self._img_embed or {}
        if not sup:
            return None
        fmt = self.IMG_FORMATS[sup["fmt"]]
        out = []
        for i in range(0, len(paths), 8):
            batch = [fmt(image_to_data_url(p, max_side=1024)) for p in paths[i:i + 8]]
            data = self._post("/embeddings", {"model": sup["model"], "input": batch}, f"이미지 임베딩({sup['model']})", timeout=300)
            out += [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]
        return out

    def embed_query_for_images(self, texts: list[str]) -> list[list[float]] | None:
        """이미지 벡터와 비교할 질의 벡터 (같은 이미지 임베딩 모델의 텍스트 쪽)."""
        sup = self._img_embed or {}
        if not sup:
            return None
        data = self._post("/embeddings", {"model": sup["model"], "input": texts}, "질의 임베딩", timeout=120)
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d.get("index", 0))]

    _rerank_path: str | None = None

    def rerank(self, query: str, docs: list[str], model: str | None = None) -> list[tuple[int, float]] | None:
        """리랭커로 (문서 번호, 점수) 순위. 리랭커가 없거나 경로가 안 맞으면 None."""
        if not docs:
            return []
        av = self.models().available
        model = model or os.getenv("OASH_RERANK_MODEL") or next((m for m in av if "rerank" in m.lower() and m.lower().endswith("-ko")), None) \
            or next((m for m in av if "rerank" in m.lower()), None)
        if not model or self._rerank_path == "":
            return None
        paths = [self._rerank_path] if self._rerank_path else [f"{self.base}/rerank", f"{self.root}/rerank", f"{self.root}/v1/rerank"]
        for url in paths:
            r = self._req("POST", url, headers=self._h(), json={"model": model, "query": query, "documents": [d[:4000] for d in docs]}, timeout=180)
            if r.status_code == 200:
                self._rerank_path = url
                res = r.json().get("results") or r.json().get("data") or []
                return [(int(x["index"]), float(x.get("relevance_score", x.get("score", 0)))) for x in res]
        self._rerank_path = ""
        log("    (리랭커 경로를 찾지 못해 임베딩 점수만 사용)")
        return None

    _asr_ok = True

    def transcribe(self, audio: Path, language: str = "ko") -> dict | None:
        """Whisper 받아쓰기 (OpenAI 호환 /audio/transcriptions). 구간(segments)이 있으면 시각도 함께."""
        m = next((x for x in self.models().available if "whisper" in x.lower()), None)
        if not m or not self._asr_ok:
            return None
        raw = Path(audio).read_bytes()                     # 바이트로 보내야 재시도 때도 같은 내용이 간다
        def call(fmt: str | None):
            data = {"model": m, "language": language} | ({"response_format": fmt} if fmt else {})
            return self._req("POST", f"{self.base}/audio/transcriptions", headers={"Authorization": f"Bearer {self.key}"},
                             files={"file": (Path(audio).name, raw, "audio/wav")}, data=data, timeout=self.timeout)
        r = call("verbose_json")                                   # 구간 시각까지 받기
        if r.status_code in (400, 415, 422):
            r = call(None)                                         # 실제 Hub 점검에서 확인된 기본 형식으로 다시
        if r.status_code >= 400:
            self._asr_ok = False
            log(f"    (음성 받아쓰기 미지원 HTTP {r.status_code} — 영상의 말소리는 건너뜀)")
            return None
        try:
            j = r.json()
        except ValueError:
            return {"text": r.text.strip(), "segments": []}
        return {"text": (j.get("text") or "").strip(), "segments": j.get("segments") or []}

    def _download(self, url: str) -> bytes:
        if not url.startswith(("http://", "https://")):
            url = self.root + (url if url.startswith("/") else "/" + url)
        r = self._req("GET", url, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout)
        if r.status_code >= 400:
            raise HubError(f"다운로드 실패 (HTTP {r.status_code}): {url}")
        return r.content
