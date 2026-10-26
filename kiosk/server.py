"""전시회 키오스크 웹서비스 — 관람객이 사진 한 장과 원하는 움직임을 넣으면, 모델이 사진을 읽고 → 요청을 이해하고 →
검사 질문을 만들고 → 프롬프트를 설계하고 → 영상을 만들고 → 스스로 검사·반성하는 과정을 화면에 실시간으로 보여 준다.

  python video_studio.py web                 # 이 PC에서만 열림: 브라우저로 http://localhost:8000
  python video_studio.py web --mock          # 연습 모드: Hub 없이 가짜 모델로 화면 흐름만 확인

구성
  POST /api/jobs                 사진 + 요청 → 대기열 (Hub 키의 동시 요청 한도에 맞춰 한 번에 1건씩 처리)
  GET  /api/jobs/{id}/events     진행 상황 실시간 스트림 (Server-Sent Events, 끊겨도 이어 받기)
  GET  /api/jobs/{id}/file/...   진행 중 만들어진 프레임·영상 파일
  DELETE /api/jobs/{id}          내 사진·영상 지우기
관람객 사진·영상은 VS_KIOSK_KEEP_HOURS(기본 24시간)가 지나면 자동으로 지운다.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import secrets
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline.utils import log

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(os.getenv("VS_KIOSK_DIR", "output/kiosk"))
SAMPLES = Path(os.getenv("VS_KIOSK_SAMPLES", "kiosk_samples"))
KEEP_HOURS = float(os.getenv("VS_KIOSK_KEEP_HOURS", "24"))
MAX_QUEUE = int(os.getenv("VS_KIOSK_MAX_QUEUE", "8"))
MAX_REQUEST_CHARS = 200
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

PRESETS = [
    {"label": "바람에 머리카락이 날려요", "request": "바람이 불어 머리카락과 옷자락이 자연스럽게 날리고, 카메라는 천천히 다가가요"},
    {"label": "환하게 웃어요", "request": "사람이 카메라를 보며 천천히 환하게 미소 짓고 눈을 깜빡여요"},
    {"label": "카메라가 빙 돌아요", "request": "카메라가 피사체 주위를 천천히 오른쪽으로 돌며 보여줘요"},
    {"label": "하늘의 구름이 흘러가요", "request": "하늘의 구름이 천천히 흘러가고 빛이 부드럽게 바뀌어요"},
    {"label": "서버실을 날아가듯", "request": "카메라가 서버 랙 사이 통로를 따라 천천히 앞으로 날아가듯 전진하고 불빛이 깜빡여요"},
    {"label": "물결이 출렁여요", "request": "물이 잔잔하게 출렁이고 반짝이는 빛이 물결 위에서 움직여요"},
]


# ------------------------------------------------------------------ 작업
@dataclass
class Job:
    id: str
    request: str
    image: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"            # queued | running | done | error
    events: list = field(default_factory=list)
    run_dir: Path | None = None
    final: str = ""
    error: str = ""
    started: float = 0.0
    finished: float = 0.0

    @property
    def dir(self) -> Path:
        return ROOT / self.id

    def push(self, ev: dict) -> None:
        ev = {**ev, "t": round(time.time() - self.created, 1)}
        if ev.get("stage") == "start" and ev.get("data", {}).get("run"):
            self.run_dir = self.dir / "runs" / ev["data"]["run"]
        if ev.get("stage") == "done" and ev.get("data", {}).get("final"):
            self.final = ev["data"]["final"]
        self.events.append(ev)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "request": self.request, "position": position(self.id),
                "final": f"/api/jobs/{self.id}/file/{self.final}" if self.final else "", "error": self.error,
                "events": len(self.events), "created": self.created,
                "elapsed": round((self.finished or time.time()) - (self.started or time.time()), 1) if self.started else 0}


JOBS: dict[str, Job] = {}
QUEUE: deque[str] = deque()
LOCK = threading.Lock()
WAKE = threading.Event()
DURATIONS: deque[float] = deque([180.0], maxlen=10)        # 최근 작업 소요 시간 (대기 시간 안내용)
SETTINGS = {"rounds": 2, "threshold": 0.8, "mock": False}


def position(jid: str) -> int:
    """내 앞에 있는 작업 수 (처리 중인 것 포함). 처리 중이면 0, 대기열에 없으면 -1."""
    with LOCK:
        running = [j for j in JOBS.values() if j.status == "running"]
        if any(j.id == jid for j in running):
            return 0
        if jid not in QUEUE:
            return -1
        return QUEUE.index(jid) + len(running)


def est_video_seconds() -> float:
    """영상 작업 1건 예상 시간 (화면의 진행률 추정용). 실제 Hub 실측 평균 45~50초."""
    if SETTINGS["mock"]:
        return float(os.getenv("MOCK_JOB_SECONDS", "8")) + 5
    return float(os.getenv("VS_EST_JOB_SECONDS", "50"))


def avg_seconds() -> float:
    return sum(DURATIONS) / len(DURATIONS)


# ------------------------------------------------------------------ 처리 스레드 (Hub 동시 1건)
def worker() -> None:
    from pipeline.hub import Hub, HubError
    from pipeline.orchestrator import RunConfig, run
    hub = None
    while True:
        WAKE.wait(timeout=2)
        WAKE.clear()
        while True:
            with LOCK:
                jid = QUEUE.popleft() if QUEUE else None
                job = JOBS.get(jid) if jid else None
                if job:
                    job.status, job.started = "running", time.time()
            if not job:
                break
            log(f"[kiosk] 작업 {job.id} 시작: {job.request}")
            try:
                hub = hub or Hub()
                cfg = RunConfig(image=job.image, request=job.request, rounds=SETTINGS["rounds"], candidates=1,
                                threshold=SETTINGS["threshold"], seconds=None, frames=5,
                                out_root=job.dir / "runs", name="kiosk", on_event=job.push)
                run(cfg, hub)
                job.status = "done" if job.final else "error"
                if not job.final:
                    job.error = "영상이 만들어지지 않았습니다."
            except (HubError, ValueError, OSError) as e:
                job.status, job.error = "error", str(e)
            except Exception as e:  # noqa: BLE001 — 한 관람객의 실패가 키오스크를 멈추지 않게
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
            job.finished = time.time()
            if job.status == "done":
                DURATIONS.append(job.finished - job.started)
            job.push({"stage": "end", "message": job.status, "data": {"status": job.status, "error": job.error}})
            log(f"[kiosk] 작업 {job.id} {job.status} ({job.finished - job.started:.0f}초) {job.error}")


def cleaner() -> None:
    """보관 시간이 지난 관람객 사진·영상 삭제."""
    while True:
        cut = time.time() - KEEP_HOURS * 3600
        for jid, j in list(JOBS.items()):
            if j.status in ("done", "error") and j.finished and j.finished < cut:
                delete_job(jid)
        if ROOT.exists():                                 # 서버를 다시 켜기 전에 남은 폴더도
            for d in ROOT.iterdir():
                if d.is_dir() and d.name not in JOBS and d.stat().st_mtime < cut:
                    shutil.rmtree(d, ignore_errors=True)
        time.sleep(600)


def delete_job(jid: str) -> None:
    with LOCK:
        j = JOBS.pop(jid, None)
        if jid in QUEUE:
            QUEUE.remove(jid)
    if j:
        shutil.rmtree(j.dir, ignore_errors=True)


# ------------------------------------------------------------------ 앱
def create_app(port: int = 8000) -> FastAPI:
    app = FastAPI(title="VIDEO Studio 키오스크", docs_url=None, redoc_url=None)
    ROOT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True, name="kiosk-worker").start()
    threading.Thread(target=cleaner, daemon=True, name="kiosk-cleaner").start()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def get(jid: str) -> Job:
        j = JOBS.get(jid)
        if not j:
            raise HTTPException(404, "작업이 없거나 이미 지워졌습니다.")
        return j

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        samples = sorted(p.name for p in SAMPLES.iterdir() if p.suffix.lower() in IMG_EXT)[:12] if SAMPLES.exists() else []
        with LOCK:
            waiting = len(QUEUE) + sum(1 for j in JOBS.values() if j.status == "running")
        return {"presets": PRESETS, "samples": [f"/samples/{n}" for n in samples], "waiting": waiting,
                "avg_seconds": round(avg_seconds()), "keep_hours": KEEP_HOURS, "mock": SETTINGS["mock"],
                "max_queue": MAX_QUEUE, "rounds": SETTINGS["rounds"],
                "threshold": SETTINGS["threshold"], "est_video_seconds": est_video_seconds()}

    @app.get("/samples/{name}")
    def sample(name: str):
        p = (SAMPLES / name).resolve()
        if p.parent != SAMPLES.resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.post("/api/jobs")
    async def create(request: str = Form(...), image: UploadFile | None = File(None), sample: str = Form("")):
        from PIL import Image, ImageOps, UnidentifiedImageError
        request = " ".join(request.split())[:MAX_REQUEST_CHARS]
        if len(request) < 2:
            raise HTTPException(400, "원하는 움직임을 입력해 주세요.")
        with LOCK:
            if len(QUEUE) >= MAX_QUEUE:
                raise HTTPException(429, f"지금 대기 중인 분이 많아요 ({len(QUEUE)}명). 잠시 후 다시 시도해 주세요.")
        if image is not None and image.filename:
            raw = await image.read()
        elif sample:
            p = (SAMPLES / Path(sample).name).resolve()
            if p.parent != SAMPLES.resolve() or not p.exists():
                raise HTTPException(400, "예시 사진을 찾을 수 없습니다.")
            raw = p.read_bytes()
        else:
            raise HTTPException(400, "사진을 찍거나 골라 주세요.")
        if len(raw) > 25 * 1024 * 1024:
            raise HTTPException(400, "사진이 너무 큽니다 (25MB 이하).")
        try:
            im = Image.open(io.BytesIO(raw))
            im = ImageOps.exif_transpose(im).convert("RGB")      # 폰 사진의 회전 정보 반영
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "사진 형식을 읽을 수 없어요. JPG나 PNG로 다시 시도해 주세요.") from None
        if min(im.size) < 256:
            raise HTTPException(400, "사진이 너무 작아요 (가로·세로 256픽셀 이상).")
        im.thumbnail((4096, 4096))        # 최대 약 1,670만 픽셀 = Qwen3-VL이 받는 최대 크기. AI에는 모드에 맞춰 줄여 보낸다
        jid = secrets.token_urlsafe(6).replace("-", "a").replace("_", "b")
        d = ROOT / jid
        d.mkdir(parents=True, exist_ok=True)
        img_path = d / "input.jpg"
        im.save(img_path, quality=92)
        job = Job(id=jid, request=request, image=img_path)
        with LOCK:
            JOBS[jid] = job
            QUEUE.append(jid)
        job.push({"stage": "queued", "message": "대기열에 등록", "data": {"request": request}})
        WAKE.set()
        return job.public()

    @app.get("/api/jobs/{jid}")
    def status(jid: str):
        return get(jid).public()

    @app.get("/api/jobs/{jid}/input")
    def input_image(jid: str):
        return FileResponse(get(jid).image)

    @app.get("/api/jobs/{jid}/file/{path:path}")
    def job_file(jid: str, path: str, download: int = 0):
        j = get(jid)
        if not j.run_dir:
            raise HTTPException(404)
        p = (j.run_dir / path).resolve()
        if j.run_dir.resolve() not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, filename=f"video_studio_{jid}{p.suffix}" if download else None)

    @app.get("/api/jobs/{jid}/events")
    async def events(jid: str, since: int = 0):
        j = get(jid)

        async def stream():
            i, idle = since, 0.0
            while True:
                if jid not in JOBS:
                    yield "event: gone\ndata: {}\n\n"
                    return
                while i < len(j.events):
                    yield f"id: {i + 1}\ndata: {json.dumps(j.events[i], ensure_ascii=False)}\n\n"
                    i += 1
                    idle = 0.0
                if j.status in ("done", "error") and i >= len(j.events):
                    return
                if j.status == "queued":
                    yield f"event: queue\ndata: {json.dumps({'position': position(jid), 'avg_seconds': round(avg_seconds())})}\n\n"
                    await asyncio.sleep(1.5)
                    continue
                await asyncio.sleep(0.3)
                idle += 0.3
                if idle >= 15:                              # 연결 유지
                    yield ": keep-alive\n\n"
                    idle = 0.0

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.delete("/api/jobs/{jid}")
    def delete(jid: str):
        get(jid)
        delete_job(jid)
        return {"deleted": jid}

    @app.get("/healthz")
    def healthz():
        return JSONResponse({"ok": True, "jobs": len(JOBS), "queue": len(QUEUE)})

    return app


def _quiet_disconnects(loop, context) -> None:
    """브라우저가 연결을 먼저 끊을 때(영상 탐색·새로고침·화면 닫기) Windows가 남기는
    ConnectionResetError [WinError 10054] 로그는 정상 동작이므로 숨긴다. 다른 오류는 그대로 보여 준다."""
    if isinstance(context.get("exception"), (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return
    loop.default_exception_handler(context)


def serve(host: str = "127.0.0.1", port: int = 8000, mock: bool = False, rounds: int = 2,
          threshold: float = 0.8) -> None:
    import uvicorn
    SETTINGS.update(rounds=rounds, threshold=threshold, mock=mock)
    if mock:                                              # 연습 모드: 가짜 Hub를 띄우고 그쪽으로 연결
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
        os.environ.setdefault("MOCK_JOB_SECONDS", "8")
        os.environ.setdefault("MOCK_CHAT_SECONDS", "1.2")
        import mock_hub
        mock_hub.MOCK_JOB_SECONDS = float(os.environ["MOCK_JOB_SECONDS"])
        mock_hub.MOCK_CHAT_SECONDS = float(os.environ["MOCK_CHAT_SECONDS"])
        mock_hub.serve(8799)
        os.environ.update(OASH_BASE_URL="http://127.0.0.1:8799/v1", OASH_API_KEY="mock", VS_NET_RETRY_SLEEP="0")
        for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL"):
            os.environ.pop(k, None)
        log("[kiosk] 연습 모드: 가짜 Hub(127.0.0.1:8799) 사용 — 실제 영상은 만들지 않습니다")
    log(f"\n[kiosk] 키오스크 화면:  http://localhost:{port}   (이 PC에서만 열립니다)")
    log(f"[kiosk] 관람객 사진·영상 보관 {KEEP_HOURS:g}시간 → {ROOT.resolve()}")
    server = uvicorn.Server(uvicorn.Config(create_app(port), host=host, port=port, log_level="warning"))

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(_quiet_disconnects)
        await server.serve()

    asyncio.run(main())
