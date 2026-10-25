"""멀티모달 임베딩 파이프라인 가능성 점검 (Open AI Service Hub 실제 호출, 1~3분).

  python tools/check_multimodal.py            (.env의 OASH_API_KEY 사용)

확인하는 것
  1. 모델 목록: 임베딩·리랭커·VLM·옴니·음성인식·OCR 모델이 있는지
  2. 텍스트 임베딩 (bge-m3 · qwen3-embedding-4b · nemotron-embed-8b): 차원, 한국어 검색 순위가 맞는지
  3. 이미지 임베딩: 이미지를 넣을 수 있는지 (여러 입력 형식 시도) → 되면 '글로 이미지 찾기'가 맞는지
  4. 리랭커: /v1/rerank, /rerank
  5. VLM: 영상(mp4)을 직접 받는지, 안 되면 프레임 여러 장으로
  6. 음성: TTS(melotts-ko)로 만든 한국어 음성을 Whisper로 다시 받아쓰기 → 영상 속 말 추출 가능 여부
  7. OCR 모델: 문서 이미지에서 글자 추출
결과는 표로 출력하고 output/check_multimodal.json 에 저장.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
from pipeline import cvio  # noqa: E402
from pipeline.utils import log  # noqa: E402  (윈도우 콘솔 한글 설정 포함)

BASE = (os.getenv("OASH_BASE_URL") or "https://open.hasa.re.kr/v1").rstrip("/")
ROOT_URL = BASE.rsplit("/v1", 1)[0]
KEY = os.getenv("OASH_API_KEY", "")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
ASSETS = ROOT / "promo_projects" / "gb_aicloudfarm" / "assets"
RESULTS: list[dict] = []


def rec(area: str, item: str, ok: bool | None, detail: str) -> None:
    RESULTS.append({"area": area, "item": item, "ok": ok, "detail": detail})
    mark = {True: "가능", False: "불가", None: "확인필요"}[ok]
    log(f"  [{mark:^5}] {area:<10} {item:<28} {detail}")


def post(path: str, body: dict, timeout: int = 120, base: str = BASE):
    t = time.time()
    r = requests.post(f"{base}{path}", headers=H, json=body, timeout=timeout)
    return r, time.time() - t


def cos(a, b) -> float:
    s = sum(x * y for x, y in zip(a, b))
    return s / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


def data_url(path: Path, mime: str = "image/jpeg", max_side: int = 768) -> str:
    if mime.startswith("image"):
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        raw = buf.getvalue()
    else:
        raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def embed(model: str, inputs: list, extra: dict | None = None):
    r, dt = post("/embeddings", {"model": model, "input": inputs, **(extra or {})})
    if r.status_code != 200:
        return None, f"HTTP {r.status_code} {r.text[:120]}", dt
    data = r.json().get("data") or []
    return [d["embedding"] for d in data], "", dt


# ------------------------------------------------------------------ 1
def check_models() -> list[str]:
    r = requests.get(f"{BASE}/models", headers=H, timeout=30)
    ids = [m["id"] for m in r.json().get("data", [])]
    groups = {"임베딩": ("embed", "bge-m3"), "리랭커": ("rerank",), "VLM": ("-vl", "omni"), "음성인식": ("whisper",),
              "음성합성": ("tts", "melo", "cosyvoice"), "OCR": ("ocr",), "영상생성": ("i2v", "t2v", "ltx"), "이미지편집": ("edit",)}
    for g, kws in groups.items():
        hit = [i for i in ids if any(k in i.lower() for k in kws)]
        rec("모델", g, bool(hit), ", ".join(hit) or "없음")
    return ids


# ------------------------------------------------------------------ 2
QUERY = "고성능 GPU 서버로 AI 모델을 학습하고 싶다"
DOCS = ["H100·H200·B200 GPU 클러스터에서 대규모 모델 학습과 추론을 지원합니다.",
        "본원 주차장은 건물 뒤편에 있으며 방문객은 무료입니다.",
        "AI BUILD-UP COMMUNITY에서 기업과 전문가를 연결합니다.",
        "MLOps 기반 개발환경으로 학습부터 배포까지 한 곳에서 수행합니다."]


def check_text(models: list[str]) -> list[str]:
    ok_models = []
    for m in models:
        q = QUERY
        if "qwen3-embedding" in m:
            q = f"Instruct: Given a question, retrieve passages that answer it\nQuery:{QUERY}"
        vecs, err, dt = embed(m, [q] + DOCS)
        if not vecs:
            rec("텍스트임베딩", m, False, err)
            continue
        sims = [cos(vecs[0], v) for v in vecs[1:]]
        order = sorted(range(len(DOCS)), key=lambda i: -sims[i])
        good = order[0] in (0, 3) and order[-1] == 1
        rec("텍스트임베딩", m, good, f"{len(vecs[0])}차원 {dt:.1f}s · 순위 {[i + 1 for i in order]} (1·4가 위, 2가 맨 아래면 정상)")
        ok_models.append(m)
    return ok_models


# ------------------------------------------------------------------ 3
def check_image_embed(models: list[str]) -> None:
    imgs = [ASSETS / "pdf_p02_9.jpg", ASSETS / "pdf_p04_113.jpg", ASSETS / "pdf_p08_338.jpg"]
    if not all(p.exists() for p in imgs):
        rec("이미지임베딩", "-", None, "promo_projects/gb_aicloudfarm/assets 사진이 없어 건너뜀")
        return
    urls = [data_url(p) for p in imgs]
    texts = ["파란 하늘 아래 연구동 건물 외관", "서버 랙이 늘어선 데이터센터 복도", "대형 모니터가 있는 관제실"]
    formats = {
        "data URL 문자열": lambda u: u,
        "OpenAI 멀티모달 객체": lambda u: [{"type": "image_url", "image_url": {"url": u}}],
        "NIM <img> 태그": lambda u: f'<img src="{u}" />',
    }
    for m in models:
        done = False
        for fname, fmt in formats.items():
            try:
                vecs, err, dt = embed(m, [fmt(u) for u in urls])
            except requests.RequestException as e:
                vecs, err = None, str(e)[:100]
            if not vecs:
                continue
            tv, terr, _ = embed(m, texts)
            if not tv:
                rec("이미지임베딩", m, None, f"{fname} 형식은 받음, 텍스트 비교 실패: {terr}")
                done = True
                break
            hits = sum(1 for i in range(3) if max(range(3), key=lambda j: cos(tv[i], vecs[j])) == i)
            # 이미지가 실제로 '보였는지': 서로 다른 사진의 벡터가 거의 같으면 이미지를 무시하고 문자열로 처리한 것
            spread = 1 - min(cos(vecs[0], vecs[1]), cos(vecs[0], vecs[2]))
            real = spread > 0.02
            rec("이미지임베딩", m, real and hits >= 2,
                f"{fname} · 글→사진 찾기 {hits}/3 정답 · 사진 간 차이 {spread:.3f}"
                + ("" if real else " (사진 내용을 못 읽고 문자열로 처리한 듯)"))
            done = True
            break
        if not done:
            rec("이미지임베딩", m, False, "세 가지 입력 형식 모두 거부 → 사진은 VLM 설명문으로 바꿔 텍스트 임베딩")


# ------------------------------------------------------------------ 4
def check_rerank(ids: list[str]) -> None:
    rr = [i for i in ids if "rerank" in i.lower()]
    for m in rr:
        body = {"model": m, "query": QUERY, "documents": DOCS}
        for base, path in ((BASE, "/rerank"), (ROOT_URL, "/rerank"), (ROOT_URL, "/v1/rerank")):
            try:
                r, dt = post(path, body, base=base)
            except requests.RequestException:
                continue
            if r.status_code == 200:
                res = r.json().get("results") or r.json().get("data") or []
                top = res[0].get("index") if res else None
                rec("리랭커", m, top in (0, 3), f"{base.replace(ROOT_URL, '')}{path} · 1위 문서 {None if top is None else top + 1} · {dt:.1f}s")
                break
        else:
            rec("리랭커", m, False, "rerank 경로 응답 없음")


# ------------------------------------------------------------------ 5
def _tiny_video() -> Path | None:
    try:
        import cv2
    except ImportError:
        return None
    src = ASSETS / "pdf_p04_113.jpg"
    img = cvio.imread(str(src)) if src.exists() else None
    if img is None:
        import numpy as np
        img = np.full((360, 640, 3), 90, "uint8")
    img = cv2.resize(img, (640, 360))
    p = Path(tempfile.gettempdir()) / "vs_check.mp4"
    vw = cvio.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 360))
    for i in range(24):
        z = 1 + 0.01 * i
        M = cv2.getRotationMatrix2D((320, 180), 0, z)
        vw.write(cv2.warpAffine(img, M, (640, 360)))
    vw.release()
    return p


def check_video_vlm(ids: list[str]) -> None:
    vid = _tiny_video()
    vlms = [i for i in ids if ("-vl" in i.lower() or "omni" in i.lower()) and "ocr" not in i.lower()]
    for m in vlms:
        if vid is None:
            rec("영상이해", m, None, "opencv 없음")
            continue
        content = [{"type": "text", "text": "이 영상에서 카메라가 어떻게 움직이는지 한 문장으로."},
                   {"type": "video_url", "video_url": {"url": data_url(vid, "video/mp4")}}]
        try:
            r, dt = post("/chat/completions", {"model": m, "messages": [{"role": "user", "content": content}], "max_tokens": 120})
        except requests.RequestException as e:
            rec("영상이해", m, False, str(e)[:100])
            continue
        if r.status_code == 200:
            txt = (r.json()["choices"][0]["message"].get("content") or "").strip().replace("\n", " ")
            rec("영상이해", f"{m} (mp4 직접)", True, f"{dt:.1f}s · {txt[:70]}")
        else:
            rec("영상이해", f"{m} (mp4 직접)", False, f"HTTP {r.status_code} → 프레임을 뽑아 이미지 여러 장으로 보내면 됨")


# ------------------------------------------------------------------ 6
def check_speech(ids: list[str]) -> None:
    tts = next((i for i in ids if "melotts" in i.lower() or "tts" in i.lower() or "cosyvoice" in i.lower()), None)
    asr = next((i for i in ids if "whisper" in i.lower()), None)
    sentence = "경북 인공지능 클라우드팜에서 영상을 만들고 있습니다."
    wav = None
    if tts:
        try:
            r, dt = post("/audio/speech", {"model": tts, "input": sentence, "response_format": "wav"})
            if r.status_code == 200 and len(r.content) > 1000 and not r.headers.get("Content-Type", "").startswith("application/json"):
                wav = r.content
                rec("음성합성", tts, True, f"/audio/speech {len(r.content) // 1024}KB · {dt:.1f}s")
            else:
                rec("음성합성", tts, False, f"HTTP {r.status_code} {r.text[:80]}")
        except requests.RequestException as e:
            rec("음성합성", tts, False, str(e)[:80])
    if asr:
        if wav is None:
            import struct
            import wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes(b"".join(struct.pack("<h", 0) for _ in range(16000)))
            wav = buf.getvalue()
        try:
            t = time.time()
            r = requests.post(f"{BASE}/audio/transcriptions", headers={"Authorization": H["Authorization"]},
                              files={"file": ("a.wav", wav, "audio/wav")}, data={"model": asr, "language": "ko"}, timeout=120)
            dt = time.time() - t
            if r.status_code == 200:
                txt = (r.json().get("text") if r.headers.get("Content-Type", "").startswith("application/json") else r.text).strip()
                rec("음성인식", asr, True, f"/audio/transcriptions {dt:.1f}s · '{txt[:50]}'")
            else:
                rec("음성인식", asr, False, f"HTTP {r.status_code} {r.text[:80]}")
        except requests.RequestException as e:
            rec("음성인식", asr, False, str(e)[:80])


# ------------------------------------------------------------------ 7
def check_ocr(ids: list[str]) -> None:
    ocr = [i for i in ids if "ocr" in i.lower()]
    page = ASSETS / "pdf_p13_809.jpg"                      # 시안 마지막 쪽(지도·글자)
    for m in ocr:
        if not page.exists():
            rec("OCR", m, None, "글자 이미지 없음")
            continue
        content = [{"type": "text", "text": "이미지의 글자를 그대로 적어줘."},
                   {"type": "image_url", "image_url": {"url": data_url(page, max_side=1280)}}]
        try:
            r, dt = post("/chat/completions", {"model": m, "messages": [{"role": "user", "content": content}], "max_tokens": 300})
            ok = r.status_code == 200
            txt = (r.json()["choices"][0]["message"].get("content") or "")[:60].replace("\n", " ") if ok else r.text[:80]
            rec("OCR", m, ok and len(txt) > 5, f"{dt:.1f}s · {txt}")
        except requests.RequestException as e:
            rec("OCR", m, False, str(e)[:80])


def main() -> None:
    if not KEY:
        raise SystemExit(".env에 OASH_API_KEY가 없습니다.")
    log(f"\n=== 멀티모달 점검: {BASE}\n")
    ids = check_models()
    emb = [i for i in ids if "embed" in i.lower() or i.lower() == "bge-m3"]
    log("")
    check_text(emb)
    check_image_embed(emb)
    check_rerank(ids)
    check_video_vlm(ids)
    check_speech(ids)
    check_ocr(ids)
    out = ROOT / "output" / "check_multimodal.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\n저장: {out}  — 이 파일(또는 위 표)을 그대로 보내주시면 설계를 확정할게요.")


if __name__ == "__main__":
    main()
