"""오프라인 테스트용 가짜 Hub (OpenAI 호환 응답 + 영상 작업 흉내).

  python tests/mock_hub.py                       # http://127.0.0.1:8765/v1
  (Windows) set OASH_BASE_URL=http://127.0.0.1:8765/v1 & set OASH_API_KEY=test
  python video_studio.py run --image samples/sample.jpg --request "바람에 머리카락이 날리고 천천히 줌인"

영상 작업은 첫 프레임 이미지로 실제 mp4를 만든다. 첫 영상은 거의 정지(→ '정지' 페널티·반성 루프 확인),
그다음부터는 줌인 + 좌우 흔들림이 들어간 영상을 돌려준다.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import os as _os
import time as _time

# 연습(리허설) 모드용 지연: 실제 Hub처럼 단계가 눈에 보이게 (테스트에서는 0)
MOCK_JOB_SECONDS = float(_os.getenv("MOCK_JOB_SECONDS", "0"))
MOCK_CHAT_SECONDS = float(_os.getenv("MOCK_CHAT_SECONDS", "0"))
JOB_T: dict = {}
STATE = {"videos": 0, "answers": 0, "requests": [], "fail_from": 0, "checks": 0, "edits": 0, "tts": 0}
VIDEOS: dict[str, bytes] = {}
LOCK = threading.Lock()
_ids = itertools.count(1)


def _make_video(image_data_url: str | None, size: str, frames: int, moving: bool) -> bytes:
    import cv2
    import numpy as np
    w, h = (int(x) for x in size.split("x"))
    if image_data_url:
        raw = base64.b64decode(image_data_url.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (w, h))
    else:
        img = np.full((h, w, 3), 128, np.uint8)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
    for i in range(frames):
        t = i / max(frames - 1, 1)
        if moving:
            z = 1.0 + 0.25 * t
            dx = 18 * np.sin(t * 6.28)
            M = np.float32([[z, 0, (1 - z) * w / 2 + dx], [0, z, (1 - z) * h / 2]])
            f = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            f = img
        vw.write(f)
    vw.release()
    with open(path, "rb") as fh:
        data = fh.read()
    os.remove(path)
    return data


def _tvec(text: str, dim: int = 256) -> list[float]:
    """가짜 텍스트 임베딩: 글자 2-gram 해시 (비슷한 글이면 비슷한 벡터) + 몇 가지 동의어 묶음."""
    import math
    import zlib
    t = text.split("Query:")[-1].lower()
    for a, b in (("서버실", "server"), ("서버", "server"), ("데이터센터", "data center"), ("건물", "building"),
                 ("연구동", "building"), ("하늘", "sky"), ("gpu", "gpu")):
        t = t.replace(a, " " + b + " ")
    v = [0.0] * dim
    toks = t.split()
    for w in toks:
        v[zlib.crc32(w.encode()) % dim] += 2.0
    for i in range(len(t) - 1):
        v[zlib.crc32(t[i:i + 2].encode()) % dim] += 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1
    return [x / n for x in v]


def _brightness(data_url: str) -> float:
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("L").resize((32, 32))
        return sum(im.getdata()) / 1024
    except Exception:
        return 128


def _video_brightness(data_url: str) -> float:
    import cv2
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(data_url.split(",", 1)[1]))
    cap = cv2.VideoCapture(path)
    ok, f = cap.read()
    cap.release()
    os.remove(path)
    return float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) if ok else 128


def _big_jpg() -> bytes:
    import numpy as np
    from PIL import Image
    arr = np.zeros((600, 900, 3), np.uint8)
    arr[...] = (140, 190, 240)
    arr[200:550, 300:600] = (225, 225, 230)
    arr += np.random.default_rng(1).integers(0, 20, arr.shape, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _wav(seconds: float = 2.0) -> bytes:
    import math
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 16000 * 2 * math.pi * 440)))
                               for i in range(int(16000 * seconds))))
    return buf.getvalue()


MOCK_CTX = 131072


def promo_reply(system: str, user_text: str) -> dict | None:
    if "Stage P1" in system:
        return {"facts": [{"label": "GPU", "display": "64", "unit": "GPUs", "tags": ["infra"], "page": 1, "quote": "GPU 64GPUs"},
                          {"label": "Storage", "display": "500", "unit": "Core", "tags": ["infra"], "page": 2, "quote": "Storage 500Core"},
                          {"label": "매출", "display": "999", "unit": "억", "tags": ["infra"], "page": 1, "quote": "매출 999억"}],
                "messages": [{"tag": "org", "text": "테스트 AI 센터", "short": "", "page": 1},
                             {"tag": "tagline", "text": "함께 만드는 AI 미래", "short": "AI 미래", "page": 1},
                             {"tag": "pain", "text": "고성능 AI 인프라가 부담됩니다", "short": "", "page": 2},
                             {"tag": "mission", "text": "세계 최고의 AI 센터입니다", "short": "", "page": 1}],
                "issues": [{"page": 1, "text": "모의 이슈"}]}
    if "Stage P2" in system:
        return {"kind": "real_building", "subject": "a modern research building under blue sky", "has_text": False,
                "people": 0, "usable": True, "notes_ko": "건물 실사진"}
    if "Stage P3" in system:
        return {"beats": [{"index": 0, "asset": "zzz", "recipe": "aerial_push", "overlay_items": ["m2"],
                           "narration": ["m999", "m2"], "bridge": "2배 더 빠르게 함께"}]}
    return None


def long_reply(system: str, user_text: str) -> dict | None:
    import re
    pr = promo_reply(system, user_text)
    if pr is not None:
        return pr
    if "Stage R1" in system:                        # 이미지 설명: 밝기로 장면을 흉내 (밝음=건물, 어두움=서버실)
        bright = STATE.get("last_image_brightness", 128) > 110
        return ({"caption": "A bright six-storey research building under a blue sky", "caption_ko": "파란 하늘 아래 연구동 건물",
                 "kind": "real_photo", "subjects": ["building"], "setting": "campus", "people": 0, "has_text": False,
                 "readable_text": "", "good_first_frame": True} if bright else
                {"caption": "A dark data center server room with glowing racks", "caption_ko": "불빛이 나는 서버 랙이 있는 어두운 서버실",
                 "kind": "real_photo", "subjects": ["server racks"], "setting": "data center", "people": 0, "has_text": False,
                 "readable_text": "", "good_first_frame": True})
    if "Stage R2" in system:
        bright = STATE.get("last_image_brightness", 128) > 110
        return {"caption": "A bright building exterior" if bright else "A dark server corridor",
                "caption_ko": "밝은 건물 외관" if bright else "어두운 서버 복도",
                "motion": "camera slowly moves forward" if not bright else "clouds drift",
                "camera": "dolly in" if not bright else "static", "people": 0, "has_text": False, "good_first_frame": True}
    if "Stage L1b" in system:
        STATE["picks"] = STATE.get("picks", 0) + 1
        return {"best": "o2", "scores": {"o1": 6, "o2": 9}, "reason_ko": "요청에 더 충실함"}
    if "Stage L1" in system:
        STATE["last_l1"] = user_text
        STATE["outlines"] = STATE.get("outlines", 0) + 1
        n = int(re.search(r"exactly (\d+) act", user_text).group(1))
        return {"title": "바닷가의 하루", "logline_ko": "바닷가 마을에서 보내는 하루",
                "bible": {"characters": [{"id": "c1", "name": "woman", "look": "long black hair, white dress",
                                          "must_keep": ["face"]}],
                          "world": "seaside village", "style": "warm cinematic photo", "camera_language": "slow",
                          "tone": "calm"},
                "acts": [{"title": f"Act {i + 1}", "seconds": 10, "summary": "she walks by the sea"} for i in range(n)]}
    if "Stage L2" in system:
        lo = int(re.search(r"between (\d+) and", user_text).group(1))
        k = max(lo, 3)
        refs = re.findall(r"\[([0-9a-f]{12})\] (?:image|video_segment)", user_text)
        STATE["last_l2"] = user_text
        if refs:
            sc = [{"title": f"scene {i + 1}", "seconds": 10, "location": "data center", "time_of_day": "day",
                   "start": "new_keyframe" if i else "continue", "keyframe": "a dark server corridor" if i == 2 else "",
                   "events": ["the camera glides past servers"],
                   "mood": "calm", "narration_ko": "", "visual_ref": refs[0] if i == 1 else ("zzzz" if i == 2 else "")}
                  for i in range(k)]
            return {"scenes": sc}
        return {"scenes": [{"title": f"scene {i + 1}", "seconds": 10, "location": "beach", "time_of_day": "sunset",
                            "start": "new_keyframe" if i % 2 == 0 else "continue",
                            "keyframe": f"she sits on a bench, scene {i + 1}" if i % 2 == 0 else "",
                            "events": ["she looks at the sea", "her hair moves in the wind"], "mood": "calm",
                            "narration_ko": "바다를 바라본다"} for i in range(k)]}
    if "Stage L3" in system:
        n = int(re.search(r"EXACTLY (\d+) shots", system).group(1))
        m = max(1, n - 1)                         # 일부러 1개 모자라게 → 코드가 채우는지 확인
        return {"shots": [{"event": f"event {i + 1}", "prompt": f"The woman slowly turns, shot {i + 1}.",
                           "negative": "blur", "motion": "moderate", "end_state": f"pose {i + 1}",
                           "checks": [{"question": "Does she move?", "when": "compare_start_end"}]} for i in range(m)]}
    if "Stage L4" in system:
        with LOCK:
            STATE["edits"] += 0
        return {"matches_description": True, "style_consistent": True, "notes_ko": "설명과 일치"}
    if "Stage L5" in system:
        with LOCK:
            STATE["checks"] += 1
            n = STATE["checks"]
        bad = n % 7 == 0
        return {"summary": "she moves", "answers": {"k1": {"evidence": "x", "answer": "no" if bad else "yes"},
                                                     "k2": {"evidence": "x", "answer": "yes"}},
                "style_kept": True, "severe_artifacts": ["melted object"] if bad else [], "motion_quality": 4,
                "temporal_consistency": 4, "notes_ko": "불일치" if bad else "좋음"}
    if "Stage L6" in system:
        return {"cause": "too weak", "route": "strengthen_motion", "prompt": "Strong wind; she turns her head clearly.",
                "negative": "static", "summary_ko": "움직임 강화"}
    return None


def chat_reply(system: str, user_text: str) -> dict:
    lr = long_reply(system, user_text)
    if lr is not None:
        return lr
    if "Stage 0" in system:
        return {"caption": "A woman with long black hair stands on a windy seaside cliff at sunset.",
                "subjects": [{"id": "s1", "name": "young woman", "attributes": ["long black hair", "white dress"],
                              "position": "center, near", "movable_parts": ["hair", "dress", "head", "eyes"]}],
                "background": "ocean and orange sky", "environment_dynamics": ["waves", "clouds"],
                "camera": {"shot": "medium", "angle": "eye-level"}, "lighting": "warm backlight",
                "style": "photo", "identity_anchors": ["her face", "white dress"], "text_in_image": ""}
    if "Stage 1" in system:
        two = "8 seconds" in user_text or "seconds=8" in user_text or "about 8" in user_text
        goals = [{"id": "g1", "subject": "s1", "category": "environment", "change": "long hair blows to the right in the wind",
                  "change_ko": "긴 머리카락이 바람에 오른쪽으로 날림",
                  "when": "throughout", "intensity": "moderate", "rule": "hair and dress move in the same wind direction"},
                 {"id": "g2", "subject": "camera", "category": "camera", "change": "slow dolly-in toward the woman", "change_ko": "카메라가 천천히 다가감",
                  "when": "throughout", "intensity": "moderate", "rule": "subject grows steadily larger, no jumps"}]
        shots = [{"id": "shot1", "seconds": 4, "goals": ["g1", "g2"], "note": "hair blows, camera pushes in"}]
        if two:
            goals.append({"id": "g3", "subject": "s1", "category": "action", "change": "she turns her head and smiles", "change_ko": "고개를 돌리며 미소",
                          "when": "end", "intensity": "subtle", "rule": "face stays the same person"})
            shots.append({"id": "shot2", "seconds": 4, "goals": ["g3"], "note": "she turns and smiles"})
        return {"intent": "Animate the woman's hair in the wind while the camera slowly pushes in.",
                "intent_ko": "바람에 머리카락이 날리고 카메라가 천천히 다가간다", "total_seconds": 4 * len(shots),
                "goals": goals, "shots": shots, "preserve": ["identity of the woman", "white dress", "sunset sky"],
                "style": "keep the photo style", "assumptions": ["바람 방향은 오른쪽으로 가정"],
                "questions": [{"q": "바람은 어느 쪽으로 불까요?", "default": "오른쪽"}], "risks": []}
    if "Stage 2" in system:
        return {"questions": [
            {"id": "q1", "kind": "goal", "ref": "g1", "shot": "shot1", "when": "throughout",
             "question": "Does her hair visibly move to the right?", "question_ko": "머리카락이 오른쪽으로 날리나요?", "weight": 1.5},
            {"id": "q2", "kind": "goal", "ref": "g2", "shot": "shot1", "when": "compare_start_end",
             "question": "Does the woman appear larger in the last frame?", "question_ko": "마지막 장면에서 인물이 더 크게 보이나요?", "weight": 1.5},
            {"id": "q3", "kind": "preserve", "ref": "preserve", "shot": "all", "when": "throughout",
             "question": "Is she clearly the same person in all frames?", "question_ko": "모든 장면에서 같은 사람으로 보이나요?", "weight": 2.0},
            {"id": "q4", "kind": "quality", "ref": "quality", "shot": "all", "when": "throughout",
             "question": "Is the motion free of sudden jumps?", "question_ko": "갑자기 튀는 장면 없이 자연스럽나요?", "weight": 1.0}]}
    if "Stage 3" in system:
        return {"prompts": [
            {"prompt": "A young woman in a white dress on a seaside cliff; her long black hair streams to the right in a "
                       "steady wind while the camera slowly dollies in.", "negative": "static, blur",
             "covers": ["g1", "g2"] if "g3" not in user_text.split("goals_of_this_shot")[-1][:400] else ["g3"],
             "strategy_ko": "바람과 줌인을 앞에 배치"},
            {"prompt": "Slow dolly-in on the woman; strong wind lifts her hair and dress to the right.",
             "negative": "static", "covers": ["g1"], "strategy_ko": "카메라 먼저"}]}
    if "Stage 4a" in system:
        return {"summary": "The woman's hair moves and the camera pushes in.", "per_subject": [],
                "camera": "zoom in", "artifacts": [], "identity_kept": True}
    if "Stage 4b" in system:
        with LOCK:
            STATE["answers"] += 1
            n = STATE["answers"]
        good = n >= 2
        ans = {f"q{i}": {"evidence": "visible", "answer": "yes"} for i in range(1, 12)}
        if not good:
            ans["q1"] = {"evidence": "hair barely moves", "answer": "no"}
            ans["q2"] = {"evidence": "same size", "answer": "no"}
        return {"revised_summary": "ok", "answers": ans, "motion_quality": 4 if good else 2,
                "temporal_consistency": 5, "notes_ko": "요구사항 충족" if good else "거의 움직이지 않음"}
    if "Stage 4c" in system:
        return {"ranking": ["c2", "c1"], "reason_ko": "c2가 움직임이 더 자연스러움"}
    if "Stage 5" in system:
        return {"mismatches": ["hair does not move"], "cause": "motion written too weakly",
                "route": "strengthen_motion", "reflection": "Open with the wind: 'Strong wind blows her hair to the right'.",
                "summary_ko": "움직임 표현을 맨 앞에 강하게"}
    return {"ok": True}


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/page.html"):                             # 웹사이트 수집 테스트용
            return self._send(("<html><head><title>테스트 센터 소개</title><meta property='og:image' content='/big.jpg'></head>"
                               "<body><nav>메뉴</nav><h1>테스트 센터</h1><p>우리 센터는 경산에 있으며 GPU 클러스터를 운영합니다.</p>"
                               "<p>매년 기업 120곳의 AI 실증을 돕습니다.</p><script>var x=1;</script></body></html>").encode(),
                              ctype="text/html; charset=utf-8")
        if self.path.endswith("/big.jpg"):
            return self._send(_big_jpg(), ctype="image/jpeg")
        if self.path.endswith("/models"):
            return self._send({"data": [{"id": x} for x in (
                "qwen3-coder", "qwen3-next-80b", "paddleocr-vl", "qwen3-vl-32b", "bge-m3", "Qwen-Image",
                "Qwen-Image-Edit", "Wan2.2-T2V", "wan2.2-i2v", "LTX-2", "melotts-ko",
                "qwen3-embedding-4b", "bge-reranker-v2-m3-ko", "whisper-large-v3-turbo", "nemotron-embed-8b")]})
        if "/jobs/" in self.path:
            jid = self.path.split("/jobs/")[1]
            el = _time.time() - JOB_T.get(jid, 0)
            if MOCK_JOB_SECONDS and el < MOCK_JOB_SECONDS:
                return self._send({"job_id": jid, "status": "GENERATING" if el > 1 else "QUEUED",
                                   "progress": int(100 * el / MOCK_JOB_SECONDS)})
            return self._send({"job_id": jid, "status": "COMPLETED", "progress": 100,
                               "artifact_url": f"/files/{jid}.mp4"})
        if self.path.startswith("/files/"):
            jid = self.path.split("/files/")[1].rsplit(".", 1)[0]
            return self._send(VIDEOS.get(jid, b""), ctype="video/mp4")
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/audio/transcriptions"):                 # multipart — 내용은 보지 않고 고정 문장
            STATE["asr"] = STATE.get("asr", 0) + 1
            return self._send({"text": "경북 클라우드팜 서버실에는 GPU 112장이 있습니다.",
                               "segments": [{"start": 0, "end": 4, "text": "경북 클라우드팜 서버실에는 GPU 112장이 있습니다."}]})
        req = json.loads(raw or b"{}")
        if self.path.endswith("/chat/completions"):
            if int(req.get("max_tokens") or 0) > MOCK_CTX:           # 최대 길이 초과 → vLLM 오류 문구 흉내
                return self._send({"object": "error", "message": f"This model's maximum context length is {MOCK_CTX} tokens. "
                                   f"However, you requested {req['max_tokens']} tokens.", "code": 400}, 400)
            STATE["max_tokens_seen"] = max(STATE.get("max_tokens_seen", 0), int(req.get("max_tokens") or 0))
            msgs = req["messages"]
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            user_text = user if isinstance(user, str) else " ".join(
                p.get("text", "") for p in user if p.get("type") == "text")
            if not isinstance(user, str):
                imgs = [p["image_url"]["url"] for p in user if p.get("type") == "image_url"]
                vids = [p["video_url"]["url"] for p in user if p.get("type") == "video_url"]
                if vids:                                   # 영상 직접 입력: 첫 프레임 밝기로 장면 흉내
                    STATE["video_inputs"] = STATE.get("video_inputs", 0) + 1
                    STATE["last_image_brightness"] = _video_brightness(vids[-1])
                elif imgs:
                    STATE["last_image_brightness"] = _brightness(imgs[-1])
            if MOCK_CHAT_SECONDS:
                _time.sleep(MOCK_CHAT_SECONDS)
            content = "```json\n" + json.dumps(chat_reply(system, user_text), ensure_ascii=False) + "\n```"
            STATE["chat_calls"] = STATE.get("chat_calls", 0) + 1
            return self._send({"choices": [{"message": {"role": "assistant", "content": content}}],
                               "usage": {"prompt_tokens": len(raw) // 4, "completion_tokens": len(content) // 2}})
        if self.path.endswith("/embeddings"):
            inp = req["input"] if isinstance(req["input"], list) else [req["input"]]
            if any(not isinstance(x, str) or x.startswith("data:image") or x.startswith("<img") for x in inp):
                return self._send({"error": "text only"}, 400)          # 이미지 임베딩 미지원 흉내 → 설명문 검색 경로
            return self._send({"data": [{"index": i, "embedding": _tvec(x)} for i, x in enumerate(inp)]})
        if self.path.endswith("/rerank"):
            q = _tvec(req["query"])
            sc = [sum(a * b for a, b in zip(q, _tvec(d))) for d in req["documents"]]
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            return self._send({"results": [{"index": i, "relevance_score": sc[i]} for i in order]})
        if self.path.endswith("/images/generations"):
            import cv2
            import numpy as np
            raw_img = base64.b64decode(req["reference"].split(",", 1)[1])
            img = cv2.imdecode(np.frombuffer(raw_img, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.addWeighted(img, 0.8, np.full_like(img, (40, 80, 120)), 0.2, 0)
            STATE["edits"] += 1
            ok, png = cv2.imencode(".png", img)
            return self._send({"data": [{"b64_json": base64.b64encode(png.tobytes()).decode()}]})
        if self.path.endswith("/audio/speech"):
            STATE["tts"] += 1
            return self._send(_wav(2.0), ctype="audio/wav")
        if self.path.endswith("/videos/generations"):
            if STATE["fail_from"] and STATE["videos"] + 1 >= STATE["fail_from"]:
                return self._send({"error": "gpu node down"}, 500)
            assert "length" in req and "size" in req, req.keys()
            if req["size"] in STATE.get("reject_sizes", ()):              # 받지 않는 해상도 흉내 (세로·720P 시험)
                STATE["rejected"] = STATE.get("rejected", 0) + 1
                return self._send({"error": f"unsupported size {req['size']}"}, 400)
            STATE.setdefault("sizes_seen", []).append(req["size"])
            with LOCK:
                STATE["videos"] += 1
                moving = STATE["videos"] >= 2
                STATE["requests"].append({k: v for k, v in req.items() if k != "image"} | {"has_image": "image" in req})
            jid = f"job{next(_ids)}"
            VIDEOS[jid] = _make_video(req.get("image"), req["size"], int(req["length"]), moving)
            JOB_T[jid] = _time.time()
            return self._send({"job_id": jid, "status": "QUEUED"})
        self._send({"error": "not found"}, 404)


def serve(port: int = 8765) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    serve()
    print("mock hub on http://127.0.0.1:8765/v1  (Ctrl+C로 종료)")
    threading.Event().wait()
