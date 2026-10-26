"""키오스크 웹서비스 오프라인 테스트:  python tests/test_kiosk.py
가짜 Hub로 서버를 띄우고 관람객 흐름을 끝까지: 사진+요청 등록 → 실시간 진행 알림 → 완성 영상·QR·폰 결과 페이지 → 지우기.
입력 검사(빈 요청·사진 아닌 파일·작은 사진)와 대기열 순서·한도도 확인."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("VS_STATE_DIR", tempfile.mkdtemp())
os.environ["VS_KIOSK_DIR"] = tempfile.mkdtemp()
os.environ["VS_KIOSK_SAMPLES"] = tempfile.mkdtemp()
os.environ.update(OASH_API_KEY="test", VS_NET_RETRY_SLEEP="0")
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL"):
    os.environ.pop(k, None)

import requests                                      # noqa: E402

import mock_hub                                      # noqa: E402
from test_offline import sample_image                 # noqa: E402

HUB_PORT, WEB_PORT = 8771, 8772
os.environ["OASH_BASE_URL"] = f"http://127.0.0.1:{HUB_PORT}/v1"
BASE = f"http://127.0.0.1:{WEB_PORT}"


def jpg_bytes(path: Path) -> bytes:
    return path.read_bytes()


class Kiosk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        from kiosk import server
        cls.server = server
        server.SETTINGS.update(rounds=2, threshold=0.8, mock=False)
        cls.hub = mock_hub.serve(HUB_PORT)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.photo = sample_image(cls.tmp / "p.jpg")
        sample_image(Path(os.environ["VS_KIOSK_SAMPLES"]) / "예시1.jpg")
        cfg = uvicorn.Config(server.create_app(WEB_PORT), host="127.0.0.1", port=WEB_PORT, log_level="warning")
        cls.web = uvicorn.Server(cfg)
        threading.Thread(target=cls.web.run, daemon=True).start()
        for _ in range(100):
            try:
                if requests.get(BASE + "/healthz", timeout=1).ok:
                    break
            except requests.ConnectionError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.web.should_exit = True
        cls.hub.shutdown()

    def post(self, request="바람에 머리카락이 날리고 카메라가 천천히 다가가요", image: bytes | None = None, sample=""):
        files = {"image": ("photo.jpg", image if image is not None else jpg_bytes(self.photo), "image/jpeg")} if image is not False else None
        return requests.post(BASE + "/api/jobs", data={"request": request, "sample": sample}, files=files, timeout=30)

    def wait(self, jid, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = requests.get(f"{BASE}/api/jobs/{jid}", timeout=10).json()
            if st["status"] in ("done", "error"):
                return st
            time.sleep(0.3)
        self.fail("시간 초과")

    def test_1_validation(self):
        self.assertEqual(self.post(request=" ").status_code, 400)
        r = self.post(image=b"not an image")
        self.assertEqual(r.status_code, 400)
        self.assertIn("JPG", r.json()["detail"])
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (100, 80)).save(buf, format="JPEG")
        self.assertEqual(self.post(image=buf.getvalue()).status_code, 400)          # 너무 작은 사진
        self.assertEqual(self.post(image=False).status_code, 400)                    # 사진 없음
        cfg = requests.get(BASE + "/api/config", timeout=10).json()
        self.assertEqual(len(cfg["presets"]), 6)
        self.assertEqual(cfg["samples"], ["/samples/예시1.jpg"])
        self.assertNotIn("base_url", cfg)                                           # 외부 접속 주소·QR 없음 (localhost 전용)
        self.assertEqual(requests.get(BASE + "/samples/..%2F..%2Fetc%2Fpasswd", timeout=10).status_code, 404)

    def test_2_full_flow(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        jid = r.json()["id"]
        # 실시간 알림을 끝까지 받는다
        stages = []
        with requests.get(f"{BASE}/api/jobs/{jid}/events", stream=True, timeout=180) as s:
            for line in s.iter_lines(decode_unicode=True):
                if line and line.startswith("data: ") and line != "data: {}":
                    ev = json.loads(line[6:])
                    if "stage" in ev:
                        stages.append(ev)
        names = [e["stage"] for e in stages]
        for want in ("queued", "start", "observe_done", "understand_done", "checklist_done", "design_done",
                     "progress", "verify_done", "done", "end"):
            self.assertIn(want, names)
        self.assertLess(names.index("observe_done"), names.index("understand_done"))
        self.assertLess(names.index("checklist_done"), names.index("verify_done"))
        und = next(e for e in stages if e["stage"] == "understand_done")["data"]
        self.assertTrue(all(g.get("change_ko") for g in und["goals"]))               # 화면에 보일 한국어
        chk = next(e for e in stages if e["stage"] == "checklist_done")["data"]["questions"]
        self.assertTrue(chk[0]["question_ko"])
        ver = [e["data"] for e in stages if e["stage"] == "verify_done"]
        self.assertTrue(ver[0]["answers"] and ver[0]["frames"] and ver[0]["video"])
        self.assertTrue(all("question_ko" in a for a in ver[0]["answers"]))
        if len(ver) > 1:                                                              # 1차 미달 → 반성 → 2차
            self.assertIn("reflect_done", names)
        st = self.wait(jid)
        self.assertEqual(st["status"], "done", st)
        # 진행 중 파일·완성 영상·QR·폰 결과 페이지
        f = requests.get(f"{BASE}/api/jobs/{jid}/file/{ver[0]['frames'][0]}", timeout=10)
        self.assertEqual(f.status_code, 200)
        self.assertEqual(f.headers["content-type"], "image/jpeg")
        v = requests.get(BASE + st["final"], timeout=10)
        self.assertEqual(v.status_code, 200)
        self.assertGreater(len(v.content), 1000)
        self.assertEqual(requests.get(f"{BASE}/api/jobs/{jid}/file/../input.jpg", timeout=10).status_code, 404)
        self.assertEqual(requests.get(f"{BASE}/api/jobs/{jid}/qr.png", timeout=10).status_code, 404)
        dl = requests.get(BASE + st["final"] + "?download=1", timeout=10)                # 영상 파일로 저장
        self.assertIn("attachment", dl.headers.get("content-disposition", ""))
        self.assertIn(f"video_studio_{jid}.mp4", dl.headers["content-disposition"])
        res = next(e for e in stages if e["stage"] == "start")["data"]["resolution"]
        self.assertEqual(res["input"], [900, 1200])                                   # 원본 크기 그대로
        self.assertEqual(next(e for e in stages if e["stage"] == "start")["data"]["size"], "480x640")   # 세로 3:4 → 세로 영상
        # 끊겼다 다시 연결하면 이어서 받는다 (since)
        n = st["events"]
        with requests.get(f"{BASE}/api/jobs/{jid}/events?since={n - 1}", stream=True, timeout=30) as s:
            got = [ln for ln in s.iter_lines(decode_unicode=True) if ln and ln.startswith("data: ")]
        self.assertEqual(len(got), 1)
        # 지우기
        d = Path(os.environ["VS_KIOSK_DIR"]) / jid
        self.assertTrue(d.exists())
        self.assertEqual(requests.delete(f"{BASE}/api/jobs/{jid}", timeout=10).status_code, 200)
        self.assertFalse(d.exists())
        self.assertEqual(requests.get(f"{BASE}/api/jobs/{jid}", timeout=10).status_code, 404)
        self.assertEqual(requests.get(f"{BASE}/r/{jid}", timeout=10).status_code, 404)

    def test_3_queue_order_and_limit(self):
        s = self.server
        old = s.MAX_QUEUE
        s.MAX_QUEUE = 2
        try:
            ids = []
            for i in range(4):                                  # 처리 중 1건 + 대기 2건 → 4번째는 거절
                ids.append(self.post(request=f"카메라가 천천히 다가가요 {i}"))
            codes = [r.status_code for r in ids]
            self.assertIn(429, codes)                                                 # 한도를 넘으면 거절 (안내 문구)
            ok = [r.json()["id"] for r in ids if r.status_code == 200]
            positions = [requests.get(f"{BASE}/api/jobs/{j}", timeout=10).json()["position"] for j in ok]
            self.assertEqual(positions, sorted(positions))                            # 먼저 온 순서
            for j in ok:
                self.assertEqual(self.wait(j)["status"], "done")
        finally:
            s.MAX_QUEUE = old

    def test_4_sample_photo(self):
        r = self.post(image=False, sample="예시1.jpg")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.wait(r.json()["id"])["status"], "done")
        self.assertEqual(self.post(image=False, sample="../../secret.jpg").status_code, 400)

    def test_5_hub_failure(self):
        """Hub 영상 생성이 계속 실패하면: 작업은 오류로 끝나고 이유가 남으며, 다음 관람객은 정상 처리."""
        mock_hub.STATE["fail_from"] = mock_hub.STATE["videos"] + 1
        try:
            r = self.post(request="물결이 출렁여요")
            st = self.wait(r.json()["id"])
            self.assertEqual(st["status"], "error")
            self.assertTrue(st["error"])
        finally:
            mock_hub.STATE["fail_from"] = 0
        self.assertEqual(self.wait(self.post().json()["id"])["status"], "done")

    def test_6_portrait_photo(self):
        """세로 폰 사진(1080×2400): 원본 해상도 유지 → AI는 비율 그대로 크게 읽음 → 세로 영상(480×832) 요청.
        Hub가 세로 크기를 거절하면 기억하고 가로 크기에 여백을 채워 다시 요청, 다음부터는 바로 가로 크기."""
        from PIL import Image
        from pipeline import hub as hubmod
        buf = io.BytesIO()
        Image.open(self.photo).resize((1080, 2400)).save(buf, format="JPEG", quality=92)
        hubmod._sizes_path().unlink(missing_ok=True)
        mock_hub.STATE["sizes_seen"] = []
        jid = self.post(request="웃으면서 머리를 넘겨요", image=buf.getvalue()).json()["id"]
        self.assertEqual(self.wait(jid)["status"], "done")
        with requests.get(f"{BASE}/api/jobs/{jid}/events", stream=True, timeout=30) as ev:
            start = next(json.loads(ln[6:]) for ln in ev.iter_lines(decode_unicode=True)
                         if ln and ln.startswith("data: ") and '"stage": "start"' in ln)["data"]
        self.assertEqual(start["resolution"]["input"], [1080, 2400])                  # 1600px로 줄이지 않음
        vw, vh = start["resolution"]["vlm"]
        self.assertLess(vw, vh)                                                       # 비율 그대로
        self.assertGreater(vw * vh, 2_000_000)                                        # 예전(긴 변 1280px, 약 74만 픽셀)보다 크게
        self.assertEqual(start["size"], "480x832")                                    # 세로 영상
        self.assertIn("480x832", mock_hub.STATE["sizes_seen"])
        mock_hub.STATE["reject_sizes"] = {"480x832", "480x640"}                       # Hub가 세로를 거절하는 경우
        mock_hub.STATE["sizes_seen"] = []
        try:
            jid2 = self.post(request="웃으면서 머리를 넘겨요", image=buf.getvalue()).json()["id"]
            self.assertEqual(self.wait(jid2)["status"], "done")
            self.assertIn("480x832", hubmod.rejected_sizes("wan2.2-i2v"))             # 기억
            self.assertEqual(set(mock_hub.STATE["sizes_seen"]), {"640x480"})          # 비율이 가장 가까운 가로 크기 + 여백 채움
            for _ in range(2):                                                        # 받지 않는 크기는 한 번씩만 시도하고 기억
                jid3 = self.post(request="웃으면서 머리를 넘겨요", image=buf.getvalue()).json()["id"]
                self.assertEqual(self.wait(jid3)["status"], "done")
            self.assertEqual(hubmod.rejected_sizes("wan2.2-i2v"), {"480x832", "480x640"})
            n = mock_hub.STATE.get("rejected", 0)
            jid4 = self.post(request="웃으면서 머리를 넘겨요", image=buf.getvalue()).json()["id"]
            self.assertEqual(self.wait(jid4)["status"], "done")
            self.assertEqual(mock_hub.STATE.get("rejected", 0), n)                    # 그다음부터는 거절당하지 않음
        finally:
            mock_hub.STATE["reject_sizes"] = set()
            hubmod._sizes_path().unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
