"""홍보 모드 오프라인 테스트:  python tests/test_promo.py
브로셔 PDF를 만들어 → 읽기(원문 검증·규칙 점검·연락처·사진 분류) → LLM 스토리보드(검증) → 초안 렌더 →
본 렌더(AI 클립·홍보 검사·캐시·TTS)까지 가짜 Hub로 확인."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tempfile as _tf  # noqa: E402
os.environ.setdefault("VS_STATE_DIR", _tf.mkdtemp())   # 가짜 Hub의 길이·사용량이 실제 output/에 섞이지 않게
sys.path.insert(0, str(ROOT / "tests"))
os.environ.update(OASH_API_KEY="test", VS_NET_RETRY_SLEEP="0")
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL"):
    os.environ.pop(k, None)

import mock_hub                                            # noqa: E402
from pipeline import videoio                                # noqa: E402
from pipeline.promo import graphics, ingest, planner, render  # noqa: E402

os.environ["OASH_BASE_URL"] = "http://127.0.0.1:8768/v1"


def make_pdf(path: Path) -> Path:
    import numpy as np
    import pymupdf
    from PIL import Image
    arr = np.zeros((700, 1200, 3), np.uint8)
    arr[..., 2] = np.linspace(255, 120, 700)[:, None]
    arr[..., 1] = np.linspace(200, 80, 1200)[None, :]
    arr[200:600, 400:800] = (210, 210, 220)                 # 건물처럼 보이는 사각형
    for x in range(420, 780, 40):
        arr[220:580, x:x + 20] = (90, 120, 170)
    img = path.parent / "b.png"
    Image.fromarray(arr).save(img)
    doc = pymupdf.open()
    fontfile = graphics.FONT_CANDIDATES[False][2][0]
    p1 = doc.new_page()
    p1.insert_font(fontname="ko", fontfile=fontfile)
    p1.insert_text((50, 80), "테스트 AI 센터", fontname="ko", fontsize=20)
    p1.insert_text((50, 120), "함께 만드는 AI 미래", fontname="ko", fontsize=14)
    p1.insert_text((50, 160), "GPU 64GPUs", fontname="ko", fontsize=14)
    p1.insert_text((50, 200), "문의 test.example.ai 053-111-2222 hello@example.ai", fontname="ko", fontsize=12)
    p1.insert_text((50, 240), "로고예정", fontname="ko", fontsize=12)
    p2 = doc.new_page()
    p2.insert_font(fontname="ko", fontfile=fontfile)
    p2.insert_text((50, 80), "고성능 AI 인프라가 부담됩니다", fontname="ko", fontsize=14)
    p2.insert_text((50, 120), "Storage 500Core", fontname="ko", fontsize=14)
    p2.insert_image(pymupdf.Rect(50, 150, 550, 442), filename=str(img))
    doc.save(str(path))
    return path


class Promo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mock_hub.serve(8768)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.pdf = make_pdf(cls.tmp / "brochure.pdf")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_counting_and_rules(self):
        self.assertEqual(graphics.counting("2,204", 1.0), "2,204")
        self.assertTrue(graphics.counting("230+", 0.3).endswith("+"))
        self.assertEqual(graphics.counting("0", 0.5), "0")
        iss = ingest.rule_issues([{"label": "GPU ZONE Storage", "display": "2,204", "unit": "Core", "page": 7},
                                  {"label": "CPU ZONE 메모리", "display": "9,434", "unit": "GB", "page": 6}], ["x"])
        self.assertEqual(len(iss), 1)                       # 메모리 GB는 정상, Storage Core만 오기

    def test_end_to_end(self):
        proj = ingest.ingest(self.pdf, None, self.tmp / "proj")
        brief = json.loads((proj / "brief.json").read_text(encoding="utf-8"))
        st = {f["label"]: f["status"] for f in brief["facts"]}
        self.assertEqual(st, {"GPU": "ok", "Storage": "ok", "매출": "unverified"})       # 원문에 없는 숫자는 사용 안 함
        ms = {m["text"]: m["status"] for m in brief["messages"]}
        self.assertEqual(ms["세계 최고의 AI 센터입니다"], "unverified")                  # 원문에 없는 문장도
        self.assertEqual(brief["cta"]["url"], "test.example.ai")
        self.assertEqual(brief["cta"]["phone"], "053-111-2222")
        texts = " ".join(i["text"] for i in brief["issues"])
        self.assertIn("코어(Core)", texts)
        self.assertIn("로고예정", texts)
        self.assertEqual(len(brief["assets"]), 1)
        self.assertEqual(brief["assets"][0]["recipes"][0], "aerial_push")

        sb_p = planner.plan(proj, "short")                    # LLM이 잘못 고른 id·숫자 섞인 연결어는 걸러짐
        sb = json.loads(sb_p.read_text(encoding="utf-8"))
        b0 = sb["beats"][0]
        self.assertNotEqual(b0["asset"], "zzz")
        self.assertNotRegex(b0["bridge"], r"\d")
        used = {i for b in sb["beats"] for i in b["overlay"]["items"]}
        self.assertNotIn("f3", used)                          # 미확인 사실(매출)은 고를 수 없음

        draft = render.render(proj, "short", draft=True)
        self.assertGreater(videoio.media_seconds(draft), 22)
        self.assertTrue((proj / "report_short_draft.html").exists())

        before = mock_hub.STATE["videos"]
        mock_hub.STATE["fail_from"] = 0
        out = render.render(proj, "short", narration=True)    # 본 렌더: AI 클립 + 홍보 검사 + TTS
        made = mock_hub.STATE["videos"] - before
        self.assertGreater(made, 0)
        self.assertTrue(list((proj / "cache").glob("ai_*.mp4")))
        render.render(proj, "short", narration=True)          # 다시 렌더: 캐시로 영상 작업 0건
        self.assertEqual(mock_hub.STATE["videos"] - before, made)
        import subprocess
        streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
                                 capture_output=True, text=True).stdout.split()
        self.assertEqual(sorted(streams), ["audio", "video"])
        print(f"\n  AI 영상 작업 {made}건, 재렌더 추가 0건, 결과 {videoio.media_seconds(out):.1f}초")


if __name__ == "__main__":
    unittest.main(verbosity=2)
