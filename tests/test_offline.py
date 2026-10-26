"""실제 Hub 없이 전체 흐름 확인:  python tests/test_offline.py"""

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
os.environ["OASH_BASE_URL"] = "http://127.0.0.1:8766/v1"
os.environ["OASH_API_KEY"] = "test"
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL"):
    os.environ.pop(k, None)

import mock_hub                                    # noqa: E402
from pipeline import agents, lexicon, videoio                  # noqa: E402
from pipeline.hub import Hub, video_spec                       # noqa: E402
from pipeline.orchestrator import RunConfig, run, score        # noqa: E402


def sample_image(path: Path) -> Path:
    import numpy as np
    from PIL import Image, ImageDraw
    w, h = 900, 1200                                           # 세로 사진 → crop 동작 확인
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = np.linspace(255, 40, h)[:, None]
    arr[..., 1] = np.linspace(140, 60, h)[:, None]
    arr[..., 2] = 90
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.ellipse((330, 300, 570, 560), fill=(230, 200, 180))
    d.rectangle((300, 560, 600, 1100), fill=(245, 245, 245))
    for x in range(0, w, 60):                                  # 질감(광류가 잡히도록)
        d.line((x, 0, x + 200, h), fill=(255, 255, 255), width=2)
    im.save(path, quality=92)
    return path


class Offline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mock_hub.serve(8766)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.img = sample_image(cls.tmp / "photo.jpg")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_model_selection(self):
        m = Hub().models()
        self.assertEqual((m.text, m.vision, m.video), ("qwen3-next-80b", "qwen3-vl-32b", "wan2.2-i2v"))
        self.assertEqual(Hub.frames_for(4, video_spec("wan2.2-i2v")), 65)
        self.assertEqual(Hub.frames_for(10, video_spec("LTX-2")), 241)

    def test_lexicon(self):
        hits = {h["term"] for h in lexicon.match("바람에 머리카락이 흩날리고 천천히 줌 인 해줘")}
        self.assertTrue({"바람", "줌인"} <= hits, hits)

    def test_normalize_spec_splits_and_clamps(self):
        spec = agents.normalize_spec({"goals": [{"id": "g1"}, {"id": "g2", "subject": "camera", "category": "?"}],
                                      "shots": [{"seconds": 9, "goals": ["g1"]}]}, {"max_seconds": 4, "fps": 16})
        self.assertEqual(spec["shots"][0]["seconds"], 4)
        self.assertIn("g2", spec["shots"][0]["goals"])         # 빠진 목표는 첫 샷으로
        self.assertEqual(spec["goals"][1]["category"], "camera")

    def test_metrics_static_vs_moving(self):
        size = "832x480"
        first = videoio.prepare_image(self.img, size, self.tmp / "first.jpg", "crop")
        from PIL import Image
        self.assertEqual(Image.open(first).size, (832, 480))
        url = "data:image/jpeg;base64," + __import__("base64").b64encode(first.read_bytes()).decode()
        still = self.tmp / "still.mp4"
        still.write_bytes(mock_hub._make_video(url, size, 33, moving=False))
        moving = self.tmp / "moving.mp4"
        moving.write_bytes(mock_hub._make_video(url, size, 33, moving=True))
        a, b = videoio.motion_metrics(still, first), videoio.motion_metrics(moving, first)
        self.assertTrue(a["static"], a)
        self.assertFalse(b["static"], b)
        self.assertGreater(b["fidelity"], 0.8)
        fr = videoio.sample_frames(moving, self.tmp / "fr", k=5)
        self.assertEqual([f["pos"] for f in fr], ["start", "middle", "middle", "middle", "end"])

    def test_score_penalties(self):
        qs = [{"id": "q1", "kind": "goal", "question": "x", "weight": 1.5},
              {"id": "q2", "kind": "preserve", "question": "same person?", "weight": 2.0}]
        goals = [{"intensity": "moderate", "category": "action"}]
        judge = {"answers": {"q1": {"answer": "yes"}, "q2": {"answer": "yes"}}, "motion_quality": 5, "temporal_consistency": 5}
        s_ok, failed, pen = score(qs, judge, {"static": False}, goals)
        self.assertEqual((s_ok, failed, pen), (1.0, [], []))
        judge["answers"]["q2"]["answer"] = "no"
        s_bad, failed, pen = score(qs, judge, {"static": True}, goals)
        self.assertLess(s_bad, 0.3)
        self.assertEqual(len(pen), 2)

    def test_full_run_reflection_loop(self):
        st = run(RunConfig(image=self.img, request="바람에 머리카락이 날리고 천천히 줌인", rounds=3, candidates=1,
                           out_root=self.tmp / "out", name="one_shot"))
        shot = st["shots"][0]
        self.assertEqual(len(shot["rounds"]), 2)                 # 1라운드 정지 → 반성 → 2라운드 통과
        r1 = shot["rounds"][0]
        self.assertEqual(r1["reflection"]["route"], "strengthen_motion")
        self.assertTrue(any("정지" in p for p in r1["candidates"][0]["penalties"]))
        self.assertGreaterEqual(shot["best"]["score"], 0.85)
        fin = Path(st["final"]["file"])
        self.assertTrue(fin.exists() and fin.stat().st_size > 1000)
        run_dir = fin.parent
        self.assertTrue((run_dir / "report.html").exists())
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(data["size"], "480x640")              # 세로 사진(3:4) → 세로 영상 (Wan 2.2는 세로도 지원)
        self.assertTrue(all(r["has_image"] for r in mock_hub.STATE["requests"]))

    def test_multi_shot_and_rank(self):
        st = run(RunConfig(image=self.img, request="머리카락이 날리고 줌인, 그다음 고개를 돌려 미소", seconds=8,
                           rounds=1, candidates=2, out_root=self.tmp / "out", name="two_shots"))
        self.assertEqual([s["shot"]["id"] for s in st["shots"]], ["shot1", "shot2"])
        self.assertTrue(Path(st["shots"][1]["first_frame"]).name.startswith("shot2_first"))
        self.assertGreater(st["final"]["info"]["seconds"], 7.0)   # 4초 × 2샷

    def test_plan_only(self):
        st = run(RunConfig(image=self.img, request="줌아웃", plan_only=True, out_root=self.tmp / "out", name="plan"))
        self.assertNotIn("final", st)
        self.assertTrue(st["checklist"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
