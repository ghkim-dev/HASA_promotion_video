"""장편 모드 오프라인 테스트:  python tests/test_long.py
1분짜리(샷 15개)로 계획 → 키프레임 → 샷 체인 → 재시도 → 중단 → 재개 → 조립(내레이션·자막)까지 확인."""

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
os.environ.setdefault("VS_STATE_DIR", _tf.mkdtemp())
os.environ["VS_TOKEN_MODE"] = "eco"   # 판정 1회 → 재시도 경로 확인 (max 모드의 다수결은 test_budget에서)   # 가짜 Hub의 길이·사용량이 실제 output/에 섞이지 않게
sys.path.insert(0, str(ROOT / "tests"))
os.environ.update(OASH_BASE_URL="http://127.0.0.1:8767/v1", OASH_API_KEY="test", VS_NET_RETRY_SLEEP="0")
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL"):
    os.environ.pop(k, None)

import mock_hub                                     # noqa: E402
from test_offline import sample_image                # noqa: E402
from pipeline import longrun, story, videoio         # noqa: E402
from pipeline.hub import HubError                    # noqa: E402

os.environ.update(OASH_BASE_URL="http://127.0.0.1:8767/v1")   # test_offline import가 덮어쓴 주소를 되돌림


class Long(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mock_hub.serve(8767)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.img = sample_image(cls.tmp / "photo.jpg")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_distribute(self):
        d = story.distribute(1200, [1, 2, 1], minimum=8)
        self.assertAlmostEqual(sum(d), 1200, places=1)
        self.assertGreater(d[1], d[0])
        self.assertTrue(all(x >= 8 for x in story.distribute(30, [100, 0.1, 0.1], minimum=8)))

    def test_long_run_interrupt_resume(self):
        cfg = longrun.LongConfig(image=self.img, request="바닷가에서 하루를 보내는 이야기", seconds=60,
                                 narration=True, out_root=self.tmp / "out", name="long")
        mock_hub.STATE.update(videos=0, fail_from=9)            # 9번째 영상 작업부터 GPU 오류 → 중단
        with self.assertRaises(HubError):
            longrun.start(cfg)
        run_dir = next((self.tmp / "out").glob("*_long"))
        plan = story.load(run_dir)
        self.assertEqual(plan["n_shots"], 15)                  # 60초 / 4초
        self.assertAlmostEqual(sum(s["seconds"] for s in plan["scenes"]), 60, places=1)
        self.assertTrue(all(len(s["shots"]) == 5 for s in plan["scenes"]))   # 모자란 샷은 코드가 채움
        prog = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        done_before = [k for k, v in prog["shots"].items() if v.get("status") == "done"]
        self.assertTrue(0 < len(done_before) < 15)
        jobs_before = mock_hub.STATE["videos"]

        prev = longrun.assemble(run_dir)                        # 중단된 상태에서도 미리보기
        self.assertTrue(prev and prev.exists())

        mock_hub.STATE["fail_from"] = 0                         # 복구 → 이어서
        longrun.resume(run_dir)
        prog = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(1 for v in prog["shots"].values() if v.get("status") == "done"), 15)
        for k in done_before:                                    # 끝난 샷은 다시 만들지 않음
            self.assertEqual(prog["shots"][k]["file"], json.loads(json.dumps(prog["shots"][k]))["file"])
        self.assertLess(mock_hub.STATE["videos"] - jobs_before, 15 * 2)
        fin = run_dir / "final.mp4"
        self.assertTrue(fin.exists())
        secs = videoio.media_seconds(fin)
        self.assertGreater(secs, 55)                             # 이음매 프레임 제거로 약간 짧아질 수 있음
        self.assertLess(secs, 63)
        self.assertTrue((run_dir / "final.srt").read_text(encoding="utf-8").strip())
        self.assertGreater(mock_hub.STATE["edits"], 0)           # new_keyframe 씬은 원본 편집
        self.assertGreater(mock_hub.STATE["tts"], 0)
        self.assertIn("남은 시간", (run_dir / "report.html").read_text(encoding="utf-8"))
        clean = [v for v in prog["shots"].values() if v["attempts"][-1]["failed"] == []]
        self.assertTrue(clean)                                   # 모두 '예'로 답한 샷은 실패 질문이 없어야 함 (답변 범위 버그 방지)
        retried = [v for v in prog["shots"].values() if len(v.get("attempts", [])) > 1]
        self.assertTrue(retried)                                 # 점수 미달 샷은 재시도
        print(f"\n  final {secs:.1f}s, video jobs {mock_hub.STATE['videos']}, retried shots {len(retried)}, "
              f"events {len(prog['events'])}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
