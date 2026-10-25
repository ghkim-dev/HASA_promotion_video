"""토큰 예산 테스트:  python tests/test_budget.py
모델 최대 길이 탐지 · 답변 길이 자동 설정 · 길이 초과 시 재요청 · 사용량 기록 · 하루 예산 초과 시 eco 전환 ·
다수결 검사 · 근거를 예산만큼 채우기 · 계획 후보 여러 개 + 심판 선택."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp())
os.environ.update(OASH_API_KEY="test", VS_STATE_DIR=str(TMP / "state"), VS_TOKEN_MODE="max")
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL", "OASH_EMBED_MODEL"):
    os.environ.pop(k, None)

import mock_hub                                     # noqa: E402
from pipeline import agents, story                  # noqa: E402
from pipeline.budget import BUDGET, MODES, est_tokens, pack  # noqa: E402
from pipeline.hub import Hub                         # noqa: E402
from pipeline.rag import Store                       # noqa: E402

PORT = 8770
os.environ["OASH_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"


class BudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mock_hub.serve(PORT)
        cls.hub = Hub()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_1_probe_and_out_tokens(self):
        msg = "This model's maximum context length is 262144 tokens. However, you requested 10000000 tokens"
        self.assertEqual(BUDGET.parse_limit(msg), 262144)
        m = self.hub.models().text
        self.assertEqual(BUDGET.context(self.hub, m), mock_hub.MOCK_CTX)          # 서버에서 읽음
        self.assertTrue((TMP / "state" / "model_limits.json").exists())           # 기억
        self.assertEqual(BUDGET.out_tokens(self.hub, m, 1000), 16384)
        self.assertEqual(BUDGET.out_tokens(self.hub, m, 125000), mock_hub.MOCK_CTX - 125000 - 512)

    def test_2_chat_uses_budget_and_records_usage(self):
        before = BUDGET.used_today()
        self.hub.chat([{"role": "user", "content": "안녕"}])
        self.assertEqual(mock_hub.STATE["max_tokens_seen"], 16384)
        self.assertGreater(BUDGET.used_today(), before)

    def test_3_retry_on_context_error(self):
        m = self.hub.models().text
        BUDGET.remember_limit(m, 10_000_000)                                       # 잘못 기억된 큰 값
        old = MODES["max"]["out_tokens"]
        MODES["max"]["out_tokens"] = 500_000
        try:
            out = self.hub.chat([{"role": "user", "content": "길이 테스트"}])      # 1차 400 → 한도 학습 → 재요청 성공
            self.assertTrue(out)
            self.assertEqual(BUDGET.limits()[m], mock_hub.MOCK_CTX)
        finally:
            MODES["max"]["out_tokens"] = old

    def test_4_votes(self):
        runs = [{"answers": {"k1": {"answer": "yes"}}, "severe_artifacts": ["melted face"], "style_kept": True, "motion_quality": 4},
                {"answers": {"k1": {"answer": "yes"}}, "severe_artifacts": [], "style_kept": True, "motion_quality": 5},
                {"answers": {"k1": {"answer": "no"}}, "severe_artifacts": [], "style_kept": False, "motion_quality": 3}]
        m = agents.merge_votes(runs)
        self.assertEqual(m["answers"]["k1"]["answer"], "yes")
        self.assertEqual(m["answers"]["k1"]["votes"], "2/3")
        self.assertEqual(m["severe_artifacts"], [])                                 # 한 번만 나온 결함은 무시
        self.assertTrue(m["style_kept"])
        self.assertEqual(m["motion_quality"], 4)

    def test_5_evidence_fills_budget(self):
        src = TMP / "big.txt"
        src.write_text("\n\n".join(f"{i}번 사실: 경북 클라우드팜의 서버 {i}대와 관련된 기록입니다." for i in range(300)), encoding="utf-8")
        st = Store(TMP / "store", self.hub)
        st.add([str(src)])
        ev = st.evidence("클라우드팜 서버 기록", token_budget=3000)
        lines = ev.splitlines()
        self.assertLessEqual(sum(est_tokens(x) for x in lines), 3000)
        self.assertGreater(len(lines), 14)                                          # 예전 상한(14개)보다 많이
        self.assertEqual(pack(["가" * 100, "나" * 10000, "다" * 50], 200), ["가" * 100, "다" * 50])

    def test_6_outline_best_of_n(self):
        mock_hub.STATE.update(outlines=0, picks=0)
        d = story.outline(self.hub, "센터 소개", {"caption": "building"}, 60)
        self.assertEqual(mock_hub.STATE["outlines"], MODES["max"]["plan_candidates"])
        self.assertEqual(mock_hub.STATE["picks"], 1)
        self.assertTrue(d["acts"])

    def test_7_daily_budget_downgrade(self):
        BUDGET.daily = 10                                                           # 이미 넘은 상태로
        BUDGET._downgraded = False
        self.assertEqual(BUDGET.mode, MODES["eco"])
        BUDGET.daily, BUDGET._downgraded = 20_000_000, False


if __name__ == "__main__":
    unittest.main(verbosity=2)
