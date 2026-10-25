"""토큰 예산 관리 — 모델별 최대 입력 길이를 알아내 그 안을 최대한 채워 쓰고, 하루 사용량을 기록한다.

모드 (VS_TOKEN_MODE, 기본 max)
  max       근거·자료를 컨텍스트의 60%까지 관련도순으로 채움, 답변 최대 16K, 검사 판정 3회 다수결,
            이야기 계획 후보 3개 중 최선 선택, 사진 설명 길게, VLM 이미지 1280px
  balanced  근거 25%, 답변 4K, 판정 1회
  eco       예전 방식 (근거 약 3,500자, 답변 2K)

모델 최대 길이: 한 번 '너무 긴 요청'을 보내 서버(vLLM)가 돌려주는 'maximum context length is N' 에서 N을 읽어
output/model_limits.json 에 기억한다. 못 읽으면 모델 카드 기준 표, 그것도 없으면 32K.
하루 사용량: 응답의 usage(prompt/completion 토큰)를 output/token_usage.json 에 날짜별로 더한다.
하루 예산(VS_DAILY_TOKEN_BUDGET, 기본 2천만)의 90%를 넘으면 그날은 자동으로 eco 모드로 내린다.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import date
from pathlib import Path

from .utils import log

MODES = {
    "max": {"out_tokens": 16384, "ctx_fill": 0.6, "votes": 3, "plan_candidates": 3, "frames": 8, "img_side": 1280,
            "evidence_k": 200, "rerank_pool": 100, "visuals_k": 30, "detailed_captions": True},
    "balanced": {"out_tokens": 4096, "ctx_fill": 0.25, "votes": 1, "plan_candidates": 1, "frames": 5, "img_side": 1024,
                 "evidence_k": 40, "rerank_pool": 50, "visuals_k": 15, "detailed_captions": False},
    "eco": {"out_tokens": 2048, "ctx_fill": 0.03, "votes": 1, "plan_candidates": 1, "frames": 4, "img_side": 768,
            "evidence_k": 14, "rerank_pool": 30, "visuals_k": 10, "detailed_captions": False},
}

# 서버에서 못 읽었을 때 쓰는 모델 카드 기준값 (Hub 배포 설정이 더 작을 수 있어 실제로는 탐지값이 우선)
KNOWN_CTX = {"qwen3-next": 262144, "qwen3-vl": 262144, "qwen3-coder": 262144, "qwen2.5-vl": 32768,
             "gpt-oss": 131072, "exaone-4.0": 131072, "llama-3.3": 131072, "deepseek": 131072,
             "kanana": 32768, "hyperclovax": 32768, "midm": 32768, "solar": 65536, "glm-4.7": 131072,
             "nemotron-super": 131072, "nemotron-nano": 131072, "nemotron-omni": 131072, "magistral": 40960,
             "ax-3.1": 32768, "schift-ocr": 32768, "paddleocr": 16384}
DEFAULT_CTX = 32768
LIMIT_RE = re.compile(r"(?:maximum context length|max_model_len|context length|context window)[^\d]{0,40}(\d{4,7})", re.I)

STATE_DIR = Path(os.getenv("VS_STATE_DIR", "output"))
_lock = threading.Lock()


def est_tokens(text: str) -> int:
    """대략적인 토큰 수 (한글 1자 ≈ 1토큰, 영문·숫자 약 3.5자 ≈ 1토큰). 넉넉하게 잡는다."""
    if not text:
        return 0
    non = sum(1 for c in text if ord(c) > 127)
    return int(non * 1.0 + (len(text) - non) / 3.5) + 8


def image_tokens(side: int) -> int:
    """VLM 이미지 1장의 대략 토큰 (28px 패치 기준)."""
    return int((side / 28) ** 2 * 0.6) + 64


class Budget:
    def __init__(self):
        self.mode_name = os.getenv("VS_TOKEN_MODE", "max").strip().lower()
        if self.mode_name not in MODES:
            self.mode_name = "max"
        self.daily = int(float(os.getenv("VS_DAILY_TOKEN_BUDGET", "20000000")))
        self._limits: dict[str, int] | None = None
        self._downgraded = False

    # ---------------------------------------------------------------- 모드
    @property
    def mode(self) -> dict:
        if not self._downgraded and self.mode_name != "eco" and self.used_today() >= 0.9 * self.daily:
            self._downgraded = True
            log(f"    (오늘 토큰 사용이 하루 예산의 90%를 넘어 eco 모드로 낮춥니다: {self.used_today():,}/{self.daily:,})")
        return MODES["eco"] if self._downgraded else MODES[self.mode_name]

    # ---------------------------------------------------------------- 모델 최대 길이
    def _limits_path(self) -> Path:
        return STATE_DIR / "model_limits.json"

    def limits(self) -> dict[str, int]:
        if self._limits is None:
            p = self._limits_path()
            self._limits = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return self._limits

    def remember_limit(self, model: str, n: int) -> None:
        with _lock:
            self.limits()[model] = int(n)
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            self._limits_path().write_text(json.dumps(self._limits, indent=1), encoding="utf-8")

    @staticmethod
    def parse_limit(text: str) -> int | None:
        m = LIMIT_RE.search(text or "")
        return int(m.group(1)) if m else None

    def context(self, hub, model: str) -> int:
        """모델 최대 길이 (토큰). 처음 한 번은 서버에 물어본다."""
        lim = self.limits().get(model)
        if lim:
            return lim
        n = None
        try:
            r = hub._req("POST", f"{hub.base}/chat/completions", headers=hub._h(), timeout=60,
                         json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10_000_000})
            n = self.parse_limit(r.text) if r.status_code >= 400 else None
            why = (f"서버 응답 {r.status_code}: {r.text[:120].strip()}" if r.status_code >= 400
                   else f"서버가 너무 긴 요청도 받아들임(응답 {r.status_code}) — Hub가 길이를 알려주지 않는 설정")
        except Exception as e:  # noqa: BLE001 — 탐지 실패는 표로 대체
            n, why = None, str(e)[:120]
        if not n:
            low = model.lower()
            n = next((v for k, v in KNOWN_CTX.items() if k in low), DEFAULT_CTX)
            log(f"    ({model} 최대 길이를 서버에서 읽지 못해 기준값 {n:,} 사용 · {why})")
        else:
            log(f"    ({model} 최대 입력 길이 {n:,} 토큰 확인)")
        self.remember_limit(model, n)
        return n

    def out_tokens(self, hub, model: str, input_tokens: int) -> int:
        ctx = self.context(hub, model)
        room = ctx - input_tokens - 512
        return max(256, min(self.mode["out_tokens"], room))

    def fill_tokens(self, hub, model: str, reserved: int = 4000) -> int:
        """근거·자료로 채울 수 있는 입력 토큰 (모드의 ctx_fill 비율, 답변·지시문 몫은 남긴다)."""
        ctx = self.context(hub, model)
        return max(1500, int(ctx * self.mode["ctx_fill"]) - reserved)

    # ---------------------------------------------------------------- 하루 사용량
    def _usage_path(self) -> Path:
        return STATE_DIR / "token_usage.json"

    def _usage(self) -> dict:
        p = self._usage_path()
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except ValueError:
            return {}

    def add_usage(self, model: str, usage: dict | None, fallback_in: int = 0, fallback_out: int = 0) -> None:
        pin = int((usage or {}).get("prompt_tokens") or fallback_in)
        pout = int((usage or {}).get("completion_tokens") or fallback_out)
        with _lock:
            u = self._usage()
            d = u.setdefault(date.today().isoformat(), {"prompt": 0, "completion": 0, "calls": 0, "by_model": {}})
            d["prompt"] += pin
            d["completion"] += pout
            d["calls"] += 1
            m = d["by_model"].setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
            m["prompt"] += pin
            m["completion"] += pout
            m["calls"] += 1
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            self._usage_path().write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8")

    def used_today(self) -> int:
        d = self._usage().get(date.today().isoformat(), {})
        return int(d.get("prompt", 0)) + int(d.get("completion", 0))

    def summary(self) -> str:
        d = self._usage().get(date.today().isoformat(), {"prompt": 0, "completion": 0, "calls": 0})
        tot = d["prompt"] + d["completion"]
        return (f"오늘 토큰 {tot:,} / {self.daily:,} ({100 * tot / max(1, self.daily):.1f}%) · 입력 {d['prompt']:,} · "
                f"출력 {d['completion']:,} · 호출 {d['calls']}회 · 모드 {('eco(자동)' if self._downgraded else self.mode_name)}")


BUDGET = Budget()


def pack(lines: list[str], token_budget: int) -> list[str]:
    """관련도순 줄들을 예산 안에서 최대한 담는다."""
    out, n = [], 0
    for ln in lines:
        t = est_tokens(ln)
        if n + t > token_budget:
            continue                                   # 긴 것 하나 때문에 멈추지 않고 뒤의 짧은 것을 더 담는다
        out.append(ln)
        n += t
    return out
