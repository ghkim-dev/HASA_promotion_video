"""움직임·카메라 용어집 (knowledge/motion_terms.json + 선택적으로 knowledge/custom_terms.json).

요청 문장 속 한국어 표현을 영상 모델이 잘 따르는 영어 표현과, 프레임 비교로 확인할 수 있는 검증 질문으로 바꾼다.
LLM이 이해 단계에서 참고하고(힌트), 체크리스트 단계에서 질문을 그대로 쓴다(생성·검증이 같은 정의를 공유).
"""

from __future__ import annotations

import json
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def load_terms() -> list[dict]:
    terms: dict[str, dict] = {}
    for name in ("motion_terms.json", "custom_terms.json"):     # custom이 같은 term을 덮어씀
        p = KB_DIR / name
        if p.exists():
            for t in json.loads(p.read_text(encoding="utf-8")).get("terms", []):
                terms[t["term"]] = t
    return list(terms.values())


def match(request: str) -> list[dict]:
    """요청 문장에 별칭이 들어 있는 용어. 긴 별칭이 먼저 잡히도록 정렬하고, 같은 범주의 상반된 용어가 둘 다 잡히면 둘 다 남긴다."""
    text = request.replace(" ", "")
    hits = []
    for t in load_terms():
        for a in sorted(t.get("aliases", []), key=len, reverse=True):
            if a.replace(" ", "") in text:
                hits.append({**t, "matched": a})
                break
    return hits
