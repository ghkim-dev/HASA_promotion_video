"""홍보 모드 지식: 형식 템플릿(knowledge/promo/templates.json)과 움직임 레시피(motion_recipes.json)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

KB = Path(__file__).resolve().parents[2] / "knowledge" / "promo"
ASSET_KINDS = ("real_building", "real_interior", "real_people", "real_landscape", "concept_tech", "concept_city",
               "graphic_text", "logo", "blank")
DEFAULT_BRAND = {"primary": "#2f5bd3", "secondary": "#5b3fd0", "accent": "#36c6f0", "dark": "#101a3d",
                 "light": "#eef3ff", "text": "#ffffff", "font": "", "font_bold": "", "logo": ""}


@lru_cache(None)
def templates() -> dict:
    return json.loads((KB / "templates.json").read_text(encoding="utf-8"))


@lru_cache(None)
def recipes() -> dict:
    return json.loads((KB / "motion_recipes.json").read_text(encoding="utf-8"))


def recipes_for(kind: str, subject: str) -> list[str]:
    """사진 종류에 맞는 레시피. 설명(subject)에 키워드가 맞는 것을 앞으로."""
    rs = recipes()["recipes"]
    subj = (subject or "").lower()
    ok = [k for k, r in rs.items() if kind in r["kinds"]]
    ok.sort(key=lambda k: -sum(1 for w in rs[k].get("match", []) if w in subj))
    return ok or [recipes()["fallback"]]
