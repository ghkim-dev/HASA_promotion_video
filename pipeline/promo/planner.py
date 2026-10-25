"""brief.json + 형식 템플릿 → storyboard_<형식>.json.

LLM은 '고르기'만 한다: 비트마다 사진(asset id)·움직임 레시피·보여줄 사실/문구 id·읽을 문구 id.
숫자와 문장은 brief에서 원문 확인(status=ok)된 것만 고를 수 있으므로 영상에 사실이 아닌 말이 들어갈 수 없다.
LLM 없이(--rules) 또는 LLM 결과가 규칙에 어긋나면 규칙 기반 기본값을 쓴다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .. import prompts as P
from ..agents import _ask_json, _j
from ..hub import Hub, HubError
from ..utils import log
from . import kb

NARR_TAG = {"tagline": ["tagline"], "pain": ["pain"], "mission": ["mission", "org"], "infra": ["infra", "offering"],
            "process": ["process", "offering"], "community": ["community"], "offering": ["offering"], "cta": ["cta"]}


def load_brief(project: Path) -> dict:
    f = Path(project) / "brief.json"
    if not f.exists():
        raise SystemExit(f"{f} 이 없습니다. 자료 읽기(promo ingest)가 끝나지 않았습니다. 먼저 실행해 주세요:\n"
                         f"    python video_studio.py promo ingest --input <자료 폴더> --name {Path(project).name}\n"
                         f"  마지막 줄에 '[promo] → …brief.json'이 나오면 성공입니다.")
    return json.loads(f.read_text(encoding="utf-8"))


def _cands(brief: dict, beat: dict, used: set[str]) -> dict:
    assets = [a for a in brief["assets"] if a.get("usable")]
    pref = beat.get("asset_kinds", [])
    want = beat.get("prefer_recipes", [])

    def rank(a):                                       # 안 쓴 사진 → 이 장면에 어울리는 레시피 → 사진 종류 순
        r = min([want.index(x) for x in a["recipes"][:1] if x in want] or [len(want)])
        return (a["id"] in used, r, pref.index(a["kind"]))
    alist = sorted([a for a in assets if a["kind"] in pref], key=rank)
    msgs = [m for m in brief["messages"] if m["status"] == "ok" and m["tag"] in beat.get("message_tags", [])]
    facts = [f for f in brief["facts"] if f["status"] == "ok" and set(f.get("tags", [])) & set(beat.get("fact_tags", []))]
    facts.sort(key=lambda f: f.get("priority", 9))     # 대표 숫자(priority 낮을수록)를 먼저 — 숏폼처럼 칸이 적을 때
    return {"assets": alist, "messages": msgs, "facts": facts}


def _fit_narration(ids: list[str], msgs: dict, budget: float) -> list[str]:
    out, n = [], 0
    for i in ids:
        if i in msgs and n + len(msgs[i]["text"]) <= budget:
            out.append(i)
            n += len(msgs[i]["text"])
    return out


def rule_plan(brief: dict, fmt: str) -> list[dict]:
    T = kb.templates()
    cps = T["narration_chars_per_second"]
    msgs = {m["id"]: m for m in brief["messages"] if m["status"] == "ok"}
    used: set[str] = set()
    out = []
    for i, b in enumerate(T["formats"][fmt]["beats"]):
        c = _cands(brief, b, used)
        asset = c["assets"][0] if (b["visual"] == "photo" and c["assets"]) else None
        recipe = None
        if asset:
            used.add(asset["id"])
            recipe = next((r for r in b.get("prefer_recipes", []) if r in asset["recipes"]), asset["recipes"][0])
        items = []
        if b.get("fact_tags"):
            items = [f["id"] for f in c["facts"][: b.get("max_items", 4)]]
        elif b.get("message_tags"):
            ms = c["messages"]
            if "nth" in b:
                ms = ms[b["nth"]:b["nth"] + 1]
            items = [m["id"] for m in ms[: b.get("max_items", 1)]]
        narr = []
        if b["narration"] != "none":
            pool = [m["id"] for t in NARR_TAG.get(b["narration"], []) for m in brief["messages"]
                    if m["status"] == "ok" and m["tag"] == t]
            if "nth" in b:
                pool = pool[b["nth"]:] + pool[:b["nth"]]
            narr = _fit_narration(pool, msgs, b["seconds"] * cps)
        out.append({"index": i, "asset": asset["id"] if asset else None,
                    "recipe": recipe, "overlay_items": items,
                    "narration": narr, "bridge": ""})
    return out


def llm_plan(hub: Hub, brief: dict, fmt: str, base: list[dict]) -> list[dict]:
    T = kb.templates()
    beats = []
    used: set[str] = set()
    for i, b in enumerate(T["formats"][fmt]["beats"]):
        c = _cands(brief, b, used)
        beats.append({"index": i, "role": b["role"], "seconds": b["seconds"], "visual": b["visual"],
                      "overlay": b["overlay"], "max_items": b.get("max_items", 1), "narration": b["narration"],
                      "allowed_assets": [{"id": a["id"], "kind": a["kind"], "subject": a["subject"],
                                          "recipes": a["recipes"]} for a in c["assets"]],
                      "allowed_items": [{"id": m["id"], "text": m["text"]} for m in c["messages"]] +
                                       [{"id": f["id"], "fact": f"{f['label']} {f['display']}{f['unit']}"} for f in c["facts"]]})
    all_msgs = [{"id": m["id"], "tag": m["tag"], "text": m["text"]} for m in brief["messages"] if m["status"] == "ok"]
    user = f"Organisation: {brief['org']}\n\nBeats:\n{_j(beats)}\n\nAll verified messages (for narration):\n{_j(all_msgs)}"
    d = _ask_json(hub, P.PROMO_PLAN.replace("{cps}", str(T["narration_chars_per_second"])), user, temperature=0.4)
    msgs = {m["id"]: m for m in brief["messages"] if m["status"] == "ok"}
    got = {int(x.get("index", -1)): x for x in d.get("beats") or [] if isinstance(x, dict)}
    out = []
    for b, tb, base_b in zip(beats, T["formats"][fmt]["beats"], base):
        x, r = got.get(b["index"], {}), dict(base_b)
        aids = {a["id"]: a for a in b["allowed_assets"]}
        if x.get("asset") in aids:
            r["asset"] = x["asset"]
            r["recipe"] = x["recipe"] if x.get("recipe") in aids[x["asset"]]["recipes"] else aids[x["asset"]]["recipes"][0]
        allowed = {it["id"] for it in b["allowed_items"]}
        items = [i for i in (x.get("overlay_items") or []) if i in allowed][: b["max_items"]]
        if items:
            r["overlay_items"] = items
        if tb["narration"] != "none" and x.get("narration"):
            narr = _fit_narration([i for i in x["narration"] if i in msgs], msgs, tb["seconds"] * T["narration_chars_per_second"])
            if narr:
                r["narration"] = narr
        bridge = re.sub(r"\d", "", str(x.get("bridge") or ""))[:15].strip()
        r["bridge"] = bridge if tb["narration"] != "none" else ""
        out.append(r)
    return out


def _join(parts: list[str]) -> str:
    """문장을 이을 때 끝에 문장부호가 없으면 마침표를 넣는다."""
    out = []
    for t in parts:
        t = t.strip()
        if t:
            out.append(t if t[-1] in ".!?…," else t + ".")
    return " ".join(out)


def resolve(brief: dict, fmt: str, picks: list[dict]) -> dict:
    """id를 실제 내용으로 펼친 스토리보드 (사람이 읽고 고치기 쉽게)."""
    T = kb.templates()
    f = T["formats"][fmt]
    assets = {a["id"]: a for a in brief["assets"]}
    items = {m["id"]: {"type": "message", **m} for m in brief["messages"]} | {x["id"]: {"type": "fact", **x} for x in brief["facts"]}
    beats, t = [], 0.0
    for b, p in zip(f["beats"], picks):
        a = assets.get(p["asset"]) if p["asset"] else None
        parts = ([p["bridge"]] if p.get("bridge") else []) + [items[i]["text"] for i in p["narration"] if i in items]
        narr = _join(parts)
        # 화면 그래픽에 이미 같은 문장이 있으면 자막은 생략 (내레이션 음성에는 그대로 씀)
        sub = _join([items[i]["text"] for i in p["narration"] if i in items and i not in p["overlay_items"]])
        beats.append({"index": p["index"], "role": b["role"], "start": round(t, 2), "seconds": b["seconds"],
                      "visual": b["visual"], "asset": a["id"] if a else None, "asset_file": a["file"] if a else None,
                      "asset_kind": a["kind"] if a else None, "asset_subject": a["subject"] if a else None,
                      "recipe": p["recipe"], "overlay": {"kind": b["overlay"], "items": p["overlay_items"]},
                      "narration_ids": p["narration"], "bridge": p.get("bridge", ""), "narration": narr, "subtitle": sub})
        t += b["seconds"]
    return {"format": fmt, "label": f["label"], "aspect": f["aspect"], "canvas": f["canvas"], "fps": f["fps"],
            "seconds": round(t, 2), "beats": beats}


def plan(project: Path, fmt: str, hub: Hub | None = None, use_llm: bool = True) -> Path:
    if fmt not in kb.templates()["formats"]:
        raise SystemExit(f"형식은 {', '.join(kb.templates()['formats'])} 중 하나입니다.")
    brief = load_brief(project)
    picks = rule_plan(brief, fmt)
    if use_llm:
        try:
            picks = llm_plan(hub or Hub(), brief, fmt, picks)
            log("[promo] 스토리보드: LLM이 사진·문구를 골랐습니다 (원문 확인된 것만).")
        except (HubError, ValueError) as e:
            log(f"[promo] 스토리보드: LLM 실패로 규칙 기반 구성을 씁니다 ({e})")
    sb = resolve(brief, fmt, picks)
    out = project / f"storyboard_{fmt}.json"
    out.write_text(json.dumps(sb, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[promo] {sb['label']} {sb['seconds']:g}초 · 비트 {len(sb['beats'])}개 → {out}")
    for b in sb["beats"]:
        log(f"  {b['start']:>5.1f}s {b['role']:<12} {str(b['asset']):<4} {str(b['recipe']):<14} {b['overlay']['kind']:<11} {b['narration'][:40]}")
    return out
