"""장편(10~20분) 계층 계획: 이야기 설정(bible)·막(act) → 씬(scene) → 샷(shot).

한 번의 LLM 호출로 수백 개 샷을 쓰게 하면 빠뜨리거나 반복하므로, Vlogger(CVPR'24)처럼 위에서 아래로 나눠 계획한다.
막 1회 + 막마다 씬 1회 + 씬마다 샷 1회. 길이는 코드가 정확히 맞춘다(LLM이 준 초를 비율대로 다시 나눔).
계획은 단계마다 plan.json에 저장되므로 계획 도중 끊겨도 이어서 계획한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import prompts as P
from .agents import _ask_json, _j
from .hub import Hub, HubError, video_spec
from .utils import log

CHARS_PER_SEC = 4           # 한국어 내레이션 글자 수 상한 (초당)
MAX_SHOTS_PER_SCENE = 15    # 씬 하나에 샷이 이보다 많으면 '이어지는 씬'으로 쪼갠다 (샷 계획 호출 크기 제한)


def distribute(total: float, weights: list[float], minimum: float) -> list[float]:
    """total을 weights 비율로 나누되 각 몫이 minimum 이상이 되게 (합은 정확히 total)."""
    n = len(weights)
    if n == 0:
        return []
    minimum = min(minimum, total / n)
    w = [max(float(x or 0), 0.01) for x in weights]
    spare = total - minimum * n
    s = sum(w)
    out = [minimum + spare * x / s for x in w]
    out[-1] += total - sum(out)
    return [round(x, 2) for x in out]


def save(plan: dict, run_dir: Path) -> None:
    tmp = run_dir / "plan.json.tmp"
    tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(run_dir / "plan.json")


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ L1 막
def outline(hub: Hub, request: str, scene: dict, total: float, kb=None) -> dict:
    spec = video_spec(hub.models().video)
    n_acts = max(1, min(8, round(total / 180)))
    user = (f"Photo analysis:\n{_j(scene)}\n\nUser request: {request}\n\nTotal length: {total:g} seconds. "
            f"Use exactly {n_acts} act(s). Model limits: each clip at most {spec['max_seconds']} s, {spec['fps']} fps.")
    if kb is not None:                                     # 사용자 자료에서 찾은 근거 (사실·장소·사건)
        ev = kb.evidence(request)                          # 모델 최대 길이 × 모드 비율까지 관련도순으로
        if ev:
            user += f"\n\nSource material (from the user's documents, videos and websites; ids in brackets):\n{ev}"
    from .budget import BUDGET
    n_cand = BUDGET.mode["plan_candidates"]
    cands = []
    for i in range(n_cand):                                # 후보 여러 개 (온도를 달리해 다양하게)
        try:
            cands.append(_ask_json(hub, P.OUTLINE, user, temperature=0.6 if i == 0 else 0.9))
        except (HubError, ValueError) as e:
            log(f"    (계획 후보 {i + 1} 실패: {e})")
    if not cands:
        raise HubError("이야기 계획을 만들지 못했습니다.")
    d = cands[0]
    if len(cands) > 1:                                     # 심판 호출로 가장 나은 후보 선택
        lines = {f"o{i + 1}": {"title": c.get("title"), "logline_ko": c.get("logline_ko"), "bible": c.get("bible"),
                               "acts": c.get("acts")} for i, c in enumerate(cands)}
        try:
            pick = _ask_json(hub, P.OUTLINE_PICK, f"{user}\n\nCandidates:\n{_j(lines)}", temperature=0.1)
            best = str(pick.get("best", "o1"))
            if best in lines:
                d = cands[int(best[1:]) - 1]
                log(f"    (계획 후보 {len(cands)}개 중 {best} 선택: {pick.get('reason_ko', '')})")
        except (HubError, ValueError) as e:
            log(f"    (후보 선택 실패, 첫 후보 사용: {e})")
    acts = (d.get("acts") or [])[:8] or [{"title": "Act 1", "summary": request}]
    secs = distribute(total, [a.get("seconds", 1) for a in acts], minimum=8)
    for i, (a, s) in enumerate(zip(acts, secs), 1):
        a["id"], a["seconds"] = f"a{i}", s
    d["acts"] = acts
    d.setdefault("bible", {})
    return d


# ------------------------------------------------------------------ L2 씬
def scenes_for_act(hub: Hub, plan: dict, act: dict, prev_summary: str, kb=None) -> list[dict]:
    lo, hi = max(1, math.floor(act["seconds"] / 45)), max(1, math.ceil(act["seconds"] / 15))
    user = (f"Story bible:\n{_j(plan['bible'])}\n\nThis act ({act['seconds']:g} s): {_j(act)}\n\n"
            f"What happened just before: {prev_summary or '(start of the film; the first frame is the original photo)'}\n\n"
            f"Write between {lo} and {hi} scenes.")
    vis_ids: dict[str, dict] = {}
    if kb is not None:
        q = f"{act.get('title', '')} {act.get('summary', '')}"
        from .budget import BUDGET
        ev = kb.evidence(q, token_budget=BUDGET.fill_tokens(hub, hub.models().text) // 2)
        if ev:
            user += f"\n\nSource material for this act:\n{ev}"
        vis = kb.visuals(q)                                # 사용자 사진·영상 구간 = 씬 첫 프레임 후보
        if vis:
            vis_ids = {v["id"]: v for v in vis}
            lines = [f"[{v['id']}] {v['modality']} · {v['source']} {v['loc']} · {v['text']}" for v in vis]
            user += "\n\nSource visuals you may use as a scene's first frame (visual_ref):\n" + "\n".join(lines)
    d = _ask_json(hub, P.SCENES.replace("{chars_per_sec}", str(CHARS_PER_SEC)), user, temperature=0.6)
    raw = d.get("scenes") or [{"title": act.get("title", ""), "events": [act.get("summary", "")]}]
    secs = distribute(act["seconds"], [s.get("seconds", 1) for s in raw], minimum=4)
    out = []
    for s, sec in zip(raw, secs):
        vref = str(s.get("visual_ref") or "")
        s = {k: s.get(k, "") for k in ("title", "location", "time_of_day", "start", "keyframe", "mood", "narration_ko")} | \
            {"events": [str(e) for e in (s.get("events") or [])], "seconds": sec, "act": act["id"],
             "visual_ref": vref if vref in vis_ids else ""}      # 목록에 없는 id는 버린다
        if s["visual_ref"]:
            s["start"] = "new_keyframe"
        if not s["visual_ref"]:
            s["start"] = "continue" if s["start"] == "continue" or not s["keyframe"] else "new_keyframe"
        s["narration_ko"] = str(s["narration_ko"])[: int(sec * CHARS_PER_SEC)]
        out.append(s)
    return out


def _split_long(scenes: list[dict], max_s: float) -> list[dict]:
    res = []
    for s in scenes:
        n = math.ceil(s["seconds"] / max_s - 1e-6)
        parts = math.ceil(n / MAX_SHOTS_PER_SCENE)
        if parts <= 1:
            res.append(s)
            continue
        ev = s["events"]
        for p in range(parts):
            q = dict(s)
            q["seconds"] = round(s["seconds"] / parts, 2)
            q["events"] = ev[p * len(ev) // parts:(p + 1) * len(ev) // parts] or ev[-1:]
            if p:
                q.update(start="continue", keyframe="", narration_ko="", visual_ref="", title=f"{s['title']} ({p + 1})")
            res.append(q)
    return res


# ------------------------------------------------------------------ L3 샷
def shots_for_scene(hub: Hub, plan: dict, sc: dict, prev_end: str, kb=None) -> list[dict]:
    spec = video_spec(hub.models().video)
    n = max(1, math.ceil(sc["seconds"] / spec["max_seconds"] - 1e-6))
    sec = round(sc["seconds"] / n, 2)
    style = P.STYLE_GUIDES.get(spec["family"], P.STYLE_GUIDES["generic"])
    system = P.SHOTS.format(model=hub.models().video, n=n, sec=sec, style_guide=style)
    first = ("the original photo" if sc["index"] == 0 else
             ("the last frame of the previous scene: " + prev_end) if sc["start"] == "continue" else
             "a new keyframe: " + sc["keyframe"])
    whole = [{"id": x["id"], "title": x.get("title"), "events": x.get("events")} for x in plan.get("scenes", [])]
    user = (f"Story bible:\n{_j(plan['bible'])}\n\nWhole film (for continuity; you write only the scene below):\n{_j(whole)}"
            f"\n\nScene:\n{_j({k: sc[k] for k in ('id', 'title', 'location', 'time_of_day', 'events', 'mood', 'seconds')})}"
            f"\n\nScene first frame: {first}")
    if sc.get("visual_ref") and sc.get("visual_desc"):
        user += f" (a real frame from the user's material: {sc['visual_desc']})"
    if kb is not None:
        mo = kb.motions(" ".join(sc.get("events") or [sc.get("title", "")]), k=3)
        if mo:
            user += "\n\nreference_motion (from the user's source videos):\n- " + "\n- ".join(mo)
    d = _ask_json(hub, system, user, temperature=0.6)
    shots = [s for s in (d.get("shots") or []) if isinstance(s, dict) and s.get("prompt")]
    if not shots:
        shots = [{"event": e, "prompt": e} for e in (sc["events"] or [sc["title"]])]
    while len(shots) < n:                                  # 모자라면 마지막 샷을 '천천히 이어지는' 샷으로 늘린다
        last = dict(shots[-1])
        last["event"] = "continues: " + str(last.get("event", ""))
        last["prompt"] = str(last["prompt"]) + " The motion continues slowly and smoothly; the camera drifts gently."
        shots.append(last)
    out = []
    for i, s in enumerate(shots[:n], 1):
        checks = [c for c in (s.get("checks") or []) if isinstance(c, dict) and c.get("question")][:2]
        out.append({"id": f"{sc['id']}_s{i:02d}", "seconds": sec, "event": s.get("event", ""), "prompt": s["prompt"],
                    "negative": s.get("negative", ""), "end_state": s.get("end_state", ""),
                    "motion": s.get("motion") if s.get("motion") in ("subtle", "moderate", "strong") else "moderate",
                    "checks": checks or [{"question": f"Does the clip show: {s.get('event') or s['prompt'][:120]}?",
                                          "when": "throughout"}]})
    return out


# ------------------------------------------------------------------ 전체
def make_plan(hub: Hub, run_dir: Path, request: str, scene_analysis: dict, total: float, kb=None) -> dict:
    """plan.json이 있으면 빠진 단계만 이어서 계획한다."""
    max_s = video_spec(hub.models().video)["max_seconds"]
    p = run_dir / "plan.json"
    plan = load(run_dir) if p.exists() else {}
    if not plan.get("acts"):
        log("[plan] L1 이야기 설정·막 구성")
        o = outline(hub, request, scene_analysis, total, kb)
        plan = {"request": request, "total_seconds": total, "title": o.get("title", ""),
                "logline_ko": o.get("logline_ko", ""), "bible": o["bible"], "acts": o["acts"], "scenes": []}
        save(plan, run_dir)
    done_acts = {s["act"] for s in plan["scenes"]}
    for a in plan["acts"]:
        if a["id"] in done_acts:
            continue
        log(f"[plan] L2 {a['id']} ({a['seconds']:g}초) 씬 나누기")
        prev = plan["scenes"][-1]["events"][-1] if plan["scenes"] and plan["scenes"][-1]["events"] else ""
        plan["scenes"] += _split_long(scenes_for_act(hub, plan, a, prev, kb), max_s)
        for i, s in enumerate(plan["scenes"]):
            s["id"], s["index"] = f"sc{i + 1:02d}", i
        plan["scenes"][0].update(start="continue", keyframe="", visual_ref="")   # 첫 씬은 입력 사진에서 시작
        save(plan, run_dir)
    prev_end = ""
    for sc in plan["scenes"]:
        if not sc.get("shots"):
            log(f"[plan] L3 {sc['id']} {sc['title']} ({sc['seconds']:g}초) 샷 계획")
            if sc.get("visual_ref") and kb is not None and not sc.get("visual_desc"):
                v = kb.get(sc["visual_ref"])
                sc["visual_desc"] = (v or {}).get("meta", {}).get("caption", "") if v else ""
            sc["shots"] = shots_for_scene(hub, plan, sc, prev_end, kb)
            save(plan, run_dir)
        prev_end = sc["shots"][-1].get("end_state", "")
    plan["n_shots"] = sum(len(s["shots"]) for s in plan["scenes"])
    save(plan, run_dir)
    return plan
