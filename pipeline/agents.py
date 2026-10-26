"""LLM·VLM 에이전트: 관찰 → 이해 → 체크리스트 → 설계 → (묘사 → 답변 → 비교) → 반성.
각 함수는 Hub 호출 1회(+JSON 파싱 실패 시 1회 재요청)이고, 결과를 코드에서 한 번 더 정규화한다."""

from __future__ import annotations

import json
from pathlib import Path

from . import prompts as P
from .hub import Hub, video_spec
from .utils import extract_json, log

ROUTES = ("strengthen_motion", "fix_binding", "simplify", "fix_camera", "fix_physics", "protect_identity",
          "reduce_artifacts", "clarify")
CATEGORIES = ("action", "motion_direction", "camera", "dynamic_attribute", "interaction", "environment", "spatial",
              "numeracy")


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1)


def _ask_json(hub: Hub, system: str, user: str, images: list[tuple[str, Path]] | None = None,
              temperature: float = 0.3, video: Path | None = None) -> dict:
    for attempt in range(2):
        if video is not None:
            raw = hub.vision_video(user, video, images=images, system=system, temperature=temperature)
        elif images:
            raw = hub.vision(user, images, system=system, temperature=temperature)
        else:
            raw = hub.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           temperature=temperature)
        try:
            data = extract_json(raw)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
        log("    (JSON을 반환하지 못해 한 번 더 요청)")
        user = user + "\n\nReturn ONLY one valid JSON object. No explanation."
    raise ValueError("모델이 JSON을 반환하지 못했습니다. .env의 OASH_TEXT_MODEL / OASH_VISION_MODEL을 더 큰 모델로 바꿔 보세요.")


# ------------------------------------------------------------------ 0. 관찰 (RPG: recaption)
def observe(hub: Hub, image: Path) -> dict:
    d = _ask_json(hub, P.OBSERVE, "Analyse this image for animation.", images=[("", image)], temperature=0.1)
    subs = d.get("subjects") or []
    for i, s in enumerate(subs, 1):
        s.setdefault("id", f"s{i}")
    d["subjects"] = subs[:6]
    return d


# ------------------------------------------------------------------ 1. 이해
def understand(hub: Hub, request: str, scene: dict, hints: list[dict], limits: dict,
               answers: list[dict] | None = None) -> dict:
    hint_txt = [{"user_word": h["matched"], "category": h["category"], "means": h["en"]} for h in hints]
    user = (f"Scene analysis:\n{_j(scene)}\n\nUser request: {request}\n\n"
            f"Glossary hints:\n{_j(hint_txt) if hint_txt else '[]'}\n\n"
            f"Model limits: max_seconds_per_shot={limits['max_seconds']}, fps={limits['fps']}"
            + (f", user wants about {limits['seconds']} seconds in total" if limits.get("seconds") else ""))
    if answers:
        user += f"\n\nUser's answers to clarification questions (follow them):\n{_j(answers)}"
    spec = _ask_json(hub, P.UNDERSTAND, user, temperature=0.2)
    return normalize_spec(spec, limits)


def normalize_spec(spec: dict, limits: dict) -> dict:
    """LLM 출력 보정: 목표 id·범주·시점, 샷 길이 상한, 모든 목표가 어떤 샷에 속하도록."""
    maxs = float(limits["max_seconds"])
    goals = []
    for i, g in enumerate(spec.get("goals") or [], 1):
        g = dict(g)
        g["id"] = str(g.get("id") or f"g{i}")
        if g.get("category") not in CATEGORIES:
            g["category"] = "camera" if str(g.get("subject")) == "camera" else "action"
        if g.get("when") not in ("start", "middle", "end", "throughout"):
            g["when"] = "throughout"
        if g.get("intensity") not in ("subtle", "moderate", "strong"):
            g["intensity"] = "moderate"
        goals.append(g)
    spec["goals"] = goals
    ids = [g["id"] for g in goals]

    shots = []
    for i, s in enumerate((spec.get("shots") or [])[:4], 1):
        s = dict(s)
        s["id"] = f"shot{i}"
        try:
            sec = float(s.get("seconds") or maxs)
        except (TypeError, ValueError):
            sec = maxs
        s["seconds"] = max(1.0, min(sec, maxs))
        s["goals"] = [g for g in (s.get("goals") or []) if g in ids]
        shots.append(s)
    if not shots:
        want = float(limits.get("seconds") or maxs)
        shots = [{"id": "shot1", "seconds": min(want, maxs), "goals": ids, "note": spec.get("intent", "")}]
    assigned = {g for s in shots for g in s["goals"]}
    for gid in ids:                                       # 어느 샷에도 없는 목표는 첫 샷에 넣는다
        if gid not in assigned:
            shots[0]["goals"].append(gid)
    spec["shots"] = shots
    spec["total_seconds"] = round(sum(s["seconds"] for s in shots), 1)
    for k in ("preserve", "assumptions", "questions", "risks"):
        spec[k] = list(spec.get(k) or [])
    return spec


# ------------------------------------------------------------------ 2. 체크리스트
def checklist(hub: Hub, spec: dict, hints: list[dict]) -> list[dict]:
    hint_checks = [{"user_word": h["matched"], "means": h["en"], "check": h["check"]} for h in hints]
    user = (f"Video specification:\n{_j({k: spec[k] for k in ('intent', 'goals', 'shots', 'preserve', 'style')})}\n\n"
            f"Glossary check questions (reuse for matching goals):\n{_j(hint_checks) if hint_checks else '[]'}")
    d = _ask_json(hub, P.CHECKLIST, user, temperature=0.2)
    return normalize_checklist(d.get("questions") or [], spec)


def normalize_checklist(qs: list[dict], spec: dict) -> list[dict]:
    shot_of = {g: s["id"] for s in spec["shots"] for g in s["goals"]}
    out = []
    for i, q in enumerate(qs, 1):
        q = dict(q)
        q["id"] = f"q{i}"
        q["kind"] = q.get("kind") if q.get("kind") in ("goal", "preserve", "quality") else "quality"
        if q["kind"] == "goal" and q.get("ref") in shot_of:
            q["shot"] = shot_of[q["ref"]]
        elif q.get("shot") not in {s["id"] for s in spec["shots"]}:
            q["shot"] = "all"
        if q.get("when") not in ("start", "middle", "end", "throughout", "compare_start_end"):
            q["when"] = "throughout"
        try:
            q["weight"] = float(q.get("weight") or 1.0)
        except (TypeError, ValueError):
            q["weight"] = 1.0
        out.append(q)
    covered = {q.get("ref") for q in out if q["kind"] == "goal"}
    for g in spec["goals"]:                               # 질문이 빠진 목표는 기본 질문을 붙인다 (TIFA: 목표마다 1개 이상)
        if g["id"] not in covered:
            out.append({"id": f"q{len(out) + 1}", "kind": "goal", "ref": g["id"], "shot": shot_of.get(g["id"], "all"),
                        "when": "compare_start_end" if g["when"] != "start" else "start",
                        "question": f"Does the video show this: {g.get('change', '')}?",
                        "question_ko": f"{g['change_ko']} — 영상에 보이나요?" if g.get("change_ko") else "", "weight": 1.5})
    return out


# ------------------------------------------------------------------ 3. 설계
def design(hub: Hub, spec: dict, shot: dict, scene: dict, n: int, memory: list[dict],
           hints: list[dict]) -> list[dict]:
    model = hub.models().video
    fam = video_spec(model)["family"]
    system = P.DESIGN.format(model=model, style_guide=P.STYLE_GUIDES.get(fam, P.STYLE_GUIDES["generic"]), n=n)
    goals = [g for g in spec["goals"] if g["id"] in shot["goals"]]
    ctx = {"scene": {k: scene.get(k) for k in ("caption", "subjects", "background", "style")},
           "shot": {"id": shot["id"], "seconds": shot["seconds"], "note": shot.get("note", "")},
           "goals_of_this_shot": goals, "preserve": spec["preserve"], "style": spec.get("style", ""),
           "glossary": [{"user_word": h["matched"], "means": h["en"]} for h in hints]}
    user = f"{_j(ctx)}\n\nMemory of earlier rounds (oldest first):\n{_j(memory) if memory else '[] (first round)'}"
    cands = _clean_prompts(_ask_json(hub, system, user, temperature=0.7).get("prompts") or [], goals)
    missing_all = all(c["missing"] for c in cands) if cands else True
    if missing_all:                                       # VPO 정확성: 목표를 빠뜨린 후보뿐이면 한 번 고쳐 쓰게 한다
        miss = sorted({m for c in cands for m in c["missing"]}) or [g["id"] for g in goals]
        log(f"    (프롬프트가 목표 {', '.join(miss)}를 빠뜨려 다시 작성)")
        user += f"\n\nYour previous candidates did not cover goals {miss}. Every candidate must cover ALL goals of this shot."
        again = _clean_prompts(_ask_json(hub, system, user, temperature=0.5).get("prompts") or [], goals)
        cands = again or cands
    cands.sort(key=lambda c: len(c["missing"]))
    return cands[:n] if cands else [{"prompt": spec.get("intent", ""), "negative": "", "covers": [],
                                     "missing": [g["id"] for g in goals], "strategy_ko": "기본값"}]


def _clean_prompts(items: list[dict], goals: list[dict]) -> list[dict]:
    ids = {g["id"] for g in goals}
    out = []
    for c in items:
        if not isinstance(c, dict) or not str(c.get("prompt", "")).strip():
            continue
        covers = [x for x in (c.get("covers") or []) if x in ids]
        out.append({"prompt": c["prompt"].strip(), "negative": str(c.get("negative") or "").strip(),
                    "covers": covers, "missing": sorted(ids - set(covers)),
                    "strategy_ko": c.get("strategy_ko", "")})
    return out


# ------------------------------------------------------------------ 4. 검증
def _frame_images(frames: list[dict], input_img: Path | None) -> list[tuple[str, Path]]:
    imgs = [("INPUT (original still image):", input_img)] if input_img else []
    imgs += [(f"Frame at t={f['t']:.1f}s ({f['pos']}):", f["path"]) for f in frames]
    return imgs


def describe(hub: Hub, frames: list[dict], input_img: Path | None) -> dict:
    return _ask_json(hub, P.DESCRIBE, "Describe this video.", images=_frame_images(frames, input_img),
                     temperature=0.1)


def answer(hub: Hub, frames: list[dict], description: dict, questions: list[dict],
           input_img: Path | None) -> dict:
    qs = [{"id": q["id"], "when": q["when"], "question": q["question"]} for q in questions]
    user = f"Your earlier description:\n{_j(description)}\n\nQuestions:\n{_j(qs)}"
    d = _ask_json(hub, P.ANSWER, user, images=_frame_images(frames, input_img), temperature=0.1)
    ans = {}
    for q in questions:
        a = (d.get("answers") or {}).get(q["id"]) or {}
        val = str(a.get("answer", "no")).strip().lower()
        ans[q["id"]] = {"answer": "yes" if val.startswith("y") else "no", "evidence": a.get("evidence", "")}
    d["answers"] = ans
    for k in ("motion_quality", "temporal_consistency"):
        try:
            d[k] = max(1, min(5, int(d.get(k) or 3)))
        except (TypeError, ValueError):
            d[k] = 3
    return d


def rank(hub: Hub, sheets: list[tuple[str, Path]], request: str) -> dict:
    imgs = [(f"Candidate {cid}:", p) for cid, p in sheets]
    d = _ask_json(hub, P.RANK, f"Request: {request}", images=imgs, temperature=0.1)
    valid = [cid for cid, _ in sheets]
    order = [c for c in (d.get("ranking") or []) if c in valid]
    d["ranking"] = order + [c for c in valid if c not in order]
    return d


# ------------------------------------------------------------------ 5. 반성
def reflect(hub: Hub, spec: dict, shot: dict, cand: dict, history: list[dict]) -> dict:
    goals = [g for g in spec["goals"] if g["id"] in shot["goals"]]
    ctx = {"goals": goals, "preserve": spec["preserve"], "prompt_used": cand["prompt"],
           "observed": cand.get("judge", {}).get("revised_summary") or cand.get("description", {}).get("summary", ""),
           "observed_detail": cand.get("description", {}),
           "failed_checks": cand.get("failed", []), "metrics": cand.get("metrics", {}),
           "history": history}
    d = _ask_json(hub, P.REFLECT, _j(ctx), temperature=0.3)
    if d.get("route") not in ROUTES:
        d["route"] = "strengthen_motion" if cand.get("metrics", {}).get("static") else "fix_binding"
    return d


def merge_votes(runs: list[dict]) -> dict:
    """여러 번 받은 검사 결과를 다수결로 합친다 (Self-Consistency).
    질문 답·스타일 유지·심각한 결함은 과반일 때만 인정, 1~5 점수는 평균."""
    if len(runs) == 1:
        return runs[0]
    half = len(runs) / 2
    out = dict(runs[0])
    qids = {k for r in runs for k in (r.get("answers") or {})}
    ans = {}
    for q in qids:
        votes = [str(((r.get("answers") or {}).get(q) or {}).get("answer", "no")).strip().lower().startswith("y") for r in runs]
        yes = sum(votes)
        ev = next((((r.get("answers") or {}).get(q) or {}).get("evidence", "") for r, v in zip(runs, votes) if v == (yes > half)), "")
        ans[q] = {"answer": "yes" if yes > half else "no", "evidence": ev, "votes": f"{yes}/{len(runs)}"}
    out["answers"] = ans
    out["style_kept"] = sum(bool(r.get("style_kept", True)) for r in runs) > half
    sev = [r.get("severe_artifacts") or [] for r in runs]
    out["severe_artifacts"] = next((x for x in sev if x), []) if sum(1 for x in sev if x) > half else []
    for k in ("motion_quality", "temporal_consistency"):
        vals = [int(r[k]) for r in runs if str(r.get(k, "")).isdigit()]
        if vals:
            out[k] = round(sum(vals) / len(vals))
    out["_votes"] = len(runs)
    return out


def ask_clip(hub: Hub, system: str, user: str, anchor: tuple[str, Path] | None, frames: list[dict],
             clip: Path | None, temperature: float = 0.1) -> tuple[dict, str]:
    """생성 영상 검사: 가능하면 영상 자체를 VLM에 보내고(움직임을 직접 봄), 안 되면 시각 표시 프레임으로.
    토큰 모드의 votes만큼 여러 번 물어 다수결로 합친다 (max 모드 3회).
    돌려주는 값: (응답 JSON, 'video' | 'frames')."""
    from .budget import BUDGET
    n = BUDGET.mode["votes"]
    runs, mode = [], ""
    for i in range(n):
        try:
            r, mode = _ask_clip_once(hub, system, user, anchor, frames, clip, temperature if i == 0 else 0.6)
            runs.append(r)
        except (HubError, ValueError):
            if not runs and i == n - 1:
                raise
    return merge_votes(runs), mode


def _ask_clip_once(hub: Hub, system: str, user: str, anchor: tuple[str, Path] | None, frames: list[dict],
                   clip: Path | None, temperature: float = 0.1) -> tuple[dict, str]:
    if clip is not None and hub.video_ok:
        try:
            note = "\n(The generated clip itself is attached as VIDEO instead of separate frames.)"
            return _ask_json(hub, system, user + note, images=[anchor] if anchor else None, video=clip,
                             temperature=temperature), "video"
        except (HubError, ValueError) as e:
            log(f"    (영상 직접 입력 실패 → 프레임으로 검사: {str(e)[:120]})")
            hub.video_ok = False
    imgs = ([anchor] if anchor else []) + [(f"t={f['t']:.1f}s:", f["path"]) for f in frames]
    return _ask_json(hub, system, user, images=imgs, temperature=temperature), "frames"
