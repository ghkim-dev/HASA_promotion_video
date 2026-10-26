"""이미지 + 요구사항 → 영상.

관찰(0) → 이해(1) → 체크리스트(2) → 샷마다 [ 설계(3) → 생성 → 검증(4: 묘사·답변·지표·비교) → 반성(5) ] × 라운드
→ 샷 이어 붙이기 → report.html / run.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import agents, lexicon, videoio
from .hub import MAX_CONCURRENCY, Hub, HubError, video_spec
from .report import write_report
from .utils import log


@dataclass
class RunConfig:
    image: Path
    request: str
    rounds: int = 2                 # 샷당 설계→생성→검증→반성 최대 반복
    candidates: int = 1             # 라운드당 후보 영상 수 (best-of-N)
    threshold: float = 0.85         # 이 점수 이상이면 그 샷은 조기 종료
    seconds: float | None = None    # 원하는 전체 길이(초). 샷 최대 길이보다 길면 여러 샷으로 나눔
    size: str | None = None         # 비우면 모델 해상도 중 사진 비율과 가장 가까운 것
    fit: str = "crop"               # crop | pad | none — 사진을 영상 해상도에 맞추는 방식
    frames: int = 5                 # 검증에 쓰는 프레임 수
    plan_only: bool = False         # 0~2단계만 (생성 없음)
    out_root: Path = Path("output")
    name: str = "run"
    ask_fn: Callable[[list[dict]], list[dict] | None] | None = None   # 확인 질문에 사용자가 답하는 함수 (CLI --ask)
    on_event: Callable[[dict], None] | None = None


# ------------------------------------------------------------------ 점수
def severe_artifacts(chk: dict) -> list:
    """심각한 결함만. 실제 Hub 실행에서 VLM이 사소한 흠까지 'artifacts'로 적어 샷의 절반이 불필요하게
    재시도됐으므로, severe_artifacts가 있으면 그것만 본다 (옛 형식 artifacts는 심각 표현이 있을 때만)."""
    if "severe_artifacts" in chk:
        return [a for a in (chk.get("severe_artifacts") or []) if str(a).strip()]
    words = ("melt", "duplicat", "extra", "missing limb", "distort", "garbled", "jump", "broken", "merg", "deform")
    return [a for a in (chk.get("artifacts") or []) if any(w in str(a).lower() for w in words)]



def score(questions: list[dict], judge: dict, metrics: dict, goals: list[dict]) -> tuple[float, list[dict], list[str]]:
    """TIFA식 가중 정답률 → 품질(1~5) 소폭 반영 → 하드 페널티(정체성·정지·컷 튐·첫 프레임 이탈)."""
    ans = judge.get("answers", {})
    tot = sum(q["weight"] for q in questions) or 1.0
    got = sum(q["weight"] for q in questions if ans.get(q["id"], {}).get("answer") == "yes")
    base = got / tot
    qual = (judge.get("motion_quality", 3) + judge.get("temporal_consistency", 3)) / 10.0
    s = 0.85 * base + 0.15 * qual
    failed = [{"id": q["id"], "kind": q["kind"], "ref": q.get("ref"), "question": q["question"],
               "evidence": ans.get(q["id"], {}).get("evidence", "")}
              for q in questions if ans.get(q["id"], {}).get("answer") != "yes"]
    pen = []
    if any(f["kind"] == "preserve" and next(q for q in questions if q["id"] == f["id"])["weight"] >= 2 for f in failed):
        s *= 0.6
        pen.append("정체성 유지 실패 ×0.6")
    wants_motion = any(g["intensity"] != "subtle" or g["category"] == "camera" for g in goals)
    if metrics.get("static") and wants_motion:
        s *= 0.7
        pen.append(f"거의 정지(움직임 {metrics.get('motion')}) ×0.7")
    if metrics.get("jump"):
        s *= 0.85
        pen.append("급격한 프레임 변화(컷 튐) ×0.85")
    fid = metrics.get("fidelity")
    if fid is not None and fid < 0.5:
        s *= 0.85
        pen.append(f"첫 프레임이 입력과 다름(유사도 {fid}) ×0.85")
    return round(s, 3), failed, pen


# ------------------------------------------------------------------ 실행
def run(cfg: RunConfig, hub: Hub | None = None) -> dict:
    hub = hub or Hub()

    def emit(stage: str, message: str, **extra) -> None:
        log(f"[{stage}] {message}")
        if cfg.on_event:
            try:
                cfg.on_event({"stage": stage, "message": message, **extra})
            except Exception:
                pass

    m = hub.models()
    if not m.video:
        raise HubError("영상 모델을 찾지 못했습니다. .env의 OASH_VIDEO_MODEL에 wan2.2-i2v 등을 지정하세요.")
    spec_v = video_spec(m.video)
    run_dir = cfg.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_img = run_dir / f"input{cfg.image.suffix.lower() or '.jpg'}"
    shutil.copy(cfg.image, input_img)
    size = cfg.size or videoio.choose_size(input_img, spec_v["sizes"])

    state: dict = {"config": {k: str(v) for k, v in cfg.__dict__.items() if k not in ("ask_fn", "on_event")},
                   "models": {"text": m.text, "vision": m.vision, "video": m.video},
                   "video_spec": spec_v, "size": size, "input": str(input_img),
                   "started": datetime.now().isoformat(timespec="seconds"), "shots": []}

    def save() -> None:
        state["hub_calls"] = dict(hub.calls)
        (run_dir / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
        write_report(state, run_dir)

    log(f"\n=== {cfg.name} | 저장={run_dir}\n    모델: text={m.text} vision={m.vision} video={m.video} ({size})")

    def rel(p) -> str:                                    # 화면(웹)에서 쓰는 run_dir 기준 상대 경로
        try:
            return Path(p).resolve().relative_to(run_dir.resolve()).as_posix()
        except (ValueError, TypeError):
            return ""

    from PIL import Image as _Image
    from .budget import vlm_pixels
    from .utils import fit_pixels, vlm_tokens
    with _Image.open(input_img) as _im:
        in_w, in_h = _im.size
    vw, vh = fit_pixels(in_w, in_h, vlm_pixels(1))
    state["resolution"] = {"input": [in_w, in_h], "vlm": [vw, vh], "vlm_tokens": vlm_tokens(vw, vh), "video": size}
    log(f"    해상도: 입력 {in_w}×{in_h} → AI가 보는 사진 {vw}×{vh} (약 {vlm_tokens(vw, vh):,}토큰) → 영상 {size}")
    emit("start", "시작", data={"run": run_dir.name, "input": rel(input_img), "size": size,
                                "models": state["models"], "request": cfg.request, "resolution": state["resolution"]})

    # 0. 관찰
    emit("observe", "사진 장면 분석 (RPG식 재캡션)")
    scene = agents.observe(hub, input_img)
    state["scene"] = scene
    emit("observe_done", scene.get("caption", ""), data={k: scene.get(k) for k in (
        "caption", "subjects", "background", "environment_dynamics", "camera", "lighting", "style", "identity_anchors")})
    # 0.5 용어
    hints = lexicon.match(cfg.request)
    state["terms"] = hints
    if hints:
        emit("terms", "용어 인식: " + ", ".join(f"{h['matched']}→{h['term']}" for h in hints))
    # 1. 이해
    limits = {"max_seconds": spec_v["max_seconds"], "fps": spec_v["fps"], "seconds": cfg.seconds}
    emit("understand", "요구사항 이해 (움직임 중심 분해·범주·물리 규칙·샷 계획)")
    spec = agents.understand(hub, cfg.request, scene, hints, limits)
    if spec["questions"] and cfg.ask_fn:
        answers = cfg.ask_fn(spec["questions"])
        if answers:
            state["clarifications"] = answers
            emit("understand", "확인 답변을 반영해 다시 이해")
            spec = agents.understand(hub, cfg.request, scene, hints, limits, answers=answers)
    state["spec"] = spec
    emit("understand_done", spec.get("intent_ko") or spec.get("intent", ""), data={k: spec.get(k) for k in (
        "intent", "intent_ko", "goals", "shots", "preserve", "assumptions", "risks")})
    # 2. 체크리스트
    emit("checklist", "검증 질문 작성 (TIFA·Video-Bench·VBench)")
    state["checklist"] = agents.checklist(hub, spec, hints)
    emit("checklist_done", f"질문 {len(state['checklist'])}개", data={"questions": state["checklist"]})
    save()
    if cfg.plan_only:
        emit("done", f"계획만 실행 완료 → {run_dir / 'report.html'}")
        return state

    # 3~5. 샷별 루프
    first = videoio.prepare_image(input_img, size, run_dir / "shot1_first.jpg", cfg.fit)
    finals: list[Path] = []
    for si, shot in enumerate(spec["shots"]):
        sdir = run_dir / shot["id"]
        goals = [g for g in spec["goals"] if g["id"] in shot["goals"]]
        qs = [q for q in state["checklist"] if q["shot"] in (shot["id"], "all")]
        srec: dict = {"shot": shot, "first_frame": str(first), "rounds": [], "best": None}
        state["shots"].append(srec)
        memory: list[dict] = []
        emit("shot", f"{shot['id']} ({shot['seconds']:g}초) 목표 {', '.join(shot['goals'])}")

        for r in range(1, cfg.rounds + 1):
            rdir = sdir / f"round{r}"
            emit("design", f"{shot['id']} 라운드 {r}: 프롬프트 후보 {cfg.candidates}개 설계")
            cands = agents.design(hub, spec, shot, scene, cfg.candidates, memory, hints)
            for ci, c in enumerate(cands, 1):
                c["id"] = f"c{ci}"
            emit("design_done", f"{shot['id']} 라운드 {r} 프롬프트", data={
                "shot": shot["id"], "round": r, "first": rel(first),
                "candidates": [{k: c.get(k) for k in ("id", "prompt", "strategy_ko", "covers", "missing")} for c in cands]})
            _generate_all(hub, cands, rdir, first, shot["seconds"], size, emit)

            for c in cands:
                if not c.get("file"):
                    c["score"] = 0.0
                    continue
                emit("verify", f"{shot['id']} 라운드 {r} {c['id']}: 프레임 묘사 → 질문 답변 → 지표")
                vid = Path(c["file"])
                frames = videoio.sample_frames(vid, rdir / "frames", k=cfg.frames)
                c["frames"] = [str(f["path"]) for f in frames]
                sheet = videoio.contact_sheet([f["path"] for f in frames], rdir / f"{c['id']}_sheet.jpg")
                c["sheet"] = str(sheet) if sheet else ""
                c["metrics"] = videoio.motion_metrics(vid, first)
                try:
                    c["description"] = agents.describe(hub, frames, first)
                    c["judge"] = agents.answer(hub, frames, c["description"], qs, first)
                except (HubError, ValueError) as e:
                    c["error"] = f"검증 실패: {e}"
                    c["score"] = 0.0
                    continue
                c["score"], c["failed"], c["penalties"] = score(qs, c["judge"], c["metrics"], goals)
                if c.get("missing"):                      # VPO 정확성: 프롬프트가 목표를 빠뜨렸으면 소폭 감점
                    c["score"] = round(c["score"] * 0.97 ** len(c["missing"]), 3)
                    c["penalties"].append(f"프롬프트에 목표 {', '.join(c['missing'])} 누락 ×{0.97 ** len(c['missing']):.2f}")
                emit("verify", f"    {c['id']} 점수 {c['score']:.2f}" + (f" ({'; '.join(c['penalties'])})" if c["penalties"] else ""))
                ans = c["judge"].get("answers", {})
                emit("verify_done", f"{c['id']} 점수 {c['score']:.2f}", data={
                    "shot": shot["id"], "round": r, "cand": c["id"], "score": c["score"], "video": rel(vid),
                    "sheet": rel(c["sheet"]) if c["sheet"] else "", "frames": [rel(f) for f in c["frames"]],
                    "summary": c["description"].get("summary", ""), "camera": c["description"].get("camera", ""),
                    "answers": [{"id": q["id"], "kind": q["kind"], "question": q["question"], "question_ko": q.get("question_ko", ""),
                                 "answer": ans.get(q["id"], {}).get("answer", "no"),
                                 "evidence": ans.get(q["id"], {}).get("evidence", "")} for q in qs],
                    "motion_quality": c["judge"].get("motion_quality"), "temporal_consistency": c["judge"].get("temporal_consistency"),
                    "notes_ko": c["judge"].get("notes_ko", ""), "metrics": c["metrics"], "penalties": c["penalties"],
                    "threshold": cfg.threshold})

            ok = [c for c in cands if c.get("file") and "error" not in c]
            ranking = None
            if len(ok) >= 2:                              # 점수가 비슷하면 Video-Bench식 상대 비교로 결정
                top = sorted(ok, key=lambda c: -c["score"])
                if top[0]["score"] - top[1]["score"] < 0.05 and all(c.get("sheet") for c in top):
                    try:
                        ranking = agents.rank(hub, [(c["id"], Path(c["sheet"])) for c in top], cfg.request)
                        first_id = ranking["ranking"][0]
                        for c in ok:
                            c["rank_bonus"] = 0.01 if c["id"] == first_id else 0.0
                    except (HubError, ValueError) as e:
                        log(f"    (후보 비교 생략: {e})")
            best = max(cands, key=lambda c: c.get("score", 0) + c.get("rank_bonus", 0))
            rec = {"round": r, "candidates": cands, "ranking": ranking, "best": best["id"]}
            srec["rounds"].append(rec)
            if best.get("file") and (srec["best"] is None or best["score"] > srec["best"]["score"]):
                srec["best"] = {"round": r, **best}
            save()

            if srec["best"] and srec["best"]["score"] >= cfg.threshold:
                emit("select", f"{shot['id']} 기준 점수 {cfg.threshold} 도달 → 종료")
                break
            if r == cfg.rounds or not best.get("file") or "judge" not in best:
                continue
            emit("reflect", f"{shot['id']} 라운드 {r}: 실패 원인 분석·수정 경로 선택 (GenMAC·PhyT2V)")
            history = [{"round": h["round"], "prompt": h["prompt"], "score": h["score"], "route": h.get("route")}
                       for h in memory]
            refl = agents.reflect(hub, spec, shot, best, history)
            rec["reflection"] = refl
            memory.append({"round": r, "prompt": best["prompt"], "score": best["score"],
                           "failed": [f["question"] for f in best.get("failed", [])],
                           "metrics": {k: best["metrics"].get(k) for k in ("motion", "static", "jump", "fidelity")},
                           "route": refl["route"], "reflection": refl.get("reflection", "")})
            if refl["route"] == "clarify":
                state.setdefault("needs_clarification", []).append(
                    {"shot": shot["id"], "round": r, "why": refl.get("summary_ko", ""), "mismatches": refl.get("mismatches", [])})
            emit("reflect", f"    경로={refl['route']} — {refl.get('summary_ko', '')}")
            emit("reflect_done", refl.get("summary_ko", ""), data={
                "shot": shot["id"], "round": r, "route": refl["route"], "cause": refl.get("cause", ""),
                "mismatches": refl.get("mismatches", []), "summary_ko": refl.get("summary_ko", "")})
            save()

        if not srec["best"]:
            raise HubError(f"{shot['id']}: 생성된 영상이 없습니다. 위 오류 메시지를 확인하세요.")
        finals.append(Path(srec["best"]["file"]))
        if si + 1 < len(spec["shots"]):                  # Vlogger식: 다음 샷은 이번 샷 마지막 프레임에서 시작
            first = videoio.last_frame(finals[-1], run_dir / f"{spec['shots'][si + 1]['id']}_first.jpg", size)

    emit("compose", f"샷 {len(finals)}개 합치기")
    final = videoio.concat_videos(finals, run_dir / "final.mp4")
    state["final"] = {"file": str(final), "info": videoio.probe(final),
                      "scores": [s["best"]["score"] for s in state["shots"]]}
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    emit("done", f"완료 → {final}  (보고서 {run_dir / 'report.html'})", data={
        "final": rel(final), "scores": state["final"]["scores"], "seconds": (state["final"]["info"] or {}).get("seconds")})
    return state


def _generate_all(hub: Hub, cands: list[dict], rdir: Path, first: Path, seconds: float, size: str, emit) -> None:
    """개발키는 동시 1건이라 순서대로. 키 한도가 늘면(OASH_MAX_CONCURRENCY) 병렬로 보낸다."""
    def one(c: dict) -> None:
        def prog(status: str, pct) -> None:
            emit("progress", f"    {c['id']} {status} {pct}%", data={"cand": c["id"], "status": status, "percent": pct})
        try:
            res = hub.generate_video(c["prompt"], rdir / c["id"], first, seconds, size, negative=c.get("negative"),
                                     on_progress=prog)
            c.update({"file": str(res["file"]), "seed": res["seed"], "job_id": res["job_id"],
                      "elapsed": res["elapsed"]})
        except HubError as e:
            c["error"] = str(e)
            emit("error", f"    {c['id']} 생성 실패: {e}")

    emit("generate", f"영상 생성 {len(cands)}건 (첫 프레임 고정, {seconds:g}초, {size})")
    if MAX_CONCURRENCY > 1 and len(cands) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
            list(ex.map(one, cands))
    else:
        for c in cands:
            one(c)
