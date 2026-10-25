"""장편 모드: 사진 + 이야기 요청 → 10~20분 영상.

  계획   L1 막 → L2 씬 → L3 샷 (story.py, 결과는 plan.json — 사람이 고쳐도 됨)
  생성   씬마다 기준 프레임(앵커) → 샷 체인 생성 [생성 → 프레임 검사 → (미달 시 프롬프트 수정 후 재시도)]
         · 단기 기억: 다음 샷은 이전 샷 마지막 프레임에서 시작 (StreamingT2V의 CAM 역할을 첫 프레임으로)
         · 장기 기억: 씬 앵커 기준 색 보정 (StreamingT2V의 APM 역할을 색 수준에서)
                     (StreamingT2V의 APM, VideoStudio의 참조 이미지)
         · 일관성 단위 = 씬 (Long Context Tuning). 장소·시간이 바뀌는 씬은 원본 사진을 편집해 새 키프레임
  조립   샷 정규화(이음매 프레임 제거) → 씬 클립(+페이드, +내레이션) → final.mp4 + 자막 + 챕터
  재개   progress.json에 샷마다 기록 → 끊겨도 `resume`으로 남은 샷부터
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import agents, prompts as P, story, videoio
from .agents import _ask_json, _j, ask_clip
from .hub import MAX_CONCURRENCY, Hub, HubError, video_spec
from .orchestrator import score, severe_artifacts as _severe
from .utils import log

EST_JOB_SECONDS = float(os.getenv("VS_EST_JOB_SECONDS", "50"))    # 실측 전 영상 1건 예상 시간 (실제 Hub 실측 평균 45~50초)
DRIFT_DELTA_E = float(os.getenv("VS_DRIFT_DELTA_E", "6"))          # 앵커 대비 색 거리가 이보다 크면 색 보정


@dataclass
class LongConfig:
    image: Path | None = None
    request: str = ""
    seconds: float = 600
    retries: int = 1              # 샷당 재시도 (점수 미달 시)
    min_score: float = 0.6        # 이 점수 미만이면 재시도
    frames: int = 4               # 샷 검사 프레임 수
    size: str | None = None
    fit: str = "crop"
    narration: bool = False       # Hub TTS로 내레이션 음성 (미지원이면 자막만)
    voice: str | None = None
    plan_only: bool = False
    out_root: Path = Path("output")
    name: str = "long"
    kb: str | None = None         # 멀티모달 지식 저장소 (rag_stores/<이름> 또는 경로) — 근거·첫 프레임·움직임 참고
    extra: dict = field(default_factory=dict)


class Progress:
    """progress.json — 샷·씬 상태. 여러 작업자가 동시에 쓰므로 잠금 + 원자적 저장."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "progress.json"
        self.lock = threading.RLock()
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else \
            {"shots": {}, "scenes": {}, "jobs": {"count": 0, "seconds": 0.0}, "events": []}

    def save(self) -> None:
        with self.lock:
            self.data["updated"] = datetime.now().isoformat(timespec="seconds")
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            tmp.replace(self.path)

    def shot(self, sid: str) -> dict:
        return self.data["shots"].get(sid, {})

    def scene(self, scid: str) -> dict:
        with self.lock:
            return self.data["scenes"].setdefault(scid, {})

    def event(self, msg: str) -> None:
        with self.lock:
            self.data["events"].append(f"{datetime.now():%m-%d %H:%M} {msg}")
        log(f"    ⚑ {msg}")

    def add_job(self, seconds: float) -> None:
        with self.lock:
            self.data["jobs"]["count"] += 1
            self.data["jobs"]["seconds"] += seconds


# ------------------------------------------------------------------ 시작 / 재개
def start(cfg: LongConfig, hub: Hub | None = None) -> Path:
    hub = hub or Hub()
    m = hub.models()
    spec = video_spec(m.video)
    run_dir = cfg.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    inp = run_dir / f"input{cfg.image.suffix.lower() or '.jpg'}"
    shutil.copy(cfg.image, inp)
    size = cfg.size or videoio.choose_size(inp, spec["sizes"])
    meta = {"config": {k: str(v) for k, v in cfg.__dict__.items()}, "size": size, "fps": spec["fps"],
            "models": {k: getattr(m, k) for k in ("text", "vision", "video", "image_edit", "tts")},
            "input": inp.name, "started": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\n=== 장편 {cfg.seconds / 60:.1f}분 | {run_dir}\n    모델: {meta['models']} ({size})")
    log("[observe] 사진 장면 분석")
    scene = agents.observe(hub, inp)
    (run_dir / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1), encoding="utf-8")
    return resume(run_dir, hub, cfg)


def resume(run_dir: Path, hub: Hub | None = None, cfg: LongConfig | None = None) -> Path:
    hub = hub or Hub()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    c = meta["config"]
    cfg = cfg or LongConfig(request=c["request"], seconds=float(c["seconds"]), retries=int(c["retries"]),
                            min_score=float(c["min_score"]), frames=int(c["frames"]), fit=c["fit"],
                            narration=c["narration"] == "True", voice=None if c["voice"] == "None" else c["voice"],
                            kb=None if c.get("kb", "None") == "None" else c.get("kb"))
    scene = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    kb = None
    if cfg.kb:
        from .rag import Store
        kb = Store(cfg.kb, hub)
        log(f"[rag] 지식 저장소 사용: {kb.dir} (조각 {len(kb.chunks)}개)")
    plan = story.make_plan(hub, run_dir, cfg.request, scene, cfg.seconds, kb)
    prog = Progress(run_dir)
    _estimate(plan, prog, meta)
    _report(run_dir, plan, prog, meta)
    if cfg.plan_only:
        log(f"[plan] 계획 완료 → {run_dir / 'report.html'}  (plan.json을 고친 뒤 `resume {run_dir}`)")
        return run_dir
    Runner(run_dir, hub, cfg, plan, prog, meta, kb).run()
    return run_dir


def _estimate(plan: dict, prog: Progress, meta: dict) -> dict:
    done = sum(1 for s in prog.data["shots"].values() if s.get("status") == "done")
    j = prog.data["jobs"]
    avg = j["seconds"] / j["count"] if j["count"] else EST_JOB_SECONDS
    tries = (j["count"] / done) if done else 1.3                          # 재시도 포함 샷당 영상 작업 수
    left = (plan["n_shots"] - done) * tries * avg / MAX_CONCURRENCY
    est = {"shots_total": plan["n_shots"], "shots_done": done, "avg_job_seconds": round(avg),
           "jobs_per_shot": round(tries, 2), "eta_seconds": round(left), "measured": bool(j["count"])}
    meta["estimate"] = est
    log(f"[진행] 샷 {done}/{plan['n_shots']} ({100 * done / max(1, plan['n_shots']):.0f}%) · 영상 작업 평균 "
        f"{_hms(avg)}{'' if j['count'] else '(추정)'} · 샷당 {tries:.2f}건 · 남은 시간 약 {_hms(left)}"
        f" (동시 {MAX_CONCURRENCY})")
    from .budget import BUDGET
    log(f"[토큰] {BUDGET.summary()}")
    meta["tokens"] = BUDGET.summary()
    return est


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}시간 {s % 3600 // 60}분" if s >= 3600 else f"{s // 60}분 {s % 60}초"


def _report(run_dir, plan, prog, meta):
    from .longreport import write
    with prog.lock:
        write(run_dir, plan, prog.data, meta)


# ------------------------------------------------------------------ 생성
class Runner:
    def __init__(self, run_dir: Path, hub: Hub, cfg: LongConfig, plan: dict, prog: Progress, meta: dict, kb=None):
        self.d, self.hub, self.cfg, self.plan, self.prog, self.meta = run_dir, hub, cfg, plan, prog, meta
        self.kb = kb
        self.size, self.fps = meta["size"], meta["fps"]
        self.input = run_dir / meta["input"]
        self.ref = run_dir / "anchors" / "input_first.jpg"        # 원본 사진을 영상 해상도로 맞춘 것 = 스타일 기준
        if not self.ref.exists():
            videoio.prepare_image(self.input, self.size, self.ref, cfg.fit)
        self.last_report = 0.0

    def run(self) -> None:
        videoio.need_ffmpeg()
        seqs, cur = [], []
        for sc in self.plan["scenes"]:                           # 'continue'로 이어진 씬끼리 한 줄(sequence)
            if cur and sc["start"] == "new_keyframe":
                seqs.append(cur)
                cur = []
            cur.append(sc)
        seqs.append(cur)
        log(f"[generate] 씬 {len(self.plan['scenes'])}개 · 독립 줄기 {len(seqs)}개 · 동시 {MAX_CONCURRENCY}")
        errors = []
        if MAX_CONCURRENCY > 1 and len(seqs) > 1:                # 키가 허용하면 독립 줄기를 병렬로
            with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
                for f in [ex.submit(self._sequence, s) for s in seqs]:
                    try:
                        f.result()
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
        else:
            for s in seqs:
                self._sequence(s)
        if errors:
            raise errors[0]
        self.assemble(final=True)

    # ---- 씬 줄기
    def _sequence(self, scenes: list[dict]) -> None:
        prev_last: Path | None = None
        color_ref: Path | None = None
        for sc in scenes:
            st = self.prog.scene(sc["id"])
            if st.get("status") == "done" and st.get("clip") and Path(st["clip"]).exists():   # 재개: 끝난 씬은 건너뜀
                last = self.prog.shot(sc["shots"][-1]["id"]).get("next_first")
                prev_last = Path(last) if last else prev_last
                color_ref = Path(st["color_ref"]) if st.get("color_ref") else color_ref
                continue
            anchor = self._anchor(sc, prev_last)
            if sc["start"] == "new_keyframe" or color_ref is None:
                color_ref = anchor
            st.update(anchor=str(anchor), color_ref=str(color_ref))
            prev_last = self._scene(sc, anchor, color_ref)
            st["status"] = "done"
            self._build_scene_clip(sc)
            self.prog.save()

    def _anchor(self, sc: dict, prev_last: Path | None) -> Path:
        st = self.prog.scene(sc["id"])
        if st.get("anchor") and Path(st["anchor"]).exists():
            return Path(st["anchor"])
        out = self.d / "anchors" / f"{sc['id']}.jpg"
        if sc["index"] == 0:
            shutil.copy(self.ref, out)
            return out
        if sc["start"] == "continue" and prev_last:
            shutil.copy(prev_last, out)
            return out
        if sc.get("visual_ref") and self.kb is not None:       # 사용자 자료의 실제 사진·영상 프레임을 첫 프레임으로
            v = self.kb.get(sc["visual_ref"])
            if v and v.get("path") and Path(v["path"]).exists():
                videoio.prepare_image(Path(v["path"]), self.size, out, "crop")
                self.prog.event(f"{sc['id']} 첫 프레임 = 사용자 자료 {v['source']} {v['loc']}")
                self.prog.scene(sc["id"])["visual_ref"] = {"id": v["id"], "source": v["source"], "loc": v["loc"]}
                return out
        instr = sc["keyframe"] or sc["title"]
        if self.kb is not None:                                # visual_ref가 없어도 자료에서 가장 맞는 프레임을 먼저 찾는다
            hits = self.kb.visuals(f"{sc.get('location', '')} {instr} {' '.join(sc.get('events') or [])}", k=1)
            if hits and hits[0].get("path") and Path(hits[0]["path"]).exists():
                videoio.prepare_image(Path(hits[0]["path"]), self.size, out, "crop")
                self.prog.event(f"{sc['id']} 첫 프레임 = 사용자 자료(검색) {hits[0]['source']} {hits[0]['loc']}")
                self.prog.scene(sc["id"])["visual_ref"] = {"id": hits[0]["id"], "source": hits[0]["source"], "loc": hits[0]["loc"]}
                return out
        log(f"[keyframe] {sc['id']} 새 장면 키프레임 (원본 사진 편집 — 자료에 맞는 프레임이 없을 때)")
        return self._keyframe(instr, out, sc["id"])

    def _keyframe(self, instruction: str, out: Path, tag: str) -> Path:
        look = "; ".join(f"{c.get('name', '')}: {c.get('look', '')}" for c in self.plan["bible"].get("characters", []))
        prompt = (f"{instruction}. Keep the look of the main subjects ({look}). "
                  f"Style: {self.plan['bible'].get('style', 'same as the photo')}. No text, no watermark.")
        best = None
        for attempt in range(2):
            raw = self.hub.edit_image(self.input, prompt, out.with_name(out.stem + f"_raw{attempt}.png"))
            img = videoio.prepare_image(raw, self.size, out.with_name(out.stem + f"_try{attempt}.jpg"), "crop")
            try:
                chk = _ask_json(self.hub, P.KEYFRAME_CHECK, f"Description: {instruction}",
                                images=[("ORIGINAL:", self.ref), ("GENERATED FIRST FRAME:", img)], temperature=0.1)
            except (HubError, ValueError):
                chk = {"matches_description": True}
            ok = chk.get("matches_description", True) is not False and chk.get("style_consistent", True) is not False
            best = img
            if ok:
                break
            self.prog.event(f"{tag} 키프레임이 설명·스타일과 다름 → 다시 생성 ({chk.get('notes_ko', '')})")
        shutil.copy(best, out)
        return out

    # ---- 씬 안의 샷 체인
    def _scene(self, sc: dict, anchor: Path, color_ref: Path) -> Path:
        first = anchor
        for i, sh in enumerate(sc["shots"]):
            rec = self.prog.shot(sh["id"])
            if rec.get("status") == "done" and Path(rec.get("next_first", "")).exists():
                first = Path(rec["next_first"])
                continue
            rec = self._shot(sc, sh, first, i)
            last = videoio.last_frame(Path(rec["file"]), self.d / "frames" / f"{sh['id']}_last.jpg", self.size)
            drift = videoio.color_distance(last, color_ref)
            nxt = last
            if drift > DRIFT_DELTA_E:                           # 장기 기억: 색 누적 오차 되돌리기
                nxt = videoio.color_match(last, color_ref, self.d / "frames" / f"{sh['id']}_next.jpg")
                rec["color_fixed"] = True
            rec.update(last_frame=str(last), next_first=str(nxt), drift=drift, status="done")
            with self.prog.lock:
                self.prog.data["shots"][sh["id"]] = rec
            self.prog.save()
            first = nxt
            self._tick()
        return first

    def _shot(self, sc: dict, sh: dict, first: Path, idx: int) -> dict:
        qs = [{"id": f"k{j + 1}", "kind": "goal", "weight": 1.5, "question": c["question"], "when": c.get("when", "throughout")}
              for j, c in enumerate(sh["checks"])]
        qs += [{"id": "kst", "kind": "quality", "weight": 1.0, "question": "Same visual style as the ANCHOR?"},
               {"id": "kar", "kind": "quality", "weight": 1.0, "question": "Free of severe artifacts?"}]
        checks = [q for q in qs if q["id"] not in ("kst", "kar")]          # 샷 고유 질문 (모델이 답하는 것)
        goals = [{"intensity": sh["motion"], "category": "action"}]
        prompt, negative = sh["prompt"], sh.get("negative") or None
        attempts, best = [], None
        sdir = self.d / "shots" / sc["id"]
        for a in range(1 + self.cfg.retries):
            log(f"[shot] {sh['id']} ({sh['seconds']:g}초) 시도 {a + 1}: {sh['event'][:70]}")
            t0 = time.time()
            last_err = None
            for net_try in range(3):                              # Hub 일시 오류는 같은 프롬프트로 다시
                try:
                    res = self.hub.generate_video(prompt, sdir / f"{sh['id']}_a{a + 1}", first, sh["seconds"],
                                                  self.size, negative=negative)
                    break
                except HubError as e:
                    last_err = e
                    log(f"    (생성 실패 {net_try + 1}/3: {str(e)[:160]})")
                    time.sleep(float(os.getenv("VS_NET_RETRY_SLEEP", "10")) * (net_try + 1))
            else:
                raise HubError(f"{sh['id']} 생성이 계속 실패합니다. 원인을 해결한 뒤 resume 하세요: {last_err}")
            self.prog.add_job(time.time() - t0)
            vid = Path(res["file"])
            from .budget import BUDGET
            frames = videoio.sample_frames(vid, sdir / "frames", k=max(self.cfg.frames, BUDGET.mode["frames"]))
            sheet = videoio.contact_sheet([f["path"] for f in frames], sdir / f"{sh['id']}_a{a + 1}_sheet.jpg", height=120)
            metrics = videoio.motion_metrics(vid, first)
            try:
                clip = videoio.small_clip(vid, sdir / f"{sh['id']}_a{a + 1}_check.mp4") if self.hub.video_ok else None
                chk, mode = ask_clip(self.hub, P.SHOT_CHECK,
                                     f"Shot event: {sh['event']}\nQuestions:\n{_j([{k: q[k] for k in ('id', 'question')} | {'when': q.get('when', 'throughout')} for q in checks])}",
                                     ("ANCHOR:", self.ref), frames, clip)
                chk["_mode"] = mode
            except (HubError, ValueError) as e:
                chk = {"answers": {}, "notes_ko": f"검사 실패: {e}"}
            ans = {q["id"]: {"answer": "yes" if str((chk.get("answers") or {}).get(q["id"], {}).get("answer", "no")).strip().lower().startswith("y") else "no",
                             "evidence": (chk.get("answers") or {}).get(q["id"], {}).get("evidence", "")} for q in checks}
            ans["kst"] = {"answer": "yes" if chk.get("style_kept", True) else "no", "evidence": ""}
            ans["kar"] = {"answer": "no" if _severe(chk) else "yes", "evidence": ", ".join(map(str, _severe(chk)))}
            judge = {"answers": ans, "motion_quality": _int15(chk.get("motion_quality")),
                     "temporal_consistency": _int15(chk.get("temporal_consistency"))}
            s, failed, pen = score(qs, judge, metrics, goals)
            att = {"attempt": a + 1, "file": str(vid), "prompt": prompt, "seed": res["seed"], "score": s,
                   "failed": [f["question"] for f in failed], "penalties": pen, "summary": chk.get("summary", ""),
                   "notes_ko": chk.get("notes_ko", ""), "metrics": metrics, "sheet": str(sheet) if sheet else "",
                   "elapsed": round(time.time() - t0)}
            attempts.append(att)
            log(f"    점수 {s:.2f}" + (f" ({'; '.join(pen)})" if pen else ""))
            if best is None or s > best["score"]:
                best = att
            if s >= self.cfg.min_score or a == self.cfg.retries:
                break
            try:                                                   # 미달 → 원인·경로·새 프롬프트 (GenMAC + PhyT2V)
                fx = _ask_json(self.hub, P.RETRY, _j({"shot": sh, "prompt_used": prompt, "observed": att["summary"],
                                                      "failed_checks": att["failed"], "metrics": metrics,
                                                      "earlier": [{k: x[k] for k in ("prompt", "score")} for x in attempts]}))
                prompt = fx.get("prompt") or prompt
                negative = fx.get("negative") or negative
                att["route"] = fx.get("route", "")
                att["fix_ko"] = fx.get("summary_ko", "")
            except (HubError, ValueError):
                pass
        return {"status": "running", "scene": sc["id"], "file": best["file"], "score": best["score"],
                "sheet": best["sheet"], "attempts": attempts,
                "chosen": best["attempt"]}

    def _tick(self) -> None:
        if time.time() - self.last_report > 20:
            _estimate(self.plan, self.prog, self.meta)
            _report(self.d, self.plan, self.prog, self.meta)
            self.last_report = time.time()

    # ---- 조립
    def _scene_parts(self, sc: dict) -> list[Path]:
        """씬의 샷을 같은 형식으로 정규화. 앞 샷에서 이어진 샷은 첫 프레임(이음매)을 버린다."""
        parts = []
        for i, sh in enumerate(sc["shots"]):
            rec = self.prog.shot(sh["id"])
            if rec.get("status") != "done":
                break
            out = self.d / "segments" / f"{sh['id']}_c{rec.get('chosen', 1)}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                chained = i > 0 or (sc["start"] == "continue" and sc["index"] > 0)
                videoio.normalize_segment(Path(rec["file"]), out, self.size, self.fps, drop_first=chained)
            parts.append(out)
        return parts

    def _build_scene_clip(self, sc: dict, partial: bool = False) -> Path | None:
        parts = self._scene_parts(sc)
        if not parts:
            return None
        sdir = self.d / "scenes"
        sdir.mkdir(exist_ok=True)
        raw = videoio.concat_copy(parts, sdir / f"{sc['id']}_raw.mp4")
        audio = None
        if self.cfg.narration and sc.get("narration_ko") and not partial:
            wav = sdir / f"{sc['id']}_narration.wav"
            audio = wav if wav.exists() else self.hub.tts(sc["narration_ko"], wav, self.cfg.voice)
        scenes = self.plan["scenes"]
        nxt = scenes[sc["index"] + 1] if sc["index"] + 1 < len(scenes) else None
        out = videoio.finish_scene(raw, sdir / f"{sc['id']}.mp4",
                                   fade_in=sc["start"] == "new_keyframe" and sc["index"] > 0,
                                   fade_out=nxt is None or nxt["start"] == "new_keyframe", audio=audio)
        with self.prog.lock:
            self.prog.scene(sc["id"]).update(clip=str(out), seconds=videoio.media_seconds(out),
                                             narration_audio=str(audio) if audio else "")
        return out

    def assemble(self, final: bool = False) -> Path | None:
        """끝난 씬까지 이어 붙인다. 진행 중에도 `assemble` 명령으로 미리보기를 만들 수 있다."""
        clips, srt, chapters, t = [], [], [], 0.0
        for sc in self.plan["scenes"]:
            st = self.prog.scene(sc["id"])
            clip = Path(st["clip"]) if st.get("status") == "done" and st.get("clip") else None
            if clip is None or not clip.exists():
                clip = self._build_scene_clip(sc, partial=True)
                if clip is None:
                    break
            dur = videoio.media_seconds(clip)
            chapters.append(f"{_ts(t, False)} {sc['id']} {sc['title']}")
            if sc.get("narration_ko"):
                srt.append(f"{len(srt) + 1}\n{_ts(t + 0.3)} --> {_ts(t + dur - 0.2)}\n{sc['narration_ko']}\n")
            clips.append(clip)
            t += dur
            if st.get("status") != "done":
                break
        if not clips:
            log("[assemble] 아직 완성된 샷이 없습니다.")
            return None
        name = "final.mp4" if final else "preview.mp4"
        out = videoio.concat_copy(clips, self.d / name)
        (self.d / out.with_suffix(".srt").name).write_text("\n".join(srt), encoding="utf-8")
        (self.d / "chapters.txt").write_text("\n".join(chapters), encoding="utf-8")
        with self.prog.lock:
            self.prog.data["final" if final else "preview"] = {"file": str(out), "seconds": round(t, 1),
                                                                "made": datetime.now().isoformat(timespec="seconds")}
        self.prog.save()
        _estimate(self.plan, self.prog, self.meta)
        _report(self.d, self.plan, self.prog, self.meta)
        log(f"[assemble] {out.name} {t / 60:.1f}분 ({len(clips)}개 씬) → {out}")
        return out


def assemble(run_dir: Path, hub: Hub | None = None) -> Path | None:
    """진행 중이거나 중단된 실행에서 지금까지 만든 부분으로 preview.mp4."""
    hub = hub or Hub()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    c = meta["config"]
    cfg = LongConfig(request=c["request"], seconds=float(c["seconds"]), fit=c["fit"])
    r = Runner(run_dir, hub, cfg, story.load(run_dir), Progress(run_dir), meta)
    return r.assemble(final=False)


def _int15(x) -> int:
    try:
        return max(1, min(5, int(x)))
    except (TypeError, ValueError):
        return 3


def _ts(sec: float, ms: bool = True) -> str:
    h, rem = divmod(max(0.0, sec), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",") if ms else f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
