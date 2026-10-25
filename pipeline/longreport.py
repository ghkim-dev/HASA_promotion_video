"""장편 진행 보고서 (report.html). 생성 중에는 60초마다 자동 새로고침된다."""

from __future__ import annotations

import html
import os
from pathlib import Path


def _rel(p, base: Path) -> str:
    if not p:
        return ""
    try:
        return html.escape(Path(os.path.relpath(str(p), base)).as_posix())
    except ValueError:
        return html.escape(str(p))


def _hms(s) -> str:
    s = int(s or 0)
    return f"{s // 3600}시간 {s % 3600 // 60}분" if s >= 3600 else f"{s // 60}분 {s % 60}초"


def write(run_dir: Path, plan: dict, prog: dict, meta: dict) -> Path:
    e = lambda x: html.escape(str(x if x is not None else ""))  # noqa: E731
    est = meta.get("estimate", {})
    total, done = plan.get("n_shots", 0), est.get("shots_done", 0)
    fin = prog.get("final") or prog.get("preview")
    P = [f"<header><h1>{e(plan.get('title') or '장편')}</h1><p class='req'>{e(plan.get('logline_ko'))}</p>"
         f"<p class='meta'>요청: {e(plan.get('request'))}<br>목표 {plan.get('total_seconds', 0) / 60:.1f}분 · 씬 "
         f"{len(plan.get('scenes', []))}개 · 샷 {total}개 · {e(meta.get('size'))} · 모델 {e(meta.get('models'))}</p></header>"]
    pct = 100 * done / max(1, total)
    P.append(f"<section class='prog'><div class='bar'><i style='width:{pct:.1f}%'></i></div>"
             f"<p>샷 <b>{done}/{total}</b> ({pct:.0f}%) · 영상 작업 평균 {_hms(est.get('avg_job_seconds'))}"
             f"{'' if est.get('measured') else ' (추정값)'} · 샷당 작업 {est.get('jobs_per_shot', '-')}건 · "
             f"<b>남은 시간 약 {_hms(est.get('eta_seconds'))}</b></p></section>")
    if fin:
        P.append(f"<section><h2>{'최종 영상' if prog.get('final') else '미리보기 (끝난 부분까지)'} · {fin['seconds'] / 60:.1f}분</h2>"
                 f"<video src='{_rel(fin['file'], run_dir)}' controls playsinline></video>"
                 f"<p class='small'>자막: {_rel(Path(fin['file']).with_suffix('.srt'), run_dir)} · 챕터: chapters.txt</p></section>")
    b = plan.get("bible", {})
    P.append("<h2>이야기 설정 (bible)</h2><table>")
    for c in b.get("characters", []):
        P.append(f"<tr><th>{e(c.get('name'))}</th><td>{e(c.get('look'))}</td></tr>")
    for k, ko in (("world", "세계"), ("style", "스타일"), ("camera_language", "카메라"), ("tone", "톤")):
        P.append(f"<tr><th>{ko}</th><td>{e(b.get(k))}</td></tr>")
    P.append("</table>")
    P.append("<h2>막 구성</h2><ol>" + "".join(
        f"<li><b>{e(a.get('title'))}</b> ({a['seconds'] / 60:.1f}분) — {e(a.get('summary'))}</li>" for a in plan.get("acts", [])) + "</ol>")

    P.append("<h2>씬</h2>")
    t = 0.0
    for sc in plan.get("scenes", []):
        st = prog.get("scenes", {}).get(sc["id"], {})
        shots = sc.get("shots", [])
        recs = [prog.get("shots", {}).get(s["id"], {}) for s in shots]
        ok = [r for r in recs if r.get("status") == "done"]
        avg = sum(r["score"] for r in ok) / len(ok) if ok else 0
        retries = sum(len(r.get("attempts", [])) - 1 for r in ok)
        flags = "자료" if st.get("visual_ref") else ""
        state = "완료" if st.get("status") == "done" else (f"{len(ok)}/{len(shots)}" if ok else "대기")
        P.append(f"<details class='scene'><summary><span class='tc'>{int(t // 60):02d}:{int(t % 60):02d}</span>"
                 + (f"<img class='kf' src='{_rel(st.get('anchor'), run_dir)}'>" if st.get("anchor") else "<span class='kf'></span>")
                 + f"<span><b>{e(sc['id'])} {e(sc.get('title'))}</b> <small>{sc['seconds']:g}초 · "
                 f"{'새 키프레임' if sc['start'] == 'new_keyframe' else '이어짐'} · 샷 {len(shots)}</small><br>"
                 f"<small>{e(' → '.join(sc.get('events', [])))}</small></span>"
                 f"<span class='st'>{state}{f' · {avg:.2f}' if ok else ''}{f' · 재시도 {retries}' if retries else ''}"
                 f"{' · 첫 프레임=자료' if flags else ''}</span></summary>")
        if sc.get("narration_ko"):
            P.append(f"<p class='nar'>내레이션: {e(sc['narration_ko'])}</p>")
        if sc.get("keyframe"):
            P.append(f"<p class='small'>키프레임 지시: {e(sc['keyframe'])}</p>")
        if st.get("clip"):
            P.append(f"<video class='clip' src='{_rel(st['clip'], run_dir)}' controls muted playsinline></video>")
        P.append("<table class='shots'><tr><th>샷</th><th>사건 / 프롬프트</th><th>프레임</th><th>점수</th></tr>")
        for s, r in zip(shots, recs):
            att = r.get("attempts", [])
            sheet = next((a["sheet"] for a in att if a["attempt"] == r.get("chosen")), "")
            score_txt = f"{r['score']:.2f}" if r.get("score") is not None else "-"
            sheet_html = f"<img class='sheet' src='{_rel(sheet, run_dir)}'>" if sheet else ""
            notes = "".join(f"<div class='small'>시도 {a['attempt']}: {a['score']:.2f} {e(a.get('notes_ko'))}"
                            f"{(' → ' + e(a['fix_ko'])) if a.get('fix_ko') else ''}</div>" for a in att)
            P.append(f"<tr><td><code>{e(s['id'][-3:])}</code><br><small>{s['seconds']:g}초</small></td>"
                     f"<td>{e(s.get('event'))}<details><summary class='small'>프롬프트</summary>"
                     f"<p class='small'>{e(s.get('prompt'))}</p></details>{notes}</td>"
                     f"<td>{sheet_html}</td><td>{score_txt}"
                     f"{'<br><small>색 보정</small>' if r.get('color_fixed') else ''}"
                     "</td></tr>")
        P.append("</table></details>")
        t += sc["seconds"]
    if prog.get("events"):
        P.append("<h2>이벤트</h2><ul class='small'>" + "".join(f"<li>{e(x)}</li>" for x in prog["events"][-200:]) + "</ul>")

    css = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e2e1dc;--accent:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9c9b95;--card:#20201e;--line:#34332f;--accent:#7cc3ad}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,'Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}
h1{margin:0}h2{margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:6px}
.req{font-size:1.1em}.meta,.small,small{color:var(--muted);font-size:.85em}
.bar{height:12px;background:var(--line);border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
video{width:100%;border-radius:8px;background:#000}.clip{max-width:560px}
table{border-collapse:collapse;width:100%;font-size:.9em}td,th{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
th{background:var(--card)}
.scene{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0;padding:6px 10px}
.scene>summary{display:grid;grid-template-columns:48px 96px 1fr auto;gap:10px;align-items:center;cursor:pointer;list-style:none}
.tc{font-variant-numeric:tabular-nums;color:var(--muted)}.kf{width:96px;height:54px;object-fit:cover;border-radius:4px;display:block;background:var(--line)}
.st{font-weight:600;color:var(--accent);white-space:nowrap}.sheet{width:260px;border-radius:4px}.nar{font-style:italic}
@media (max-width:700px){.scene>summary{grid-template-columns:40px 1fr}.kf,.st{display:none}.sheet{width:150px}}
"""
    refresh = "" if prog.get("final") else "<meta http-equiv='refresh' content='60'>"
    page = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"{refresh}<title>장편 진행</title><style>{css}</style></head><body>{''.join(P)}</body></html>")
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
