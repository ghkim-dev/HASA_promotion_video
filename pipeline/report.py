"""실행 기록(run.json)을 브라우저로 보는 report.html로 정리. 경로는 run 폴더 기준 상대 경로."""

from __future__ import annotations

import html
import os
from pathlib import Path

CAT_KO = {"action": "동작", "motion_direction": "이동 방향", "camera": "카메라", "dynamic_attribute": "속성 변화",
          "interaction": "상호작용", "environment": "환경", "spatial": "공간 관계", "numeracy": "개수"}
WHEN_KO = {"start": "시작", "middle": "중간", "end": "끝", "throughout": "전체", "compare_start_end": "처음↔끝 비교"}
ROUTE_KO = {"strengthen_motion": "움직임 강화", "fix_binding": "주체·방향 바로잡기", "simplify": "동작 단순화",
            "fix_camera": "카메라 수정", "fix_physics": "물리 수정", "protect_identity": "정체성 보호",
            "reduce_artifacts": "결함 줄이기", "clarify": "사용자 확인 필요"}


def _rel(path, base: Path) -> str:
    if not path:
        return ""
    try:
        return Path(os.path.relpath(str(path), base)).as_posix()
    except ValueError:
        return Path(str(path)).as_posix()


def _media(path, base: Path, cls: str = "") -> str:
    if not path:
        return '<div class="empty">없음</div>'
    src = html.escape(_rel(path, base))
    if str(path).endswith((".mp4", ".webm", ".mov")):
        return f'<video class="{cls}" src="{src}" controls muted loop playsinline></video>'
    return f'<img class="{cls}" src="{src}" alt="">'


def write_report(state: dict, run_dir: Path) -> Path:
    e = lambda x: html.escape(str(x if x is not None else ""))  # noqa: E731
    spec = state.get("spec", {})
    scene = state.get("scene", {})
    P: list[str] = []
    cfg = state.get("config", {})
    P.append(f"<header><h1>{e(cfg.get('name', 'run'))}</h1><p class='req'>“{e(cfg.get('request'))}”</p>"
             f"<p class='meta'>text {e(state['models'].get('text'))} · vision {e(state['models'].get('vision'))} · "
             f"video {e(state['models'].get('video'))} · {e(state.get('size'))} · "
             f"Hub 호출 {e(state.get('hub_calls', {}))}</p></header>")

    fin = state.get("final")
    P.append("<section class='hero'><div><h3>입력 사진</h3>" + _media(state.get("input"), run_dir) + "</div>"
             "<div><h3>" + ("최종 영상 · 샷 점수 " + ", ".join(f"{s:.2f}" for s in fin["scores"]) if fin else "최종 영상 (아직 없음)")
             + "</h3>" + (_media(fin["file"], run_dir) if fin else "<div class='empty'>계획만 실행했거나 진행 중</div>") + "</div></section>")
    if state.get("needs_clarification"):
        P.append("<div class='warn'><b>사용자 확인이 필요해 보입니다</b><ul>" + "".join(
            f"<li>{e(n['shot'])} 라운드 {e(n['round'])}: {e(n['why'])}</li>" for n in state["needs_clarification"]) + "</ul></div>")

    P.append("<h2>0. 장면 관찰 <small>RPG(ICML 2024)식 재캡션</small></h2>")
    P.append(f"<p>{e(scene.get('caption'))}</p><table><tr><th>id</th><th>대상</th><th>속성</th><th>움직일 수 있는 부분</th></tr>")
    for s in scene.get("subjects", []):
        P.append(f"<tr><td><code>{e(s.get('id'))}</code></td><td>{e(s.get('name'))}<br><small>{e(s.get('position'))}</small></td>"
                 f"<td>{e(', '.join(map(str, s.get('attributes', []))))}</td><td>{e(', '.join(map(str, s.get('movable_parts', []))))}</td></tr>")
    P.append(f"</table><p class='small'>배경: {e(scene.get('background'))} · 환경 요소: "
             f"{e(', '.join(map(str, scene.get('environment_dynamics', []))))} · 스타일: {e(scene.get('style'))}</p>")
    if state.get("terms"):
        P.append("<p class='small'>용어 인식: " + " · ".join(
            f"<b>{e(t['matched'])}</b>→{e(t['en'])}" for t in state["terms"]) + "</p>")

    if spec:
        P.append(f"<h2>1. 요구사항 이해</h2><p><b>{e(spec.get('intent_ko'))}</b><br><small>{e(spec.get('intent'))}</small></p>")
        P.append("<table><tr><th>목표</th><th>주체</th><th>범주<br><small>T2V-CompBench</small></th><th>시점</th><th>강도</th>"
                 "<th>변화</th><th>물리·상식 규칙<br><small>PhyT2V</small></th></tr>")
        for g in spec.get("goals", []):
            P.append(f"<tr><td><code>{e(g['id'])}</code></td><td>{e(g.get('subject'))}</td><td>{e(CAT_KO.get(g['category'], g['category']))}</td>"
                     f"<td>{e(WHEN_KO.get(g['when'], g['when']))}</td><td>{e(g['intensity'])}</td><td>{e(g.get('change'))}</td>"
                     f"<td class='small'>{e(g.get('rule'))}</td></tr>")
        P.append("</table><p><b>샷 계획</b> <small>Vlogger</small>: " + " → ".join(
            f"{e(s['id'])}({s['seconds']:g}초: {e(', '.join(s['goals']))})" for s in spec.get("shots", [])) + "</p>")
        for title, key in (("보존", "preserve"), ("가정", "assumptions"), ("위험", "risks")):
            if spec.get(key):
                P.append(f"<p><b>{title}</b></p><ul>" + "".join(f"<li>{e(x)}</li>" for x in spec[key]) + "</ul>")
        if spec.get("questions"):
            P.append("<p><b>확인 질문</b> (기본값으로 진행)</p><ul>" + "".join(
                f"<li>{e(q.get('q'))} → <i>{e(q.get('default'))}</i></li>" for q in spec["questions"]) + "</ul>")
        if state.get("clarifications"):
            P.append("<p><b>사용자 답변</b></p><ul>" + "".join(
                f"<li>{e(a.get('q'))} → <b>{e(a.get('a'))}</b></li>" for a in state["clarifications"]) + "</ul>")

    qs = state.get("checklist", [])
    if qs:
        P.append("<h2>2. 검증 체크리스트 <small>TIFA · Video-Bench · VBench</small></h2><table><tr><th>id</th><th>종류</th>"
                 "<th>대상</th><th>샷</th><th>프레임</th><th>질문</th><th>가중치</th></tr>")
        for q in qs:
            P.append(f"<tr><td><code>{e(q['id'])}</code></td><td>{e(q['kind'])}</td><td>{e(q.get('ref'))}</td><td>{e(q['shot'])}</td>"
                     f"<td>{e(WHEN_KO.get(q['when'], q['when']))}</td><td>{e(q['question'])}</td><td>{q['weight']:g}</td></tr>")
        P.append("</table>")

    for srec in state.get("shots", []):
        sh = srec["shot"]
        P.append(f"<h2>3–5. {e(sh['id'])} · {sh['seconds']:g}초 <small>{e(sh.get('note'))}</small></h2>"
                 f"<div class='first'>이 샷의 첫 프레임<br>{_media(srec.get('first_frame'), run_dir, 'thumb')}</div>")
        for rec in srec["rounds"]:
            P.append(f"<h3>라운드 {rec['round']} — 선택 {e(rec['best'])}</h3><div class='cands'>")
            for c in rec["candidates"]:
                mt = c.get("metrics", {})
                fails = "".join(f"<li><code>{e(f['id'])}</code> {e(f['question'])}<br><small>{e(f.get('evidence'))}</small></li>"
                                for f in c.get("failed", []))
                P.append(
                    f"<div class='cand{' best' if c['id'] == rec['best'] else ''}'>"
                    f"<div class='score'>{e(c['id'])} · {c.get('score', 0):.2f}</div>"
                    + _media(c.get("file"), run_dir)
                    + (_media(c.get("sheet"), run_dir, "sheet") if c.get("sheet") else "")
                    + (f"<p class='err'>{e(c['error'])}</p>" if c.get("error") else "")
                    + f"<p class='strategy'>{e(c.get('strategy_ko'))}"
                    + (f" <span class='err'>누락 목표: {e(', '.join(c['missing']))}</span>" if c.get("missing") else "") + "</p>"
                    f"<details><summary>프롬프트</summary><p class='prompt'>{e(c.get('prompt'))}</p>"
                    f"<p class='small'>negative: {e(c.get('negative'))}</p></details>"
                    + (f"<p class='small'>움직임 {e(mt.get('motion'))}{' (정지)' if mt.get('static') else ''} · 깜빡임 {e(mt.get('flicker'))}"
                       f" · 컷 튐 {'있음' if mt.get('jump') else '없음'} · 첫 프레임 유사도 {e(mt.get('fidelity'))} · seed {e(c.get('seed'))}</p>" if mt else "")
                    + (f"<p class='small'>판정: {e(c.get('judge', {}).get('notes_ko'))}<br>관찰: {e(c.get('judge', {}).get('revised_summary') or c.get('description', {}).get('summary'))}</p>" if c.get("judge") else "")
                    + ("<p class='small'>" + " · ".join(map(e, c.get("penalties", []))) + "</p>" if c.get("penalties") else "")
                    + (f"<details><summary>실패한 질문 {len(c.get('failed', []))}개</summary><ul>{fails}</ul></details>" if fails else "")
                    + "</div>")
            P.append("</div>")
            if rec.get("ranking"):
                P.append(f"<p class='small'>후보 비교: {e(' > '.join(rec['ranking']['ranking']))} — {e(rec['ranking'].get('reason_ko'))}</p>")
            if rec.get("reflection"):
                rf = rec["reflection"]
                P.append(f"<div class='reflect'><b>반성 → {e(ROUTE_KO.get(rf['route'], rf['route']))}</b> {e(rf.get('summary_ko'))}"
                         f"<br><small>원인: {e(rf.get('cause'))}</small><br><small>다음 지시: {e(rf.get('reflection'))}</small></div>")

    css = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e2e1dc;--accent:#2f6f5e;--warn:#9a4b00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9c9b95;--card:#20201e;--line:#34332f;--accent:#7cc3ad;--warn:#f0a35a}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,'Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:1200px;margin:0 auto;padding:24px 16px}
h1{margin:0}h2{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:6px}h2 small,h3 small{color:var(--muted);font-weight:400;font-size:.75em}
.req{font-size:1.15em}.meta,.small{color:var(--muted);font-size:.85em}
table{border-collapse:collapse;width:100%;font-size:.9em}td,th{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
th{background:var(--card)}code{font-size:.9em}
.hero{display:grid;grid-template-columns:1fr 1.6fr;gap:16px;margin-top:16px}.hero img,.hero video{width:100%;border-radius:8px}
.cands{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.cand{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}.cand.best{border:2px solid var(--accent)}
.cand video{width:100%;border-radius:6px}.sheet{width:100%;margin-top:6px;border-radius:4px}
.score{font-weight:700;color:var(--accent);margin-bottom:6px}.prompt{white-space:pre-wrap;font-size:.85em}
.thumb{max-width:260px;border-radius:6px}.first{color:var(--muted);font-size:.85em}
.reflect{background:var(--card);border-left:4px solid var(--accent);padding:8px 12px;margin:10px 0}
.warn{border-left:4px solid var(--warn);background:var(--card);padding:8px 12px;margin-top:12px}.err{color:var(--warn)}
.empty{padding:40px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:8px}
@media (max-width:700px){.hero{grid-template-columns:1fr}}
"""
    page = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>VIDEO Studio 보고서</title><style>{css}</style></head><body>{''.join(P)}</body></html>")
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
