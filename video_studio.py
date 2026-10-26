"""VIDEO Studio — 사진 한 장 + 원하는 움직임(한국어) → 영상.  Open AI Service Hub(open.hasa.re.kr) 모델 사용.

  python video_studio.py models
  python video_studio.py run --image 내사진.jpg --request "바람에 머리카락이 날리고 천천히 줌인" --plan-only
  python video_studio.py run --image 내사진.jpg --request "..." --seconds 8 --rounds 2 --candidates 2
  python video_studio.py terms

  # 장편 (10~20분): 계획만 먼저 보고 → 생성 → 끊기면 이어서 → 진행 중 미리보기
  python video_studio.py long --image 내사진.jpg --request "이 사람이 바닷가 마을에서 하루를 보내는 이야기" --minutes 15 --plan-only
  python video_studio.py resume output\20261025_160000_long
  python video_studio.py assemble output\20261025_160000_long

  # 멀티모달 지식 저장소: 사진·영상·문서·웹 → 조각·임베딩 → 검색 → 장편 영상에 연결
  python video_studio.py rag add 우리자료 자료폴더 소개영상.mp4 https://example.com
  python video_studio.py rag ask 우리자료 "데이터센터 서버실"
  python video_studio.py long --image 내사진.jpg --request "..." --minutes 3 --kb 우리자료

  # 전시회 키오스크 웹서비스 (이 PC의 브라우저에서 http://localhost:8000)
  python video_studio.py web            (연습: python video_studio.py web --mock)

  # 입력 폴더 하나에 PDF 여러 개·사진·동영상·문서(docx/pptx/xlsx/hwpx/txt)·웹 주소(.url, urls.txt)를 모아 두고
  python video_studio.py promo ingest --input input\gbcloud_pdf --name 클라우드팜
  python video_studio.py long --input input\gbcloud_pdf --request "..." --minutes 5      (첫 프레임도 자료에서 자동 선택)

  # 홍보영상: 브로셔(PDF)·사진 → 사실 확인 → 스토리보드 → 초안(GPU 없음) → 본 렌더
  python video_studio.py promo ingest --pdf 브로셔.pdf --photos 사진폴더 --name 우리기관
  python video_studio.py promo plan promo_projects\우리기관 --format main
  python video_studio.py promo render promo_projects\우리기관 --format main --draft
  python video_studio.py promo render promo_projects\우리기관 --format main --narration --bgm 음악.mp3
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from pipeline import lexicon                                   # noqa: E402
from pipeline.hub import Hub, HubError, accepts_image, video_spec  # noqa: E402
from pipeline.utils import log                                  # noqa: E402


def cmd_models(_a) -> None:
    hub = Hub()
    m = hub.models()
    log("사용 가능 모델:\n  " + "\n  ".join(m.available))
    log(f"\n역할별 선택:\n  text   = {m.text}\n  vision = {m.vision}\n  video  = {m.video}")
    if m.video:
        s = video_spec(m.video)
        log(f"\n영상 규격: {s['fps']}fps, 샷당 최대 {s['max_seconds']}초, 해상도 {', '.join(s['sizes'])}, "
            f"첫 프레임 이미지 {'받음' if accepts_image(m.video) else '안 받음 (OASH_VIDEO_ACCEPTS_IMAGE=1로 강제 가능)'}")


def _ask(questions: list[dict]) -> list[dict]:
    if not sys.stdin.isatty():
        return []
    log("\n모델이 확인하고 싶은 것이 있습니다. (엔터 = 기본값)")
    out = []
    for q in questions:
        a = input(f"  - {q.get('q')} [{q.get('default')}]: ").strip()
        out.append({"q": q.get("q"), "a": a or q.get("default")})
    return out


def cmd_run(a) -> None:
    from pipeline.orchestrator import RunConfig, run
    img = Path(a.image)
    if not img.exists():
        raise SystemExit(f"사진이 없습니다: {img}")
    if a.seconds and a.seconds > 16:                     # 샷 4개(약 16초)를 넘으면 장편 모드가 맞다
        log(f"\n{a.seconds:g}초는 장편 모드로 진행합니다 (막→씬→샷 계획, 씬 키프레임, 이어서 생성 가능).")
        a.minutes, a.retries, a.min_score, a.narration, a.voice = a.seconds / 60, 1, 0.6, False, None
        a.frames = min(a.frames, 4)
        return cmd_long(a)
    cfg = RunConfig(image=img, request=a.request, rounds=a.rounds, candidates=a.candidates, threshold=a.threshold,
                    seconds=a.seconds, size=a.size, fit=a.fit, frames=a.frames, plan_only=a.plan_only,
                    out_root=Path(a.out), name=a.name or img.stem, ask_fn=_ask if a.ask else None)
    run(cfg)


def _kb_from_input(a) -> None:
    """--input 폴더·파일을 지식 저장소에 넣고(이미 넣은 건 건너뜀), 사진이 없으면 자료에서 첫 프레임을 고른다."""
    from pipeline import inputs
    from pipeline.rag import Store
    found = inputs.scan(a.input)
    inputs.log_found(found)
    a.kb = a.kb or Path(a.input[0].rstrip("\\/")).name or "input"
    st = Store(a.kb)
    st.add(a.input)
    if not a.image:
        hits = st.search(a.request, k=1, modalities=("image", "video_segment"), first_frame=True) or \
            st.search(a.request, k=1, modalities=("image", "video_segment"))
        if not hits:
            raise SystemExit("자료에 첫 프레임으로 쓸 사진·영상이 없습니다. --image로 사진을 지정해 주세요.")
        a.image = str(st.files / hits[0]["file"])
        log(f"[입력] 첫 프레임 = {hits[0]['source']} {hits[0]['loc']} ({hits[0]['text'][:60]}…)")


def cmd_long(a) -> None:
    from pipeline import longrun
    if getattr(a, "input", None):
        _kb_from_input(a)
    if not a.image:
        raise SystemExit("--image 또는 --input 중 하나는 필요합니다.")
    img = Path(a.image)
    if not img.exists():
        raise SystemExit(f"사진이 없습니다: {img}")
    cfg = longrun.LongConfig(image=img, request=a.request, seconds=a.minutes * 60, retries=a.retries,
                             min_score=a.min_score, frames=a.frames, size=a.size, fit=a.fit, narration=a.narration,
                             voice=a.voice, plan_only=a.plan_only, out_root=Path(a.out), name=a.name or "long",
                             kb=getattr(a, "kb", None))
    longrun.start(cfg)


def cmd_resume(a) -> None:
    from pipeline import longrun
    longrun.resume(Path(a.run_dir))


def cmd_assemble(a) -> None:
    from pipeline import longrun
    longrun.assemble(Path(a.run_dir))


def cmd_promo(a) -> None:
    from pipeline.promo import ingest, planner, render
    if a.pcmd == "ingest":
        if not a.pdf and not a.photos and not a.input:
            raise SystemExit("--input, --pdf, --photos 중 하나는 필요합니다.")
        ingest.ingest(Path(a.pdf) if a.pdf else None, Path(a.photos) if a.photos else None,
                      Path(a.out) / a.name, org=a.org, inputs=a.input)
    elif a.pcmd in ("plan", "render") and not Path(a.project).exists() and (Path("promo_projects") / a.project).exists():
        a.project = str(Path("promo_projects") / a.project)                 # 이름만 줘도 된다 (예: 클라우드팜)
    if a.pcmd == "plan":
        for fmt in a.format.split(","):
            planner.plan(Path(a.project), fmt.strip(), use_llm=not a.rules)
    elif a.pcmd == "render":
        for fmt in a.format.split(","):
            render.render(Path(a.project), fmt.strip(), draft=a.draft, narration=a.narration,
                          subtitles=not a.no_subtitles, bgm=Path(a.bgm) if a.bgm else None)


def cmd_rag(a) -> None:
    from pipeline.rag import Store
    st = Store(a.store)
    if a.rcmd == "add":
        st.add(a.items, speech=not a.no_speech)
    elif a.rcmd == "ask":
        hits = st.search(a.query, k=a.k, modalities=tuple(a.only.split(",")) if a.only else None)
        for h in hits:
            log(f"  {h['score']:<8} {h['modality']:<13} {h['source']} {h['loc']}\n           {h['text'][:150]}")
        out = st.results_html(a.query, hits, st.dir / "last_search.html")
        log(f"\n사진·프레임까지 보려면: {out}")
    elif a.rcmd == "info":
        cnt = {}
        for c in st.chunks:
            cnt[c["modality"]] = cnt.get(c["modality"], 0) + 1
        log(f"{st.dir}\n  조각 {len(st.chunks)}개 {cnt}\n  텍스트 임베딩 {st.meta.get('text_model')} · "
            f"이미지 임베딩 {st.meta.get('image_model') or '미사용(설명문으로 검색)'}\n  넣은 입력 {len(st.sources)}개")
        for v in st.sources.values():
            log(f"    - {v['item']} ({v['chunks']}조각)")


def cmd_tokens(a) -> None:
    from pipeline.budget import BUDGET, MODES
    hub = Hub()
    m = hub.models()
    if a.probe:
        targets = [x for x in m.available if not any(k in x.lower() for k in
                   ("embed", "rerank", "whisper", "tts", "melotts", "cosyvoice", "i2v", "t2v", "ltx", "image", "cuquantum", "groot", "pii", "bge"))]
        log("모델 최대 입력 길이 확인 중…")
        for x in targets:
            BUDGET.context(hub, x)
    log(f"\n모드: {BUDGET.mode_name}  ({', '.join(f'{k}={v}' for k, v in MODES[BUDGET.mode_name].items())})")
    log(BUDGET.summary())
    lim = BUDGET.limits()
    if lim:
        log("\n모델 최대 입력 길이 (토큰):")
        for k, v in sorted(((k, v) for k, v in lim.items() if not k.startswith("_")), key=lambda x: -x[1]):
            mark = " ← 글" if k == m.text else (" ← 비전" if k == m.vision else "")
            log(f"  {k:<28} {v:>9,}{mark}")
    else:
        log("\n아직 확인한 모델이 없습니다. python video_studio.py tokens --probe")


def cmd_web(a) -> None:
    try:
        from kiosk.server import serve
    except ImportError as e:
        raise SystemExit(f"웹서비스에 필요한 패키지가 없습니다 ({e.name}). 설치: pip install -r requirements.txt") from None
    serve(host=a.host, port=a.port, mock=a.mock, rounds=a.rounds, threshold=a.threshold)


def cmd_terms(_a) -> None:
    for t in lexicon.load_terms():
        log(f"[{t['category']:<17}] {t['term']:<8} ← {', '.join(t['aliases'])}\n{'':20}→ {t['en']}")


def main() -> None:
    faulthandler.enable()          # PDF·영상 라이브러리 내부에서 프로그램이 통째로 꺼질 때도 어디서 꺼졌는지 출력
    ap = argparse.ArgumentParser(description="사진 + 요구사항 → 영상 (Open AI Service Hub)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="키로 쓸 수 있는 모델과 역할별 자동 선택 결과")
    sub.add_parser("terms", help="등록된 움직임·카메라 용어")
    r = sub.add_parser("run", help="실행")
    r.add_argument("--image", required=True, help="입력 사진 (첫 프레임)")
    r.add_argument("--request", required=True, help="원하는 움직임·연출 (한국어 가능)")
    r.add_argument("--seconds", type=float, default=None, help="전체 길이(초). 샷 최대 길이보다 길면 여러 샷으로 나눔")
    r.add_argument("--rounds", type=int, default=2, help="샷당 최대 반복 (기본 2)")
    r.add_argument("--candidates", type=int, default=1, help="라운드당 후보 영상 수 (기본 1)")
    r.add_argument("--threshold", type=float, default=0.85, help="이 점수 이상이면 조기 종료 (기본 0.85)")
    r.add_argument("--size", default=None, help="해상도 (예: 832x480). 비우면 사진 비율에 맞춰 자동")
    r.add_argument("--fit", choices=("crop", "pad", "none"), default="crop", help="사진을 영상 비율에 맞추는 방식")
    r.add_argument("--frames", type=int, default=5, help="검증에 쓰는 프레임 수 (기본 5)")
    r.add_argument("--plan-only", action="store_true", help="이해·체크리스트까지만 (영상 생성 없음)")
    r.add_argument("--ask", action="store_true", help="모호한 점이 있으면 터미널에서 물어봄")
    r.add_argument("--name", default=None)
    r.add_argument("--out", default="output")
    lg = sub.add_parser("long", help="장편 (수 분~20분): 막→씬→샷 계획 후 생성")
    lg.add_argument("--image", default=None, help="첫 프레임 사진 (--input을 주면 생략 가능: 자료에서 자동 선택)")
    lg.add_argument("--input", nargs="+", default=None,
                    help="자료 폴더·파일·웹 주소 → 지식 저장소(이름 = 폴더 이름)로 넣고 근거·첫 프레임에 사용")
    lg.add_argument("--request", required=True, help="이야기·연출 요청 (한국어 가능)")
    lg.add_argument("--minutes", type=float, required=True, help="영상 길이(분)")
    lg.add_argument("--retries", type=int, default=1, help="샷 점수 미달 시 재시도 횟수 (기본 1)")
    lg.add_argument("--min-score", type=float, default=0.6, help="이 점수 미만이면 재시도 (기본 0.6)")
    lg.add_argument("--frames", type=int, default=4, help="샷 검사 프레임 수 (기본 4)")
    lg.add_argument("--size", default=None)
    lg.add_argument("--fit", choices=("crop", "pad", "none"), default="crop")
    lg.add_argument("--narration", action="store_true", help="씬 내레이션을 Hub TTS로 음성 합성 (미지원이면 자막만)")
    lg.add_argument("--voice", default=None)
    lg.add_argument("--plan-only", action="store_true", help="계획(plan.json)·예상 시간까지만")
    lg.add_argument("--kb", default=None, help="멀티모달 지식 저장소 이름/경로 (rag add로 만든 것)")
    lg.add_argument("--name", default=None)
    lg.add_argument("--out", default="output")
    rs = sub.add_parser("resume", help="장편: 계획 확인 후 생성 시작 / 끊긴 곳부터 이어서")
    rs.add_argument("run_dir")
    asm = sub.add_parser("assemble", help="장편: 지금까지 만든 부분으로 preview.mp4")
    asm.add_argument("run_dir")
    wb = sub.add_parser("web", help="전시회 키오스크 웹서비스 (사진+요청 → 생성 과정 실시간 시각화)")
    wb.add_argument("--host", default="127.0.0.1", help="기본 127.0.0.1 = 이 PC에서만 접속 (localhost)")
    wb.add_argument("--port", type=int, default=8000)
    wb.add_argument("--mock", action="store_true", help="연습 모드: Hub 없이 가짜 모델로 화면 흐름만 확인")
    wb.add_argument("--rounds", type=int, default=2, help="관람객 1명당 최대 시도 횟수 (기본 2)")
    wb.add_argument("--threshold", type=float, default=0.8, help="이 점수 이상이면 바로 완성 (기본 0.8)")
    tk = sub.add_parser("tokens", help="오늘 토큰 사용량 · 모드 · 모델별 최대 입력 길이")
    tk.add_argument("--probe", action="store_true", help="대화·비전 모델의 최대 입력 길이를 서버에 물어 기록")
    rg = sub.add_parser("rag", help="멀티모달 지식 저장소: 사진·영상·문서·웹을 넣고 검색")
    rs2 = rg.add_subparsers(dest="rcmd", required=True)
    ra = rs2.add_parser("add", help="파일·폴더·URL을 넣기 (이미 넣은 것은 건너뜀)")
    ra.add_argument("store", help="저장소 이름 (rag_stores 아래) 또는 경로")
    ra.add_argument("items", nargs="+", help="이미지·영상·PDF·docx·txt·md 파일, 폴더, http(s) 주소")
    ra.add_argument("--no-speech", action="store_true", help="영상의 말소리 받아쓰기 생략")
    rq = rs2.add_parser("ask", help="검색해 보기 (결과 사진까지 last_search.html)")
    rq.add_argument("store")
    rq.add_argument("query")
    rq.add_argument("--k", type=int, default=8)
    rq.add_argument("--only", default=None, help="text,image,video_segment,speech 중 일부만")
    ri = rs2.add_parser("info", help="저장소 상태")
    ri.add_argument("store")
    pr = sub.add_parser("promo", help="홍보영상: 브로셔·사진 → 사실 확인된 숏폼/메인/설명형")
    ps = pr.add_subparsers(dest="pcmd", required=True)
    pi = ps.add_parser("ingest", help="브로셔 읽기: 숫자·문구 원문 확인, 사진 분류, 연락처, 오류 점검 → brief.json")
    pi.add_argument("--input", nargs="+", default=None,
                    help="자료 폴더(또는 파일·웹 주소 여러 개): PDF·사진·동영상·docx/pptx/xlsx/hwpx/txt·.url/urls.txt")
    pi.add_argument("--pdf")
    pi.add_argument("--photos", help="실사진 폴더 (건물·전산실·행사 등). 브로셔 사진보다 우선 사용")
    pi.add_argument("--name", required=True, help="프로젝트 이름 (promo_projects 아래 폴더)")
    pi.add_argument("--org", default=None, help="기관 이름 (비우면 브로셔에서)")
    pi.add_argument("--out", default="promo_projects")
    pp = ps.add_parser("plan", help="스토리보드: 형식 템플릿에 사진·사실·문구 배치 → storyboard_<형식>.json")
    pp.add_argument("project")
    pp.add_argument("--format", default="main", help="short | main | explainer (쉼표로 여러 개)")
    pp.add_argument("--rules", action="store_true", help="LLM 없이 규칙으로만 구성")
    prr = ps.add_parser("render", help="영상 만들기")
    prr.add_argument("project")
    prr.add_argument("--format", default="main")
    prr.add_argument("--draft", action="store_true", help="GPU 없이 카메라 움직임만으로 빠른 미리보기")
    prr.add_argument("--narration", action="store_true", help="Hub TTS로 내레이션 음성")
    prr.add_argument("--no-subtitles", action="store_true")
    prr.add_argument("--bgm", default=None, help="배경음악 파일 (mp3/wav)")
    a = ap.parse_args()
    try:
        {"web": cmd_web, "models": cmd_models, "run": cmd_run, "terms": cmd_terms, "long": cmd_long, "resume": cmd_resume,
         "assemble": cmd_assemble, "promo": cmd_promo, "rag": cmd_rag, "tokens": cmd_tokens}[a.cmd](a)
    except HubError as e:
        raise SystemExit(f"\n[Hub 오류] {e}")


if __name__ == "__main__":
    main()
