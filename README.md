# VIDEO Studio

사진 한 장과 원하는 움직임을 한국어로 적으면, Open AI Service Hub(open.hasa.re.kr)의 LLM·VLM이 **요구사항을 움직임
단위로 이해하고 → wan2.2-i2v로 영상을 만들고 → 프레임을 보고 스스로 검사·반성해서** 가장 잘 맞는 영상을 골라 주는
도구입니다. 짧은 영상(4~16초)은 정밀 모드(`run`), **수 분~20분 영상은 장편 모드(`long`)**, **기관·사업 홍보영상은 홍보 모드(`promo`)** 로 만듭니다.
구조와 근거 논문(ICCV·ECCV·CVPR·AAAI·ICML)은 [PIPELINE.md](PIPELINE.md)에 있습니다.

## 설치 (Windows 기준)

```
cd %USERPROFILE%\Downloads\VIDEO_Studio
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        :: .env 의 OASH_API_KEY 에 개발키 입력
```

ffmpeg: 짧은 모드는 없어도 되지만 **장편 모드는 필요**합니다 (`winget install Gyan.FFmpeg` 후 새 터미널).

## 1. 모델 확인

```
python video_studio.py models
```

자동 선택 기본값: text `qwen3-next-80b` · vision `qwen3-vl-32b` · video `wan2.2-i2v`(16fps, 샷당 최대 4초,
832x480 / 640x480). 바꾸려면 `.env`의 `OASH_TEXT_MODEL`, `OASH_VISION_MODEL`, `OASH_VIDEO_MODEL`에 직접 적으세요.

## 2. 요구사항이 어떻게 이해됐는지 먼저 보기 (생성 없음)

```
python video_studio.py run --image 내사진.jpg --request "바람에 머리카락이 날리고 카메라가 천천히 다가가 줘" --plan-only
```

`output\…\report.html`에 장면 관찰(대상 s1…), 목표 분해(범주·시점·강도·물리 규칙), 샷 계획, 보존 항목, 가정,
확인 질문, 위험, 검증 체크리스트가 나옵니다. 여기서 이해가 틀렸으면 요청 문장을 고치고 다시 보세요.

## 3. 실행

```
:: 기본: 1샷(4초), 2라운드, 라운드당 후보 1개
python video_studio.py run --image 내사진.jpg --request "바람에 머리카락이 날리고 천천히 줌인"

:: 8초 → 샷 2개로 나눠 이어 붙임 (2번째 샷은 1번째 샷의 마지막 프레임에서 시작)
python video_studio.py run --image 내사진.jpg --request "처음엔 바람에 머리카락이 날리다가, 그다음 고개를 돌려 미소" --seconds 8

:: 모호한 점은 터미널에서 물어보게, 후보 2개씩 비교
python video_studio.py run --image 내사진.jpg --request "분위기 있게 움직여줘" --ask --candidates 2

:: 등록된 움직임·카메라 용어
python video_studio.py terms
```

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--seconds` | 샷 최대 길이 | 전체 길이. 샷 최대 길이(4초)보다 길면 샷으로 분할 (최대 4샷) |
| `--rounds` | 2 | 샷당 설계→생성→검증→반성 최대 반복 |
| `--candidates` | 1 | 라운드당 후보 영상 수 (영상 1건 = 수 분) |
| `--threshold` | 0.85 | 이 점수 이상이면 그 샷은 조기 종료 |
| `--size` | 자동 | 사진 비율에 가장 가까운 모델 해상도 |
| `--fit` | crop | 사진을 영상 비율에 맞추는 방식: crop(가운데 자름) / pad(흐린 여백) / none(늘림) |
| `--frames` | 5 | 검증에 쓰는 프레임 수. 눈 깜빡임처럼 짧은 동작은 8 이상 권장 |
| `--plan-only` | – | 0~2단계만 |
| `--ask` | – | 확인 질문이 있으면 터미널에서 답을 받아 다시 이해 |

결과 폴더: `final.mp4`(최종), `report.html`(이해·체크리스트·라운드별 후보 영상·프레임 시트·점수·실패 질문·반성),
`run.json`(전체 기록), `shotN/roundR/cK.*`(후보 영상).

## 4. 장편 (수 분~20분)

```
:: ① 계획만: 이야기 설정·막·씬·샷(plan.json)과 예상 시간 확인
python video_studio.py long --image 내사진.jpg --request "이 사람이 바닷가 마을에서 하루를 보내는 이야기. 아침 산책, 시장, 노을" --minutes 15 --plan-only

:: ② plan.json을 필요하면 고친 뒤 생성 시작 (같은 명령으로 끊긴 곳부터 이어서)
python video_studio.py resume output\20261025_160000_long

:: ③ 생성 중에도 지금까지 만든 부분으로 미리보기 (preview.mp4)
python video_studio.py assemble output\20261025_160000_long
```

`--plan-only` 없이 실행하면 계획 후 바로 생성합니다. `run --seconds 60`처럼 16초를 넘기면 자동으로 장편 모드로 넘어갑니다.

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--minutes` | (필수) | 영상 길이(분). 20분 ≈ 막 7개 · 씬 20~40개 · 샷 약 300개 |
| `--retries` | 1 | 샷 점수가 `--min-score` 미만이면 원인을 분석해 프롬프트를 고쳐 다시 생성하는 횟수 |
| `--min-score` | 0.6 | 재시도 기준 |
| `--narration` | – | 씬 내레이션을 Hub TTS(melotts-ko)로 음성 합성. 안 되면 자막(final.srt)만 |
| `--frames` | 4 | 샷 검사 프레임 수 |

결과 폴더: `final.mp4`, `final.srt`(내레이션 자막), `chapters.txt`(씬 시작 시각), `report.html`(진행률·남은 시간·
씬별 키프레임·점수·재시도·재고정, 생성 중 60초마다 새로고침), `plan.json`(계획), `progress.json`(샷별 기록), `scenes/`(씬 클립).

**얼마나 걸리나:** 20분 ≈ 영상 작업 330~390건. 실제 Hub 실측으로 1건 평균 45~50초라, 개발키(동시 1건)로 약 4~5시간입니다. 실행하면 실측
시간으로 남은 시간을 다시 계산합니다. 줄이는 방법은 PIPELINE.md의 '시간과 호출 수'를 보세요 (동시 한도 상향 시 씬 줄기 병렬,
LTX-2 이미지 입력이 되면 샷당 10초).

**끊겼을 때:** 네트워크·GPU 오류는 같은 샷을 3번까지 다시 시도하고, 그래도 안 되면 멈춥니다. 원인이 풀리면 `resume`으로
끝난 샷은 건너뛰고 이어서 만듭니다. 컴퓨터를 꺼도 폴더만 있으면 됩니다.

## 5. 홍보영상 (브로셔·사진 → 사실 확인된 홍보영상)

기관·사업을 알리는 영상에 특화된 모드입니다. 일반 모드와 달리 **AI는 브로셔에서 원문이 확인된 숫자와 문장만 고를 수 있고**,
숫자·글자·로고·QR은 생성 모델이 아니라 코드가 그립니다. 실제 시설 사진은 모양을 바꾸지 않는 느린 카메라 움직임만 씁니다.

```
:: ① 브로셔 읽기: 숫자·문구 원문 확인, 사진 분류, 연락처, 시안 오류 점검 → brief.json
python video_studio.py promo ingest --pdf 브로셔.pdf --photos 실사진폴더 --name 우리기관

:: ② 스토리보드 (short=숏폼 24초 세로 · main=메인 70초 · explainer=설명형 약 3분)
python video_studio.py promo plan promo_projects\우리기관 --format short,main

:: ③ 초안: GPU 없이 카메라 움직임으로 몇 분 안에 전체 미리보기 (그래픽·자막·엔딩카드 확인용)
python video_studio.py promo render promo_projects\우리기관 --format main --draft

:: ④ 본 렌더: 사진을 AI로 움직이고 홍보 검사 (+ 내레이션, 배경음악)
python video_studio.py promo render promo_projects\우리기관 --format main --narration --bgm 음악.mp3
```

**경북AI클라우드팜 사례가 미리 들어 있습니다** (`promo_projects\gb_aicloudfarm`). 시안(261019_시안02)의 숫자 17개·문구 28개가
원문 확인을 통과했고, 사진 12장이 분류되어 있으며, 숏폼·메인·설명형 스토리보드와 초안 영상이 있습니다.
`report_main_draft.html`을 열면 초안 영상, 장면별 출처 쪽수, 사실 시트, 시안에서 확인할 점이 한눈에 보입니다.

```
python video_studio.py promo render promo_projects\gb_aicloudfarm --format main            :: 본 렌더 (Hub 필요)
python tools\build_case_gb_aicloudfarm.py                                                 :: brief.json 다시 만들기
```

고치는 방법:
- 문구·숫자·연락처·엔딩 문장·색·로고: `brief.json` (`brand.logo`에 로고 파일 경로를 넣으면 엔딩에 표시)
- 장면 순서·사진·움직임·보여줄 항목: `storyboard_<형식>.json` 을 고친 뒤 render만 다시. AI 영상은 `cache\`에 남아
  있어 글자·숫자만 바꾸면 GPU를 다시 쓰지 않습니다.
- 실제 전산실·행사 사진을 `--photos`로 주면 브로셔 사진보다 먼저 씁니다. 사람 얼굴이 나오는 사진은 공개 동의를 받은 것만 쓰세요.
- AI가 움직인 장면에는 오른쪽 위에 'AI 생성 장면'이 표시됩니다 (`brand.ai_label`, 빈 값이면 끔).

## 6. 여러 자료로 만들기 (멀티모달 지식 저장소)

사진·동영상·문서(PDF·docx·txt·md)·웹사이트를 한 저장소에 넣으면, 조각으로 나눠 벡터로 저장해 두고
장편 영상을 만들 때 **장면마다 맞는 자료를 찾아 Wan에 넘깁니다.**

```
:: ① 자료 넣기 (폴더·파일·주소를 섞어서, 여러 번 나눠 넣어도 됨 — 이미 넣은 것은 건너뜀)
python video_studio.py rag add 클라우드팜 자료폴더 홍보영상.mp4 시안.pdf https://hafa.gbcloud.ai

:: ② 잘 찾아지는지 확인 (결과 사진까지 rag_stores\클라우드팜\last_search.html)
python video_studio.py rag ask 클라우드팜 "서버 랙이 있는 전산실"
python video_studio.py rag info 클라우드팜

:: ③ 장편 영상에 연결
python video_studio.py long --image 내사진.jpg --request "센터를 소개하는 이야기" --minutes 3 --kb 클라우드팜 --plan-only
```

| 자료 | 저장되는 조각 | 영상에서 쓰이는 곳 |
|---|---|---|
| 사진 | 사진 + VLM 설명문 | 씬의 첫 프레임 후보 (장소·사물 장면) |
| 동영상 | 장면 전환마다 구간(시작 프레임 + 움직임·카메라 설명) + 말소리 받아쓰기 | 첫 프레임 후보, 샷 프롬프트의 '참고 움직임', 근거 |
| PDF | 쪽별 글 + 사진 | 근거(사실·장소·이름), 첫 프레임 후보 |
| 문서·웹 | 본문 글 (+ 웹 큰 사진) | 이야기 계획의 근거 |

- 검색 = 글 벡터(+이미지 벡터) 순위 합치기 → 리랭커(bge-reranker-v2-m3-ko) → 거의 같은 조각 제거
- 실제 Hub 점검 결과 이미지 임베딩은 지원되지 않아, 사진·영상 프레임은 VLM 설명문(글자 있는 사진은 OCR 포함)으로 찾습니다.
  영상은 구간을 잘라 VLM(qwen3-vl-32b)에 직접 보여 움직임·카메라를 설명하게 합니다. 생성 영상 검사도 같은 방식입니다.
- 새 장면의 첫 프레임은 ① 계획 단계가 고른 자료(visual_ref) ② 없으면 저장소에서 가장 맞는 사진·영상 프레임을 검색 ③ 그래도 없을 때만 입력 사진을 편집해 만듭니다.

## 입력 폴더 하나로 넣기 (`--input`)

자료를 한 폴더에 모아 두면 안에 있는 파일을 종류별로 읽습니다 (하위 폴더 포함).

| 넣을 수 있는 것 | 읽는 방법 |
|---|---|
| PDF (여러 개 가능) | 쪽별 글 + 큰 사진 |
| 사진 .jpg .png .webp | 실사진으로 우선 사용 |
| 동영상 .mp4 .mov 등 | 장면마다 대표 프레임(사진 후보) + 말소리 받아쓰기 |
| 문서 .docx .pptx .xlsx .hwpx .txt .md .csv .json | 글 (pptx는 슬라이드별, hwpx는 구역별) |
| 웹 주소 | `.url`(인터넷 바로가기), 또는 주소만 한 줄씩 적은 `urls.txt` → 본문 글 + 큰 사진 |
| 구형 `.hwp` | 읽을 수 없어 건너뜀 → 한글에서 `.hwpx`나 PDF로 저장해서 넣기 |

```
:: 홍보영상
python video_studio.py promo ingest --input input\gbcloud_pdf --name 클라우드팜
python video_studio.py promo plan promo_projects\클라우드팜 --format short,main
python video_studio.py promo render promo_projects\클라우드팜 --format main --draft
python video_studio.py promo render promo_projects\클라우드팜 --format main --narration

:: 장편 (자료 폴더 → 지식 저장소 'gbcloud_pdf' 자동 생성, 첫 프레임도 자료에서 자동 선택)
python video_studio.py long --input input\gbcloud_pdf --request "경북 초거대 AI 클라우드팜 소개" --minutes 5 --plan-only
```

- `brief.json`의 사실·문구·확인할 점마다 `where`(예: `GB-HAFA-addon-leaflet.pdf 2쪽`)가 붙어 어느 파일에서 왔는지 보입니다.
- 폴더에 파일을 더 넣고 다시 실행하면, 장편 모드의 지식 저장소는 새 파일만 추가로 읽습니다.

## 7. 토큰 예산 (모델 최대 길이 활용)

기본은 **max 모드**입니다. 모델마다 최대 입력 길이를 서버에서 알아내 그 안을 채워 씁니다.

```
python video_studio.py tokens --probe     :: 대화·비전 모델의 최대 입력 길이를 확인해 기록 (처음 한 번)
python video_studio.py tokens             :: 오늘 사용량 / 하루 예산 · 모드 · 모델별 최대 길이
```

| 항목 | eco (예전) | max (기본) |
|---|---|---|
| 계획에 넣는 근거 | 약 3,500자, 14개 | 글 모델 최대 길이의 60%까지 관련도순 (최대 200조각, 조각 내용은 자르지 않음) |
| 브로셔 읽기 | 4쪽씩, 쪽당 3,500자에서 자름 | 쪽을 자르지 않고 예산에 들어가는 만큼 한 번에 |
| 답변 길이 | 3,000토큰 고정 | 모델 최대 길이 − 입력 길이 (상한 16K) |
| 생성 영상 검사 | 1회 판정 | 3회 판정 다수결 (사소한 흠으로 인한 헛재시도 감소) |
| 이야기 계획 | 1개 | 후보 3개 → 심판 호출로 최선 선택 |
| 자료 사진·영상 설명 | 한두 문장 | 상세 설명(120~200단어 + 한국어)까지 → 검색 단서 증가 |
| VLM 이미지 해상도 | 768px | 1280px |

- 응답의 usage를 `output\token_usage.json`에 날짜별로 더하고, 하루 예산(`VS_DAILY_TOKEN_BUDGET`, 기본 2천만)의 90%를 넘으면 그날은 자동으로 eco로 내립니다.
- 모드는 `.env`의 `VS_TOKEN_MODE=max|balanced|eco`로 바꿉니다.
- 대략적인 사용량(추정): 자료 저장소를 쓰는 3분 장편 한 편에 수백만 토큰(검사 다수결과 근거 채우기가 대부분). 실제 값은 `tokens` 명령으로 확인하세요.

## 움직임·카메라 용어집

`knowledge/motion_terms.json` — "줌인, 팬, 틸트, 오빗, 고정 카메라, 바람, 물결, 눈 깜빡임…" 25개. 요청 문장에서 잡히면
영상 모델이 잘 따르는 영어 표현과 **프레임 비교로 확인할 수 있는 검증 질문**을 같이 씁니다. 현장 표현을 추가하려면
같은 형식으로 `knowledge/custom_terms.json`을 만들면 됩니다(같은 term이면 덮어씀).

## 실제 Hub 없이 흐름만 테스트

```
python tests\test_offline.py      :: 짧은 모드
python tests\test_long.py         :: 장편 모드: 1분(샷 15개) 계획 → 중간에 GPU 오류로 중단 → 미리보기 → 재개 → 내레이션·자막 조립
python tests\test_promo.py        :: 홍보 모드: 브로셔 읽기(가짜 숫자·문장 거르기) → 스토리보드 → 초안 → 본 렌더 → 캐시 재사용
python tests\test_rag.py          :: 지식 저장소: 사진·영상(장면 전환·말소리)·텍스트·웹 → 검색 → 장편 모드 연결
python tests\test_inputs.py       :: 입력 폴더: PDF 여러 개·사진·영상·docx/pptx/xlsx/hwpx·.url → 홍보·장편
python tests\test_budget.py       :: 토큰 예산: 최대 길이 탐지·재요청·사용량·다수결·근거 채우기·계획 후보 선택
```

가짜 Hub(`tests/mock_hub.py`)가 첫 프레임으로 실제 mp4를 만들어 돌려줍니다. 첫 영상은 일부러 거의 정지 →
'정지' 감점과 반성(움직임 강화) → 2라운드 통과까지, 그리고 2샷 이어 붙이기·후보 비교를 확인합니다.

## 실제 Hub에서 처음 확인할 것

- `wan2.2-i2v`가 세로 해상도(480x832)를 받는지. 받으면 `--size 480x832`로 세로 사진을 자르지 않고 쓸 수 있습니다.
- `LTX-2`가 첫 프레임 이미지를 받는지(`.env`에 `OASH_VIDEO_MODEL=LTX-2`, `OASH_VIDEO_ACCEPTS_IMAGE=1`). 받으면 샷당 최대 10초.
- `/audio/speech`로 melotts-ko 음성 합성이 되는지(`--narration`). 안 되면 자막만 만들어집니다.
- 영상 1건 실제 소요 시간 → `.env`의 `VS_EST_JOB_SECONDS`에 넣으면 장편 계획 단계의 예상 시간이 정확해집니다.
- `VS_STATIC_THRESHOLD`(기본 0.35): 몇 번 돌려 보고 report.html의 '움직임' 값을 보며 정지 판정 기준을 맞추세요.

## 오류가 날 때

| 증상 | 조치 |
|---|---|
| 404 | 등록되지 않은 모델 → `python video_studio.py models` |
| 429 / 503 | 한도 초과 또는 GPU 준비 중 → 자동으로 기다렸다 재시도 |
| 401/403 | 개발키 만료·폐기, 또는 해당 모델 권한 없음 |
| `JSON을 반환하지 못했습니다` | `OASH_TEXT_MODEL`을 더 큰 모델(gpt-oss-120b 등)로 |
| 점수가 이상함 | report.html의 '관찰'·'실패한 질문' 근거 확인, `--frames` 늘리기, `OASH_VISION_MODEL` 변경 |
| 얼굴이 바뀜 | `--rounds`를 늘리면 반성 단계가 '정체성 보호' 경로로 프롬프트를 고침 |
| 영상이 거의 안 움직임 | 반성 단계가 '움직임 강화'로 고침. 계속이면 요청에 강도("크게", "빠르게")를 적어 보세요 |
