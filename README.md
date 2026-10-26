# VIDEO Studio

사진 한 장(또는 자료 묶음)과 원하는 내용을 한국어로 적으면, Open AI Service Hub(open.hasa.re.kr)의 모델이
**요구사항을 움직임 단위로 이해하고 → 영상을 만들고 → 프레임을 보고 스스로 채점·반성해서** 가장 잘 맞는 영상을 골라 주는 도구입니다.
구조와 근거 논문(ICCV·ECCV·CVPR·AAAI·ICML)은 [PIPELINE.md](PIPELINE.md)에 있습니다.

## 목차

1. [시작하기](#1-시작하기)
2. [모드 한눈에 보기](#2-모드-한눈에-보기)
3. [짧은 영상 (`run`)](#3-짧은-영상-run)
4. [장편 영상 (`long`)](#4-장편-영상-long)
5. [홍보영상 (`promo`)](#5-홍보영상-promo)
6. [자료 넣기 (`--input`, `rag`)](#6-자료-넣기---input-rag)
7. [전시회 키오스크 (`web`)](#7-전시회-키오스크-web)
8. [모델 활용 설정](#8-모델-활용-설정)
9. [Hub 없이 테스트](#9-hub-없이-테스트)
10. [문제 해결](#10-문제-해결)

---

## 1. 시작하기

### 1.1 설치 (Windows)

```
git clone https://github.com/ghkim-dev/HASA_promotion_video.git
cd HASA_promotion_video
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        :: .env 의 OASH_API_KEY 에 개발키 입력
```

- **ffmpeg**: 짧은 영상(`run`)은 없어도 되지만 장편·홍보영상·키오스크는 필요합니다 (`winget install Gyan.FFmpeg` 후 새 터미널).
- 가상환경에 `pymupdf4llm`·`pymupdf-layout`·`onnxruntime`이 있어도 됩니다. PDF를 읽을 때 이 부가 기능은 불러오지 않습니다.

### 1.2 `.env` 주요 설정

| 설정 | 기본 | 설명 |
|---|---|---|
| `OASH_API_KEY` | (필수) | Hub 개발키 |
| `OASH_TEXT_MODEL` / `OASH_VISION_MODEL` / `OASH_VIDEO_MODEL` | 자동 | 비우면 qwen3-next-80b / qwen3-vl-32b / wan2.2-i2v |
| `OASH_VIDEO_QUALITY` | 480p | `720p`면 1280×720·720×1280 우선 (생성 시간 약 2~4배) |
| `OASH_MAX_CONCURRENCY` | 1 | 키의 동시 요청 한도. 늘면 후보 영상을 병렬 생성 |
| `VS_TOKEN_MODE` | max | 토큰 사용 방식 `max` / `balanced` / `eco` ([8.2](#82-토큰-예산)) |
| `VS_DAILY_TOKEN_BUDGET` | 20000000 | 하루 토큰 예산. 90%를 넘으면 그날은 자동 eco |
| `VS_EST_JOB_SECONDS` | 50 | 영상 1건 예상 시간(초). 장편 계획의 예상 시간 계산용 |

전체 목록과 설명은 `.env.example`에 있습니다.

### 1.3 모델 확인

```
python video_studio.py models      :: 쓸 수 있는 모델과 역할별 자동 선택 결과
python video_studio.py tokens --probe   :: 대화·비전 모델의 최대 입력 길이 확인 (처음 한 번)
```

---

## 2. 모드 한눈에 보기

| 모드 | 이럴 때 | 길이 | 명령 |
|---|---|---|---|
| 짧은 영상 | 사진 한 장을 원하는 대로 움직이기 | 4~16초 | `run` |
| 장편 영상 | 사진 한 장(+자료)으로 이야기가 있는 영상 | 수 분~20분 | `long` → `resume` |
| 홍보영상 | 브로셔·자료의 사실만으로 기관·사업 홍보 | 24초 / 70초 / 약 3분 | `promo ingest` → `plan` → `render` |
| 지식 저장소 | 사진·영상·문서·웹을 모아 장편에 근거·첫 장면으로 연결 | – | `rag add` / `ask` |
| 키오스크 | 전시장에서 관람객이 직접 체험 | 4초 | `web` |

---

## 3. 짧은 영상 (`run`)

### 3.1 이해 결과 먼저 보기 (생성 없음)

```
python video_studio.py run --image 내사진.jpg --request "바람에 머리카락이 날리고 카메라가 천천히 다가가 줘" --plan-only
```

`output\…\report.html`에 장면 관찰(대상 s1…), 목표 분해(범주·시점·강도·물리 규칙), 샷 계획, 보존 항목, 가정,
확인 질문, 위험, 검증 체크리스트가 나옵니다. 이해가 틀렸으면 요청 문장을 고쳐 다시 보세요.

### 3.2 실행

```
:: 기본: 1샷(4초), 2라운드, 라운드당 후보 1개
python video_studio.py run --image 내사진.jpg --request "바람에 머리카락이 날리고 천천히 줌인"

:: 8초 → 샷 2개로 나눠 이어 붙임 (2번째 샷은 1번째 샷의 마지막 프레임에서 시작)
python video_studio.py run --image 내사진.jpg --request "처음엔 바람에 머리카락이 날리다가, 그다음 고개를 돌려 미소" --seconds 8

:: 모호한 점은 터미널에서 물어보게, 후보 2개씩 비교
python video_studio.py run --image 내사진.jpg --request "분위기 있게 움직여줘" --ask --candidates 2
```

### 3.3 옵션

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--seconds` | 샷 최대 길이 | 전체 길이. 4초보다 길면 샷으로 분할 (최대 4샷). 16초를 넘기면 장편 모드로 넘어감 |
| `--rounds` | 2 | 샷당 설계→생성→검증→반성 최대 반복 |
| `--candidates` | 1 | 라운드당 후보 영상 수 |
| `--threshold` | 0.85 | 이 점수 이상이면 그 샷은 조기 종료 |
| `--size` | 자동 | 사진 비율에 가장 가까운 해상도 (세로 사진은 세로 영상) |
| `--fit` | crop | crop(가운데 자름, 비율 차이가 크면 자동으로 pad) / pad(흐린 여백) / none(늘림) |
| `--frames` | 5 | 검증 프레임 수. 눈 깜빡임처럼 짧은 동작은 8 이상 권장 |
| `--plan-only` | – | 이해·체크리스트까지만 |
| `--ask` | – | 확인 질문이 있으면 터미널에서 답을 받아 다시 이해 |

### 3.4 결과

`final.mp4`(최종), `report.html`(이해·체크리스트·라운드별 후보 영상·프레임 시트·점수·실패 질문·반성),
`run.json`(전체 기록), `shotN/roundR/cK.*`(후보 영상).

---

## 4. 장편 영상 (`long`)

### 4.1 실행

```
:: ① 계획만: 이야기 설정·막·씬·샷(plan.json)과 예상 시간 확인
python video_studio.py long --image 내사진.jpg --request "이 사람이 바닷가 마을에서 하루를 보내는 이야기. 아침 산책, 시장, 노을" --minutes 15 --plan-only

:: ② plan.json을 필요하면 고친 뒤 생성 시작 (같은 명령으로 끊긴 곳부터 이어서)
python video_studio.py resume output\20261025_160000_long

:: ③ 생성 중에도 지금까지 만든 부분으로 미리보기 (preview.mp4)
python video_studio.py assemble output\20261025_160000_long

:: 자료 폴더를 함께 주면 지식 저장소를 만들어 근거·첫 장면에 사용 (--image 생략 가능)
python video_studio.py long --input input\자료폴더 --request "센터 소개" --minutes 5 --plan-only
```

`--plan-only` 없이 실행하면 계획 후 바로 생성합니다.

### 4.2 옵션

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--minutes` | (필수) | 영상 길이(분). 20분 ≈ 막 7개 · 씬 20~40개 · 샷 약 300개 |
| `--retries` | 1 | 샷 점수가 `--min-score` 미만이면 원인을 분석해 다시 생성하는 횟수 |
| `--min-score` | 0.6 | 재시도 기준 |
| `--narration` | – | 씬 내레이션을 Hub TTS로 음성 합성. 안 되면 자막(final.srt)만 |
| `--kb` | – | 지식 저장소 이름 ([6.2](#62-지식-저장소-rag)) |
| `--input` | – | 자료 폴더·파일·주소 → 저장소 자동 생성 (이름 = 폴더 이름) |
| `--frames` | 4 | 샷 검사 프레임 수 |

### 4.3 결과·시간·이어 하기

- **결과 폴더**: `final.mp4`, `final.srt`(내레이션 자막), `chapters.txt`(씬 시작 시각), `report.html`(진행률·남은 시간·씬별 키프레임·점수·재시도, 생성 중 60초마다 새로고침), `plan.json`, `progress.json`, `scenes/`.
- **걸리는 시간**: 20분 ≈ 영상 작업 330~390건. 실측 1건 평균 45~50초라 개발키(동시 1건)로 약 4~5시간. 실행 중 실측 시간으로 남은 시간을 다시 계산합니다.
- **끊겼을 때**: 네트워크·GPU 오류는 같은 샷을 3번까지 다시 시도하고, 그래도 안 되면 멈춥니다. `resume`으로 끝난 샷은 건너뛰고 이어 만듭니다.

---

## 5. 홍보영상 (`promo`)

기관·사업 홍보에 특화된 모드입니다. **AI는 자료에서 원문이 확인된 숫자와 문장만 고를 수 있고**, 숫자·글자·로고·QR은
생성 모델이 아니라 코드가 그립니다. 실제 시설 사진은 모양을 바꾸지 않는 느린 카메라 움직임만 씁니다.

### 5.1 만드는 순서

```
:: ① 자료 읽기: 숫자·문구 원문 확인, 사진 분류, 연락처, 시안 오류 점검 → brief.json
python video_studio.py promo ingest --input input\자료폴더 --name 우리기관
:: (PDF 하나와 사진 폴더만 줄 때)  promo ingest --pdf 브로셔.pdf --photos 실사진폴더 --name 우리기관

:: ② 스토리보드
python video_studio.py promo plan 우리기관 --format short,main

:: ③ 초안: Hub 영상 생성 없이 카메라 움직임만으로 몇 분 안에 미리보기 (그래픽·자막·엔딩카드 확인용)
python video_studio.py promo render 우리기관 --format main --draft

:: ④ 본 렌더: 사진을 AI로 움직이고 홍보 검사 (+ 내레이션, 배경음악)
python video_studio.py promo render 우리기관 --format main --narration --bgm 음악.mp3
```

| 형식 | 길이 | 화면 | 용도 |
|---|---|---|---|
| `short` | 약 24초 | 세로 9:16 | SNS·박람회 반복 재생 |
| `main` | 약 70초 | 가로 16:9 | 홈페이지·행사 오프닝 |
| `explainer` | 약 2분 50초 | 가로 16:9 | 상담·설명회 |

### 5.2 결과 확인·수정

- `promo_projects\우리기관\report_<형식>_draft.html`: 초안 영상, 장면별 출처(파일명·쪽), 사실 시트, 자료에서 확인할 점.
- 문구·숫자·연락처·엔딩 문장·색·로고: `brief.json` (`brand.logo`에 로고 파일 경로를 넣으면 엔딩에 표시).
- 장면 순서·사진·움직임·보여줄 항목: `storyboard_<형식>.json`을 고친 뒤 render만 다시. AI 영상은 `cache\`에 남아 있어 글자·숫자만 바꾸면 영상을 다시 만들지 않습니다.
- 실제 전산실·행사 사진을 주면 브로셔 사진보다 먼저 씁니다. 사람 얼굴이 나오는 사진은 공개 동의를 받은 것만 쓰세요.
- AI가 움직인 장면에는 오른쪽 위에 'AI 생성 장면'이 표시됩니다 (`brand.ai_label`, 빈 값이면 끔).
- 경북AI클라우드팜 시안으로 사례 프로젝트를 다시 만들려면: `python tools\build_case_gb_aicloudfarm.py --pdf 261019_시안02.pdf` (결과물 폴더 `promo_projects`는 저장소에 올리지 않습니다).

---

## 6. 자료 넣기 (`--input`, `rag`)

### 6.1 입력 폴더 (`--input`)

자료를 한 폴더에 모아 두면 안의 파일을 종류별로 읽습니다 (하위 폴더 포함). `promo ingest`와 `long`에서 씁니다.

| 넣을 수 있는 것 | 읽는 방법 |
|---|---|
| PDF (여러 개 가능) | 쪽별 글 + 큰 사진 |
| 사진 .jpg .png .webp | 실사진으로 우선 사용 |
| 동영상 .mp4 .mov 등 | 장면마다 대표 프레임(사진 후보) + 말소리 받아쓰기 |
| 문서 .docx .pptx .xlsx .hwpx .txt .md .csv .json | 글 (pptx는 슬라이드별, hwpx는 구역별) |
| 웹 주소 | `.url`(인터넷 바로가기), 또는 주소만 한 줄씩 적은 `urls.txt` → 본문 글 + 큰 사진 |
| 구형 `.hwp` | 읽을 수 없어 건너뜀 → 한글에서 `.hwpx`나 PDF로 저장해서 넣기 |

- `brief.json`의 사실·문구·확인할 점마다 `where`(예: `GB-HAFA-addon-leaflet.pdf 2쪽`)가 붙어 출처가 보입니다.
- 자료가 많으면 모델 답이 길어져 형식이 깨지거나 잘릴 수 있는데, 이때는 완성된 항목은 건지고 쪽을 반씩 나눠 다시 읽습니다. 끝내 못 읽은 쪽은 `debug\facts_raw_*.txt`에 원문 답이 남습니다.

### 6.2 지식 저장소 (`rag`)

사진·동영상·문서·웹을 한 저장소에 넣으면 조각으로 나눠 벡터로 저장해 두고,
장편 영상을 만들 때 **장면마다 맞는 자료를 찾아 영상 모델에 넘깁니다.**

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
| 사진 | 사진 + VLM 설명문 (글자 있으면 OCR 포함) | 씬의 첫 프레임 후보 |
| 동영상 | 장면 전환마다 구간(시작 프레임 + 움직임·카메라 설명) + 말소리 받아쓰기 | 첫 프레임 후보, 샷 프롬프트의 '참고 움직임', 근거 |
| PDF | 쪽별 글 + 사진 | 근거(사실·장소·이름), 첫 프레임 후보 |
| 문서·웹 | 본문 글 (+ 웹 큰 사진) | 이야기 계획의 근거 |

- 검색 = 글 벡터 순위 합치기 → 리랭커(bge-reranker-v2-m3-ko) → 거의 같은 조각 제거.
- Hub는 이미지 임베딩을 지원하지 않아, 사진·영상 프레임은 VLM 설명문으로 찾습니다. 영상 구간은 VLM(qwen3-vl-32b)에 직접 보여 움직임·카메라를 설명하게 합니다.
- 새 장면의 첫 프레임은 ① 계획 단계가 고른 자료 ② 없으면 저장소에서 가장 맞는 사진·프레임 ③ 그래도 없을 때만 입력 사진을 편집해 만듭니다.

---

## 7. 전시회 키오스크 (`web`)

관람객이 사진을 찍거나 고르고 원하는 움직임을 고르면, AI가 **사진 읽기 → 요청 이해 → 채점 질문 만들기 → 지시문 쓰기 →
영상 만들기 → 스스로 채점 → 고쳐서 다시 → 완성** 과정을 화면에 실시간으로 보여 줍니다. 단계마다 참고한 논문을 함께 표시하고,
완성되면 "영상 파일로 저장" 버튼으로 내려받을 수 있습니다.

### 7.1 실행

```
python video_studio.py web --mock          :: 연습: Hub 없이 가짜 모델로 화면 흐름만 (영상은 입력 사진을 살짝 움직인 것)
python video_studio.py web                 :: 실제 운영 → 이 PC의 브라우저에서 http://localhost:8000
chrome --kiosk http://localhost:8000       :: 전체 화면 (끝내기 Alt+F4)
```

- **이 PC에서만 열립니다**(localhost). 다른 기기에서는 접속되지 않습니다.
- 예시 사진: `kiosk_samples` 폴더에 jpg/png를 넣으면 "예시 사진으로 해 보기"에 나옵니다 (최대 12장).

### 7.2 운영

- **대기열**: Hub 키의 동시 요청 한도(개발키 1건)에 맞춰 한 번에 1명씩 처리하고, 기다리는 분께 순서와 예상 시간을 보여 줍니다.
- **1명당 시간**: 영상 생성 약 50초 × 최대 2번 시도 + 분석·채점 → 대략 2~4분. 줄이 길면 `--rounds 1`.
- **개인정보**: 관람객 사진·영상은 `output\kiosk`에 저장되고 24시간 뒤 자동으로 지워집니다. 결과 화면의 "내 사진·영상 지우기"로 바로 지울 수도 있습니다. 얼굴 사진은 개인정보이므로 수집·이용 안내와 동의(만 14세 미만은 보호자 동의)를 받으세요. Hub에는 이용량(요청 수·토큰 수)이 집계됩니다.
- **콘솔의 `ConnectionResetError [WinError 10054]`**: 브라우저가 연결을 먼저 끊을 때(영상 탐색·새로고침) Windows가 남기는 정상 로그라 숨겼습니다.

| 옵션 / 설정 | 기본 | 설명 |
|---|---|---|
| `--port` | 8000 | 접속 포트 |
| `--rounds` | 2 | 1명당 최대 시도 횟수 |
| `--threshold` | 0.8 | 이 점수 이상이면 바로 완성 |
| `VS_KIOSK_KEEP_HOURS` | 24 | 관람객 자료 보관 시간 |
| `VS_KIOSK_MAX_QUEUE` | 8 | 최대 대기 인원 |
| `VS_KIOSK_SAMPLES` | kiosk_samples | 예시 사진 폴더 |

---

## 8. 모델 활용 설정

### 8.1 사진·영상 크기

| | 모델 한계 (공개 자료) | 이 프로그램 |
|---|---|---|
| 사진 이해 (Qwen3-VL) | 32×32픽셀당 1토큰, 사진 1장 최소 65,536 ~ 최대 16,777,216픽셀(약 4096×4096) | 비율 그대로 픽셀 수로 맞춤. max 모드 1장 약 420만 픽셀(약 4천 토큰), 프레임 여러 장이면 장당 약 92만 픽셀. `VS_VLM_MAX_PIXELS`·`VS_VLM_FRAME_PIXELS`로 조정 |
| 영상 생성 (Wan 2.2 I2V) | 480P·720P, 결과 비율은 입력 사진을 따름 (가로 832×480·1280×720, 세로 480×832·720×1280) | 사진 비율에 가장 가까운 크기: 가로 832×480·640×480, 세로 480×832·480×640. `OASH_VIDEO_QUALITY=720p`면 720P 우선 |
| 키오스크 업로드 | | 최대 4096px까지 원본 유지 |

- Hub가 어떤 크기(예: 세로)를 받지 않으면 `output\video_sizes.json`에 기억하고, 비율이 가장 가까운 크기에 흐린 여백을 채워 다시 요청합니다.
- 사진과 영상 비율이 1.3배 이상 다르면 가운데만 잘라내지 않고 여백을 채웁니다.
- Hub가 사진이 너무 크다고 거절하면 픽셀을 1/4로 줄여 다시 보내고 기억합니다.

### 8.2 토큰 예산

기본은 **max 모드**입니다. 모델마다 최대 입력 길이를 서버에서 알아내 그 안을 채워 씁니다.

```
python video_studio.py tokens --probe     :: 대화·비전 모델의 최대 입력 길이를 확인해 기록 (처음 한 번)
python video_studio.py tokens             :: 오늘 사용량 / 하루 예산 · 모드 · 모델별 최대 길이
```

| 항목 | eco | max (기본) |
|---|---|---|
| 계획에 넣는 근거 | 약 3,500자, 14개 | 글 모델 최대 길이의 60%까지 관련도순 (최대 200조각, 조각은 자르지 않음) |
| 자료 읽기 | 4쪽씩, 쪽당 3,500자에서 자름 | 쪽을 자르지 않고 예산에 들어가는 만큼 한 번에 |
| 답변 길이 | 2,048토큰 | 모델 최대 길이 − 입력 길이 (상한 16K) |
| 생성 영상 검사 | 1회 판정 | 3회 판정 다수결 |
| 이야기 계획 | 1개 | 후보 3개 → 심판 호출로 최선 선택 |
| 자료 사진·영상 설명 | 한두 문장 | 상세 설명(120~200단어 + 한국어) |
| VLM 사진 크기 | 장당 약 59만 픽셀 | 1장 약 420만 픽셀 / 여러 장이면 장당 약 92만 픽셀 |

- 응답의 usage를 `output\token_usage.json`에 날짜별로 더하고, 하루 예산의 90%를 넘으면 그날은 자동으로 eco로 내립니다.
- 모드는 `.env`의 `VS_TOKEN_MODE=max|balanced|eco`로 바꿉니다.

### 8.3 움직임·카메라 용어집

`knowledge/motion_terms.json` — "줌인, 팬, 틸트, 오빗, 고정 카메라, 바람, 물결, 눈 깜빡임…" 25개. 요청 문장에서 잡히면
영상 모델이 잘 따르는 영어 표현과 **프레임 비교로 확인할 수 있는 검증 질문**을 같이 씁니다. 현장 표현을 추가하려면
같은 형식으로 `knowledge/custom_terms.json`을 만들면 됩니다(같은 term이면 덮어씀). 목록 보기: `python video_studio.py terms`

---

## 9. Hub 없이 테스트

```
python tests\test_offline.py      :: 짧은 모드: 정지 감점 → 반성 → 2라운드 통과, 2샷 이어 붙이기, 후보 비교
python tests\test_long.py         :: 장편: 1분 계획 → 중간에 GPU 오류로 중단 → 미리보기 → 재개 → 내레이션·자막
python tests\test_promo.py        :: 홍보: 브로셔 읽기(가짜 숫자·문장 거르기) → 스토리보드 → 초안 → 본 렌더 → 캐시 재사용
python tests\test_rag.py          :: 지식 저장소: 사진·영상(장면 전환·말소리)·텍스트·웹 → 검색 → 장편 연결
python tests\test_inputs.py       :: 입력 폴더: PDF 여러 개·사진·영상·docx/pptx/xlsx/hwpx·.url, 깨진 답 복구, 한글 경로
python tests\test_kiosk.py        :: 키오스크: 등록 → 실시간 알림 → 완성·저장 → 지우기, 입력 검사, 대기열, Hub 실패, 세로 사진
python tests\test_budget.py       :: 토큰 예산: 최대 길이 탐지·재요청·사용량·다수결·근거 채우기·계획 후보 선택
```

가짜 Hub(`tests/mock_hub.py`)가 첫 프레임으로 실제 mp4를 만들어 돌려줍니다.

---

## 10. 문제 해결

### 10.1 실제 Hub에서 처음 확인할 것

- `wan2.2-i2v`가 세로(480×832)·720P를 받는지. 받지 않으면 콘솔에 `Hub가 480x832 크기를 받지 않아…`가 나오고 자동으로 가로 크기로 바뀝니다.
- `LTX-2`가 첫 프레임 이미지를 받는지(`.env`에 `OASH_VIDEO_MODEL=LTX-2`, `OASH_VIDEO_ACCEPTS_IMAGE=1`). 받으면 샷당 최대 10초.
- 영상 1건 실제 소요 시간 → `.env`의 `VS_EST_JOB_SECONDS`에 넣으면 예상 시간이 정확해집니다.
- `VS_STATIC_THRESHOLD`(기본 0.35): report.html의 '움직임' 값을 보며 정지 판정 기준을 맞추세요.

### 10.2 오류가 날 때

| 증상 | 조치 |
|---|---|
| 404 | 등록되지 않은 모델 → `python video_studio.py models` |
| 429 / 503 | 한도 초과 또는 GPU 준비 중 → 자동으로 기다렸다 재시도 |
| 401 / 403 | 개발키 만료·폐기, 또는 해당 모델 권한 없음 |
| `No module named 'pymupdf'` | 가상환경에서 `pip install -r requirements.txt` |
| 메시지 없이 꺼짐 / `access violation` | 화면에 찍힌 마지막 위치를 보내 주세요 (PDF 부가 기능 충돌은 이미 막아 둠) |
| `JSON을 반환하지 못했습니다` | 자료 읽기는 자동으로 나눠 다시 읽음. 계속되면 `OASH_TEXT_MODEL`을 더 큰 모델로 |
| 한글 경로에서 사진·영상을 못 읽음 | 이미 우회 처리됨. 계속되면 영문 경로로 옮겨 시험 |
| 점수가 이상함 | report.html의 '관찰'·'실패한 질문' 근거 확인, `--frames` 늘리기, `OASH_VISION_MODEL` 변경 |
| 얼굴이 바뀜 | `--rounds`를 늘리면 반성 단계가 '정체성 보호' 경로로 프롬프트를 고침 |
| 영상이 거의 안 움직임 | 반성 단계가 '움직임 강화'로 고침. 계속이면 요청에 강도("크게", "빠르게")를 적어 보세요 |
