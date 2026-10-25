# 파이프라인 구조와 근거 논문

사진 한 장과 "바람에 머리카락이 날리고 천천히 다가가 줘" 같은 말은 영상 모델(wan2.2-i2v)이 그대로 알아듣기 어렵습니다.
특히 이미지→영상(I2V)에서는 **무엇이 있는지는 사진이 이미 정했고, 사용자의 말은 대부분 '어떻게 움직일지'** 입니다.
그래서 Hub의 LLM(qwen3-next-80b)과 VLM(qwen3-vl-32b)을 영상 모델 앞뒤에 붙여 **사진을 먼저 읽고 → 요구사항을
움직임 단위로 쪼개 이해하고 → 프레임으로 확인할 수 있는 질문을 만들고 → 생성 후 스스로 검사·반성해 다시 만드는**
구조로 짰습니다. 근거 논문은 요청대로 **ICCV · ECCV · CVPR · AAAI · ICML** 게재 논문만 썼습니다.

```
사진 + 요구사항(한국어)
   │
   ▼
[0]   관찰 Observe       VLM  장면 재캡션: 대상(s1,s2…)·움직일 수 있는 부분·환경 요소·정체성 단서   ← RPG (ICML'24)
[0.5] 용어               지식 "줌인/팬/틸트/바람/깜빡임…" → 모델이 따르는 영어 표현 + 프레임 검증 질문
[1]   이해 Understand    LLM  ① 움직임 중심으로 재서술                                          ← TIP-I2V (ICCV'25)
                              ② 원자적 목표(g1,g2…)로 분해, 목표마다 주체 1개·변화 1개           ← X-Planner (AAAI'26)
                              ③ 목표 범주 태깅(동작/방향/카메라/속성 변화/상호작용/환경/공간/개수) ← T2V-CompBench (CVPR'25)
                              ④ 목표마다 물리·상식 규칙                                          ← PhyT2V (CVPR'25)
                              ⑤ 샷 계획: 모델 최대 길이 초과·순차 사건이면 샷으로 분할            ← Vlogger (CVPR'24)
                              ⑥ 보존 목록·가정·확인 질문·위험
[2]   체크리스트         LLM  프레임 비교로 답할 수 있는 yes/no 질문 + 가중치                   ← TIFA (ICCV'23)
                              보존(정체성 2.0)·품질(주체/배경 일관성, 깜빡임, 물리) 질문        ← VBench (CVPR'24), Video-Bench (CVPR'25)
   │
   ├── 샷마다 ── 라운드 루프 (최대 --rounds) ───────────────────────────────────────────────┐
   │ [3] 설계 Design   LLM  영상 모델별 문체로 후보 프롬프트 N개, 목표 누락 검사·재작성          ← Prompt-A-Video, VPO (ICCV'25)
   │                        이전 라운드 메모리(프롬프트·점수·실패·반성) 반영                    ← Idea2Img (ECCV'24), Reflect-DiT (ICCV'25)
   │     생성              wan2.2-i2v: 첫 프레임 = 입력 사진(또는 이전 샷의 마지막 프레임)
   │ [4] 검증 Verify   VLM  4a 프레임(시각 표시) 묘사 → 4b 묘사 재검토 후 질문 답변          ← Video-Bench chain-of-query, TIFA
   │                  코드  움직임·깜빡임·컷 튐·첫 프레임 유사도 (모델 없는 근사 지표)          ← VBench 차원 근사
   │                  VLM  4c 점수가 비슷한 후보끼리 상대 비교                               ← Video-Bench few-shot scoring
   │ [5] 반성 Reflect  LLM  불일치 확인 → 상위 원인(step-back) → 수정 경로 1개 선택 → 다음 지시 ← GenMAC (AAAI'26), PhyT2V
   └── 점수 ≥ --threshold 이면 조기 종료 ──────────────────────────────────────────────────────┘
   ▼
샷 이어 붙이기 → final.mp4 + report.html + run.json
```

## 단계별로 가져온 아이디어

| 단계 | 논문 (학회) | 논문에서 가져온 것 | 이 코드에서 |
|---|---|---|---|
| 0 관찰 | [RPG — Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs](https://proceedings.mlr.press/v235/yang24ai.html) (ICML 2024) | MLLM이 짧은 입력을 풍부한 설명으로 다시 쓰고(recaption), 영역별로 계획한 뒤 생성 | `agents.observe()` — 사진을 대상별(s1…) 속성·위치·**움직일 수 있는 부분**·환경 요소·정체성 단서로 구조화. 이후 모든 단계가 이 id로 주체를 가리킴 |
| 1 이해 ① | [TIP-I2V](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_TIP-I2V_A_Million-Scale_Real_Text_and_Image_Prompt_Dataset_for_ICCV_2025_paper.html) (ICCV 2025) | 실제 사용자 I2V 프롬프트 약 170만 건 분석: I2V 텍스트는 장면을 새로 묘사하기보다 **이미지 속 요소를 어떻게 움직일지** 지시하는 경향. 자주 쓰는 동작은 줌·걷기·눈 깜빡임 등 | 목표를 "사진 속 무엇이 시간에 따라 어떻게 변하는가"로만 적게 함. 정적 내용 재서술 금지. 용어집(`knowledge/motion_terms.json`)도 이 분포(카메라·사람 동작·환경) 위주로 구성 |
| 1 이해 ② | [X-Planner](https://ojs.aaai.org/index.php/AAAI/article/view/38187) (AAAI 2026) | 복잡·간접 지시를 CoT로 단순 하위 지시로 쪼개고, 하위 지시마다 유형·영역을 붙임 | `goals[{id, subject, category, change, when, intensity, rule}]`. 목표 하나 = 주체 하나 + 변화 하나. "평화로운 느낌" 같은 간접 표현은 보이는 움직임으로 번역 |
| 1 이해 ③ | [T2V-CompBench](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.html) (CVPR 2025) | 영상 조합성 7범주: 일관 속성·동적 속성·공간 관계·움직임·동작·상호작용·개수 | 목표마다 범주 태그(`camera` 추가). 범주는 체크리스트 질문 형태와 반성 단계의 원인 분석에 쓰임 |
| 1 이해 ④ / 5 반성 | [PhyT2V](https://openaccess.thecvf.com/content/CVPR2025/html/Xue_PhyT2V_LLM-Guided_Iterative_Self-Refinement_for_Physics-Grounded_Text-to-Video_Generation_CVPR_2025_paper.html) (CVPR 2025) | 라운드마다 ① 객체·물리 규칙 추출 ② 생성 영상 캡션과 프롬프트의 불일치 판정 ③ step-back 추론으로 프롬프트 수정. 이전 점수가 낮으면 다른 추론 경로를 택하게 함. 대부분의 개선은 초반 2라운드에 나타남 | ①은 이해 단계의 `rule`, ②는 검증 4a의 영상 묘사 + 실패 질문, ③은 `reflect()`의 cause(상위 원인) → route. 점수가 안 오르면 지난번과 다른 경로를 고르게 함. 기본 `--rounds 2` |
| 1 이해 ⑤ | [Vlogger](https://openaccess.thecvf.com/content/CVPR2024/html/Zhuang_Vlogger_Make_Your_Dream_A_Vlog_CVPR_2024_paper.html) (CVPR 2024) | LLM을 감독처럼 써서 긴 요청을 샷 단위 대본으로 나누고 샷별로 생성 | 원하는 길이가 샷 최대 길이(wan2.2-i2v 4초)를 넘거나 "먼저 … 그다음 …" 같은 순차 사건이면 샷으로 분할. **다음 샷의 첫 프레임 = 이전 샷 최고 후보의 마지막 프레임**으로 이어 정체성 유지 |
| 2 체크리스트 · 4 검증 | [TIFA](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_TIFA_Accurate_and_Interpretable_Text-to-Image_Faithfulness_Evaluation_with_Question_Answering_ICCV_2023_paper.html) (ICCV 2023) | 요청에서 질문을 만들고 VQA 모델이 답하게 해 충실도를 해석 가능한 점수로 | 목표마다 질문 1개 이상(빠지면 코드가 기본 질문을 붙임), 가중 정답률. 질문마다 **프레임 시점(시작/끝/처음↔끝 비교/전체)** 을 붙여 영상에 맞게 확장 |
| 2 체크리스트 · 4 지표 | [VBench](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) (CVPR 2024) | 영상 품질을 주체 일관성·배경 일관성·깜빡임·움직임 부드러움·움직임 정도(dynamic degree) 등 차원별로 분해 | 품질 질문 항목 + 코드 지표: 광류 기반 **움직임 정도**(정지 영상 판별), 밝기 차 **깜빡임**, 프레임 차 **컷 튐**, 입력 사진과 첫 프레임 **유사도** |
| 4 검증 | [Video-Bench](https://openaccess.thecvf.com/content/CVPR2025/html/Han_Video-Bench_Human-Aligned_Video_Generation_Benchmark_CVPR_2025_paper.html) (CVPR 2025) | chain-of-query: MLLM이 영상을 먼저 글로 묘사 → 질문 생성 → 묘사를 재검토한 뒤 답변 → 채점. few-shot scoring: 같은 프롬프트의 여러 영상을 한 묶음으로 서로를 기준 삼아 채점 | 4a `describe()` → 4b `answer()`(revised_summary로 묘사 재검토 후 답변). 4c `rank()`: 점수 차가 0.05 미만인 후보들의 컨택트 시트를 함께 보여 상대 순위로 결정 |
| 3 설계 | [Prompt-A-Video](https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Prompt-A-Video_Prompt_Your_Video_Diffusion_Model_via_Preference-Aligned_LLM_ICCV_2025_paper.html) (ICCV 2025) | 같은 요청이라도 **영상 모델마다 잘 먹히는 프롬프트가 다르므로**, 모델 선호에 맞춘 프롬프트 재작성기 학습 | 학습 대신 모델별 문체 가이드(`prompts.STYLE_GUIDES`: Wan=핵심 움직임을 앞에 둔 50~120단어 한 문단, LTX=시간순 서술 4~8문장)를 프롬프트에 주입 |
| 3 설계 | [VPO](https://iccv.thecvf.com/virtual/2025/poster/694) (ICCV 2025) | 프롬프트 최적화의 세 원칙: 무해성·**정확성(사용자 의도 왜곡 금지)**·**유용성(빠진 세부 보완)** | 후보마다 `covers`(다룬 목표 id)를 받아 코드가 누락을 검사 → 모든 후보가 누락이면 재작성 요청, 남은 누락은 점수 소폭 감점. "요청에 없는 동작·물체·카메라 추가 금지" 규칙 |
| 3 설계 | [Idea2Img](https://ecva.net/papers/eccv_2024/papers_ECCV/html/5515_ECCV_2024_paper.php) (ECCV 2024) | 프롬프트 생성 → 초안 선택 → 피드백 반영을 반복하며 탐색 이력을 메모리로 유지 | 라운드 메모리 `{prompt, score, failed, metrics, route, reflection}`를 다음 설계에 제공. 후보 N개 중 최고 점수 선택 |
| 3 설계 | [Reflect-DiT](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Reflect-DiT_Inference-Time_Scaling_for_Text-to-Image_Diffusion_Transformers_via_In-Context_Reflection_ICCV_2025_paper.html) (ICCV 2025) | 단순 best-of-N 대신 이전 생성물과 텍스트 피드백을 맥락으로 넣어 다음 생성을 개선 | 생성 모델은 그대로 두고, 반성 텍스트를 프롬프트 설계의 맥락으로 넣는 수준에서 적용 |
| 5 반성 | [GenMAC](https://ojs.aaai.org/index.php/AAAI/article/view/37418) (AAAI 2026) | Design → Generation → Redesign 반복. Redesign을 검증·제안·보정·구조화로 나누고 상황에 맞는 보정 에이전트를 스스로 고르는 self-routing | `reflect()` — mismatches(검증) → cause → **route 8종**(움직임 강화 / 주체·방향 바로잡기 / 단순화 / 카메라 / 물리 / 정체성 보호 / 결함 줄이기 / 사용자 확인) → reflection(다음 지시), JSON으로 구조화 |

## 점수 계산

```
기본   = Σ(맞힌 질문 가중치) / Σ(전체 가중치)                     (TIFA)
점수   = 0.85 × 기본 + 0.15 × (움직임 품질 + 시간 일관성) / 10      (VLM 1~5점, Video-Bench 품질 차원)
감점   정체성 질문(가중치 2.0) 실패 ×0.6 · 움직임을 요구했는데 거의 정지 ×0.7
       컷 튐 ×0.85 · 첫 프레임이 입력과 다름(유사도<0.5) ×0.85 · 프롬프트 목표 누락 1개당 ×0.97
```

## Hub 호출 수 (기본값: 1샷, 2라운드, 후보 1)

관찰 1 + 이해 1 + 체크리스트 1 + 라운드마다 [설계 1 + 영상 N + 비전 2N + (비교 1) + 반성 1] ≈ **채팅·비전 10회, 영상 2건**.
영상 1건이 수 분 걸리므로 실행 시간은 거의 영상 생성이 좌우합니다. 개발키는 동시 1건이라 후보를 순서대로 생성하고,
`OASH_MAX_CONCURRENCY`를 올리면 병렬로 보냅니다.

## 논문과 다르게 한 부분 (Hub 환경에 맞춘 단순화)

- **학습하지 않습니다.** Prompt-A-Video·VPO는 프롬프트 재작성 LLM을 SFT·DPO로 학습하는 방법이지만, 여기서는 그 원칙(모델별 선호, 정확성·유용성)을 시스템 프롬프트와 코드 검사로만 반영했습니다.
- **영상은 프레임으로 봅니다.** qwen3-vl-32b에 시각을 새긴 프레임 5장(+입력 사진)을 보내 판정합니다. 프레임 사이에서 일어나는 짧은 동작(눈 깜빡임 등)은 놓칠 수 있어, 그런 목표는 `--frames 8`처럼 늘리는 편이 낫습니다.
- **VBench 지표는 근사치입니다.** VBench는 RAFT·DINO·CLIP 등 별도 모델로 측정하지만, 여기서는 opencv 광류·밝기 차·상관계수로 '정지·깜빡임·컷 튐·첫 프레임 이탈'만 거릅니다. 절대 점수가 아니라 거름망입니다. 정지 기준(`VS_STATIC_THRESHOLD`)은 실제 결과를 보고 조정하세요.
- **PhyT2V의 물리 점수 모델(VideoCon-Physics)·캡셔너(Tarsier)는 없어서** VLM 묘사와 체크리스트 점수로 대신합니다.
- **GenMAC의 레이아웃(바운딩 박스) 제어, X-Planner의 마스크 생성은 뺐습니다.** Hub 영상 API가 텍스트·첫 프레임만 받기 때문입니다.
- **확인 질문(`--ask`)은 논문 근거가 아닌 실무용 추가 기능**입니다. 이해 단계가 "틀리면 결과를 망칠 모호함"만 질문으로 내고, 답하지 않으면 기본값으로 진행합니다.

---

# 장편 모드 (10~20분) — `video_studio.py long`

wan2.2-i2v는 한 번에 최대 4초입니다. 20분이면 **샷 약 300개**를 만들어 이어야 하고, 이때 문제는 세 가지입니다.
① LLM에게 300개 샷을 한 번에 쓰게 하면 빠뜨리거나 반복한다 ② 마지막 프레임을 계속 이어 붙이면 색·얼굴이 조금씩
틀어져 누적된다 ③ 몇 시간~하루가 걸리므로 중간에 끊겨도 이어서 할 수 있어야 한다. 이를 아래처럼 풀었습니다.

```
사진 + 이야기 요청 + 길이(분)
   │
[0]  관찰                VLM  (짧은 모드와 같음)
[L1] 이야기 설정·막      LLM  bible(인물의 고정 외형·세계·스타일·카메라 언어) + 막(약 3분 단위)   ← Vlogger, VideoStudio
[L2] 씬 (막마다 1회)     LLM  장소·시간 단위 씬, 시작 방식(이어짐/새 키프레임), 사건 목록, 내레이션 ← VideoStudio, Long Context Tuning, Mind the Time
[L3] 샷 (씬마다 1회)     LLM  정확히 n개(= 씬 길이 ÷ 4초), 샷 하나 = 사건 하나, end_state, 검사 질문 ← Mind the Time, StreamingT2V
   │                         → plan.json (사람이 고쳐도 됨) · 예상 시간
   ├── 씬 줄기(이어진 씬끼리)마다 ───────────────────────────────────────────────────────────────┐
   │ 앵커   첫 씬 = 입력 사진 / 새 장소 씬 = 원본 사진을 Qwen-Image-Edit로 편집 → VLM 정체성 확인 ← VideoStudio(참조 이미지)
   │ 샷     생성 → 프레임 4장 + 원본 사진으로 검사(한 번의 VLM 호출) → 점수<0.6이면 원인·경로·새 프롬프트로 재시도
   │        다음 샷 첫 프레임 = 이 샷 마지막 프레임 (단기 기억)                           ← StreamingT2V CAM의 역할
   │        색 거리가 커지면 앵커 쪽으로 색 보정, 정체성 2샷 연속 실패면 원본으로 키프레임 재고정 ← StreamingT2V APM의 역할
   │        샷마다 progress.json 저장 → 끊기면 resume으로 남은 샷부터
   │ 씬 조립 정규화(해상도·fps 통일, 이음매 중복 프레임 제거) → 이어 붙임 → 장소 바뀌는 곳만 페이드 → 내레이션(TTS) 또는 무음 트랙
   └──────────────────────────────────────────────────────────────────────────────────────────┘
   ▼
final.mp4 + final.srt(내레이션 자막) + chapters.txt + report.html(진행률·남은 시간·씬별 점수, 60초마다 새로고침)
```

| 문제 | 논문 (학회) | 논문에서 가져온 것 | 이 코드에서 |
|---|---|---|---|
| 300샷을 어떻게 계획하나 | [Vlogger](https://openaccess.thecvf.com/content/CVPR2024/html/Zhuang_Vlogger_Make_Your_Dream_A_Vlog_CVPR_2024_paper.html) (CVPR 2024) | LLM 감독이 긴 영상을 대본 → 샷으로 위에서 아래로 나눔 | 막 → 씬 → 샷 3단 계획(막 1회 + 막마다 1회 + 씬마다 1회 호출). 20분 기준 LLM 약 30회. 길이는 코드가 비율 재분배로 정확히 맞춤 |
| 여러 씬에서 같은 인물 유지 | [VideoStudio](https://eccv2024.ecva.net/virtual/2024/poster/728) (ECCV 2024) | LLM이 다장면 대본과 공통 등장 대상의 상세 묘사를 만들고, 대상별 참조 이미지를 조건으로 써서 장면 간 일관성 유지 | 'bible'에 인물 외형을 한 번 정의해 모든 샷 프롬프트에 재사용. 새 장소 씬의 첫 프레임은 **원본 사진을 참조로 편집**해 만들고, VLM이 원본과 같은 인물인지 확인 |
| 일관성의 단위 | [Long Context Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Long_Context_Tuning_for_Video_Generation_ICCV_2025_paper.html) (ICCV 2025) | 한 샷이 아니라 **씬 안의 모든 샷**에 걸쳐 시각·동작 일관성을 맞춰야 함 (모델의 주의 범위를 씬 전체로 확장) | 모델은 고칠 수 없으므로 구조로 대응: 씬 안에서는 샷을 끊김 없이 체인, 씬이 바뀔 때만 새 키프레임. 샷 계획도 씬 단위로 한 번에 써서 샷끼리 이어지게 함 |
| 한 샷에 여러 사건 | [Mind the Time](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Mind_the_Time_Temporally-Controlled_Multi-Event_Video_Generation_CVPR_2025_paper.html) (CVPR 2025) | 여러 사건을 한 문단으로 주면 일부를 빠뜨리거나 순서를 틀림 → 사건마다 시간 구간을 따로 지정 | 씬의 사건을 목록으로 받고, **샷 하나에 사건 하나**를 순서대로 배정. 사건이 샷보다 적으면 새 동작 대신 느린 연속 동작·카메라·환경 움직임으로 채움 |
| 이어 붙일수록 틀어짐 | [StreamingT2V](https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html) (CVPR 2025) | 단기 기억(직전 구간으로 이음매를 매끄럽게) + 장기 기억(첫 구간의 특징을 계속 참조해 초기 장면을 잊지 않게). I2V 이어 붙이기의 정체(stagnation) 문제 | 단기: 직전 샷 마지막 프레임 + end_state 문장. 장기: 씬 앵커 기준 LAB 색 보정(ΔE>6), 모든 샷 검사에 원본 사진을 기준 이미지로 제시, 정체성 2연속 실패 시 원본에서 재고정. 정체는 움직임 지표로 감지해 재시도 |

## 시간과 호출 수 (20분, wan2.2-i2v, 개발키 동시 1건)

- 샷 약 300개 × (재시도 포함 약 1.1~1.3건) = 영상 작업 약 330~390건. **실제 Hub 실측 1건 평균 45~50초 → 약 4~5시간**입니다.
- 비전 호출 ≈ 영상 작업 수, 채팅 ≈ 30 + 재시도 수, 이미지 편집 ≈ 새 장소 씬 수(×1~2).
- 시간을 줄이는 방법: 키 동시 한도를 올리면(`OASH_MAX_CONCURRENCY`) **장소가 바뀌는 씬 줄기끼리 병렬**로 생성 /
  LTX-2가 첫 프레임을 받으면 샷당 10초라 샷 수가 약 40%로 줄어듦 / `--retries 0`.
- 실행하면 실제 작업 시간으로 남은 시간을 다시 계산해 터미널과 report.html에 표시합니다.

## 장편 모드에서 논문과 다르게 한 부분

- StreamingT2V·Long Context Tuning은 **모델 내부(어텐션·조건 모듈)를 바꾸는 방법**입니다. Hub 영상 API는 첫 프레임과 텍스트만 받으므로, 같은 목적(단기·장기 기억, 씬 단위 일관성)을 첫 프레임 선택·색 보정·참조 사진 검사·재고정이라는 **파이프라인 수준**으로 옮겼습니다.
- VideoStudio의 참조 이미지는 생성 조건으로 들어가지만, 여기서는 키프레임을 만들 때(이미지 편집의 참조)와 검사할 때만 씁니다.
- 내레이션 음성은 Hub의 TTS 모델(melotts-ko)을 OpenAI 호환 `/audio/speech`로 호출해 봅니다. **이 경로는 아직 확인되지 않았고**, 안 되면 자동으로 자막(final.srt)만 만듭니다.

---

# 홍보 모드 — `video_studio.py promo`

일반 모드는 "무엇이든 만드는" 주방이고, 홍보 모드는 같은 주방에 **홍보영상 전문점의 레시피·메뉴·맛 기준**을 붙인 것입니다.
학회 기법(계획·검사·반성)은 그대로 쓰고, 그 안에 담는 내용을 홍보 분야 것으로 바꿨습니다.

| 전문점 요소 | 일반 모드 | 홍보 모드 |
|---|---|---|
| 손님·숙제 | 아무 요청 | 기관·사업 홍보, 형식 3종(숏폼 24초 세로 / 메인 70초 / 설명형 약 3분) |
| 레시피 (지식) | 일반 카메라 용어 25개 | `knowledge/promo/templates.json` 홍보 문법(후킹→문제→우리는→증거→지원 과정→성과→행동 유도), `motion_recipes.json` 사진 종류별 움직임(건물=드론 접근, 서버=복도 전진, 관제실=화면 갱신 …)과 검사 질문 |
| 재료 | 사진 한 장 | 브로셔 PDF → 숫자(facts)·문구(messages)·사진·연락처·브랜드 색 |
| 사실 지키기 | — | 숫자·문장은 **원문에 그대로 있는 것만** 사용(코드가 대조). LLM은 id로 고르기만 하고 새 문장·숫자를 쓸 수 없음. 연결어에 숫자가 들어가면 지움 |
| 글자 | 모델이 그릴 수도 있음 | 숫자 카운트업·문구·단계·엔딩카드·QR·자막은 코드(Pillow)가 그림. 생성 장면에 글자가 생기면 검사에서 감점(가중치 2) |
| 맛 기준 (검사) | 요청한 움직임이 나왔나 | 건물 모양·창 배치 유지, 새 글자·간판 없음, 사람 수·얼굴 유지, 움직임 존재 |
| 시안 점검 | — | 단위가 이름과 안 맞는 숫자, 같은 숫자의 중복 사용, '로고예정' 같은 자리표시를 자동으로 찾음 |
| 반복 비용 | 매번 생성 | AI 클립을 사진·레시피·길이 기준으로 캐시 → 문구·숫자 수정은 GPU 없이 재렌더. `--draft`는 GPU 없이 전체 미리보기 |

| 단계 | 근거 (학회) | 홍보 모드에서 |
|---|---|---|
| 브로셔 읽기 | RPG (ICML'24) 재캡션 | 사진마다 종류·내용을 구조화해 레시피와 연결 |
| 사실·문구 고르기 | VPO (ICCV'25) 정확성 원칙 | '의도 왜곡 금지'를 한 단계 더: 원문 대조로 숫자·문장을 잠그고 id로만 선택 |
| 장면 순서 | Vlogger (CVPR'24), Mind the Time (CVPR'25) | 형식 템플릿(비트) 안에서 비트 하나 = 메시지 하나 |
| 움직임 지시 | Prompt-A-Video (ICCV'25) 모델별 선호, TIP-I2V (ICCV'25) 움직임 중심 | 사진 종류별 레시피 프롬프트 — 무엇이 움직이고 무엇은 그대로인지 명시 |
| 생성 후 검사 | TIFA (ICCV'23), Video-Bench (CVPR'25), VBench (CVPR'24) | 레시피의 검사 질문 + 움직임·첫 프레임 지표, 미달이면 '첫 프레임 그대로, 카메라와 빛만' 지시로 재생성 |
| 긴 비트 | StreamingT2V (CVPR'25) | 4초보다 긴 비트는 마지막 프레임에서 이어 생성하고 최대 1.5배까지 늘려 샷 수를 줄임 |

---

# 멀티모달 지식 저장소 — `video_studio.py rag` + `long --kb`

wan2.2-i2v가 받는 것은 **첫 프레임 1장 + 글 프롬프트**뿐입니다. 그래서 임베딩의 역할은 영상 모델에 직접 넣는 것이 아니라,
수많은 자료 중에서 **장면마다 가장 맞는 첫 프레임·사실·움직임을 골라내는 것**입니다.

```
사진 · 동영상 · PDF · docx/txt/md · 웹사이트
  │  sources.py   종류별로 조각내기 (원본 위치 기억)
  │               동영상: HSV 히스토그램으로 장면 전환 → 구간마다 시작 프레임 + VLM 움직임·카메라 설명
  │                       + ffmpeg로 소리 추출 → Whisper 받아쓰기(30초 단위)
  ▼
rag_stores/<이름>  store.py   글 벡터(qwen3-embedding-4b 등) + 이미지 벡터(지원 시) + 원본 프레임
  │
  ▼  검색: 글·이미지 순위 합치기(RRF) → 리랭커 → 중복 제거
장편 모드 (long --kb)
  L1 막 구성   ← 요청으로 찾은 근거 14개 (사실·장소·이름을 지어내지 않도록)
  L2 씬 구성   ← 막마다 근거 6개 + 첫 프레임 후보 사진·구간 10개 → 씬이 visual_ref로 고름 (목록 밖 id는 버림)
  L3 샷 구성   ← 씬의 사건으로 찾은 '참고 움직임' 3개 (사용자 영상의 카메라 언어 재사용)
  생성         새 장면 첫 프레임: visual_ref → (없으면) 저장소 검색 1위 프레임 → (그래도 없으면) 입력 사진 편집
```

| 단계 | 근거 (학회) | 적용 |
|---|---|---|
| 참고 움직임 검색 | DreamRunner (AAAI 2026) — 장면마다 관련 영상을 검색해 그 움직임을 생성에 반영 | 논문은 검색한 영상으로 움직임을 학습(적응)하지만, 여기서는 학습 없이 검색한 구간의 움직임·카메라 설명을 샷 프롬프트에 넣음 |
| 장면 단위 참조 | VideoStudio (ECCV 2024) — 공통 대상의 참조 이미지로 장면 간 일관성 | 자료 속 실제 장소·사물 프레임을 씬의 첫 프레임(참조)으로. 계획이 지정하지 않아도 검색으로 채움 |
| 근거 기반 계획 | Vlogger (CVPR 2024) 계층 계획 + VPO (ICCV 2025) 정확성 | 계획 단계마다 출처 id가 붙은 근거를 주고, 고를 수 있는 시각 자료를 목록으로 제한 |

## 실제 Hub 실행에서 배운 것 (2026-10-25, 3분 장편 2회)

| 관찰 | 고친 것 |
|---|---|
| 영상 1건 평균 45~50초 (예상 3분보다 빠름) | 예상 시간 기본값 50초 |
| 재시도의 대부분이 '결함 없음?' 실패 (71건 중 57건) — VLM이 사소한 흠까지 결함으로 적음 | 검사에서 '심각한 결함'과 '사소한 흠'을 나눠 받고, 심각한 것만 감점 |
| "머리카락 한 가닥만", "정확히 한 번" 같은 너무 미세한 검사 질문 | 4프레임으로 보이는 주요 사건만 묻도록 규칙 추가 |
| 사진에 없는 인물(남자친구)이 나오는 씬에서 정체성 실패 8건 | 인물 씬 처리는 이 버전에서 제외(장소·사물·자료 중심). 얼굴 정체성 검사와 재고정을 없애고, 샷 프롬프트는 얼굴 클로즈업을 피하도록 함 |

## 실제 Hub 점검 결과 반영 (tools/check_multimodal.py, 2026-10-25)

| 항목 | 결과 | 설계 |
|---|---|---|
| 텍스트 임베딩 | bge-m3(1024)·qwen3-embedding-4b(2560)·nemotron-embed-8b(4096) 모두 동작. 한국어 순위는 qwen3·nemotron이 정확 | 기본 qwen3-embedding-4b |
| 이미지 임베딩 | 3개 모델 모두 이미지 입력 거부 | 사진·프레임은 VLM 설명문(+OCR)을 텍스트 임베딩 |
| 리랭커 | `/v1/rerank` 동작 (bge-reranker-v2-m3-ko) | 검색 후보 30개를 다시 정렬 |
| 영상 이해 | qwen3-vl-32b가 mp4를 직접 받음 (3.2초, 카메라 이동을 정확히 설명). qwen2.5-vl-72b도 가능하나 14.9초 | 저장소의 영상 구간 설명과 생성 영상 검사를 **영상 직접 입력**으로 (실패하면 자동으로 프레임 방식) |
| 음성 | cosyvoice3 `/audio/speech` → whisper `/audio/transcriptions` 왕복 성공 | TTS는 cosyvoice3 우선, 받아쓰기는 구간 시각 형식이 안 되면 기본 형식 |
| OCR | schift-ocr-1-beta 정확, paddleocr-vl은 엉뚱한 글자 | 글자 있는 사진·도표는 schift-ocr로 글자를 뽑아 조각 텍스트에 포함 |

---

# 토큰 예산 — 모델 최대 길이를 채워 쓰기 (`pipeline/budget.py`)

토큰을 많이 쓰는 것 자체가 정확도를 올리지는 않습니다. 관련 없는 내용으로 채우면 중요한 정보가 묻힙니다.
그래서 **정확도에 효과가 확인된 방식에만** 토큰을 씁니다.

| 쓰는 곳 | 방식 | 왜 도움이 되나 |
|---|---|---|
| 근거 채우기 | 검색 → 리랭크 순으로 모델 최대 길이의 60%까지 (조각을 자르지 않음) | 계획이 지어내지 않고 실제 자료의 사실·장소를 쓰게 함. 관련도순이라 앞쪽이 가장 중요한 근거 |
| 판정 다수결 (Self-Consistency) | 생성 영상 검사를 3번 받아 질문별 과반, 심각한 결함도 과반일 때만 | 실제 실행에서 VLM의 사소한 흠 지적이 재시도의 대부분이었음 → 한 번의 잡음 판정을 걸러냄 |
| 계획 후보 + 심판 (Best-of-N, LLM-as-judge) | 개요를 3개 만들고 별도 호출이 요청·근거 충실도로 고름 | 한 번에 나온 계획보다 나은 것을 고를 기회 |
| 답변 길이 | 최대 길이 − 입력, 상한 16K | 긴 JSON 계획이 중간에 잘려 다시 요청하는 일을 막음 |
| 상세 설명 | 자료 사진·영상 구간마다 120~200단어 | 검색할 때 걸릴 단서가 늘어남 (이미지 임베딩이 없으므로 설명문이 곧 검색 대상) |

최대 길이 탐지: vLLM 계열 서버는 너무 큰 max_tokens 요청에 `maximum context length is N`을 돌려주므로, 모델마다 한 번
그렇게 물어 N을 기억합니다. 실제 요청이 길이 초과로 거절돼도 같은 방식으로 한도를 배우고 줄여서 다시 보냅니다.
