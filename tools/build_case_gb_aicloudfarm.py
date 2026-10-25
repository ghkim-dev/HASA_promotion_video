"""경북 초거대 AI 클라우드팜 사업단 홍보 프로젝트(promo_projects/gb_aicloudfarm) 만들기.

브로셔 시안(261019_시안02.pdf)의 숫자·문구를 사람이 옮겨 적은 '추출 결과'를, LLM 추출 때와 똑같이
ingest.verify()에 통과시켜 원문에 그대로 있는 것만 쓰게 한다. 사진 분류도 사람이 확인한 값.
Hub 없이 만들 수 있으므로 바로 `promo plan --rules`, `promo render --draft`로 미리보기가 가능하다.

  python tools/build_case_gb_aicloudfarm.py --pdf 261019_시안02.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.promo import ingest, kb  # noqa: E402

OUT = ROOT / "promo_projects" / "gb_aicloudfarm"

FACTS = [  # label, display, unit, tags, page, quote(원문 그대로)
    ("CPU ZONE 코어", "2,204", "Core", ["infra"], 6, "2,204Core"),
    ("CPU ZONE 메모리", "9,434", "GB", ["infra"], 6, "9,434GB"),
    ("CPU ZONE HDD", "2,376", "TB", ["infra_cpu"], 6, "2,376 TB"),
    ("CPU ZONE 'CPU' 표기 항목", "69", "TB", ["check"], 6, "69TB"),
    ("GPU", "112", "GPUs", ["infra"], 7, "112 56 48 8GPUs"),
    ("H100", "56", "", ["gpu_detail"], 7, "112 56 48 8GPUs"),
    ("H200", "48", "", ["gpu_detail"], 7, "112 56 48 8GPUs"),
    ("B200", "8", "", ["gpu_detail"], 7, "112 56 48 8GPUs"),
    ("GPU ZONE Storage", "2,204", "Core", ["check"], 7, "2,204Core"),
    ("GPU ZONE Network", "100", "G", ["infra_gpu"], 7, "100G"),
    ("참여기업", "230+", "개사", ["community"], 11, "참여기업\n230+개사"),
    ("발굴수요", "120+", "건", ["community"], 11, "발굴수요\n120+건"),
    ("PoC 진행", "45+", "과제", ["community"], 11, "PoC 진행\n45+과제"),
    ("공동 R&D", "20+", "과제", ["community"], 11, "공동 R&D\n20+과제"),
    ("본원 연면적", "6,967", "m²", ["facility"], 3, "6,967m²"),
    ("연구동 연면적", "5,234", "m²", ["facility"], 3, "5,234m²"),
    ("연구지원 장비", "80", "여종", ["facility"], 3, "연구지원 장비 80여종"),
]

MESSAGES = [  # tag, text(원문 그대로), short, page
    ("org", "경북 초거대 AI 클라우드팜 사업단", "", 1),
    ("location", "경북AI클라우드팜센터", "", 3),
    ("tagline", "경북에서 시작하는 AI 미래", "경북에서 시작하는 AI 미래", 1),
    ("tagline", "경북에서 시작하는 AI 혁신의 물결", "", 2),
    ("mission", "인프라를 구축하는 것을 넘어, 경북에서 AI가 실제로 실행되도록 합니다.", "", 4),
    ("mission", "경북 초거대AI클라우드팜센터를 거점으로 AI를 개발하고 서비스하며 전국으로 확산합니다", "", 2),
    ("pain", "고성능 AI 인프라가 부담됩니다", "", 12),
    ("pain", "데이터는 있지만 AI로 연결하기 어렵습니다", "", 12),
    ("pain", "어떤 모델과 환경이 필요한지 모르겠습니다", "", 12),
    ("pain", "개발한 AI를 실제 서비스로 만들기 어렵습니다", "", 12),
    ("infra", "두 개의 ZONE, 하나의 AI 실행 인프라", "", 5),
    ("infra", "CPU ZONE에서 데이터를 운영하고, GPU ZONE에서 AI를 학습·추론합니다.", "", 5),
    ("offering", "고성능 GPU 인프라와 MLOps 기반 개발환경으로 AI 모델 학습부터 추론·서비스까지 하나의 환경에서 수행합니다.", "", 7),
    ("offering", "필요한 서비스부터 시작하고, GBCLOUD.AI 위에서 확장합니다.", "", 8),
    ("offering", "기술기획·공동 R&D 발굴·실증·확산을 하나의 지원체계로 연결합니다.", "", 10),
    ("process", "아이디어가 AI 서비스가 되기까지", "", 9),
    ("process", "AI 도입부터 학습·실증·운영·확산까지, GB HAFA가 전 과정을 함께합니다.", "", 9),
    ("step", "DISCOVER | 진단", "진단", 9),
    ("step", "DESIGN | 설계", "설계", 9),
    ("step", "BUILD | 개발", "개발", 9),
    ("step", "COMPUTE | 학습·추론", "학습·추론", 9),
    ("step", "VALIDATE | 실증", "실증", 9),
    ("step", "SERVE | 운영", "운영", 9),
    ("step", "GROW | 확산", "확산", 9),
    ("community", "기업이 연결되면, 경북의 AI 생태계가 만들어집니다", "", 11),
    ("community", "사람은 Community에서 연결하고, 기술은 GBCLOUD.AI에서 실행합니다.", "", 11),
    ("cta", "AI를 위한 모든 환경을 직접 갖출 필요는 없습니다.", "", 12),
    ("cta", "필요한 환경을 선택하고, GBCLOUD.AI에서 시작하세요.", "", 12),
]

ASSETS = {  # 파일명: (kind, subject) — 사람이 확인한 분류
    "pdf_p02_9.jpg": ("real_building", "a modern six-storey research building with glass facade under a blue sky with white clouds"),
    "pdf_p03_25.jpg": ("real_building", "a research building seen from the street corner under a bright sky (portrait)"),
    "pdf_p04_113.jpg": ("concept_tech", "a futuristic data center corridor with server racks leading to a glowing digital globe"),
    "pdf_p05_220.jpg": ("concept_tech", "a glowing blue network mesh of connected light particles"),
    "pdf_p06_270.jpg": ("concept_tech", "a translucent glowing cloud icon floating above a circuit board"),
    "pdf_p07_317.jpg": ("concept_tech", "a close-up of a computer chip with glowing orange circuit lines"),
    "pdf_p08_338.jpg": ("concept_tech", "a control room with large monitor walls showing data dashboards and an operator"),
    "pdf_p09_374.jpg": ("concept_tech", "streams of colourful data light rushing through a digital tunnel"),
    "pdf_p10_456.jpg": ("concept_city", "a bright futuristic city skyline with glass towers and glowing streets"),
    "pdf_p11_675.jpg": ("concept_city", "a hazy city and mountains under a wide light blue sky"),
    "pdf_p12_776.jpg": ("concept_tech", "a hand holding a glowing cloud hologram on a dark blue background"),
    "pdf_p13_809.jpg": ("graphic_text", "a map with Korean text showing the institute location"),
}

BRAND = {  # 시안 표지·GPU ZONE 페이지에서 고른 색
    "primary": "#2b5be0", "secondary": "#6b3fd4", "accent": "#3cc8f5", "dark": "#0f1d4f", "light": "#eef3ff",
    "text": "#ffffff", "font": "", "font_bold": "", "logo": "", "ai_label": "AI 생성 장면",
}


def build(pdf: Path | None) -> Path:
    adir = OUT / "assets"
    if pdf:
        pages, _ = ingest.read_pdf(pdf, adir)
        (OUT / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    pages = json.loads((OUT / "pages.json").read_text(encoding="utf-8"))
    raw = {"facts": [dict(label=a, display=b, unit=c, tags=d, page=e, quote=f) for a, b, c, d, e, f in FACTS],
           "messages": [dict(tag=a, text=b, short=c, page=d) for a, b, c, d in MESSAGES], "issues": []}
    ver = ingest.verify(raw, pages)
    for f in ver["facts"]:                             # 대표 숫자 우선순위: GPU 112장 → 코어 → 메모리 / 참여기업 → PoC
        f["priority"] = {"GPU": 1, "CPU ZONE 코어": 2, "CPU ZONE 메모리": 3, "참여기업": 1, "PoC 진행": 2}.get(f["label"], 5)
    bad = [x for x in ver["facts"] + ver["messages"] if x["status"] != "ok"]
    for x in bad:
        print("  원문 불일치:", x.get("label") or x.get("text"))
    issues = [{**i, "by": "규칙"} for i in ingest.rule_issues(ver["facts"], pages)]
    issues += [
        {"page": 3, "text": "3쪽은 'H200/B200 GPU 클라우드 인프라 구축'으로만 적혀 있고, 7쪽은 H100 56장을 포함한 112장입니다. 표현을 맞추세요.", "by": "검토"},
        {"page": 11, "text": "커뮤니티 성과 수치가 '2026. 08 기준'입니다. 영상 공개 시점 기준으로 갱신할지 정하세요.", "by": "검토"},
        {"page": 12, "text": "브로셔는 gbcloud.ai만 안내합니다. Open AI Service Hub(open.hasa.re.kr)를 영상에 함께 넣을지 정하세요.", "by": "검토"},
    ]
    assets = []
    for i, (fname, (kind, subject)) in enumerate(ASSETS.items(), 1):
        if not (adir / fname).exists():
            continue
        assets.append({"id": f"a{i:02d}", "file": fname, "source": f"PDF {int(fname[5:7])}쪽", "kind": kind,
                       "subject": subject, "has_text": kind == "graphic_text", "people": 1 if "operator" in subject else 0,
                       "usable": kind not in ("graphic_text", "logo", "blank"), "recipes": kb.recipes_for(kind, subject)})
    brief = {
        "org": {"name": "경북 초거대 AI 클라우드팜 사업단", "tagline": "경북에서 시작하는 AI 미래"},
        "brand": BRAND, "facts": ver["facts"], "messages": ver["messages"],
        "cta": {"url": "hafa.gbcloud.ai", "phone": "053-245-5040", "email": "contact@gbcloud.ai",
                "found": ingest.find_contacts(pages)},
        "made_with": "이 영상은 경북AI클라우드팜 GPU에서 동작하는 AI 모델로 만들었습니다.",
        "assets": assets, "issues": issues,
    }
    (OUT / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"brief.json: 사실 {sum(f['status'] == 'ok' for f in ver['facts'])}/{len(ver['facts'])} · 문구 "
          f"{sum(m['status'] == 'ok' for m in ver['messages'])}/{len(ver['messages'])} 원문 확인 · 사진 {len(assets)}장 · 확인할 점 {len(issues)}개")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None, help="시안 PDF (없으면 저장된 pages.json·assets 사용)")
    a = ap.parse_args()
    build(Path(a.pdf) if a.pdf else None)
