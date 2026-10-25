"""입력 폴더 오프라인 테스트:  python tests/test_inputs.py
PDF 2개·사진·동영상·docx·pptx·xlsx·hwpx·txt·.url·구형 .hwp가 섞인 폴더 하나를
  ① 종류별로 찾고  ② 문서 글을 읽고  ③ 홍보 모드(promo ingest --input)로 brief.json을 만들고
  ④ 장편 모드(long --input)에서 지식 저장소 + 첫 프레임 자동 선택까지 가짜 Hub로 확인."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VS_STATE_DIR", tempfile.mkdtemp())
os.environ["VS_RAG_DIR"] = tempfile.mkdtemp()           # 저장소도 임시 폴더에
sys.path.insert(0, str(ROOT / "tests"))
os.environ.update(OASH_API_KEY="test", VS_NET_RETRY_SLEEP="0")
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL", "OASH_EMBED_MODEL"):
    os.environ.pop(k, None)

import mock_hub                                       # noqa: E402
from test_offline import sample_image                  # noqa: E402
from test_promo import make_pdf                        # noqa: E402
from test_rag import dark_image, make_video            # noqa: E402
from pipeline import inputs                            # noqa: E402
from pipeline.promo import ingest                      # noqa: E402

PORT = 8770
os.environ["OASH_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"


def zip_doc(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
    return path


def make_folder(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    make_pdf(d / "리플렛.pdf")
    sub = d / "추가자료"
    sub.mkdir()
    make_pdf(sub / "협력안.pdf")
    (d / "b.png").unlink(missing_ok=True)                # make_pdf가 남긴 임시 사진
    (sub / "b.png").unlink(missing_ok=True)
    sample_image(d / "건물사진.jpg")
    tmp = Path(tempfile.mkdtemp())
    make_video(d / "소개영상.mp4", sample_image(tmp / "a.jpg"), dark_image(tmp / "b.jpg"))
    zip_doc(d / "사업소개.docx", {"word/document.xml": "<w:document><w:body><w:p><w:r><w:t>도내 기업 120곳 AI 실증 지원</w:t></w:r></w:p>"
                                                       "<w:p><w:t>GPU 64GPUs 무상 제공</w:t></w:p></w:body></w:document>"})
    zip_doc(d / "발표.pptx", {"ppt/slides/slide1.xml": "<p:sld><a:p><a:t>첫 슬라이드 제목</a:t></a:p></p:sld>",
                             "ppt/slides/slide2.xml": "<p:sld><a:p><a:t>두번째 슬라이드 GPU 클러스터</a:t></a:p></p:sld>"})
    zip_doc(d / "현황.xlsx", {"xl/sharedStrings.xml": "<sst><si><t>항목</t></si><si><t>GPU 수</t></si></sst>",
                             "xl/worksheets/sheet1.xml": "<worksheet><sheetData><row r='1'><c r='A1' t=\"s\"><v>1</v></c>"
                                                         "<c r='B1'><v>112</v></c></row></sheetData></worksheet>"})
    zip_doc(d / "공고문.hwpx", {"Contents/section0.xml": "<hs:sec><hp:p><hp:t>모집 공고 내용</hp:t></hp:p></hs:sec>"})
    (d / "메모.txt").write_text("경북 클라우드팜은 경산에 있습니다.\n누구나 신청할 수 있습니다.", encoding="utf-8")
    (d / "홈페이지.url").write_text(f"[InternetShortcut]\nURL=http://127.0.0.1:{PORT}/page.html\n", encoding="utf-8")
    (d / "옛문서.hwp").write_bytes(b"\xd0\xcf\x11\xe0 old hwp")
    (d / "~$임시.docx").write_bytes(b"lock")                # 열려 있는 오피스 문서의 잠금 파일 → 무시
    return d


class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mock_hub.serve(PORT)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.folder = make_folder(cls.tmp / "gbcloud_pdf")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_scan_and_docs(self):
        f = inputs.scan([self.folder])
        self.assertEqual(len(f["pdf"]), 2)                # 하위 폴더까지
        self.assertEqual(len(f["image"]), 1)
        self.assertEqual(len(f["video"]), 1)
        self.assertEqual(sorted(p.suffix for p in f["doc"]), [".docx", ".hwpx", ".pptx", ".txt", ".xlsx"])
        self.assertEqual(f["url"], [f"http://127.0.0.1:{PORT}/page.html"])
        self.assertEqual([p.suffix for p in f["skipped"]], [".hwp"])
        self.assertIn(".hwpx나 PDF", inputs.describe(f))
        self.assertIn("120곳", inputs.doc_pages(self.folder / "사업소개.docx")[0][1])
        sl = inputs.doc_pages(self.folder / "발표.pptx")
        self.assertEqual([x[0] for x in sl], ["슬라이드 1", "슬라이드 2"])
        self.assertIn("GPU 수 | 112", inputs.doc_pages(self.folder / "현황.xlsx")[0][1])
        self.assertIn("모집 공고", inputs.doc_pages(self.folder / "공고문.hwpx")[0][1])
        u = self.tmp / "urls.txt"
        u.write_text("https://a.example.kr\n\nhttps://b.example.kr/x\n", encoding="utf-8")
        self.assertTrue(inputs.is_url_list(u))
        self.assertFalse(inputs.is_url_list(self.folder / "메모.txt"))

    def test_promo_ingest_folder(self):
        proj = ingest.ingest(None, None, self.tmp / "proj", inputs=[str(self.folder)])
        brief = json.loads((proj / "brief.json").read_text(encoding="utf-8"))
        labels = json.loads((proj / "pages_index.json").read_text(encoding="utf-8"))
        joined = " ".join(labels)
        for want in ("리플렛.pdf 1쪽", "협력안.pdf 2쪽", "사업소개.docx", "발표.pptx 슬라이드 2", "현황.xlsx",
                     "공고문.hwpx 구역 1", "메모.txt", "소개영상.mp4 말소리", "웹 http://"):
            self.assertIn(want, joined)
        self.assertTrue(all(f.get("where") for f in brief["facts"]))          # 어느 파일 몇 쪽인지
        self.assertTrue(brief["facts"][0]["where"].startswith("리플렛.pdf"))
        src = [a["source"] for a in brief["assets"]]
        self.assertTrue(src[0].startswith("사진 건물사진"))                      # 실사진·영상 프레임이 앞
        self.assertTrue(any(s.startswith("소개영상.mp4") for s in src))
        self.assertTrue(any(s.startswith("리플렛.pdf") for s in src))
        self.assertTrue(any(s.startswith("협력안.pdf") for s in src))
        self.assertTrue(any(s.startswith("웹 ") for s in src))
        files = [a["file"] for a in brief["assets"]]
        self.assertEqual(len(files), len(set(files)))                          # PDF 두 개의 사진 이름이 겹치지 않음
        self.assertTrue(all((proj / "assets" / x).exists() for x in files))
        self.assertIn("video", brief["inputs"])

    def test_long_input_kb(self):
        import video_studio
        a = SimpleNamespace(input=[str(self.folder)], kb=None, image=None, request="데이터센터 서버실")
        video_studio._kb_from_input(a)
        self.assertEqual(a.kb, "gbcloud_pdf")                                   # 저장소 이름 = 폴더 이름
        self.assertTrue(Path(a.image).exists())                                 # 자료에서 첫 프레임 자동 선택
        from pipeline.rag import Store
        st = Store("gbcloud_pdf")
        mods = {c["modality"] for c in st.chunks}
        self.assertTrue({"text", "image", "video_segment"} <= mods)
        locs = " ".join(c["loc"] for c in st.chunks)
        self.assertIn("슬라이드 2", locs)
        n = len(st.chunks)
        b = SimpleNamespace(input=[str(self.folder)], kb=None, image="x.jpg", request="r")
        video_studio._kb_from_input(b)                                          # 다시 넣으면 건너뜀
        self.assertEqual(len(Store("gbcloud_pdf").chunks), n)
        self.assertEqual(b.image, "x.jpg")

    def test_facts_cut_or_broken(self):
        """답이 최대 길이에서 잘리거나 형식이 깨져도: 완성된 항목은 건지고, 반씩 나눠 다시 읽어 결국 모두 얻는다."""
        cut = ('{"facts": [{"label": "GPU", "display": "112", "unit": "장", "tags": ["infra"], "page": 1, "quote": "H100 112장"},'
               ' {"label": "CPU", "display": "2,2')
        self.assertEqual(len(ingest.salvage(cut)["facts"]), 1)                 # 잘린 두 번째 항목은 버림
        self.assertIsNone(ingest.salvage("죄송합니다. 표로 정리하면..."))
        from pipeline.utils import extract_json                                 # PDF의 제어 문자(\x07)가 따옴표 안에 있어도
        self.assertEqual(extract_json('{"facts": [{"quote": "설비 | \x07장비 80여종"}]}')["facts"][0]["quote"][-7:], "장비 80여종")
        self.assertEqual(ingest.clean("설비 | \x07교육관\n줄"), "설비 | 교육관\n줄")

        class FakeHub:
            calls = 0
            last_finish = ""

            def models(self):
                return SimpleNamespace(text="fake")

            def chat(self, messages, temperature=0.1):
                FakeHub.calls += 1
                part = messages[1]["content"]
                pages = [int(x) for x in __import__("re").findall(r"=== page (\d+)", part)]
                if len(pages) > 2:                                              # 길면 잘린 답
                    self.last_finish = "length"
                    return cut
                self.last_finish = "stop"
                if pages == [4]:
                    return "형식 없는 답"
                return json.dumps({"facts": [{"label": f"항목{p}", "display": str(100 + p), "unit": "개", "page": p,
                                              "quote": f"항목 {100 + p}개"} for p in pages], "messages": [], "issues": []})

        from pipeline import budget
        old = budget.BUDGET.fill_tokens
        budget.BUDGET.fill_tokens = lambda hub, model, reserved=4000: 10**6
        try:
            dbg = self.tmp / "dbg"
            r = ingest.extract(FakeHub(), [f"항목 {100 + i}개" for i in range(1, 7)], debug_dir=dbg)
        finally:
            budget.BUDGET.fill_tokens = old
        got = {f["display"] for f in r["facts"]}
        self.assertTrue({"101", "102", "103", "105", "106"} <= got)            # 4쪽만 끝내 실패
        self.assertNotIn("104", got)
        self.assertIn("112", got)                                               # 잘린 답에서 건진 항목
        self.assertTrue(list(dbg.glob("facts_raw_p4-4.txt")))                   # 원인 확인용 원문 답

    def test_korean_project_draft(self):
        """한글 프로젝트 이름(클라우드팜)·한글 사진 이름으로 스토리보드 → 미리보기 (Windows OpenCV 한글 경로 우회 강제)."""
        from pipeline import videoio
        from pipeline.promo import planner, render
        os.environ["VS_FORCE_ASCII_IO"] = "1"
        try:
            proj = ingest.ingest(None, None, self.tmp / "promo_projects" / "클라우드팜", inputs=[str(self.folder)])
            planner.plan(proj, "short")
            out = render.render(proj, "short", draft=True)
        finally:
            os.environ.pop("VS_FORCE_ASCII_IO", None)
        self.assertTrue(Path(out).exists())
        self.assertGreater(videoio.media_seconds(out), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
