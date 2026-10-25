"""멀티모달 지식 저장소 오프라인 테스트:  python tests/test_rag.py
사진 2장 · 장면 전환+소리가 있는 영상 · 텍스트 파일 · 웹페이지 → 조각·임베딩 → 검색 → 장편 모드 연결."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tempfile as _tf  # noqa: E402
os.environ.setdefault("VS_STATE_DIR", _tf.mkdtemp())   # 가짜 Hub의 길이·사용량이 실제 output/에 섞이지 않게
sys.path.insert(0, str(ROOT / "tests"))
os.environ.update(OASH_API_KEY="test", VS_NET_RETRY_SLEEP="0")
for k in ("OASH_TEXT_MODEL", "OASH_VISION_MODEL", "OASH_VIDEO_MODEL", "OASH_EMBED_MODEL"):
    os.environ.pop(k, None)

import mock_hub                                    # noqa: E402
from test_offline import sample_image               # noqa: E402
from pipeline import longrun, story                 # noqa: E402
from pipeline.rag import Store, sources             # noqa: E402

PORT = 8769
os.environ["OASH_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"


def dark_image(path: Path) -> Path:
    import numpy as np
    from PIL import Image
    arr = np.full((480, 800, 3), 18, np.uint8)
    for x in range(40, 800, 90):
        arr[60:440, x:x + 50] = (30, 40, 60)
        arr[80:420:20, x + 10:x + 40] = (40, 220, 240)        # 서버 LED
    Image.fromarray(arr).save(path, quality=92)
    return path


def make_video(path: Path, bright: Path, dark: Path) -> Path:
    """앞 3초 어두운 서버실(전진) → 컷 → 뒤 3초 밝은 건물. 사인파 소리 포함."""
    import cv2
    tmp = path.with_suffix(".noaudio.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 12, (640, 360))
    for src, n in ((dark, 36), (bright, 36)):
        img = cv2.resize(cv2.imread(str(src)), (640, 360))
        for i in range(n):
            M = cv2.getRotationMatrix2D((320, 180), 0, 1 + 0.004 * i)
            vw.write(cv2.warpAffine(img, M, (640, 360)))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    tmp.unlink()
    return path


class Rag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = mock_hub.serve(PORT)
        cls.tmp = Path(tempfile.mkdtemp())
        mat = cls.tmp / "materials"
        mat.mkdir()
        cls.bright = sample_image(mat / "building.jpg")
        cls.dark = dark_image(mat / "server_room.jpg")
        make_video(mat / "tour.mp4", cls.bright, cls.dark)
        (mat / "facts.txt").write_text("경북 클라우드팜 센터는 GPU 112장을 운영합니다.\n\n참여기업은 230곳이며 PoC 45건을 진행했습니다.\n\n"
                                       "주차장은 본관 뒤에 있습니다.", encoding="utf-8")
        cls.mat = mat
        cls.store_dir = cls.tmp / "store"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_1_split_text(self):
        parts = sources.split_text("가" * 1500, size=600, overlap=80)
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(len(p) <= 600 for p in parts))

    def test_2_add_and_search(self):
        st = Store(self.store_dir)
        n = st.add([str(self.mat), f"http://127.0.0.1:{PORT}/page.html"])
        mods = {}
        for c in st.chunks:
            mods[c["modality"]] = mods.get(c["modality"], 0) + 1
        self.assertGreaterEqual(mods.get("image", 0), 3)              # 사진 2 + 웹 사진 1
        self.assertGreaterEqual(mods.get("video_segment", 0), 2)      # 장면 전환으로 2구간 이상
        self.assertEqual(mods.get("speech", 0), 1)                    # 영상 말소리 받아쓰기
        self.assertGreaterEqual(mods.get("text", 0), 2)               # 텍스트 파일 + 웹 본문
        self.assertEqual(st.tv.shape[0], len(st.chunks))
        self.assertEqual(Store(self.store_dir).add([str(self.mat)]), 0)   # 다시 넣으면 건너뜀
        self.assertGreater(n, 0)

        top = st.search("어두운 서버실", k=3, modalities=("image", "video_segment"), first_frame=True)
        self.assertIn("server", top[0]["text"].lower())
        self.assertTrue(Path(top[0]["path"]).exists())
        bld = st.search("밝은 연구동 건물", k=1, modalities=("image",))
        self.assertIn("building", bld[0]["text"].lower())
        gpu = st.search("GPU는 몇 장인가", k=2, modalities=("text", "speech"))
        self.assertTrue(any("112" in h["text"] for h in gpu))
        web = st.search("센터 위치 경산", k=3, modalities=("text",))
        self.assertTrue(any("page.html" in h["source"] for h in web))
        segs = [c for c in st.chunks if c["modality"] == "video_segment"]
        self.assertTrue(all(c["meta"]["described_from"] == "video" for c in segs))   # 구간 영상을 VLM에 직접
        self.assertGreater(mock_hub.STATE.get("video_inputs", 0), 0)
        mo = st.motions("서버 복도를 따라 전진", k=2)
        self.assertTrue(any("dolly" in m for m in mo))
        ev = st.evidence("GPU 인프라", k=4)
        self.assertIn("[", ev)
        st.results_html("서버실", top, st.dir / "t.html")

    def test_3_long_with_kb(self):
        st = Store(self.store_dir)
        if not st.chunks:
            st.add([str(self.mat)])
        cfg = longrun.LongConfig(image=self.bright, request="센터를 소개하는 이야기", seconds=24, out_root=self.tmp / "out",
                                 name="kb", kb=str(self.store_dir))
        mock_hub.STATE.update(fail_from=0, videos=10)
        longrun.start(cfg)
        run = next((self.tmp / "out").glob("*_kb"))
        plan = story.load(run)
        self.assertIn("Source material", mock_hub.STATE["last_l1"])           # 계획에 근거가 들어감
        self.assertIn("Source visuals", mock_hub.STATE["last_l2"])
        refs = [s for s in plan["scenes"] if s.get("visual_ref")]
        self.assertEqual(len(refs), 1)                                          # 목록에 없는 id(zzzz)는 버려짐
        self.assertNotIn("needs_character", refs[0])
        prog = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        self.assertTrue(any("사용자 자료" in e for e in prog["events"]))         # 실제 자료 프레임을 첫 프레임으로
        self.assertTrue(any("사용자 자료(검색)" in e for e in prog["events"]))     # visual_ref가 없는 새 장면도 자료에서 검색
        self.assertTrue((run / "final.mp4").exists())
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["config"]["kb"], str(self.store_dir))


if __name__ == "__main__":
    unittest.main(verbosity=2)
