"""OpenCV 파일 입출력 — Windows에서 한글·특수문자가 든 경로도 열리게.

Windows용 OpenCV는 cv2.imread / imwrite / VideoCapture / VideoWriter에 한글 경로를 주면
조용히 실패한다(None 또는 빈 영상). 그래서
  사진: 파일을 바이트로 읽어 imdecode / imencode 후 바이트로 쓴다 (경로를 OpenCV에 넘기지 않음)
  영상: 영문 이름의 임시 파일로 복사해 열고, 쓰기는 임시 파일에 쓴 뒤 원래 경로로 옮긴다
리눅스·맥이나 영문 경로는 그대로 OpenCV에 넘긴다. VS_FORCE_ASCII_IO=1이면 테스트를 위해 항상 우회한다.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

_TEMPS: list[Path] = []


def _needs_detour(p: str | Path) -> bool:
    if os.getenv("VS_FORCE_ASCII_IO") == "1":
        return True
    return os.name == "nt" and not str(p).isascii()


def _tmp_dir() -> Path:
    d = Path(tempfile.gettempdir())
    if not str(d).isascii():                        # 사용자 이름이 한글이면 임시 폴더도 한글 → 드라이브 루트 아래로
        d = Path(Path.cwd().anchor or "/") / "vs_tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _new_tmp(suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix, prefix="vs_", dir=_tmp_dir())
    os.close(fd)
    p = Path(name)
    _TEMPS.append(p)
    return p


@atexit.register
def _cleanup() -> None:
    for p in _TEMPS:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


# ------------------------------------------------------------------ 사진
def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    p = Path(path)
    if not _needs_detour(p):
        return cv2.imread(str(p), flags)
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def imwrite(path: str | Path, img, params: list[int] | None = None) -> bool:
    p = Path(path)
    if not _needs_detour(p):
        return cv2.imwrite(str(p), img, params or [])
    ok, enc = cv2.imencode(p.suffix or ".jpg", img, params or [])
    if ok:
        enc.tofile(str(p))
    return bool(ok)


# ------------------------------------------------------------------ 영상
def VideoCapture(path: str | Path):
    p = Path(path)
    if not _needs_detour(p) or not p.exists():
        return cv2.VideoCapture(str(p))
    tmp = _new_tmp(p.suffix or ".mp4")
    shutil.copyfile(p, tmp)
    return cv2.VideoCapture(str(tmp))


class _Writer:
    """임시 영문 경로에 쓰고 release() 때 원래 경로로 옮긴다."""

    def __init__(self, target: Path, *args):
        self.target = target
        self.tmp = _new_tmp(target.suffix or ".mp4")
        self.w = cv2.VideoWriter(str(self.tmp), *args)

    def isOpened(self) -> bool:  # noqa: N802 — OpenCV와 같은 이름
        return self.w.isOpened()

    def write(self, frame) -> None:
        self.w.write(frame)

    def release(self) -> None:
        self.w.release()
        if self.tmp.exists() and self.tmp.stat().st_size:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.tmp), str(self.target))


def VideoWriter(path: str | Path, *args):
    p = Path(path)
    if not _needs_detour(p):
        return cv2.VideoWriter(str(p), *args)
    return _Writer(p, *args)
