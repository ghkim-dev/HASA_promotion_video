"""멀티모달 지식 저장소 — rag_stores/<이름>/

  chunks.jsonl     조각 목록 (원본 위치 포함)
  text_vecs.npy    모든 조각의 글 벡터 (설명문·본문·받아쓰기)
  img_vecs.npy     사진 조각의 이미지 벡터 (Hub가 이미지 임베딩을 지원할 때만)
  files/           사진·영상 구간 시작 프레임 (Wan 첫 프레임 후보)
  sources.json     이미 넣은 입력 (같은 파일을 다시 넣으면 건너뜀)

검색 = 글 벡터 유사도 (+ 이미지 벡터 유사도) → 순위 합치기(RRF) → 리랭커 → 중복 제거(MMR)
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np

from ..hub import Hub, HubError
from ..utils import log
from . import sources as S

STORE_ROOT = Path(os.getenv("VS_RAG_DIR", "rag_stores"))
VISUAL = ("image", "video_segment")


def _norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


class Store:
    def __init__(self, name_or_path: str | Path, hub: Hub | None = None):
        p = Path(name_or_path)
        # 경로(폴더가 있거나 경로 구분자를 포함)면 그대로, 이름만 주면 rag_stores/<이름>
        self.dir = p if (p.exists() or len(p.parts) > 1) else STORE_ROOT / p.name
        self.files = self.dir / "files"
        self.files.mkdir(parents=True, exist_ok=True)
        self._hub = hub
        self.chunks: list[dict] = []
        self.tv = np.zeros((0, 0), "float32")
        self.iv = np.zeros((0, 0), "float32")
        self.img_ids: list[str] = []
        self.sources: dict = {}
        self.meta: dict = {}
        self._load()

    @property
    def hub(self) -> Hub:
        if self._hub is None:
            self._hub = Hub()
        return self._hub

    # ---------------------------------------------------------------- 저장/불러오기
    def _load(self) -> None:
        cj = self.dir / "chunks.jsonl"
        if cj.exists():
            self.chunks = [json.loads(x) for x in cj.read_text(encoding="utf-8").splitlines() if x.strip()]
        if (self.dir / "text_vecs.npy").exists():
            self.tv = np.load(self.dir / "text_vecs.npy")
        if (self.dir / "img_vecs.npy").exists():
            self.iv = np.load(self.dir / "img_vecs.npy")
            self.img_ids = json.loads((self.dir / "img_ids.json").read_text(encoding="utf-8"))
        for name, attr in (("sources.json", "sources"), ("meta.json", "meta")):
            if (self.dir / name).exists():
                setattr(self, attr, json.loads((self.dir / name).read_text(encoding="utf-8")))

    def save(self) -> None:
        (self.dir / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in self.chunks), encoding="utf-8")
        np.save(self.dir / "text_vecs.npy", self.tv)
        if len(self.img_ids):
            np.save(self.dir / "img_vecs.npy", self.iv)
            (self.dir / "img_ids.json").write_text(json.dumps(self.img_ids), encoding="utf-8")
        (self.dir / "sources.json").write_text(json.dumps(self.sources, ensure_ascii=False, indent=1), encoding="utf-8")
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 넣기
    @staticmethod
    def _fingerprint(item: str) -> str:
        p = Path(item)
        if p.exists() and p.is_file():
            st = p.stat()
            return f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        return item

    def add(self, items: list[str], speech: bool = True) -> int:
        """파일·폴더·URL을 넣는다. 이미 넣은 것은 건너뛴다. 넣은 조각 수를 돌려준다."""
        hub = self.hub
        model = hub.embed_model()
        if self.meta.get("text_model") and self.meta["text_model"] != model:
            raise HubError(f"이 저장소는 {self.meta['text_model']}로 만들었습니다. 같은 모델을 쓰거나 새 저장소를 만드세요.")
        new: list[dict] = []
        expanded = []
        for it in items:
            p = Path(it)
            expanded += [str(f) for f in sorted(p.rglob("*")) if f.is_file() and not f.name.startswith(("~$", "."))] \
                if p.is_dir() else [it]                  # ~$ = 열려 있는 오피스 문서의 잠금 파일
        for it in expanded:
            fp = self._fingerprint(it)
            if fp in self.sources:
                continue
            log(f"[rag] 넣는 중: {it}")
            try:
                got = S.collect(hub, it, self.files, speech)
            except Exception as e:  # noqa: BLE001 — 입력 하나 실패가 전체를 멈추지 않게
                log(f"    (실패, 건너뜀: {e})")
                continue
            have = {c["id"] for c in self.chunks}
            got = [c for c in got if c["id"] not in have]
            new += got
            self.sources[fp] = {"item": it, "chunks": len(got)}
        if not new:
            log("[rag] 새로 넣을 것이 없습니다.")
            self.save()
            return 0
        log(f"[rag] 조각 {len(new)}개 임베딩 ({model})")
        vecs = np.asarray(hub.embed_texts([c["text"] for c in new]), "float32")
        self.tv = vecs if self.tv.size == 0 else np.vstack([self.tv, vecs])
        vis = [c for c in new if c["modality"] in VISUAL and c.get("file")]
        if vis:
            probe = [self.files / c["file"] for c in vis[:2]]
            if hub.image_embed_support(probe):
                iv = np.asarray(hub.embed_images([self.files / c["file"] for c in vis]), "float32")
                self.iv = iv if self.iv.size == 0 else np.vstack([self.iv, iv])
                self.img_ids += [c["id"] for c in vis]
                self.meta["image_model"] = hub._img_embed
        self.chunks += new
        self.meta["text_model"] = model
        self.save()
        cnt = {}
        for c in self.chunks:
            cnt[c["modality"]] = cnt.get(c["modality"], 0) + 1
        log(f"[rag] 저장소 {self.dir}: 조각 {len(self.chunks)}개 {cnt}")
        return len(new)

    # ---------------------------------------------------------------- 찾기
    def search(self, query: str, k: int = 8, modalities: tuple[str, ...] | None = None, first_frame: bool = False,
               rerank: bool = True, pool: int | None = None) -> list[dict]:
        if not self.chunks:
            return []
        from ..budget import BUDGET
        pool = max(pool or BUDGET.mode["rerank_pool"], k)
        hub = self.hub
        idx = [i for i, c in enumerate(self.chunks)
               if (not modalities or c["modality"] in modalities)
               and (not first_frame or (c.get("file") and c["meta"].get("good_first_frame", True)))]
        if not idx:
            return []
        q = _norm(np.asarray(hub.embed_texts([query], query=True), "float32"))[0]
        ts = _norm(self.tv[idx]) @ q
        rank_t = {idx[j]: r for r, j in enumerate(np.argsort(-ts))}
        fused = {i: 1 / (60 + rank_t[i]) for i in idx}
        tscore = {idx[j]: float(ts[j]) for j in range(len(idx))}
        if len(self.img_ids) and (self.meta.get("image_model") or {}):
            hub._img_embed = self.meta["image_model"]
            qi = hub.embed_query_for_images([query])
            if qi:
                qv = _norm(np.asarray(qi, "float32"))[0]
                pos = {cid: n for n, cid in enumerate(self.img_ids)}
                cand = [i for i in idx if self.chunks[i]["id"] in pos]
                if cand:
                    isc = _norm(self.iv[[pos[self.chunks[i]["id"]] for i in cand]]) @ qv
                    for r, j in enumerate(np.argsort(-isc)):
                        fused[cand[j]] += 1 / (60 + r)            # 글·이미지 두 순위를 합친다 (RRF)
        top = sorted(fused, key=lambda i: -fused[i])[:pool]
        order = top
        scores = {i: fused[i] for i in top}
        if rerank:
            rr = hub.rerank(query, [self.chunks[i]["text"] for i in top])
            if rr:
                order = [top[j] for j, _ in sorted(rr, key=lambda x: -x[1])]
                scores = {top[j]: s for j, s in rr}
        picked: list[int] = []
        for i in order:                                          # 거의 같은 조각은 하나만 (MMR식 중복 제거)
            c = self.chunks[i]
            dup = False
            for j in picked:
                d = self.chunks[j]
                same_src = c["source"] == d["source"] and c["modality"] == d["modality"] == "video_segment" and \
                    abs(c["meta"].get("t0", 0) - d["meta"].get("t0", 0)) < 4
                if same_src or float(_norm(self.tv[i]) @ _norm(self.tv[j])) > 0.97:
                    dup = True
                    break
            if not dup:
                picked.append(i)
            if len(picked) >= k:
                break
        return [{**self.chunks[i], "score": round(float(scores.get(i, 0)), 4), "text_sim": round(tscore.get(i, 0), 4),
                 "path": str(self.files / self.chunks[i]["file"]) if self.chunks[i].get("file") else ""} for i in picked]

    # ---------------------------------------------------------------- 영상 만들기에 쓰는 묶음
    def evidence(self, query: str, token_budget: int | None = None, k: int | None = None) -> str:
        """계획 단계에 줄 근거 묶음 (출처 표시). 관련도순으로 토큰 예산을 최대한 채운다 (조각 내용은 자르지 않음).
        token_budget을 비우면 글 모델 최대 길이 × 모드 비율 (max 모드 60%)."""
        from ..budget import BUDGET, pack
        m = BUDGET.mode
        if token_budget is None:
            token_budget = BUDGET.fill_tokens(self.hub, self.hub.models().text)
        hits = self.search(query, k=k or m["evidence_k"], pool=max(m["rerank_pool"], k or 0))
        lines = [f"[{h['id']}] ({h['modality']}, {h['source']} {h['loc']}) {h['text']}" for h in hits]
        got = pack(lines, token_budget)
        log(f"    (근거 {len(got)}/{len(self.chunks)}조각 · 약 {sum(len(x) for x in got):,}자 · 예산 {token_budget:,}토큰)")
        return "\n".join(got)

    def visuals(self, query: str, k: int | None = None) -> list[dict]:
        from ..budget import BUDGET
        return self.search(query, k=k or BUDGET.mode["visuals_k"], modalities=VISUAL, first_frame=True)

    def motions(self, query: str, k: int = 3) -> list[str]:
        hits = self.search(query, k=k, modalities=("video_segment",))
        return [f"{h['meta'].get('motion', '')} (camera: {h['meta'].get('camera', '')})".strip() for h in hits
                if h["meta"].get("motion") or h["meta"].get("camera")]

    def get(self, cid: str) -> dict | None:
        return next(({**c, "path": str(self.files / c["file"]) if c.get("file") else ""} for c in self.chunks if c["id"] == cid), None)

    # ---------------------------------------------------------------- 결과 보기
    def results_html(self, query: str, hits: list[dict], out: Path) -> Path:
        e = html.escape
        rows = []
        for h in hits:
            img = f"<img src='files/{e(h['file'])}'>" if h.get("file") else ""
            rows.append(f"<div class='hit'>{img}<div><b>{e(h['modality'])}</b> · {e(h['source'])} {e(h['loc'])} · "
                        f"점수 {h['score']}<p>{e(h['text'][:400])}</p></div></div>")
        page = (f"<!doctype html><meta charset='utf-8'><title>검색 결과</title><style>"
                "body{font:15px/1.5 'Malgun Gothic',sans-serif;max-width:1000px;margin:0 auto;padding:20px}"
                ".hit{display:flex;gap:14px;border-bottom:1px solid #ddd;padding:10px 0}.hit img{width:240px;border-radius:6px}"
                "p{margin:4px 0;color:#444}@media (prefers-color-scheme:dark){body{background:#151515;color:#eee}p{color:#bbb}}</style>"
                f"<h2>“{e(query)}”</h2>{''.join(rows)}")
        out.write_text(page, encoding="utf-8")
        return out
