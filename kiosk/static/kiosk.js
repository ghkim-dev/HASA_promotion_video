/* VIDEO Studio 키오스크 화면. 서버의 진행 알림(SSE)을 받아 단계별로 그린다. 외부 라이브러리 없음. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  const STAGES = [
    { key: "observe", title: "사진 읽기", why: "무엇이 찍혔고 무엇이 움직일 수 있는지 살펴봐요", ref: "RPG (ICML 2024)의 재캡션 방식" },
    { key: "understand", title: "요청 이해", why: "원하는 움직임을 작은 목표로 나누고 지켜야 할 물리 규칙을 정해요", ref: "TIP-I2V (ICCV 2025), X-Planner (AAAI 2026), PhyT2V (CVPR 2025)" },
    { key: "checklist", title: "채점 질문 만들기", why: "완성된 영상을 채점할 질문을 먼저 정해 둬요", ref: "TIFA (ICCV 2023), VBench (CVPR 2024)" },
    { key: "design", title: "지시문 쓰기", why: "영상 모델이 알아듣는 영어 지시문을 써요. 중요한 움직임을 앞에 둬요", ref: "Prompt-A-Video, VPO (ICCV 2025)" },
    { key: "generate", title: "영상 만들기", why: "GPU 서버에서 Wan 2.2 모델이 사진을 첫 장면으로 영상을 그려요", ref: "" },
    { key: "verify", title: "스스로 채점", why: "영상의 장면들을 보고 채점 질문에 하나씩 답해요", ref: "Video-Bench (CVPR 2025), TIFA (ICCV 2023)" },
    { key: "reflect", title: "고쳐서 다시", why: "점수가 모자라면 이유를 찾아 지시문을 고쳐요", ref: "GenMAC (AAAI 2026), Reflect-DiT (ICCV 2025)" },
    { key: "compose", title: "완성", why: "가장 점수가 높은 영상을 골라 마무리해요", ref: "" },
  ];
  const CAT = { action: "동작", motion_direction: "이동", camera: "카메라", dynamic_attribute: "변화", interaction: "상호작용",
    environment: "주변", spatial: "위치", numeracy: "개수" };
  const ROUTE = { strengthen_motion: "움직임을 더 크고 분명하게", fix_binding: "움직일 대상·방향을 바로잡기", simplify: "동시에 하는 동작 줄이기",
    fix_camera: "카메라 움직임 바로잡기", fix_physics: "물리적으로 자연스럽게", protect_identity: "얼굴·모습이 바뀌지 않게",
    reduce_artifacts: "깜빡임·뭉개짐 줄이기", clarify: "요청을 다시 해석하기" };
  const RESET_SECONDS = 120;

  let cfg = { presets: [], samples: [], threshold: 0.8, keep_hours: 24 };
  let picked = null;           // {file} | {sample}
  let job = null;              // 서버 작업 정보
  let es = null;               // EventSource
  let seen = 0;                // 받은 진행 알림 수 (끊겼다 다시 연결할 때 이어 받기)
  let questions = [];
  let round = 1;
  let resetTimer = null;
  let genTimer = null, realPct = 0;
  let flipTimer = null;
  let res = null;                     // 해상도 정보: 입력 사진 → AI가 보는 크기 → 영상 크기               // 영상을 브라우저가 재생하지 못하면 장면 사진을 넘겨 보여 준다   // Hub는 몇 초에 한 번만 진행률을 알려 주므로, 사이는 평소 걸리는 시간으로 추정해 채운다

  // ---------------------------------------------------------------- 화면 전환
  function show(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("is-on", s.id === "s-" + name));
    window.scrollTo(0, 0);
    if (name === "start") resetAll();
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-go]");
    if (b) show(b.dataset.go);
  });

  async function loadConfig() {
    try {
      cfg = await (await fetch("/api/config")).json();
    } catch { return; }
    $("#mock-flag").hidden = !cfg.mock;
    $("#start-queue").textContent = cfg.waiting ? `지금 ${cfg.waiting}명이 영상을 만들고 있어요. 한 사람에 약 ${Math.max(1, Math.round(cfg.avg_seconds / 60))}분 걸려요.` : "";
    $("#privacy").textContent = `사진은 영상을 만드는 데만 쓰이고 ${cfg.keep_hours}시간 뒤 자동으로 지워져요. 다 본 뒤 바로 지울 수도 있어요.`;
    const pr = $("#presets");
    pr.replaceChildren(...cfg.presets.map((p) => {
      const c = el("button", "chip", p.label);
      c.type = "button"; c.setAttribute("aria-pressed", "false"); c.dataset.request = p.request;
      c.addEventListener("click", () => {
        pr.querySelectorAll(".chip").forEach((x) => x.setAttribute("aria-pressed", String(x === c)));
        $("#req-text").value = p.request;
      });
      return c;
    }));
    const row = $("#sample-row");
    row.replaceChildren(...cfg.samples.map((src) => {
      const b = el("button"); b.type = "button"; b.setAttribute("aria-label", "예시 사진 사용");
      const im = el("img"); im.src = src; im.alt = ""; b.append(im);
      b.addEventListener("click", () => choose({ sample: src.split("/").pop(), url: src }));
      return b;
    }));
    $("#samples").hidden = !cfg.samples.length;
  }

  // ---------------------------------------------------------------- 사진·요청
  for (const id of ["#in-camera", "#in-file"]) {
    $(id).addEventListener("change", (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) choose({ file: f, url: URL.createObjectURL(f) });
      e.target.value = "";
    });
  }
  function choose(p) {
    picked = p;
    $("#req-img").src = p.url;
    $("#req-err").textContent = "";
    show("request");
  }

  $("#req-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#req-text").value.trim();
    if (text.length < 2) { $("#req-err").textContent = "움직임을 고르거나 직접 써 주세요."; return; }
    const fd = new FormData();
    fd.append("request", text);
    if (picked.file) fd.append("image", picked.file, picked.file.name || "photo.jpg");
    else fd.append("sample", picked.sample);
    const btn = $("#req-go");
    btn.disabled = true; btn.textContent = "보내는 중";
    try {
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "보내지 못했어요.");
      job = d;
      startRun(text);
    } catch (err) {
      $("#req-err").textContent = err.message;
    } finally {
      btn.disabled = false; btn.textContent = "영상 만들기";
    }
  });

  // ---------------------------------------------------------------- 진행 화면
  function buildLog() {
    const log = $("#log");
    log.replaceChildren(...STAGES.map((s) => {
      const li = el("li", "st is-pending"); li.id = "st-" + s.key;
      li.append(el("h3", null, s.title), el("p", "why", s.why));
      if (s.ref) li.append(el("p", "ref", "참고: " + s.ref));
      li.append(el("div", "body"));
      return li;
    }));
  }
  const stEl = (k) => $("#st-" + k);
  const body = (k) => stEl(k).querySelector(".body");
  function activate(k) {
    const i = STAGES.findIndex((s) => s.key === k);
    STAGES.forEach((s, j) => {
      const li = stEl(s.key);
      if (j < i && (li.classList.contains("is-active") || li.classList.contains("is-pending"))) {
        li.classList.remove("is-active", "is-pending"); li.classList.add(s.key === "reflect" && !li.dataset.used ? "is-skip" : "is-done");
      }
      if (j === i) { li.classList.remove("is-pending", "is-done", "is-skip"); li.classList.add("is-active"); }
      if (j > i && li.classList.contains("is-active")) { li.classList.remove("is-active"); li.classList.add("is-pending"); }
    });
    // 작업 기록 안에서만 스크롤한다 (세로 화면에서 페이지가 영상 아래로 밀려 내려가지 않게)
    const li = stEl(k), log = $("#log");
    if (log.scrollHeight > log.clientHeight + 4) {
      log.scrollTo({ top: Math.max(0, li.offsetTop - 12), behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }
  }
  function keep(k) { body(k).classList.add("keep"); }

  function startRun(text) {
    clearTimeout(resetTimer);
    seen = 0; questions = []; round = 1;
    buildLog();
    $("#run-req").textContent = text;
    $("#run-round").textContent = "";
    $("#run-img").src = `/api/jobs/${job.id}/input`;
    $("#run-img").hidden = false;
    const v = $("#run-video"); v.hidden = true; v.removeAttribute("src");
    $("#tags").replaceChildren(); $("#tags").hidden = false; $("#strip").replaceChildren(); $("#caption").textContent = "";
    $("#gen").hidden = true; $("#scan").hidden = true;
    waiting(job.position);
    show("run");
    connect();
  }

  function waiting(pos, avg) {
    const w = $("#waiting");
    if (pos == null || pos < 0) { w.hidden = true; return; }
    w.hidden = false;
    if (pos === 0) { $("#w-big").textContent = "곧 시작해요"; $("#w-small").textContent = ""; return; }
    const min = Math.max(1, Math.round((pos * (avg || cfg.avg_seconds || 180)) / 60));
    $("#w-big").textContent = `앞에 ${pos}명이 기다리고 있어요`;
    $("#w-small").textContent = `약 ${min}분 뒤에 시작해요. 이 화면에서 기다려 주세요.`;
  }

  function connect() {
    if (es) es.close();
    es = new EventSource(`/api/jobs/${job.id}/events?since=${seen}`);
    es.onmessage = (m) => { seen += 1; try { handle(JSON.parse(m.data)); } catch (err) { console.error(err); } };
    es.addEventListener("queue", (m) => { const d = JSON.parse(m.data); waiting(d.position, d.avg_seconds); });
    es.addEventListener("gone", () => { es.close(); show("start"); });
    es.onerror = () => {
      es.close();
      if (job && !["done", "error"].includes(job.status)) setTimeout(connect, 2000);   // 와이파이가 잠깐 끊겨도 이어 받기
    };
  }

  const file = (p) => (p ? `/api/jobs/${job.id}/file/${p}` : "");

  // 대상 표시를 사진이 실제로 보이는 영역(object-fit: contain) 안에 놓는다 — 세로 사진이면 가운데 좁은 영역
  function placeTags() {
    const img = $("#run-img"), box = $("#table"), tags = $("#tags");
    if (!img.naturalWidth) return;
    const W = box.clientWidth, H = box.clientHeight;
    const s = Math.min(W / img.naturalWidth, H / img.naturalHeight);
    const w = img.naturalWidth * s, h = img.naturalHeight * s;
    Object.assign(tags.style, { left: (W - w) / 2 + "px", top: (H - h) / 2 + "px", width: w + "px", height: h + "px", right: "auto", bottom: "auto" });
  }
  $("#run-img").addEventListener("load", placeTags);
  window.addEventListener("resize", placeTags);
  const POS = { left: 25, center: 50, right: 75, near: 62, far: 34 };

  function handle(ev) {
    const d = ev.data || {};
    switch (ev.stage) {
      case "queued": break;
      case "start":
        res = d.resolution || null;
        waiting(null); break;
      case "observe":
        activate("observe"); $("#scan").hidden = false; break;
      case "observe_done": {
        $("#scan").hidden = true;
        $("#caption").textContent = d.caption || "";
        const tags = $("#tags"); tags.replaceChildren();
        (d.subjects || []).slice(0, 4).forEach((s, i) => {
          const pos = String(s.position || "center").toLowerCase();
          const x = POS[["left", "right", "center"].find((k) => pos.includes(k)) || "center"];
          const y = POS[["near", "far"].find((k) => pos.includes(k)) || "near"] - 8 + i * 9;
          const t = el("div", "tag", s.name || "");
          if ((s.movable_parts || []).length) t.append(el("small", null, "움직일 수 있음: " + s.movable_parts.slice(0, 4).join(", ")));
          t.style.left = x + "%"; t.style.top = y + "%"; t.style.animationDelay = i * 0.15 + "s";
          tags.append(t);
        });
        const b = body("observe"); b.replaceChildren();
        const env = (d.environment_dynamics || []).slice(0, 4);
        b.append(el("p", "note", `찾은 대상 ${(d.subjects || []).length}개` + (env.length ? `, 저절로 움직일 수 있는 것: ${env.join(", ")}` : "")));
        if (res) {
          const [iw, ih] = res.input, [vw, vh] = res.vlm;
          b.append(el("p", "res", `원본 ${iw}×${ih} 사진을 ${vw}×${vh}로 비율 그대로 읽었어요 (약 ${res.vlm_tokens.toLocaleString()}토큰)`));
        }
        keep("observe");
        placeTags();
        break;
      }
      case "understand": activate("understand"); break;
      case "understand_done": {
        const b = body("understand"); b.replaceChildren();
        if (d.intent_ko) b.append(el("p", "note", d.intent_ko));
        const ul = el("ul", "goal-list");
        (d.goals || []).forEach((g) => {
          const li = el("li"); li.append(el("span", "cat", CAT[g.category] || g.category), document.createTextNode(g.change_ko || g.change || ""));
          ul.append(li);
        });
        b.append(ul); keep("understand");
        break;
      }
      case "checklist": activate("checklist"); break;
      case "checklist_done": {
        questions = d.questions || [];
        const b = body("checklist"); b.replaceChildren();
        const ul = el("ul", "q-list"); ul.id = "qlist";
        questions.forEach((q) => {
          const li = el("li"); li.dataset.q = q.id;
          const t = el("span", null, q.question_ko || q.question);
          li.append(el("span", "q-mark"), t); ul.append(li);
        });
        b.append(ul); keep("checklist");
        break;
      }
      case "design":
        if (round > 1 || /라운드 2|라운드 3/.test(ev.message)) {
          round = Number((ev.message.match(/라운드 (\d+)/) || [])[1] || round);
        }
        $("#run-round").textContent = round > 1 ? `${round}번째 시도` : "";
        activate("design"); break;
      case "design_done": {
        round = d.round || round;
        $("#run-round").textContent = round > 1 ? `${round}번째 시도` : "";
        const c = (d.candidates || [])[0] || {};
        const b = body("design"); b.replaceChildren();
        b.append(el("pre", "prompt", c.prompt || ""));
        if (c.strategy_ko) b.append(el("p", "strategy", "전략: " + c.strategy_ko));
        keep("design");
        // 다시 시도할 때는 이전 채점 표시를 지운다
        document.querySelectorAll("#qlist .q-mark").forEach((m) => { m.className = "q-mark"; m.textContent = ""; });
        document.querySelectorAll("#qlist .q-ev").forEach((x) => x.remove());
        break;
      }
      case "generate": {
        stopFlip(); $("#run-img").src = `/api/jobs/${job.id}/input`;
        activate("generate");
        $("#run-video").hidden = true; $("#run-video").pause(); $("#run-img").hidden = false; $("#tags").hidden = true;
        $("#gen").hidden = false;
        const b = body("generate"); b.replaceChildren();
        const bar = el("div", "bar-pct"); bar.append(el("i")); b.append(bar);
        if (res && res.video) {
          const [w, h] = res.video.split("x").map(Number);
          b.append(el("p", "res", `영상 크기 ${w}×${h} (${w > h ? "가로" : w < h ? "세로" : "정사각"}) · 사진 비율에 가장 가까운 크기`));
        }
        startGen();
        break;
      }
      case "progress": realPct = Math.max(realPct, Number(d.percent) || 0); break;
      case "verify":
        stopGen(); $("#gen").hidden = true; activate("verify"); break;
      case "verify_done": showVerify(d); break;
      case "reflect": stEl("reflect").dataset.used = "1"; activate("reflect"); break;
      case "reflect_done": {
        const b = body("reflect"); b.replaceChildren();
        b.append(el("p", "note", d.summary_ko || ""));
        if (d.route) b.append(el("p", "strategy", "고칠 방향: " + (ROUTE[d.route] || d.route)));
        keep("reflect");
        break;
      }
      case "select": break;
      case "compose": activate("compose"); break;
      case "done": {
        activate("compose");
        stEl("compose").classList.replace("is-active", "is-done");
        finish(d);
        break;
      }
      case "error": break;   // 후보 하나의 실패 — 다음 시도로 넘어간다
      case "end":
        job.status = d.status;
        if (d.status === "error") { $("#err-msg").textContent = friendly(d.error); show("error"); scheduleReset(); }
        if (es) es.close();
        break;
      default: break;
    }
  }

  function startGen() {
    stopGen();
    realPct = 0;
    const t0 = Date.now(), est = (cfg.est_video_seconds || 50) * 1000;
    const step = () => {
      const guess = Math.min(92, (100 * (Date.now() - t0)) / est);
      setPct(Math.max(realPct, guess));
    };
    step();
    genTimer = setInterval(step, 400);
  }
  function stopGen() { if (genTimer) { clearInterval(genTimer); genTimer = null; setPct(100); } }

  function setPct(p) {
    const ring = $("#ring");
    ring.style.strokeDashoffset = String(326.7 * (1 - Math.min(100, p) / 100));
    $("#gen-pct").textContent = Math.round(p) + "%";
    const bar = body("generate").querySelector(".bar-pct i");
    if (bar) bar.style.width = p + "%";
  }

  function showVerify(d) {
    // 왼쪽: 만든 영상 재생 + 장면 띠
    stopFlip();
    $("#tags").hidden = true;
    const strip = $("#strip"); strip.replaceChildren();
    const frames = (d.frames || []).map(file);
    frames.forEach((src, i) => {
      const fr = el("div", "frame"); fr.style.animationDelay = i * 0.08 + "s";
      const im = el("img"); im.src = src; im.alt = `${i + 1}번째 장면`; fr.append(im); strip.append(fr);
    });
    const v = $("#run-video");
    v.onerror = () => flip(frames);
    v.src = file(d.video); v.hidden = false; $("#run-img").hidden = true;
    v.play().catch(() => flip(frames));
    // 채점 질문에 ✓/✗ 찍기
    (d.answers || []).forEach((a, i) => {
      const li = document.querySelector(`#qlist li[data-q="${a.id}"]`);
      if (!li) return;
      const m = li.querySelector(".q-mark");
      setTimeout(() => {
        m.className = "q-mark " + (a.answer === "yes" ? "yes" : "no");
        m.textContent = a.answer === "yes" ? "✓" : "✗";
        if (a.answer !== "yes" && a.evidence) { const ev = el("span", "q-ev", "본 것: " + a.evidence); li.children[1].append(ev); }
      }, 250 + i * 220);
    });
    // 점수 막대
    const b = body("verify"); b.replaceChildren();
    const pass = d.score >= (d.threshold ?? cfg.threshold);
    const num = el("div", "score-num", Math.round(d.score * 100) + "점");
    num.append(el("small", null, (d.round > 1 ? `${d.round}번째 시도, ` : "") +
      (pass ? "통과 기준을 넘었어요" : `통과 기준 ${Math.round((d.threshold ?? cfg.threshold) * 100)}점에 못 미쳐요`)));
    const sc = el("div", "score"); const fill = el("i"); sc.append(fill);
    const th = el("div", "th"); th.style.left = `calc(${(d.threshold ?? cfg.threshold) * 100}% - 1px)`; th.append(el("span", null, "통과 기준")); sc.append(th);
    b.append(sc, num);
    if (d.notes_ko) b.append(el("p", "strategy", d.notes_ko));
    if ((d.penalties || []).length) b.append(el("p", "strategy", "감점: " + d.penalties.join(", ")));
    keep("verify");
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = Math.round(d.score * 100) + "%"; fill.classList.toggle("pass", pass); }));
  }

  function flip(frames) {
    stopFlip();
    if (!frames.length) return;
    $("#run-video").hidden = true;
    const img = $("#run-img"); img.hidden = false;
    const cells = [...document.querySelectorAll("#strip .frame")];
    let i = 0;
    const step = () => {
      img.src = frames[i % frames.length];
      cells.forEach((c, j) => c.classList.toggle("is-now", j === i % frames.length));
      i += 1;
    };
    step();
    flipTimer = setInterval(step, 350);
  }
  function stopFlip() {
    if (flipTimer) { clearInterval(flipTimer); flipTimer = null; }
    document.querySelectorAll("#strip .frame.is-now").forEach((c) => c.classList.remove("is-now"));
  }

  function finish(d) {
    const best = Math.max(...(d.scores || [0]));
    setTimeout(() => {
      stopFlip();
      $("#done-video").src = file(d.final);
      $("#done-req").textContent = "“" + $("#run-req").textContent + "”";
      $("#done-score").textContent = Math.round(best * 100) + "점";
      $("#done-save").href = file(d.final) + "?download=1";
      show("done");
      scheduleReset();
    }, 2200);
  }

  function scheduleReset() {
    clearTimeout(resetTimer);
    let left = RESET_SECONDS;
    const tick = () => {
      $("#done-timer").textContent = `${left}초 뒤 처음 화면으로 돌아가요`;
      if (left-- <= 0) { show("start"); return; }
      resetTimer = setTimeout(tick, 1000);
    };
    tick();
  }
  ["pointerdown", "keydown"].forEach((t) => document.addEventListener(t, () => {
    if ($("#s-done").classList.contains("is-on") || $("#s-error").classList.contains("is-on")) scheduleReset();
  }));

  $("#done-delete").addEventListener("click", async () => {
    if (!job) return;
    await fetch(`/api/jobs/${job.id}`, { method: "DELETE" }).catch(() => {});
    $("#done-video").removeAttribute("src");
    show("start");
  });

  function friendly(msg) {
    msg = String(msg || "");
    if (/HTTP 401|HTTP 403/.test(msg)) return "AI 서버 접속 키에 문제가 있어요. 직원에게 알려 주세요.";
    if (/HTTP 429/.test(msg)) return "AI 서버가 지금 바빠요. 잠시 후 다시 해 주세요.";
    if (/연결|timed out|Timeout|ConnectionError/i.test(msg)) return "AI 서버에 연결하지 못했어요. 인터넷 연결을 확인해 주세요.";
    return "다른 사진이나 다른 움직임으로 다시 해 보세요." + (msg ? ` (${msg.slice(0, 120)})` : "");
  }

  function resetAll() {
    if (es) { es.close(); es = null; }
    if (genTimer) { clearInterval(genTimer); genTimer = null; }
    stopFlip();
    clearTimeout(resetTimer);
    job = null; picked = null;
    $("#req-text").value = "";
    document.querySelectorAll("#presets .chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    $("#done-video").pause?.();
    loadConfig();
  }

  loadConfig();
})();
