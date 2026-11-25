/* Seam-carving player client.
 * The kernel owns all geometry; this script only renders state and forwards
 * gestures. Every mutation round-trips through /api/* so the displayed seam,
 * carved frame and continuity chart always match the engine's actual output.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  meta: null,
  seams: null,
  frame: 0,
  playing: false,
  timer: null,
  view: "seam",
  drag: null,
};

async function post(action, payload = {}) {
  const res = await fetch(action, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!data.ok) {
    setStatus(`内核返回失败：${data.status}`, true);
    throw new Error(data.status);
  }
  applyState(data);
  return data;
}

async function refresh() {
  const res = await fetch("/api/state");
  const data = await res.json();
  applyState(data);
}

function applyState(data) {
  state.meta = data;
  state.seams = data.seams;
  $("frameLabel").textContent = `frame ${state.frame} / ${Math.max(data.frames - 1, 0)}`;
  $("scrub").max = Math.max(data.frames - 1, 0);
  $("scrub").value = Math.min(state.frame, data.frames - 1);
  if (!$("targetWidth").value) $("targetWidth").value = data.original_width;
  const warn = (data.warnings || []).join(" | ") || "无";
  setStatus(
    `场景: ${data.name}\n原始宽度 → 当前宽度: ${data.original_width} → ${data.width}\n` +
    `已应用步骤: ${data.steps.length}（${data.steps.join(", ") || "无"}）\n` +
    `图割流量: ${Number(data.flow_value).toFixed(3)}，求解耗时: ${Number(data.elapsed).toFixed(3)}s\n` +
    `镜头切换帧间隙: ${JSON.stringify(data.shot_cuts)}\n求解器告警: ${warn}`,
    (data.warnings || []).length > 0);
  renderFrames();
  drawContinuity(data);
}

function setStatus(text, isWarn) {
  const el = $("status");
  el.textContent = text;
  el.style.color = isWarn ? "var(--warn)" : "var(--ok)";
}

function frameUrl(view, f) {
  return `/api/frame?view=${encodeURIComponent(view)}&f=${f}&t=${Date.now()}`;
}

function renderFrames() {
  if (!state.meta) return;
  const f = state.frame;
  $("seamView").src = frameUrl("seam", f);
  $("carvedView").src = frameUrl("carved", f);
  if (state.view === "energy" || state.view === "original") {
    $("seamView").src = frameUrl(state.view, f);
  }
  sizeEditCanvas();
}

function sizeEditCanvas() {
  const img = $("seamView");
  const cv = $("editCanvas");
  cv.width = img.naturalWidth || state.meta?.original_width || 320;
  cv.height = img.naturalHeight || state.meta?.height || 120;
  drawSeamOverlay();
}

function drawSeamOverlay() {
  const cv = $("editCanvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!state.seams || state.frame >= state.seams.length) return;
  const row = state.seams[state.frame];
  ctx.strokeStyle = "#ff5a3c";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  row.forEach((x, y) => {
    if (y === 0) ctx.moveTo(x + 0.5, y + 0.5);
    else ctx.lineTo(x + 0.5, y + 0.5);
  });
  ctx.stroke();
  // vertex handles for editing
  ctx.fillStyle = "#ffd1c7";
  for (let y = 0; y < row.length; y += 4) {
    ctx.fillRect(row[y] - 1.5, y - 1.5, 3, 3);
  }
}

function drawContinuity(data) {
  const cv = $("continuity");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!data.seams || data.seams.length < 2) return;
  const T = data.seams.length;
  const deltas = [];
  for (let t = 1; t < T; t++) {
    const a = data.seams[t - 1], b = data.seams[t];
    let sum = 0;
    for (let y = 0; y < a.length; y++) sum += Math.abs(a[y] - b[y]);
    deltas.push(sum / a.length);
  }
  const maxD = Math.max(1, ...deltas);
  const pad = 28;
  const w = cv.width - pad * 2, h = cv.height - pad * 2;
  ctx.strokeStyle = "#2c333d";
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + h); ctx.lineTo(pad + w, pad + h); ctx.stroke();
  ctx.strokeStyle = "#4aa3ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  deltas.forEach((d, i) => {
    const x = pad + (i / Math.max(1, deltas.length - 1)) * w;
    const y = pad + h - (d / maxD) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // shot cut markers
  ctx.fillStyle = "#ff5a3c";
  (data.shot_cuts || []).forEach((isCut, i) => {
    if (!isCut) return;
    const x = pad + (i / Math.max(1, deltas.length - 1)) * w;
    ctx.fillRect(x - 1, pad, 2, h);
  });
  ctx.fillStyle = "#97a0ad";
  ctx.font = "11px sans-serif";
  ctx.fillText(`最大相邻帧接缝位移 ${maxD.toFixed(2)} px`, pad, 14);
}

// ------------------------------------------------------------- playback
function tick() {
  if (!state.meta) return;
  state.frame = (state.frame + 1) % state.meta.frames;
  renderFrames();
  $("scrub").value = state.frame;
  $("frameLabel").textContent = `frame ${state.frame} / ${state.meta.frames - 1}`;
}

$("play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "⏸ 暂停" : "▶ 播放";
  if (state.playing) {
    const fps = parseInt($("fps").value, 10) || 10;
    state.timer = setInterval(tick, 1000 / fps);
  } else {
    clearInterval(state.timer);
  }
});
$("scrub").addEventListener("input", (e) => {
  state.frame = parseInt(e.target.value, 10);
  renderFrames();
});
$("fps").addEventListener("input", () => {
  if (state.playing) {
    clearInterval(state.timer);
    state.timer = setInterval(tick, 1000 / (parseInt($("fps").value, 10) || 10));
  }
});
$("viewMode").addEventListener("change", (e) => {
  state.view = e.target.value;
  renderFrames();
});
$("seamView").addEventListener("load", sizeEditCanvas);

// ------------------------------------------------------------- seam editing
function eventPos(e) {
  const cv = $("editCanvas");
  const r = cv.getBoundingClientRect();
  const sx = cv.width / r.width, sy = cv.height / r.height;
  return {
    x: Math.round((e.clientX - r.left) * sx),
    y: Math.round((e.clientY - r.top) * sy),
  };
}

$("editCanvas").addEventListener("mousedown", (e) => {
  const p = eventPos(e);
  state.drag = { start: p, current: p, mode: e.shiftKey ? "protect" : "seam" };
});
$("editCanvas").addEventListener("mousemove", (e) => {
  if (!state.drag) return;
  state.drag.current = eventPos(e);
  const cv = $("editCanvas");
  const ctx = cv.getContext("2d");
  drawSeamOverlay();
  const { start, current } = state.drag;
  ctx.strokeStyle = state.drag.mode === "protect" ? "#ffb648" : "#46d18c";
  ctx.strokeRect(
    Math.min(start.x, current.x), Math.min(start.y, current.y),
    Math.abs(current.x - start.x), Math.abs(current.y - start.y));
});
$("editCanvas").addEventListener("mouseup", async () => {
  if (!state.drag) return;
  const { start, current, mode } = state.drag;
  state.drag = null;
  const moved = Math.abs(current.x - start.x) + Math.abs(current.y - start.y);
  try {
    if (mode === "protect" && moved > 4) {
      await post("/api/protect", {
        frame: state.frame,
        x0: Math.max(0, Math.min(start.x, current.x)),
        y0: Math.max(0, Math.min(start.y, current.y)),
        x1: Math.min(state.meta.original_width, Math.max(start.x, current.x)),
        y1: Math.min(state.meta.height, Math.max(start.y, current.y)),
      });
    } else if (moved <= 4) {
      await post("/api/edit_seam", { frame: state.frame, row: start.y, col: start.x });
    }
  } catch (_) { /* failure already surfaced */ }
});

// ------------------------------------------------------------- toolbar
$("load").addEventListener("click", () =>
  post("/api/load", { scenario: $("scenario").value }));
$("solve").addEventListener("click", () => post("/api/solve"));
$("remove").addEventListener("click", () => post("/api/step", { kind: "remove" }));
$("insert").addEventListener("click", () => post("/api/step", { kind: "insert" }));
$("resize").addEventListener("click", () =>
  post("/api/resize", { target_width: parseInt($("targetWidth").value, 10) }));
$("reset").addEventListener("click", () => post("/api/reset"));
$("clearProtect").addEventListener("click", async () => {
  // reset clears engine including protected masks, then re-applies current size
  const target = state.meta.width;
  await post("/api/reset");
  if (target !== state.meta.original_width) {
    await post("/api/resize", { target_width: target });
  }
});

refresh().catch((e) => setStatus(String(e), true));
