"""The live call monitor page (``GET /monitor``): the browser client of ``WS /v1/stream``.

Plain HTML and JavaScript in one page, no framework and nothing loaded from the network, so it
works offline and needs no build step. It captures the microphone (or plays a chosen recording)
with an AudioWorklet at the browser's own sample rate, sends float32 frames of about 0.25 s, and
draws the scores as they arrive. The server resamples, so it works in Chrome, Edge and Firefox
(Firefox cannot feed a microphone into a 16 kHz AudioContext). Browsers allow microphone access
on localhost or HTTPS only.

Every visible string comes from ``voxguard.ui.i18n`` (the ``monitor_*`` keys, English and
Hindi), passed in as JSON and switched on the page.
"""

# ruff: noqa: E501  (the page below is HTML and JavaScript, not Python)

from __future__ import annotations

import json

from voxguard.ui.i18n import EN, HI

PREFIX = "monitor_"


SHARED = ("language",)  # keys the page shares with the Streamlit UI
SHARED_PREFIXES = ("tx_",)  # transfer decision labels, shared with the incident report


def strings() -> dict[str, dict[str, str]]:
    return {
        lang: {
            k: v for k, v in d.items() if k.startswith((PREFIX, *SHARED_PREFIXES)) or k in SHARED
        }
        for lang, d in (("en", EN), ("hi", HI))
    }


def render() -> str:
    # "</" cannot appear inside the JSON, so it cannot close the script element early
    data = json.dumps(strings(), ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__STRINGS__", data)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LineTrue live call monitor</title>
<style>
  :root { --fg:#1f2430; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; --panel:#f6f7f9;
          --safe:#dcfce7; --safe-fg:#14532d; --alert:#fee2e2; --alert-fg:#7f1d1d; --info:#e0ecff;
          --info-fg:#1e3a8a; --window:#1f6feb; --running:#7cb7ff; --thr:#888; --flag:#d62728; }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.45 system-ui, -apple-system, "Segoe UI", "Noto Sans", "Noto Sans Devanagari", sans-serif;
         color:var(--fg); background:var(--bg); }
  main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  .muted { color: var(--muted); font-size: .92rem; }
  .row { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin:16px 0; }
  label { display:flex; flex-direction:column; gap:4px; font-size:.9rem; }
  input, select, button { font: inherit; padding: 8px 10px; border:1px solid var(--line); border-radius:8px; }
  button { cursor:pointer; background:var(--panel); }
  button.primary { background:#ff4b4b; color:#fff; border-color:#ff4b4b; }
  button:disabled { opacity:.5; cursor:default; }
  #banner { padding:16px 18px; border-radius:10px; margin:16px 0; background:var(--panel); }
  #banner.safe { background:var(--safe); color:var(--safe-fg); }
  #banner.alert { background:var(--alert); color:var(--alert-fg); }
  #banner.info { background:var(--info); color:var(--info-fg); }
  #banner.caution { background:#fef3c7; color:#78350f; }
  #langnote { background:#fef3c7; color:#78350f; padding:10px 14px; border-radius:8px; }
  #banner strong { font-size:1.25rem; display:block; }
  #guidance { margin-top:10px; }
  #guidance ul { margin:4px 0 0; padding-left:20px; }
  #consent { border-left:4px solid #f59e0b; background:#fffbeb; padding:8px 12px; border-radius:6px; }
  .stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:8px 0 16px; }
  .stat { background:var(--panel); border-radius:10px; padding:10px 12px; }
  .stat span { display:block; font-size:.8rem; color:var(--muted); }
  .stat b { font-size:1.3rem; font-variant-numeric: tabular-nums; }
  canvas { width:100%; height:260px; border:1px solid var(--line); border-radius:10px; }
  .legend { display:flex; gap:16px; font-size:.85rem; color:var(--muted); margin-top:6px; flex-wrap:wrap; }
  .key { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; }
  #final { margin-top:16px; }
  #tx { margin-top:28px; border-top:1px solid var(--line); padding-top:12px; }
  #tx h2 { font-size:1.2rem; margin:0 0 4px; }
  #decision { padding:14px 16px; border-radius:10px; margin-top:8px; background:var(--panel); }
  #decision.allowed, #decision.released { background:var(--safe); color:var(--safe-fg); }
  #decision.pending { background:#fef3c7; color:#78350f; }
  #decision.held { background:var(--alert); color:var(--alert-fg); }
  #decision ul { margin:6px 0; padding-left:20px; }
</style>
</head>
<body>
<main>
  <div class="row" style="justify-content:space-between; margin-top:0">
    <div>
      <h1 data-t="monitor_title"></h1>
      <div data-t="monitor_intro"></div>
    </div>
    <label><span data-t="language"></span>
      <select id="lang"><option value="en">English</option><option value="hi">हिन्दी</option></select>
    </label>
  </div>
  <p class="muted" data-t="monitor_note"></p>
  <p id="consent" data-t="monitor_consent"></p>

  <div class="row">
    <label><span data-t="monitor_caller"></span><input id="caller" maxlength="128" autocomplete="off"></label>
    <label><span data-t="monitor_codec"></span><select id="codec"><option value="" data-t="monitor_codec_none"></option></select></label>
    <button id="start" class="primary" data-t="monitor_start_mic"></button>
    <button id="stop" data-t="monitor_stop" disabled></button>
  </div>
  <div class="row">
    <label><span data-t="monitor_file"></span><input id="file" type="file" accept="audio/*"></label>
    <button id="play" data-t="monitor_play" disabled></button>
  </div>

  <div id="banner"><strong id="headline"></strong><div id="detail"></div>
    <div id="guidance" hidden>
      <b data-t="monitor_guidance_title"></b>
      <ul>
        <li data-t="monitor_guidance_callback"></li>
        <li data-t="monitor_guidance_question"></li>
        <li data-t="monitor_guidance_report"></li>
      </ul>
    </div>
  </div>
  <div class="stats">
    <div class="stat"><span data-t="monitor_elapsed"></span><b id="elapsed">–</b></div>
    <div class="stat"><span data-t="monitor_window"></span><b id="window">–</b></div>
    <div class="stat"><span data-t="monitor_running"></span><b id="running">–</b></div>
    <div class="stat"><span data-t="monitor_threshold"></span><b id="threshold">–</b></div>
  </div>
  <canvas id="chart" width="940" height="260" aria-label="score chart"></canvas>
  <div class="legend">
    <span><i class="key" style="background:var(--window)"></i><span data-t="monitor_window"></span></span>
    <span><i class="key" style="background:var(--running)"></i><span data-t="monitor_running"></span></span>
    <span><i class="key" style="background:var(--thr)"></i><span data-t="monitor_threshold"></span></span>
    <span class="muted" data-t="monitor_seconds"></span>
  </div>
  <p id="langnote" hidden></p>
  <div id="final" class="muted"></div>
  <p id="scorenote" class="muted" hidden data-t="monitor_score_note"></p>
  <p id="reportlinks" hidden><a id="report" href="#" target="_blank" rel="noopener" data-t="monitor_report"></a>
    · <a id="reportdl" href="#" download data-t="monitor_report_download"></a></p>

  <section id="tx">
    <h2 data-t="monitor_tx_title"></h2>
    <p class="muted" data-t="monitor_tx_help"></p>
    <div class="row">
      <label><span data-t="monitor_tx_amount"></span><input id="amount" type="number" min="1" step="1" value="75000"></label>
      <label><span data-t="monitor_tx_beneficiary"></span><input id="beneficiary" maxlength="64" value="A/c XXXX1234"></label>
      <button id="approve" class="primary" data-t="monitor_tx_approve" disabled></button>
    </div>
    <div id="decision" hidden>
      <strong id="decision-head"></strong>
      <ul id="decision-reasons"></ul>
      <div id="decision-steps" hidden>
        <label><span data-t="monitor_tx_verifier"></span><input id="verifier" maxlength="64" autocomplete="off"></label>
        <div id="step-buttons" class="row"></div>
        <p id="decision-error" hidden></p>
        <p class="muted" data-t="monitor_tx_note"></p>
      </div>
    </div>
  </section>
</main>
<script>
"use strict";
const STRINGS = __STRINGS__;
const FRAME_S = 0.25;
const FLOOR = 1e-4;
let lang = "en";
try { lang = localStorage.getItem("voxguard-lang") || "en"; } catch (e) { /* private mode */ }
if (!STRINGS[lang]) lang = "en";

const $ = (id) => document.getElementById(id);
function t(key, vars) {
  let s = STRINGS[lang][key] || STRINGS.en[key] || key;
  for (const [k, v] of Object.entries(vars || {})) s = s.split("{" + k + "}").join(String(v));
  return s;
}

const state = { ws: null, ctx: null, media: null, source: null, pending: [], pendingN: 0,
                points: [], threshold: null, flagAt: null, alerted: false, last: null,
                unsupported: false, languageName: null,
                minWindows: 2, caution: 1.5, alertFactor: 1.5,
                banner: ["", "monitor_status_idle", null], final: null, seconds: 0 };

function applyLanguage() {
  document.documentElement.lang = lang;
  for (const el of document.querySelectorAll("[data-t]")) el.textContent = t(el.dataset.t);
  document.title = t("monitor_title");
  $("lang").value = lang;
  showBanner(...state.banner);
  renderFinal();
  renderDecision();
  renderLangNote();
}

function showBanner(kind, headKey, detail) {
  state.banner = [kind, headKey, detail];
  $("banner").className = kind;
  $("guidance").hidden = !state.alerted;  // what to do, once an alert has been raised
  $("headline").textContent = headKey ? t(headKey, detail && detail.vars) : "";
  $("detail").textContent = detail && detail.key ? t(detail.key, detail.vars) : (detail && detail.text) || "";
}

function setBusy(running) {
  $("start").disabled = running;
  $("play").disabled = running || !$("file").files.length;
  $("stop").disabled = !running;
  $("codec").disabled = running;
  $("caller").disabled = running;
}

async function loadCodecs() {
  try {
    const r = await fetch("/v1/codecs");
    const info = await r.json();
    for (const c of info.codecs) {
      if (!c.available) continue;
      const o = document.createElement("option");
      o.value = c.name; o.textContent = c.name;
      $("codec").appendChild(o);
    }
  } catch (e) { /* codecs are optional */ }
}

const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor("voxguard-tap", Tap);`;

async function start(kind) {
  if (!window.AudioWorkletNode || !navigator.mediaDevices || !window.WebSocket) {
    showBanner("alert", "monitor_unsupported", null); return;
  }
  reset();
  setBusy(true);
  showBanner("info", "monitor_status_connecting", null);
  try {
    state.ctx = new AudioContext();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await state.ctx.audioWorklet.addModule(url);
    const tap = new AudioWorkletNode(state.ctx, "voxguard-tap",
      { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit",
        channelInterpretation: "speakers" });  // mono downmix by the browser
    const mute = state.ctx.createGain(); mute.gain.value = 0;
    tap.connect(mute).connect(state.ctx.destination);  // keeps the worklet running
    tap.port.onmessage = (e) => queue(e.data);
    if (kind === "mic") {
      state.media = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      state.source = state.ctx.createMediaStreamSource(state.media);
    } else {
      const buffer = await state.ctx.decodeAudioData(await $("file").files[0].arrayBuffer());
      state.source = state.ctx.createBufferSource();
      state.source.buffer = buffer;
      state.source.connect(state.ctx.destination);  // the audience hears the call
      state.source.onended = () => stop();
    }
    state.source.connect(tap);
    openSocket();
    if (kind !== "mic") state.source.start();
  } catch (e) {
    const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
    teardown();
    showBanner("alert", denied ? "monitor_mic_denied" : "monitor_error",
               denied ? null : { vars: { detail: String(e && e.message || e) } });
    setBusy(false);
  }
}

function openSocket() {
  const q = new URLSearchParams({ sample_rate: String(state.ctx.sampleRate) });
  if ($("caller").value.trim()) q.set("caller_id", $("caller").value.trim());
  if ($("codec").value) q.set("codec", $("codec").value);
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/v1/stream?${q}`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => {
    if (!state.final && state.ws === ws) showBanner("alert", "monitor_disconnected", null);
    teardown(); setBusy(false);
  };
  state.ws = ws;
}

function queue(frame) {
  state.pending.push(frame); state.pendingN += frame.length;
  if (state.pendingN >= state.ctx.sampleRate * FRAME_S) flush();
}

function flush() {  // send what has been captured so far as one binary frame
  if (!state.pendingN) return;
  const out = new Float32Array(state.pendingN);
  let i = 0; for (const f of state.pending) { out.set(f, i); i += f.length; }
  state.pending = []; state.pendingN = 0;
  state.seconds += out.length / state.ctx.sampleRate;
  $("elapsed").textContent = state.seconds.toFixed(1) + " s";
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(out.buffer);
}

function onMessage(m) {
  if (m.type === "ready") {
    state.sessionId = m.session_id; $("approve").disabled = false;
    state.threshold = m.threshold; state.minWindows = m.min_alert_windows; state.caution = m.caution_factor;
    state.alertFactor = m.alert_factor;
    $("threshold").textContent = m.threshold.toFixed(6);
    showBanner("info", "monitor_status_listening", null);
  } else if (m.type === "score") {
    state.points.push(m); state.last = m;
    $("window").textContent = m.window_score.toFixed(6);
    $("running").textContent = m.running_score.toFixed(6);
    showRisk(m);
    draw();
  } else if (m.type === "alert") {
    state.alerted = true; state.flagAt = m.t_s;
    showRisk(state.last);
    draw();
  } else if (m.type === "language") {
    // speech clearly not in English or Hindi: no verdict, advise a call-back (never "safe")
    state.languageName = m.language_name || m.language;
    if (m.language_supported === false) { state.unsupported = true; renderLangNote(); }
  } else if (m.type === "busy") {
    $("final").textContent = t("monitor_busy");
  } else if (m.type === "final") {
    state.final = m; renderFinal(); teardown(); setBusy(false);
  } else if (m.type === "error") {
    state.final = { error: m.detail }; renderFinal();
    showBanner("alert", "monitor_error", { vars: { detail: m.detail } });
  }
}

function renderLangNote() {
  // a note next to the risk banner, which is unchanged: an alert still fires when the risk is high
  $("langnote").hidden = !state.unsupported;
  if (state.unsupported) {
    $("langnote").textContent = t("monitor_lang_unsupported", { language: state.languageName }) +
      " " + t("monitor_lang_unsupported_advice");
  }
}

function showRisk(m) {
  // graded: "checking" until enough windows; "elevated" over the threshold but below the alert
  // level (no alert: genuine calls often sit here); after an alert, caution or alert
  const over = m.running_score >= state.threshold;
  if (!state.alerted) {
    if (over && m.windows < state.minWindows) showBanner("info", "monitor_checking", null);
    else if (over) showBanner("info", "monitor_elevated", null);
    else showBanner("safe", "monitor_safe", null);
  } else if (!over) {
    showBanner("info", "monitor_caution", { key: "monitor_alert_cleared", vars: { seconds: state.flagAt.toFixed(1) } });
  } else if (m.running_score < state.caution * state.threshold) {
    showBanner("caution", "monitor_caution", { key: "monitor_caution_advice" });
  } else {
    showBanner("alert", "monitor_alert", { key: "monitor_alert_advice" });
  }
}

function renderFinal() {
  const f = state.final;
  $("reportlinks").hidden = !(f && f.stored);
  $("scorenote").hidden = !(f && f.stored);  // why an upload of the same file scores a little differently
  if (f && f.stored) {
    const url = `/v1/sessions/${encodeURIComponent(f.session_id)}/report?lang=${lang}`;
    $("report").href = url;
    $("reportdl").href = `${url}&download=true`;
  }
  if (!f) { $("final").textContent = ""; return; }
  if (f.error) { $("final").textContent = t("monitor_error", { detail: f.error }); return; }
  if (!f.stored && f.windows === 0) { $("final").textContent = t("monitor_final_none"); return; }
  const vars = { score: f.fake_score.toFixed(6), threshold: f.threshold.toFixed(6), session: f.session_id };
  let text = t(f.flagged ? "monitor_final_flagged" : "monitor_final_clear", vars);
  // the two-window rule gates only the live alert: a short call can be flagged with none
  if (f.flagged && f.first_flag_s === null) text += " " + t("monitor_final_no_live_alert");
  $("final").textContent = text;
}

async function approve() {
  if (!state.sessionId) { showDecisionError(t("monitor_tx_need_call")); return; }
  const body = { session_id: state.sessionId, amount_inr: Number($("amount").value),
                 beneficiary: $("beneficiary").value.trim() || null };
  await postDecision("/v1/transactions/check", body);
}

async function verify(method) {
  const name = $("verifier").value.trim();
  if (!name) { $("verifier").focus(); return; }
  await postDecision(`/v1/transactions/${encodeURIComponent(state.decision.decision_id)}/verify`,
                     { method, verified_by: name });
}

async function postDecision(url, body) {
  try {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body) });
    const d = await r.json();
    const detail = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    if (!r.ok && state.decision && !state.decision.error && url.endsWith("/verify")) {
      $("decision-error").textContent = t("monitor_error", { detail }); $("decision-error").hidden = false;
      return;  // a refused step (e.g. the same person twice): keep the decision on screen
    }
    if (!r.ok) { showDecisionError(detail); return; }
    state.decision = d; renderDecision();
  } catch (e) { showDecisionError(String(e && e.message || e)); }
}

function showDecisionError(text) {
  state.decision = { error: text }; renderDecision();
}

function renderDecision() {
  const d = state.decision, box = $("decision");
  box.hidden = !d;
  if (!d) return;
  $("decision-error").hidden = true;
  const reasons = $("decision-reasons"); reasons.replaceChildren();
  const buttons = $("step-buttons"); buttons.replaceChildren();
  if (d.error) {
    box.className = "held"; $("decision-head").textContent = t("monitor_error", { detail: d.error });
    $("decision-steps").hidden = true; return;
  }
  box.className = d.status;
  const head = { allowed: "monitor_tx_allowed", pending: "monitor_tx_step_up", held: "monitor_tx_held",
                 released: "monitor_tx_released" }[d.status];
  $("decision-head").textContent = `${t(head)} (${t("tx_tier")}: ${t("tx_tier_" + d.tier)}, ₹${d.amount_inr.toLocaleString("en-IN")})`;
  for (const code of d.reasons) {
    const li = document.createElement("li"); li.textContent = t("tx_reason_" + code); reasons.appendChild(li);
  }
  const open = d.status === "pending" || d.status === "held";
  $("decision-steps").hidden = !open;
  if (!open) return;
  for (const m of d.required_steps) {
    const b = document.createElement("button");
    const done = d.completed_steps.includes(m);
    b.textContent = (done ? "✓ " : "") + t("monitor_tx_mark") + ": " + t("tx_method_" + m);
    b.disabled = done;
    b.addEventListener("click", () => verify(m));
    buttons.appendChild(b);
  }
}

function stop() {
  flush();
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "end" }));
  stopAudio();
}

function stopAudio() {
  try { if (state.source && state.source.stop) state.source.stop(); } catch (e) { /* already stopped */ }
  if (state.media) for (const tr of state.media.getTracks()) tr.stop();
  if (state.ctx && state.ctx.state !== "closed") state.ctx.close();
  state.media = null; state.source = null;
}

function teardown() { stopAudio(); }

function reset() {
  Object.assign(state, { points: [], threshold: null, flagAt: null, alerted: false, last: null,
                         unsupported: false, languageName: null,
                         final: null, seconds: 0, pending: [], pendingN: 0, sessionId: null,
                         decision: null });
  $("approve").disabled = true; renderDecision(); $("reportlinks").hidden = true; $("scorenote").hidden = true; $("langnote").hidden = true;
  for (const id of ["elapsed", "window", "running", "threshold"]) $(id).textContent = "–";
  $("final").textContent = "";
  draw();
}

function draw() {
  const c = $("chart"), g = c.getContext("2d");
  const w = c.width = c.clientWidth * devicePixelRatio, h = c.height = c.clientHeight * devicePixelRatio;
  g.clearRect(0, 0, w, h);
  const pad = { l: 56 * devicePixelRatio, r: 12 * devicePixelRatio, t: 10 * devicePixelRatio, b: 28 * devicePixelRatio };
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const values = state.points.flatMap((p) => [p.window_score, p.running_score]);
  if (state.threshold !== null) values.push(state.threshold);
  const lo = Math.max(Math.min(...values, 0.01), FLOOR) / 1.5, hi = Math.min(Math.max(...values, 0.02) * 1.5, 1);
  const tMax = Math.max(10, ...state.points.map((p) => p.t_s));
  const x = (s) => pad.l + (w - pad.l - pad.r) * s / tMax;
  const y = (v) => { const a = Math.log10(Math.max(v, FLOOR)), l = Math.log10(lo), u = Math.log10(hi);
                     return h - pad.b - (h - pad.t - pad.b) * (a - l) / (u - l); };
  g.font = `${12 * devicePixelRatio}px system-ui, sans-serif`; g.fillStyle = css("--muted"); g.strokeStyle = css("--line");
  g.lineWidth = devicePixelRatio;
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (const m of [1, 2, 5]) {  // gridlines and labels at 1, 2 and 5 in each decade
      const v = m * 10 ** e; if (v < lo || v > hi) continue;
      g.beginPath(); g.moveTo(pad.l, y(v)); g.lineTo(w - pad.r, y(v)); g.stroke();
      g.fillText(v >= 0.01 ? String(+v.toFixed(2)) : v.toExponential(0), 4 * devicePixelRatio, y(v) + 4 * devicePixelRatio);
    }
  }
  const step = tMax > 60 ? 10 : tMax > 20 ? 5 : 2;
  for (let s = 0; s <= tMax; s += step) g.fillText(String(s), x(s) - 4 * devicePixelRatio, h - 8 * devicePixelRatio);
  if (state.threshold !== null) {
    g.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]); g.strokeStyle = css("--thr");
    g.beginPath(); g.moveTo(pad.l, y(state.threshold)); g.lineTo(w - pad.r, y(state.threshold)); g.stroke();
    g.setLineDash([]);
  }
  if (state.flagAt !== null) {
    g.setLineDash([2 * devicePixelRatio, 2 * devicePixelRatio]); g.strokeStyle = css("--flag");
    g.beginPath(); g.moveTo(x(state.flagAt), pad.t); g.lineTo(x(state.flagAt), h - pad.b); g.stroke();
    g.setLineDash([]);
  }
  for (const [key, color] of [["window_score", css("--window")], ["running_score", css("--running")]]) {
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2 * devicePixelRatio; g.beginPath();
    state.points.forEach((p, i) => { const px = x(p.t_s), py = y(p[key]); i ? g.lineTo(px, py) : g.moveTo(px, py); });
    g.stroke();
    for (const p of state.points) { g.beginPath(); g.arc(x(p.t_s), y(p[key]), 3 * devicePixelRatio, 0, 7); g.fill(); }
  }
}

$("lang").addEventListener("change", (e) => {
  lang = e.target.value;
  try { localStorage.setItem("voxguard-lang", lang); } catch (err) { /* private mode */ }
  applyLanguage();
});
$("start").addEventListener("click", () => start("mic"));
$("play").addEventListener("click", () => start("file"));
$("stop").addEventListener("click", () => stop());
$("approve").addEventListener("click", () => approve());
$("file").addEventListener("change", () => setBusy(!!state.ws && state.ws.readyState === WebSocket.OPEN));
window.addEventListener("resize", draw);
applyLanguage(); loadCodecs(); draw();
</script>
</body>
</html>
"""
