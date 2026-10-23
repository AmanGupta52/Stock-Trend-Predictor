/* Live view: price refreshed every few seconds; the quick model is trained once, then only the new price is added. */
(() => {
  const $ = id => document.getElementById(id);
  const { esc, num, pct, cls, moneyC, dirSym, dirCls, jobRun } = ST;
  let sym = null, cfgPoll = 5, timer = null, paused = false, running = false, inflight = false, chart = null;
  let ticks = [];                    // {t, p} collected in this browser session
  let failures = 0, startedAt = 0;

  const show = (el, msg) => { el.hidden = !msg; el.textContent = msg || ""; };
  const hhmmss = d => d.toLocaleTimeString([], { hour12: false });

  function stop(msg) {
    running = false; clearTimeout(timer); timer = null;
    $("lvPause").disabled = true; $("lvPause").textContent = "Pause";
    if (msg) show($("lvMsg"), msg);
  }

  async function start(e) {
    e.preventDefault();
    stop(); show($("lvMsg"), ""); paused = false; ticks = []; failures = 0;
    const query = $("lvInput").value.trim(); if (!query) return;
    $("lvGo").disabled = true; $("lvGo").textContent = "Training…"; $("lvOut").hidden = true; $("lvStatus").hidden = false;
    try {
      const res = await jobRun({ query }, (p, m) => { $("lvBar").style.width = Math.round(p * 100) + "%"; $("lvStatusMsg").textContent = `${Math.round(p * 100)}% — ${m}`; }, "/api/live/start");
      sym = res.symbol; cfgPoll = res.poll_seconds || 5; startedAt = Date.now();
      renderAccuracy(res);
      $("lvOut").hidden = false; running = true;
      $("lvPause").disabled = false;
      await tick();
    } catch (err) { show($("lvMsg"), err.message); }
    finally { $("lvStatus").hidden = true; $("lvGo").disabled = false; $("lvGo").textContent = "Start live"; }
  }

  function schedule(afterMs) {
    clearTimeout(timer);
    if (!running || paused) return;
    timer = setTimeout(tick, Math.max(0, afterMs));
  }

  async function tick() {
    if (!running || paused || inflight || document.hidden) { schedule(cfgPoll * 1000); return; }   // no polling in a hidden tab
    inflight = true; const t0 = Date.now();
    try {
      const r = await fetch("/api/live/tick?symbol=" + encodeURIComponent(sym));
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        if (j.error?.code === "not_trained") { stop("The server restarted, so the quick model was lost. Press Start live again."); return; }
        throw new Error(j.error?.message || "Request failed");
      }
      failures = 0; show($("lvMsg"), ""); render(j);
    } catch (err) {
      failures++;
      show($("lvMsg"), (err.message === "Failed to fetch" ? "Cannot reach the local server. Is app.py still running?" : err.message) + ` (retrying every ${cfgPoll}s)`);
    } finally { inflight = false; }
    schedule(cfgPoll * 1000 - (Date.now() - t0));        // a slow response never causes overlapping requests
  }

  function render(d) {
    const cur = d.currency, p = d.prediction;
    ticks.push({ t: hhmmss(new Date()), p: d.price }); if (ticks.length > 240) ticks.shift();
    const age = d.quote_age_minutes;
    const ageTxt = age == null ? "" : age < 1.5 ? " · just now" : ` · ${Math.round(age)} min old`;
    $("lvPrice").innerHTML = `<div class="sub">${esc(d.name)} (${esc(d.symbol)})</div>
      <div class="price">${moneyC(d.price, cur)}</div>
      <div class="${cls(d.change_pct)}">${d.change_abs >= 0 ? "+" : ""}${num(d.change_abs, 2)} (${pct(d.change_pct)}) <span class="mini">${esc(d.change_label)}</span></div>`;
    $("lvMeta").innerHTML = `Price time: <b>${esc(new Date(d.quote_time).toLocaleString([], { hour12: false }))}</b>${ageTxt}<br>
      Last completed close: ${moneyC(d.last_close, cur)} (${esc(d.last_close_date)}) · ${d.session_in_progress ? "session in progress" : "no newer session than last close"}<br>
      ${d.quote_stale ? '<span class="down">This price is old — the market may be closed or the feed delayed.</span><br>' : ""}${d.feed_error ? '<span class="down">Price feed error — showing the last good quote.</span><br>' : ""}
      <span class="mini">${esc(d.freshness)}</span>`;
    $("lvPred").innerHTML = `<div class="sub">Next session (${esc(p.target_date)}) — ${esc(p.model)}</div>
      <div class="price">${moneyC(p.pred_close, cur)}</div>
      <div class="${dirCls(p.direction)}">${dirSym(p.direction)} ${pct(p.change_pct)} <span class="mini">vs current price</span></div>
      <div class="sub">80% range ${moneyC(p.low, cur)} – ${moneyC(p.high, cur)}</div>
      <div class="mini">${d.recomputed ? "Recomputed for the new price" : "Price unchanged — cached forecast"} in ${d.compute_ms} ms · model trained through ${esc(d.model.trained_through)} and not retrained · refresh #${d.model.ticks_since_training}</div>
      <div class="mini">An estimate, not advice or a guarantee.</div>`;
    drawChart(d);
  }

  function drawChart(d) {
    if (chart) { try { chart.ro.disconnect(); } catch {} chart = null; }
    if (ticks.length < 2) { $("lvChart").innerHTML = '<p class="sub">Collecting prices…</p>'; return; }
    chart = new XYChart($("lvChart"), { x: ticks.map(t => t.t), series: [{ name: d.symbol, y: ticks.map(t => t.p), color: ST.COLORS[0], width: 1.8 }],
      height: 200, xFmt: v => v, tipFmt: v => moneyC(v, d.currency), ariaLabel: "Live price" });
  }

  function renderAccuracy(res) {
    const m = res.metrics, cur = res.currency, P = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
    const rows = [
      ["Model chosen (on validation inside the first 80 %)", esc(res.model)],
      ["Training period (first 80 %)", `${esc(m.train_start)} → ${esc(m.train_end)} (${res.n_train} days)`],
      ["Test period (last 20 %, never trained on)", `${esc(m.test_start)} → ${esc(m.test_end)} (${m.n} days)`],
      ["Mean absolute error (next-day close)", moneyC(m.mae, cur)],
      ["Error of “tomorrow = today” baseline", moneyC(m.naive_mae, cur)],
      ["Model error ÷ baseline error (below 1 = better)", m.rel_mae == null ? "–" : num(m.rel_mae, 3)],
      ["RMSE", moneyC(m.rmse, cur)], ["MAPE", num(m.mape, 2) + "%"],
      ["Direction accuracy (up/down)", `${P(m.directional_accuracy)} <span class="mini">(always guessing “up” would score ${P(m.up_day_rate)})</span>`],
      ["80% range actually contained the close", P(m.coverage_80)],
      ["Quick model training time", res.train_seconds + " s"]];
    $("lvAcc").innerHTML = "<table><tbody>" + rows.map(([a, b]) => `<tr><td>${a}</td><td class="num">${b}</td></tr>`).join("") + "</tbody></table>";
    $("lvVerdict").innerHTML = `<b class="${m.beats_naive ? "up" : "down"}">${esc(m.verdict)}</b> Past accuracy does not guarantee future results.`;
  }

  $("lvForm").addEventListener("submit", start);
  $("lvPause").onclick = () => {
    paused = !paused; $("lvPause").textContent = paused ? "Resume" : "Pause";
    if (!paused) tick(); else clearTimeout(timer);
  };
  document.addEventListener("visibilitychange", () => { if (!document.hidden && running && !paused) tick(); });
  document.addEventListener("viewchange", e => {           // stop polling when the user leaves the Live tab
    if (e.detail !== "live" && running) { paused = true; $("lvPause").textContent = "Resume"; clearTimeout(timer); }
  });
  document.addEventListener("viewchange", e => { if (e.detail === "live" && ST.sym && !running) $("lvInput").value = ST.sym; });
})();
