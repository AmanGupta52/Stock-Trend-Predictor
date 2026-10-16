(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const CUR = { INR: "₹", USD: "$", EUR: "€", GBP: "£", JPY: "¥", CAD: "C$", AUD: "A$", HKD: "HK$" };
  const COLORS = ["var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
  let R = null, charts = {}, pollTimer = null, evalSplit = "test", evalH = "1", curSym = "";

  const money = v => v == null ? "–" : (CUR[R?.meta.currency] ?? (R?.meta.currency ? R.meta.currency + " " : "")) + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const pct = (v, d = 2) => v == null ? "–" : (v > 0 ? "+" : "") + v.toFixed(d) + "%";
  const num = (v, d = 2) => v == null ? "–" : v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const cls = v => v > 0.05 ? "up" : v < -0.05 ? "down" : "flat";
  const dirSym = d => d === "Up" ? "▲ Up" : d === "Down" ? "▼ Down" : "◆ Flat";
  const dirCls = d => d === "Up" ? "up" : d === "Down" ? "down" : "flat";
  const fdate = s => new Date(s + "T00:00:00").toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  const fts = iso => { if (iso && iso.length === 10) return fdate(iso); try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; } };

  // ---------------- search + suggestions
  const q = $("q"), sug = $("suggest");
  let sugTimer = null, sugSeq = 0;
  q.addEventListener("input", () => {
    clearTimeout(sugTimer);
    const v = q.value.trim();
    if (v.length < 2) { sug.hidden = true; return; }
    sugTimer = setTimeout(async () => {
      const seq = ++sugSeq;
      try {
        const r = await (await fetch("/api/search?q=" + encodeURIComponent(v))).json();
        if (seq !== sugSeq) return;
        sug.innerHTML = "";
        (r.results || []).forEach(x => {
          const li = document.createElement("li"); li.setAttribute("role", "option");
          li.innerHTML = `<span><b>${esc(x.symbol)}</b> ${esc(x.name)}</span><small>${esc(x.exchange)}</small>`;
          li.onmousedown = e => { e.preventDefault(); q.value = x.symbol; sug.hidden = true; run(false); };
          sug.appendChild(li);
        });
        sug.hidden = !sug.children.length;
      } catch { sug.hidden = true; }
    }, 300);
  });
  q.addEventListener("blur", () => setTimeout(() => sug.hidden = true, 150));
  q.addEventListener("keydown", e => { if (e.key === "Escape") sug.hidden = true; });
  $("searchForm").addEventListener("submit", e => { e.preventDefault(); sug.hidden = true; run(false); });
  $("refresh").addEventListener("click", () => run(true));

  function setBusy(b) { ["go", "refresh"].forEach(i => $(i).disabled = b || (i === "refresh" && !curSym)); $("go").textContent = b ? "Working…" : "Analyze"; }
  function showError(msg) { const e = $("error"); e.hidden = !msg; e.textContent = msg || ""; }
  function setStatus(p, m) { $("status").hidden = p == null; if (p != null) { $("barfill").style.width = Math.round(p * 100) + "%"; $("statusMsg").textContent = `${Math.round(p * 100)}% — ${m}`; } }

  async function run(refresh) {
    const query = q.value.trim();
    if (!query) { showError("Enter a company name or stock symbol."); return; }
    clearInterval(pollTimer); showError(""); setBusy(true); setStatus(0.01, "Starting…");
    $("warnings").hidden = true;
    try {
      const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, period: $("period").value, refresh, retrain: $("retrain").checked }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error?.message || "Request failed");
      poll(j.job_id);
    } catch (e) { setBusy(false); setStatus(null); showError(e.message === "Failed to fetch" ? "Cannot reach the local server. Is app.py still running?" : e.message); }
  }
  function poll(id) {
    let fails = 0;
    pollTimer = setInterval(async () => {
      try {
        const j = await (await fetch("/api/jobs/" + id)).json(); fails = 0;
        if (j.error) { clearInterval(pollTimer); setBusy(false); setStatus(null); showError(j.error.message); return; }
        setStatus(j.progress, j.message);
        if (j.status === "done") { clearInterval(pollTimer); setBusy(false); setStatus(null); render(j.result); }
        if (j.status === "error") { clearInterval(pollTimer); setBusy(false); setStatus(null); showError(j.error.message); }
      } catch { if (++fails > 5) { clearInterval(pollTimer); setBusy(false); setStatus(null); showError("Lost connection to the local server."); } }
    }, 700);
  }

  // ---------------- render
  function render(res) {
    R = res; curSym = res.meta.symbol; $("refresh").disabled = false;
    $("empty").hidden = true; $("results").hidden = false;
    const w = $("warnings");
    const all = [...(res.warnings || [])];
    if (res.run?.models_from_cache) all.push(res.run.note);
    w.hidden = !all.length; w.innerHTML = all.length ? "<ul>" + all.map(x => `<li>${esc(x)}</li>`).join("") + "</ul>" : "";
    hero(); kpis(); priceCharts(); indicatorCharts(); compare(); evaluation(); backtest(); trend();
    $("disclaimer").textContent = res.disclaimer;
    document.title = `${res.meta.symbol} — Stock Trend Predictor`;
  }

  function hero() {
    const { meta, data } = R, qt = data.latest_quote;
    const shown = qt ? qt.price : data.last_close;
    $("hero").innerHTML = `
      <div><h2>${esc(meta.name)}</h2>
        <span class="tag"><b>${esc(meta.symbol)}</b></span><span class="tag">${esc(meta.exchange)}</span>
        <span class="tag">${esc(meta.currency || "currency n/a")}</span>${meta.instrument_type ? `<span class="tag">${esc(meta.instrument_type)}</span>` : ""}
        ${meta.timezone ? `<span class="tag">${esc(meta.timezone)}</span>` : ""}
        <p class="meta">${data.interpreted_from ? `Matched from “${esc(data.interpreted_from)}”. ` : ""}${altHtml()}</p></div>
      <div><div class="price">${money(shown)}</div>
        <div class="chg ${cls(data.day_change_pct)}">${pct(data.day_change_pct)} <span class="flat" style="font-weight:400">${esc(data.day_change_label || "vs prior close")}</span></div>
        <p class="meta">${qt ? `Latest available price (${esc(qt.kind)}): <b>${money(qt.price)}</b> at ${esc(fts(qt.time))} (delayed, not real-time)<br>` : ""}
        Last completed daily close: <b>${money(data.last_close)}</b> on ${esc(fdate(data.last_close_date))}<br>
        ${esc(data.freshness)}<br>
        Source: ${esc(data.provider)} · downloaded ${esc(fts(data.fetched_at))}${data.from_data_cache ? " (cached)" : ""} · ${data.rows} sessions (${esc(data.start)} → ${esc(data.end)})</p></div>`;
    $("hero").querySelectorAll("[data-alt]").forEach(a => a.onclick = () => { q.value = a.dataset.alt; run(false); });
  }
  function altHtml() {
    const a = R.data.alternatives || []; if (!a.length) return "";
    return "Other matches: " + a.slice(0, 4).map(x => `<a href="#" data-alt="${esc(x.symbol)}" onclick="return false">${esc(x.symbol)} (${esc(x.exchange)})</a>`).join(", ") + ".";
  }

  const F = () => R.forecast;
  const sel = () => $("modelSel").value || F().selected;

  function kpis() {
    const f = F(), m = f.models, s = m[f.selected];
    const mlN = f.best_ml, dlN = f.best_dl;
    const rows = k => [mlN, dlN].filter(Boolean).map(n => { const b = m[n][k]; return `<tr${n === f.selected ? ' class="best"' : ""}><td>${esc(m[n].label)}${n === f.selected ? " ★" : ""}</td><td class="num">${money(b.pred_close)}</td><td class="num ${cls(b.change_pct)}">${pct(b.change_pct)}</td><td class="${dirCls(b.direction)}">${dirSym(b.direction)}</td></tr>`; }).join("");
    const card = (title, k) => { const b = s[k]; return `<div class="card kpi"><h2>${title} — ${esc(fdate(b.date))}</h2>
      <div class="big">${money(b.pred_close)}</div>
      <div class="row"><span class="${dirCls(b.direction)}"><b>${dirSym(b.direction)}</b> ${pct(b.change_pct)}</span>
        <span>${esc(f.interval_label)}: <b>${money(b.low)}</b> – <b>${money(b.high)}</b></span>
        <span>P(up) ≈ <b>${num(b.prob_up, 0)}%</b></span></div>
      <table><thead><tr><th>ML vs DL</th><th>Predicted</th><th>Change</th><th>Direction</th></tr></thead><tbody>${rows(k)}</tbody></table>
      ${k === "next_day" ? `<p class="meta">Selected model: <b>${esc(s.label)}</b> (lowest validation error vs naive). Reliability on unseen data: <span class="pill ${f.reliability === "Moderate" ? "up" : f.reliability === "Low" ? "flat" : "down"}">${esc(f.reliability)}</span></p>` :
        `<p class="meta">Estimated month-end close after ${f.horizon} trading sessions, applied to the latest close of ${money(f.base_close)} (${esc(f.as_of)}). Estimate only — not a guaranteed outcome.</p>`}</div>`; };
    $("kpis").innerHTML = card(f.next_session_in_progress ? "Today's close (session in progress)" : "Next trading day", "next_day") + card("Next month (≈21 sessions)", "month_end");
  }

  // ---------------- price + volume
  function priceCharts() {
    const h = R.history, f = F(), n = h.dates.length;
    const ms = $("modelSel"); ms.innerHTML = Object.entries(f.models).map(([k, v]) => `<option value="${k}">${esc(v.label)}${k === f.selected ? " ★ selected" : ""}</option>`).join("");
    ms.value = f.selected; ms.onchange = drawPrice;
    $("rangeLabel").textContent = f.interval_label;
    const rg = $("ranges"); const opts = [["1M", 21], ["3M", 63], ["6M", 126], ["1Y", 252], ["3Y", 756], ["All", 1e9]];
    rg.innerHTML = ""; opts.forEach(([l, d], i) => { const b = document.createElement("button"); b.type = "button"; b.textContent = l; b.className = i === 3 ? "on" : ""; b.onclick = () => { rg.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); setRange(d); }; rg.appendChild(b); });
    drawPrice(); setRange(252);
  }
  function setRange(d) {
    const n = R.history.dates.length, fl = F().horizon, a = Math.max(0, n - Math.min(d, n));
    charts.price?.setRange(a, n + fl - 1); charts.vol?.setRange(a, n + fl - 1);
  }
  function drawPrice() {
    const h = R.history, f = F(), m = f.models[sel()], n = h.dates.length;
    const X = [...h.dates, ...m.path.dates], pad = a => [...a, ...Array(m.path.dates.length).fill(null)];
    const nulls = Array(n - 1).fill(null), last = h.close[n - 1];
    const series = [
      { name: "Actual close", y: pad(h.close), color: "var(--c1)", width: 1.7 },
      { name: "SMA 20", y: pad(h.sma_20), color: "var(--c3)", width: 1.2, hidden: true },
      { name: "SMA 50", y: pad(h.sma_50), color: "var(--c6)", width: 1.2, hidden: false },
      { name: "SMA 200", y: pad(h.sma_200), color: "var(--c4)", width: 1.2, hidden: true },
      { name: "Bollinger upper", y: pad(h.bb_up), color: "var(--muted)", width: 1, dash: "3 3", hidden: true },
      { name: "Bollinger lower", y: pad(h.bb_low), color: "var(--muted)", width: 1, dash: "3 3", hidden: true },
      { type: "band", color: "var(--c2)", lo: [...nulls, last, ...m.path.low], hi: [...nulls, last, ...m.path.high], opacity: .2, tipLabel: f.interval_label },
      { name: "Predicted close (" + m.label + ")", y: [...nulls, last, ...m.path.pred], color: "var(--c2)", width: 2.2, dash: "6 4" },
    ];
    const lastDate = h.dates[n - 1];
    const o = { x: X, series, height: 340, markers: [{ x: lastDate, label: "Forecast →" }], shade: { from: n - 1, to: X.length - 1 }, yFmt: v => v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(1), tipFmt: v => v.toLocaleString(undefined, { maximumFractionDigits: 2 }), ariaLabel: "Price history and forecast" };
    if (charts.price) { const keep = charts.price.o.series.map(s => s.hidden); charts.price.o.series = series; series.forEach((s, i) => { if (keep[i] != null) s.hidden = keep[i]; }); charts.price.update({ x: X, markers: o.markers, shade: o.shade }); }
    else charts.price = new XYChart($("priceChart"), o);
    const vcol = h.close.map((c, i) => i && c >= h.close[i - 1] ? "var(--up)" : "var(--down)");
    const vo = { x: X, series: [{ name: "Volume", type: "bar", y: pad(h.volume), colors: vcol, color: "var(--c2)", opacity: .55 }], height: 110, yZero: true, shade: o.shade, yFmt: v => v >= 1e7 ? (v / 1e7).toFixed(1) + "Cr" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "k" : v, tipFmt: v => v.toLocaleString(), ariaLabel: "Trading volume" };
    if (!charts.vol) charts.vol = new XYChart($("volChart"), vo); else charts.vol.update({ x: X, series: vo.series, shade: o.shade });
    charts.vol.legend.hidden = true;
    const act = $("ranges").querySelector(".on"); if (act) act.click();
  }

  function indicatorCharts() {
    const h = R.history, a = Math.max(0, h.dates.length - 252), s = k => h[k].slice(a), x = h.dates.slice(a);
    charts.rsi = new XYChart($("rsiChart"), { x, height: 200, yDomain: [0, 100], yRefLines: [30, 70], yFmt: v => v.toFixed(0), tipFmt: v => v.toFixed(1), series: [{ name: "RSI", y: s("rsi_14"), color: "var(--c4)", width: 1.6 }], ariaLabel: "RSI" });
    const hist = s("macd").map((m, i) => m == null || h.macd_signal[a + i] == null ? null : m - h.macd_signal[a + i]);
    charts.macd = new XYChart($("macdChart"), { x, height: 200, yFmt: v => v.toFixed(1), tipFmt: v => v.toFixed(3), yRefLines: [0], series: [
      { name: "Histogram", type: "bar", y: hist, colors: hist.map(v => v >= 0 ? "var(--up)" : "var(--down)"), color: "var(--up)", opacity: .5 },
      { name: "MACD", y: s("macd"), color: "var(--c6)", width: 1.5 }, { name: "Signal", y: s("macd_signal"), color: "var(--c3)", width: 1.5 }], ariaLabel: "MACD" });
  }

  // ---------------- comparison
  function compare() {
    const f = F(), names = Object.keys(f.models), base = f.base_close;
    const X = [f.as_of, ...f.models[names[0]].path.dates];
    const series = [{ type: "band", color: "var(--c1)", lo: [base, ...f.models[f.selected].path.low], hi: [base, ...f.models[f.selected].path.high], opacity: .09, tipLabel: "Selected model range" }];
    names.forEach((n, i) => series.push({ name: f.models[n].label + (n === f.selected ? " ★" : ""), y: [base, ...f.models[n].path.pred], color: COLORS[i % 5], width: n === f.selected ? 2.6 : 1.5, dash: f.models[n].family === "DL" ? "6 3" : "" }));
    charts.cmp = new XYChart($("cmpChart"), { x: X, series, height: 300, tipFmt: v => v.toLocaleString(undefined, { maximumFractionDigits: 2 }), ariaLabel: "Model forecast comparison" });
    $("cmpTable").innerHTML = `<table><thead><tr><th>Model</th><th>Type</th><th>Next day</th><th>Change</th><th>Month-end</th><th>Change</th><th>Month-end range</th></tr></thead><tbody>` +
      names.map(n => { const m = f.models[n]; return `<tr${n === f.selected ? ' class="best"' : ""}><td>${esc(m.label)}${n === f.selected ? " ★" : ""}</td><td>${m.family}</td><td class="num">${money(m.next_day.pred_close)}</td><td class="num ${cls(m.next_day.change_pct)}">${pct(m.next_day.change_pct)}</td><td class="num">${money(m.month_end.pred_close)}</td><td class="num ${cls(m.month_end.change_pct)}">${pct(m.month_end.change_pct)}</td><td class="num">${money(m.month_end.low)} – ${money(m.month_end.high)}</td></tr>`; }).join("") + "</tbody></table>";
  }

  // ---------------- evaluation
  function evaluation() {
    const E = R.evaluation, f = F();
    const cl = f.reliability === "Moderate" ? "moderate" : f.reliability === "Low" ? "low" : "none";
    $("verdict").innerHTML = `<div class="verdict ${cl}"><b>Reliability: ${esc(f.reliability)}.</b> ${esc(f.reliability_note)}</div>`;
    const tabs = $("evalTabs"); tabs.innerHTML = "";
    [["val", "Validation"], ["test", "Held-out test"]].forEach(([k, l]) => { const b = document.createElement("button"); b.type = "button"; b.textContent = l; b.className = k === evalSplit ? "on" : ""; b.onclick = () => { evalSplit = k; evaluation(); }; tabs.appendChild(b); });
    const hs = document.createElement("select"); hs.setAttribute("aria-label", "Forecast horizon"); E.horizons.forEach(h => { const o = document.createElement("option"); o.value = h; o.textContent = h === 1 ? "Next day (h=1)" : `${h} sessions ahead`; hs.appendChild(o); }); hs.value = evalH; hs.onchange = () => { evalH = hs.value; evaluation(); }; hs.style.marginLeft = "auto"; tabs.appendChild(hs);
    const rows = E[evalSplit], naive = rows.find(r => r.model === "naive").horizons[evalH];
    $("evalTable").innerHTML = `<table><thead><tr><th>Model</th><th>Type</th><th>MAE (price)</th><th>RMSE (price)</th><th>MAPE</th><th>Direction acc.</th><th>MAE vs naive</th><th>Interval coverage</th></tr></thead><tbody>` +
      rows.map(r => { const m = r.horizons[evalH], vs = (1 - m.mae / naive.mae) * 100, cv = evalSplit === "test" && E.coverage_test[r.model] ? E.coverage_test[r.model][evalH] : null;
        return `<tr${r.model === f.selected ? ' class="best"' : ""}><td>${esc(r.label)}${r.model === f.selected ? " ★" : ""}</td><td>${r.family}</td><td class="num">${num(m.mae, 3)}</td><td class="num">${num(m.rmse, 3)}</td><td class="num">${num(m.mape, 2)}%</td><td class="num">${m.dir_acc == null ? "–" : num(m.dir_acc, 1) + "%"}</td><td class="num ${r.model === "naive" ? "flat" : vs > 0 ? "up" : "down"}">${r.model === "naive" ? "baseline" : pct(vs, 1)}</td><td class="num">${cv == null ? "–" : num(cv, 0) + "%"}</td></tr>`; }).join("") + "</tbody></table>";
    const s = E.split, up = naive.up_share;
    const fi = (E.feature_importance || []); const mx = Math.max(...fi.map(x => x.importance), 1e-9);
    $("evalNotes").innerHTML = `
      <p><b>Chronological split</b> (a ${s.embargo}-session gap separates blocks so overlapping targets never leak): train ${s.train} samples (${s.train_range[0]} → ${s.train_range[1]}), validation ${s.val} (${s.val_range[0]} → ${s.val_range[1]}), test ${s.test} (${s.test_range[0]} → ${s.test_range[1]}).</p>
      <p>${esc(E.selection_rule)} “MAE vs naive” compares with predicting that the price stays unchanged; negative means <i>worse</i> than doing nothing. Always-up direction baseline in this block: ${up == null ? "–" : num(Math.max(up, 100 - up), 1) + "%"}.</p>
      <p>Intervals are the ${esc(f.interval_label)} from validation residuals; “coverage” is how often the real price fell inside it on the test block (ideal ≈ 80%).${E.lstm.backend ? ` LSTM backend: ${E.lstm.backend}, early-stopped at ~${E.lstm.epochs} epochs.` : ""}</p>
      ${fi.length ? `<p style="margin-top:10px"><b>Top Random Forest features</b></p><div class="fi">${fi.map(x => `<span>${esc(x.feature)}</span><div class="b" style="width:${Math.max(2, x.importance / mx * 100)}%"></div><span>${(x.importance * 100).toFixed(1)}%</span>`).join("")}</div>` : ""}`;
  }

  // ---------------- backtest
  function backtest() {
    const B = R.backtest, f = F(), names = Object.keys(B.models);
    $("btMeta").textContent = `${B.n} out-of-sample sessions (${B.dates[0]} → ${B.dates[B.n - 1]}). ${B.note}`;
    const pick = [f.best_ml, f.best_dl].filter(Boolean);
    const sp = [{ name: "Actual close", y: B.actual_close, color: "var(--c1)", width: 1.8 }, { name: "Naive (previous close)", y: B.naive_close, color: "var(--muted)", width: 1, dash: "2 3" }];
    pick.forEach((n, i) => sp.push({ name: R.forecast.models[n].label, y: B.models[n].pred_close, color: COLORS[names.indexOf(n) % 5], width: 1.4, dash: i ? "6 3" : "" }));
    charts.btp = new XYChart($("btPrice"), { x: B.dates, series: sp, height: 260, tipFmt: v => v.toLocaleString(undefined, { maximumFractionDigits: 2 }), ariaLabel: "Backtest predicted vs actual" });
    const se = [{ name: "Buy & hold", y: B.buy_hold.equity, color: "var(--c1)", width: 1.8 }];
    names.forEach((n, i) => se.push({ name: R.forecast.models[n].label, y: B.models[n].equity, color: COLORS[i % 5], width: n === f.selected ? 2.2 : 1.2, hidden: !pick.includes(n) }));
    charts.bte = new XYChart($("btEquity"), { x: B.dates, series: se, height: 260, yFmt: v => v.toFixed(2), ariaLabel: "Backtest equity curves" });
    $("btTable").innerHTML = `<table><thead><tr><th>Strategy</th><th>Total return</th><th>Max drawdown</th><th>Trades</th><th>Days long</th><th>Win rate (long days)</th></tr></thead><tbody>
      <tr><td>Buy &amp; hold</td><td class="num ${cls(B.buy_hold.total_return)}">${pct(B.buy_hold.total_return * 100)}</td><td class="num">${pct(B.buy_hold.max_drawdown * 100)}</td><td class="num">–</td><td class="num">${B.n}</td><td class="num">–</td></tr>` +
      names.map(n => { const m = B.models[n]; return `<tr${n === f.selected ? ' class="best"' : ""}><td>${esc(R.forecast.models[n].label)}</td><td class="num ${cls(m.total_return)}">${pct(m.total_return * 100)}</td><td class="num">${pct(m.max_drawdown * 100)}</td><td class="num">${m.trades}</td><td class="num">${m.days_long}</td><td class="num">${m.hit_rate == null ? "–" : num(m.hit_rate, 1) + "%"}</td></tr>`; }).join("") + "</tbody></table>";
  }

  // ---------------- trend
  function trend() {
    const A = R.analysis, S = A.stats, rc = r => /Bullish|lower|Oversold/.test(r) ? "up" : /Bearish|upper|Overbought/.test(r) ? "down" : "flat";
    $("signals").innerHTML = `<h3>Indicator readings — overall: <b>${esc(A.trend)}</b></h3><table><thead><tr><th>Indicator</th><th>Value</th><th>Reading</th></tr></thead><tbody>` +
      A.signals.map(s => `<tr><td>${esc(s.name)}</td><td class="num">${esc(s.value)}</td><td class="${rc(s.reading)}">${esc(s.reading)}</td></tr>`).join("") + `</tbody></table><p class="meta">${esc(A.note)}</p>`;
    const r = v => v == null ? "–" : pct(v * 100, 1), p = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
    $("stats").innerHTML = `<h3>Returns, volatility &amp; range</h3><table><tbody>
      <tr><td>Return 1M / 3M</td><td class="num">${r(S.return_1m)} / ${r(S.return_3m)}</td></tr>
      <tr><td>Return 6M / 1Y</td><td class="num">${r(S.return_6m)} / ${r(S.return_1y)}</td></tr>
      <tr><td>Volatility (20d, annualised)</td><td class="num">${p(S.volatility_20d)}</td></tr>
      <tr><td>Volatility (1Y, annualised)</td><td class="num">${p(S.volatility_1y)}</td></tr>
      <tr><td>Max drawdown (in history)</td><td class="num">${p(S.max_drawdown)}</td></tr>
      <tr><td>Current drawdown from peak</td><td class="num">${p(S.current_drawdown)}</td></tr>
      <tr><td>52-week high / low</td><td class="num">${money(S.high_52w)} / ${money(S.low_52w)}</td></tr>
      <tr><td>Avg volume (20d)</td><td class="num">${S.avg_volume_20d == null ? "–" : Math.round(S.avg_volume_20d).toLocaleString()}</td></tr></tbody></table>`;
  }
})();
