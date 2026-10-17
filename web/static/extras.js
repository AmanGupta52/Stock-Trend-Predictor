/* Watchlist, Compare and Forecast-tracking views. Uses helpers exposed by app.js as window.ST. */
(() => {
  const $ = id => document.getElementById(id);
  const { esc, num, pct, cls, fdate, fts, moneyC, dirSym, dirCls, COLORS } = ST;
  const api = async (url, opts) => {
    let r;
    try { r = await fetch(url, opts); } catch { throw new Error("Cannot reach the local server. Is app.py still running?"); }
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error?.message || "Request failed");
    return j;
  };
  const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const show = (el, msg) => { el.hidden = !msg; el.textContent = msg || ""; };

  // ------------------------------------------------------------------ watchlist
  let wlBusy = {};
  async function loadWatchlist(refresh) {
    const t = $("wlTable"); if (!t.innerHTML.trim()) t.innerHTML = '<p class="sub">Loading prices…</p>'; show($("wlMsg"), "");
    try { renderWatchlist((await api("/api/watchlist" + (refresh ? "?refresh=1" : ""))).items); }
    catch (e) { t.innerHTML = ""; show($("wlMsg"), e.message); }
  }
  function renderWatchlist(items) {
    const t = $("wlTable");
    if (!items.length) { t.innerHTML = '<p class="sub">Your watchlist is empty. Add a stock above, or use “☆ Add to watchlist” on the dashboard.</p>'; return; }
    t.innerHTML = `<table><thead><tr><th>Stock</th><th>Price</th><th>Change</th><th>Next trading day (forecast)</th><th>Month-end (forecast)</th><th>Reliability</th><th></th></tr></thead><tbody>` +
      items.map(i => {
        const q = i.quote, f = i.forecast, cur = q?.currency || i.currency;
        const price = q ? `<td class="num">${moneyC(q.price, cur)}<div class="mini">${esc(fdate(q.last_close_date))} close${q.stale ? " · cached" : ""}</div></td><td class="num ${cls(q.change_pct)}">${pct(q.change_pct)}<div class="mini">${esc(q.change_label)}</div></td>` : `<td colspan="2" class="down">${esc(i.error || "No data")}</td>`;
        const nd = f ? `<td class="num">${moneyC(f.next_day.pred_close, cur)} <span class="${dirCls(f.next_day.direction)}">${dirSym(f.next_day.direction)} ${pct(f.next_day.change_pct)}</span><div class="mini">${esc(f.model)} · for ${esc(f.next_day.date)}</div></td>
           <td class="num">${moneyC(f.month_end.pred_close, cur)} <span class="${dirCls(f.month_end.direction)}">${pct(f.month_end.change_pct)}</span><div class="mini">${esc(f.month_end.date)}</div></td>
           <td><span class="pill ${f.reliability === "Moderate" ? "up" : f.reliability === "Low" ? "flat" : "down"}">${esc(f.reliability)}</span><div class="mini">analysed ${esc(fts(f.analysed_at))}</div></td>` :
          `<td colspan="3" class="mini">Not analysed yet — click “Update forecast”.</td>`;
        const busy = wlBusy[i.symbol];
        return `<tr data-sym="${esc(i.symbol)}"><td><b>${esc(i.symbol)}</b><div class="mini">${esc(i.name)} · ${esc(i.exchange)}</div></td>${price}${nd}
          <td class="actions"><button class="xbtn" data-act="open">Open</button> <button class="xbtn" data-act="upd" ${busy ? "disabled" : ""}>${busy ? esc(busy) : "Update forecast"}</button> <button class="xbtn danger" data-act="rm" aria-label="Remove ${esc(i.symbol)}">✕</button></td></tr>`;
      }).join("") + "</tbody></table>";
    t.querySelectorAll("button[data-act]").forEach(b => b.onclick = () => rowAction(b.closest("tr").dataset.sym, b.dataset.act));
  }
  async function rowAction(sym, act) {
    if (act === "open") return ST.analyze(sym);
    if (act === "rm") { await api("/api/watchlist/" + encodeURIComponent(sym), { method: "DELETE" }); return loadWatchlist(false); }
    wlBusy[sym] = "0%"; loadWatchlist(false);
    try {
      await ST.jobRun({ ...ST.analyzeBody(sym, false), retrain: false }, p => { wlBusy[sym] = Math.round(p * 100) + "%"; const b = document.querySelector(`tr[data-sym="${CSS.escape(sym)}"] [data-act=upd]`); if (b) b.textContent = wlBusy[sym]; });
    } catch (e) { show($("wlMsg"), `${sym}: ${e.message}`); }
    delete wlBusy[sym]; loadWatchlist(false);
  }
  $("wlForm").onsubmit = async e => {
    e.preventDefault(); const v = $("wlInput").value.trim(); if (!v) return;
    show($("wlMsg"), "");
    try { await post("/api/watchlist", { query: v }); $("wlInput").value = ""; loadWatchlist(false); } catch (err) { show($("wlMsg"), err.message); }
  };
  $("wlRefresh").onclick = () => loadWatchlist(true);
  ST.watchAdd = async (symbol, btn) => {
    btn.disabled = true;
    try { await post("/api/watchlist", { query: symbol }); btn.textContent = "★ In watchlist"; }
    catch (e) { btn.textContent = "☆ Add to watchlist"; btn.disabled = false; alert(e.message); }
  };

  // ------------------------------------------------------------------ compare
  let cmpChart = [];
  $("cmpForm").onsubmit = async e => {
    e.preventDefault();
    const symbols = [...document.querySelectorAll(".cin")].map(i => i.value.trim()).filter(Boolean);
    show($("cmpMsg"), ""); $("cmpOut").hidden = true;
    if (symbols.length < 2) { show($("cmpMsg"), "Enter at least two stocks."); return; }
    $("cmpGo").disabled = true; $("cmpGo").textContent = "Comparing…";
    try { renderCompare(await post("/api/compare", { symbols, period: $("cmpPeriod").value })); }
    catch (err) { show($("cmpMsg"), err.message); }
    $("cmpGo").disabled = false; $("cmpGo").textContent = "Compare";
  };
  function renderCompare(c) {
    $("cmpOut").hidden = false;
    const syms = c.stats.map(s => s.symbol);
    $("cmpSummary").textContent = `${c.days} common trading days, ${c.start} → ${c.end}. Best total return: ${c.best_return}. Lowest volatility: ${c.lowest_volatility}.`;
    const series = syms.map((s, i) => ({ name: s, y: c.rebased[s], color: COLORS[i], width: 1.8 }));
    cmpChart = [new XYChart($("cmpRebased"), { x: c.dates, series, height: 320, tipFmt: v => v.toFixed(1), ariaLabel: "Rebased prices", yRefLines: [100] }),
      new XYChart($("cmpDD"), { x: c.dates, series: syms.map((s, i) => ({ name: s, y: c.drawdown[s], color: COLORS[i], width: 1.4 })), height: 200, yFmt: v => v.toFixed(0) + "%", tipFmt: v => v.toFixed(1) + "%", ariaLabel: "Drawdown" })];
    const P = v => v == null ? "–" : pct(v * 100, 1), V = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
    const rows = [["Last close", s => moneyC(s.last_close, s.currency)], ["Total return", s => P(s.total_return), s => cls(s.total_return)], ["Annualised return", s => P(s.annual_return), s => cls(s.annual_return)],
      ["Volatility (annualised)", s => V(s.volatility)], ["Return / risk (Sharpe, rf=0)", s => s.sharpe == null ? "–" : num(s.sharpe, 2)], ["Max drawdown", s => V(s.max_drawdown)], ["Current drawdown", s => V(s.current_drawdown)],
      ["Return 1M", s => P(s.ret_1m), s => cls(s.ret_1m)], ["Return 3M", s => P(s.ret_3m), s => cls(s.ret_3m)], ["Return 6M", s => P(s.ret_6m), s => cls(s.ret_6m)], ["Return 1Y", s => P(s.ret_1y), s => cls(s.ret_1y)],
      ["Best day", s => P(s.best_day), () => "up"], ["Worst day", s => P(s.worst_day), () => "down"]];
    $("cmpStats").innerHTML = `<table><thead><tr><th>Metric</th>${c.stats.map(s => `<th>${esc(s.symbol)}<div class="mini">${esc(s.name)}</div></th>`).join("")}</tr></thead><tbody>` +
      rows.map(([l, f, k]) => `<tr><td>${l}</td>${c.stats.map(s => `<td class="num ${k ? k(s) : ""}">${f(s)}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    const m = c.correlation;
    $("cmpCorr").innerHTML = `<table><thead><tr><th></th>${m.symbols.map(s => `<th>${esc(s)}</th>`).join("")}</tr></thead><tbody>` +
      m.matrix.map((r, i) => `<tr><td>${esc(m.symbols[i])}</td>${r.map(v => `<td class="num">${v.toFixed(2)}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    $("cmpNotes").innerHTML = c.notes.map(n => `<p>${esc(n)}</p>`).join("");
  }

  // ------------------------------------------------------------------ tracking
  async function loadTracking() {
    const sym = $("trkSym").value;
    try { renderTracking(await api("/api/tracking" + (sym ? "?symbol=" + encodeURIComponent(sym) : ""))); }
    catch (e) { show($("trkMsg"), e.message); }
  }
  function renderTracking(T) {
    const sel = $("trkSym"), cur = sel.value;
    sel.innerHTML = '<option value="">All stocks</option>' + T.symbols.map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join(""); sel.value = cur;
    $("trkNote").textContent = T.note;
    $("trkKpi").innerHTML = `<div><b>${T.resolved}</b><span>outcomes recorded</span></div><div><b>${T.pending}</b><span>forecasts waiting for their date</span></div>`;
    const kinds = { next_day: "Next trading day", month_end: "Month-end (≈21 sessions)" };
    if (!T.table.length) { $("trkTable").innerHTML = '<p class="sub">No outcomes yet. Every analysis you run is saved automatically; once a forecast’s date has passed, click “Update outcomes” to compare it with what really happened.</p>'; }
    else $("trkTable").innerHTML = `<table><thead><tr><th>Horizon</th><th>Model</th><th>Forecasts checked</th><th>MAE (price)</th><th>MAPE</th><th>Bias</th><th>Direction acc.</th><th>Better than “no change”</th><th>Inside range</th></tr></thead><tbody>` +
      T.table.map(r => `<tr><td>${kinds[r.kind] || r.kind}</td><td>${esc(r.label)}</td><td class="num">${r.n}</td><td class="num">${num(r.mae, 2)}</td><td class="num">${num(r.mape, 2)}%</td><td class="num">${pct(r.bias_pct, 2)}</td><td class="num">${r.dir_acc == null ? "–" : num(r.dir_acc, 0) + "%"}</td><td class="num ${r.vs_naive == null ? "" : r.vs_naive > 0 ? "up" : "down"}">${r.model === "naive" ? "baseline" : r.vs_naive == null ? "–" : pct(r.vs_naive, 1)}</td><td class="num">${r.interval_hit == null ? "–" : num(r.interval_hit, 0) + "%"}</td></tr>`).join("") + "</tbody></table>";
    $("trkRecent").innerHTML = T.recent.length ? `<table><thead><tr><th>Made</th><th>Stock</th><th>Model</th><th>Horizon</th><th>Target date</th><th>Predicted</th><th>Actual</th><th>Error</th></tr></thead><tbody>` +
      T.recent.map(r => `<tr><td>${esc(fts(r.made_at))}</td><td>${esc(r.symbol)}</td><td>${esc(r.model)}${r.selected ? " ★" : ""}</td><td>${kinds[r.kind] || r.kind}</td><td>${esc(r.target_date)}</td><td class="num">${moneyC(r.pred_close, r.currency)}</td><td class="num">${r.actual_close == null ? '<span class="mini">pending</span>' : moneyC(r.actual_close, r.currency)}</td><td class="num ${r.error_pct == null ? "" : Math.abs(r.error_pct) < 1 ? "up" : "flat"}">${r.error_pct == null ? "–" : pct(r.error_pct, 2)}</td></tr>`).join("") + "</tbody></table>" : '<p class="sub">Nothing logged yet — analyse a stock first.</p>';
  }
  $("trkSym").onchange = loadTracking;
  $("trkResolve").onclick = async () => {
    const b = $("trkResolve"); b.disabled = true; b.textContent = "Updating…"; show($("trkMsg"), "");
    try {
      const r = await post("/api/tracking/resolve", { symbol: $("trkSym").value });
      renderTracking(r.summary);
      show($("trkMsg"), r.problems.length ? r.problems.join(" · ") : r.resolved ? `${r.resolved} outcome(s) recorded.` : "No new outcomes yet — forecasts resolve once their target date has passed.");
    } catch (e) { show($("trkMsg"), e.message); }
    b.disabled = false; b.textContent = "↻ Update outcomes";
  };

  document.addEventListener("viewchange", e => {
    if (e.detail === "watchlist") loadWatchlist(false);
    if (e.detail === "tracking") loadTracking();
  });
})();
