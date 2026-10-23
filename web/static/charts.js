/* Minimal dependency-free interactive SVG chart (lines, bands, bars, hover tooltip, zoom window). */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a = {}, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };

  function niceTicks(lo, hi, n) {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const r = step0 / mag, step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const defaultFmt = v => Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) :
    Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);

  class XYChart {
    /* opts: {x:[dates], series:[{name,y,color,width,dash,type:'line'|'bar'|'band',lo,hi,opacity,hidden,tip}],
              height, yFmt, tipFmt, markers:[{x,label}], shade:{from,to}, yZero:bool, yDomain:[a,b], ariaLabel} */
    constructor(host, opts) {
      this.host = host; this.o = opts; this.range = [0, opts.x.length - 1];
      host.classList.add("xy"); host.innerHTML = "";
      this.legend = document.createElement("div"); this.legend.className = "legend"; host.appendChild(this.legend);
      this.wrap = document.createElement("div"); this.wrap.className = "plot"; host.appendChild(this.wrap);
      this.tip = document.createElement("div"); this.tip.className = "tip"; this.tip.hidden = true; this.wrap.appendChild(this.tip);
      this.svg = el("svg", { role: "img", "aria-label": opts.ariaLabel || "chart" }); this.wrap.appendChild(this.svg);
      this.buildLegend();
      this.ro = new ResizeObserver(() => this.draw()); this.ro.observe(this.wrap);
      this.svg.addEventListener("pointermove", e => this.hover(e));
      this.svg.addEventListener("pointerleave", () => this.leave());
      this.draw();
    }
    buildLegend() {
      this.legend.innerHTML = "";
      this.o.series.forEach(s => {
        if (s.type === "band" && !s.legend) return;
        const b = document.createElement("button"); b.type = "button"; b.className = "chip" + (s.hidden ? " off" : "");
        b.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
        b.onclick = () => { s.hidden = !s.hidden; b.classList.toggle("off", !!s.hidden); this.draw(); };
        this.legend.appendChild(b);
      });
    }
    setRange(a, b) { this.range = [Math.max(0, a), Math.min(this.o.x.length - 1, b)]; this.draw(); }
    update(o) { Object.assign(this.o, o); this.range = [0, this.o.x.length - 1]; this.buildLegend(); this.draw(); }
    draw() {
      const o = this.o, W = this.wrap.clientWidth || 600, Ht = o.height || 300;
      const m = { l: 56, r: 12, t: 8, b: 22 };
      this.W = W; this.H = Ht; this.m = m;
      this.svg.setAttribute("viewBox", `0 0 ${W} ${Ht}`); this.svg.setAttribute("height", Ht);
      this.svg.innerHTML = "";
      const [a, b] = this.range, n = b - a + 1, iw = W - m.l - m.r, ih = Ht - m.t - m.b;
      this.px = i => m.l + (n <= 1 ? iw / 2 : (i - a) / (n - 1) * iw);
      const vis = o.series.filter(s => !s.hidden);
      let lo = Infinity, hi = -Infinity;
      vis.forEach(s => {
        const arrs = s.type === "band" ? [s.lo, s.hi] : [s.y];
        arrs.forEach(arr => { for (let i = a; i <= b; i++) { const v = arr[i]; if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } } });
      });
      if (o.yZero) lo = Math.min(lo, 0);
      if (o.yDomain) { lo = o.yDomain[0] ?? lo; hi = o.yDomain[1] ?? hi; }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      const pad = (hi - lo) * (o.yZero ? 0.04 : 0.08) || 1; if (!o.yZero || o.yDomain == null) { hi += pad; if (!o.yZero) lo -= pad; }
      this.lo = lo; this.hi = hi;
      this.py = v => m.t + ih - (v - lo) / (hi - lo) * ih;
      const g = el("g", {}, this.svg);
      // forecast shading
      if (o.shade) {
        const i0 = Math.max(a, o.shade.from), i1 = Math.min(b, o.shade.to);
        if (i1 >= i0) el("rect", { x: this.px(i0) - (n > 1 ? iw / (n - 1) / 2 : 0), y: m.t, width: Math.max(0, this.px(i1) - this.px(i0) + (n > 1 ? iw / (n - 1) : 0)), height: ih, class: "shade" }, g);
      }
      // grid + y axis
      niceTicks(lo, hi, Math.max(3, Math.floor(ih / 48))).forEach(v => {
        const y = this.py(v); if (y < m.t - 1 || y > m.t + ih + 1) return;
        el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, g);
        const t = el("text", { x: m.l - 6, y: y + 3.5, class: "tick", "text-anchor": "end" }, g); t.textContent = (o.yFmt || defaultFmt)(v);
      });
      if (o.yRefLines) o.yRefLines.forEach(v => { if (v >= lo && v <= hi) el("line", { x1: m.l, x2: W - m.r, y1: this.py(v), y2: this.py(v), class: "ref" }, g); });
      // x axis
      const nt = Math.max(2, Math.floor(iw / 90));
      for (let k = 0; k < nt; k++) {
        const i = Math.round(a + k * (n - 1) / Math.max(1, nt - 1)); const t = el("text", { x: this.px(i), y: Ht - 6, class: "tick", "text-anchor": k === 0 ? "start" : k === nt - 1 ? "end" : "middle" }, g);
        t.textContent = o.xFmt ? o.xFmt(o.x[i]) : shortDate(o.x[i]);
      }
      // series
      const bw = Math.max(1, Math.min(8, iw / n * 0.7));
      o.series.forEach(s => {
        if (s.hidden) return;
        if (s.type === "band") {
          let up = [], dn = [];
          for (let i = a; i <= b; i++) if (s.lo[i] != null && s.hi[i] != null) { up.push([this.px(i), this.py(s.hi[i])]); dn.push([this.px(i), this.py(s.lo[i])]); }
          if (up.length > 1) el("path", { d: "M" + up.map(p => p.join(",")).join("L") + "L" + dn.reverse().map(p => p.join(",")).join("L") + "Z", fill: s.color, opacity: s.opacity ?? 0.16 }, g);
        } else if (s.type === "bar") {
          for (let i = a; i <= b; i++) { const v = s.y[i]; if (v == null) continue; const y0 = this.py(Math.max(lo, 0)), y1 = this.py(v);
            el("rect", { x: this.px(i) - bw / 2, y: Math.min(y0, y1), width: bw, height: Math.max(0.5, Math.abs(y0 - y1)), fill: (s.colors && s.colors[i]) || s.color, opacity: s.opacity ?? 0.8 }, g); }
        } else {
          let d = "", pen = false;
          for (let i = a; i <= b; i++) { const v = s.y[i]; if (v == null || !isFinite(v)) { pen = false; continue; } d += (pen ? "L" : "M") + this.px(i).toFixed(1) + "," + this.py(v).toFixed(1); pen = true; }
          if (d) el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.6, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
          if (s.dots) for (let i = a; i <= b; i++) if (s.y[i] != null) el("circle", { cx: this.px(i), cy: this.py(s.y[i]), r: 3, fill: s.color }, g);
        }
      });
      // markers
      (o.markers || []).forEach(mk => {
        const i = o.x.indexOf(mk.x); if (i < a || i > b) return;
        el("line", { x1: this.px(i), x2: this.px(i), y1: m.t, y2: m.t + ih, class: "marker" }, g);
        const t = el("text", { x: this.px(i) + 4, y: m.t + 11, class: "mlabel" }, g); t.textContent = mk.label;
      });
      this.cross = el("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" }, this.svg);
    }
    hover(e) {
      const r = this.svg.getBoundingClientRect(), x = (e.clientX - r.left) * (this.W / r.width);
      const [a, b] = this.range, n = b - a + 1, iw = this.W - this.m.l - this.m.r;
      let i = Math.round(a + (x - this.m.l) / iw * (n - 1)); i = Math.max(a, Math.min(b, i));
      const o = this.o, fmt = o.tipFmt || o.yFmt || defaultFmt;
      let rows = `<b>${o.x[i]}</b>`;
      o.series.forEach(s => {
        if (s.hidden || s.noTip) return;
        if (s.type === "band") { if (s.lo[i] != null && s.tipLabel) rows += `<div><i style="background:${s.color}"></i>${s.tipLabel}<span>${fmt(s.lo[i])} – ${fmt(s.hi[i])}</span></div>`; return; }
        const v = s.y[i]; if (v == null) return;
        rows += `<div><i style="background:${s.color}"></i>${s.name}<span>${(s.tipFmt || fmt)(v)}</span></div>`;
      });
      this.tip.innerHTML = rows; this.tip.hidden = false;
      const px = this.px(i), tw = this.tip.offsetWidth;
      this.tip.style.left = (px > this.W / 2 ? px - tw - 12 : px + 12) + "px"; this.tip.style.top = "8px";
      this.cross.setAttribute("x1", px); this.cross.setAttribute("x2", px); this.cross.setAttribute("visibility", "visible");
    }
    leave() { this.tip.hidden = true; if (this.cross) this.cross.setAttribute("visibility", "hidden"); }
  }
  function shortDate(s) { if (!s) return ""; const d = new Date(s + "T00:00:00"); return d.toLocaleDateString(undefined, { month: "short", year: "2-digit", day: s.length ? "numeric" : undefined }); }
  window.XYChart = XYChart;
})();
