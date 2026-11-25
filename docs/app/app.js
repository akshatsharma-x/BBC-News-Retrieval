// BBC News Retrieval — browser-only search interface (same design as the Streamlit app).
import { MODELS, evaluateRun, loadEngine, webLoader } from "./engine.js";

const COLOR = { "TF-IDF": "#3987e5", "BM25": "#d95926", "Sentence Transformer": "#199e70", "Hybrid RRF": "#c98500" };
const DESKS = ["Business", "Entertainment", "Politics", "Sport", "Technology"];   // index = category id
const DESK_FILTER = ["Business", "Entertainment", "Politics", "Sport", "Tech"];
const SCORE = { "TF-IDF": "cosine", "BM25": "bm25", "Sentence Transformer": "cosine", "Hybrid RRF": "rrf" };
const GRADE = { 2: ["g2", "Highly relevant"], 1: ["g1", "Partially relevant"], 0: ["g0", "Not relevant"] };
const NEEDS_DENSE = new Set(["Sentence Transformer", "Hybrid RRF"]);

const state = { query: "", model: "Hybrid RRF", k: 10, desks: new Set(), tab: "search", topic: "Q07", topicModel: "Hybrid RRF" };
let engine, bench, charts, dense = "loading", denseError = "", chartsDrawn = false, searchSeq = 0;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const dot = (m, extra = "") => `<span class="dot" style="--model:${COLOR[m]}${extra}"></span>`;

// ------------------------------------------------------------------ init ---
async function init() {
  const data = new URL("./data/", import.meta.url);
  const loader = webLoader(data);
  [engine, bench, charts] = await Promise.all([loadEngine(loader), loader.json("bench.json"), loader.json("charts.json")]);
  state.query = bench.examples[0];
  $("q").value = state.query;
  $("ticker").innerHTML = [[engine.N.toLocaleString(), "articles"], ["5", "news desks"],
    [engine.termIndex.size.toLocaleString(), "index terms"], [engine.meta.n_passages.toLocaleString(), "embedded passages"],
    ["4", "ranking models"]].map(([v, k]) => `<span><b>${v}</b> ${k}</span>`).join("") +
    `<span class="status warn" id="status">neural model: loading…</span>`;
  buildControls();
  render();
  engine.loadDense(onProgress).then(() => { dense = "ready"; setStatus("neural model: ready · runs in your browser", false); render(); })
    .catch((err) => { dense = "error"; denseError = String(err); setStatus("neural model unavailable — lexical models only", true); render(); });
}

const progress = {};
function onProgress(p) {
  if (p.status === "progress" && p.total) {
    progress[p.file] = [p.loaded, p.total];
    const [l, t] = Object.values(progress).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
    setStatus(`neural model: loading ${Math.round(100 * l / t)}% of ${(t / 1048576).toFixed(0)} MB`, true);
  } else if (p.status === "ready") setStatus("neural model: starting…", true);
}
function setStatus(text, warn) { const s = $("status"); if (s) { s.textContent = text; s.classList.toggle("warn", warn); } }

// -------------------------------------------------------------- controls ---
function buildControls() {
  $("examples").innerHTML = bench.examples.map((e) => `<button class="pill" data-example="${esc(e)}">${esc(e)}</button>`).join("");
  $("models").innerHTML = MODELS.map((m) => `<button data-model="${m}" aria-pressed="${m === state.model}">${m}</button>`).join("");
  $("ks").innerHTML = [5, 10, 20].map((k) => `<button data-k="${k}" aria-pressed="${k === state.k}">${k}</button>`).join("");
  $("desks").innerHTML = DESK_FILTER.map((d, i) => `<button class="pill" data-desk="${i}" aria-pressed="false">${d}</button>`).join("");
  let timer;
  $("q").addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => { state.query = e.target.value; render(); }, 220); });
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") { clearTimeout(timer); state.query = e.target.value; render(); } });
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button, [data-tab]");
    if (!b) return;
    if (b.dataset.example) { state.query = b.dataset.example; $("q").value = state.query; }
    else if (b.dataset.model) state.model = b.dataset.model;
    else if (b.dataset.k) state.k = Number(b.dataset.k);
    else if (b.dataset.desk) { const i = Number(b.dataset.desk); state.desks.has(i) ? state.desks.delete(i) : state.desks.add(i); }
    else if (b.dataset.topicModel) state.topicModel = b.dataset.topicModel;
    else if (b.id?.startsWith("tab-")) state.tab = b.id.slice(4);
    else return;
    render();
  });
  document.addEventListener("change", (e) => { if (e.target.id === "topic") { state.topic = e.target.value; render(); } });
}

function syncControls() {
  document.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.model === state.model));
  document.querySelectorAll("[data-k]").forEach((b) => b.setAttribute("aria-pressed", Number(b.dataset.k) === state.k));
  document.querySelectorAll("[data-desk]").forEach((b) => b.setAttribute("aria-pressed", state.desks.has(Number(b.dataset.desk))));
  for (const t of ["search", "compare", "bench"]) {
    $(`tab-${t}`).setAttribute("aria-selected", state.tab === t);
    $(`panel-${t}`).hidden = state.tab !== t;
  }
}

// ---------------------------------------------------------------- render ---
async function render() {
  syncControls();
  const seq = ++searchSeq;
  if (state.tab === "bench") return renderBench(seq);
  const q = state.query.trim();
  if (!q) {
    $("panel-search").innerHTML = $("panel-compare").innerHTML =
      `<div class="empty"><b>Start with a query</b>Type a topic, a name or a whole question above.</div>`;
    return;
  }
  const t0 = performance.now();
  const { runs, qterms } = await engine.runAll(q, engine.N);
  const ms = performance.now() - t0;
  if (seq !== searchSeq) return;                                        // a newer search has started
  if (state.tab === "search") renderSearch(runs, qterms, ms);
  else renderCompare(runs, qterms);
}

function topK(run, k) {
  const out = [];
  for (const i of run.ranking) {
    if (state.desks.size && !state.desks.has(engine.docs[i][1])) continue;
    out.push({ i, score: run.scores[i] });
    if (out.length === k) break;
  }
  return out;
}

function effectiveModel(model) {
  return NEEDS_DENSE.has(model) && dense !== "ready" ? "BM25" : model;
}

function renderSearch(runs, qterms, ms) {
  const model = effectiveModel(state.model);
  const hits = topK(runs[model], state.k);
  const qset = new Set(qterms);
  const chips = qterms.length ? qterms.map((t) => `<span class="chip">${esc(t)}</span>`).join("") : `<span class="chip">(none)</span>`;
  const notice = model !== state.model
    ? `<div class="notice">${dense === "error" ? "The neural model could not load in this browser" :
      "The neural model is still loading"} — showing <b>BM25</b> for now${dense === "error" ? "" : "; results switch automatically when it is ready"}.</div>` : "";
  const top = hits.length ? hits[0].score : 0;
  const cards = hits.map((h, n) => card(h, n + 1, model, top, qset)).join("");
  $("panel-search").innerHTML = notice +
    `<div class="runbar" style="--model:${COLOR[model]}"><span class="model">${dot(model)}${esc(model)}</span><span class="sep">|</span>` +
    `<span>${hits.length} results</span><span class="sep">|</span><span>ranked in <span class="lat">${ms.toFixed(1)} ms</span> in your browser</span>` +
    `<span class="sep">|</span><span>index terms: ${chips}</span></div>` +
    (cards || `<div class="empty"><b>No matching articles</b>This model found nothing for the query and desk filter — try the Sentence Transformer or Hybrid RRF, which also match meaning.</div>`);
}

function card(h, rank, model, top, qset) {
  const [id, cat, title, lead] = engine.docs[h.i];
  const matched = [...qset].filter((t) => engine.hasTerm(h.i, t)).sort();
  const terms = matched.length ? matched.map((t) => `<span class="chip">${esc(t)}</span>`).join("")
    : `<span class="chip">no shared terms · semantic match</span>`;
  const width = top > 0 ? Math.max(3, 100 * h.score / top) : 0;
  const score = model === "Hybrid RRF" ? h.score.toFixed(4) : h.score.toFixed(3);
  const source = `https://duckduckgo.com/?q=${encodeURIComponent(`"${title}" site:news.bbc.co.uk`)}`;
  return `<article class="card" style="--model:${COLOR[model]};animation-delay:${Math.min(rank, 10) * 30}ms">` +
    `<div class="rank">${String(rank).padStart(2, "0")}</div><div>` +
    `<div class="meta"><span class="desk">${DESKS[cat]}</span><span>${esc(id)}</span><span>${SCORE[model]} ${score}</span></div>` +
    `<div class="headline">${esc(title)}</div><p class="snippet">${highlight(lead, qset)}</p>` +
    `<div class="terms">${terms}</div>` +
    `<a class="source" href="${source}" target="_blank" rel="noopener">Find the original story on BBC News ↗</a>` +
    `<div class="scorebar"><span style="width:${width.toFixed(1)}%"></span></div></div></article>`;
}

function highlight(text, qset) {
  return text.split(/([A-Za-z0-9']+)/).map((part, i) => {
    if (i % 2 === 0) return esc(part);
    const w = engine.clean(part);
    return w && qset.has(engine.stem(w)) ? `<mark>${esc(part)}</mark>` : esc(part);
  }).join("");
}

function renderCompare(runs) {
  const avail = MODELS.filter((m) => runs[m]);
  const top5 = Object.fromEntries(avail.map((m) => [m, topK(runs[m], 5)]));
  const agree = new Map();
  for (const m of avail) for (const h of top5[m]) agree.set(h.i, (agree.get(h.i) || 0) + 1);
  const all = [...agree].filter(([, c]) => c === 4).map(([i]) => engine.docs[i][2]);
  const consensus = avail.length < 4 ? "The neural model is still loading — consensus appears when all four models are ready."
    : all.length ? `${all.length} article${all.length > 1 ? "s" : ""} in every model's top 5: ${all.map((t) => `<b>${esc(t)}</b>`).join(" · ")}`
      : "No article appears in all four top-5 lists — the models disagree on this query.";
  const cols = MODELS.map((m) => {
    const items = runs[m] ? top5[m].map((h, n) => {
      const a = agree.get(h.i);
      return `<div class="item"><div class="n">${n + 1}</div><div><div class="t">${esc(engine.docs[h.i][2])}</div>` +
        `<div class="s"><span class="desk">${DESKS[engine.docs[h.i][1]]}</span><span class="agree${a === 4 ? " all" : ""}">${a}/${avail.length} models</span></div></div></div>`;
    }).join("") : `<div class="item"><div></div><div class="s">loading the neural model…</div></div>`;
    return `<div class="deskcol" style="--model:${COLOR[m]}"><div class="deskhead">${dot(m)}${m}</div><div class="bar"></div>${items}</div>`;
  }).join("");
  $("panel-compare").innerHTML = `<div class="consensus"><span class="chip">consensus</span><span>${consensus}</span></div><div class="cols4">${cols}</div>`;
}

// ------------------------------------------------------------- benchmark ---
function table(rows, cols, digits = 4) {
  const best = Object.fromEntries(cols.map(([k]) => [k, Math.max(...Object.values(rows).map((r) => r[k]))]));
  return `<table class="mtable"><thead><tr><th>Model</th>${cols.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead><tbody>` +
    Object.entries(rows).map(([m, r]) => `<tr><td>${dot(m, ";margin-right:8px")}${esc(m)}</td>` +
      cols.map(([k]) => `<td class="${Math.abs(r[k] - best[k]) < 1e-12 ? "best" : ""}">${r[k].toFixed(digits)}</td>`).join("") + `</tr>`).join("") +
    `</tbody></table>`;
}

async function renderBench(seq) {
  const panel = $("panel-bench");
  if (!chartsDrawn) {
    const main = Object.fromEntries(bench.main.map((r) => [r.system, r]));
    const ki = Object.fromEntries(bench.known_item.table.map((r) => [r.system, r]));
    const best = (t, k) => MODELS.reduce((a, m) => (t[m][k] > t[a][k] ? m : a));
    const tiles = [[main[best(main, "AP")].AP, "Best MAP · 25 topics", best(main, "AP")],
      [main[best(main, "nDCG@10")]["nDCG@10"], "Best nDCG@10", best(main, "nDCG@10")],
      [main[best(main, "Recall@100")]["Recall@100"], "Best Recall@100", best(main, "Recall@100")],
      [ki[best(ki, "MRR")].MRR, `Known-item MRR · ${bench.known_item.queries.toLocaleString()} queries`, best(ki, "MRR")]];
    panel.innerHTML =
      `<div class="kpis">${tiles.map(([v, l, m]) => `<div class="kpi" style="--model:${COLOR[m]}"><div class="v">${v.toFixed(3)}</div><div class="l">${l}</div><div class="w">${dot(m)}${m}</div></div>`).join("")}</div>` +
      `<div class="grid2"><div class="panel"><div class="panel-title">Performance comparison</div><div class="panel-sub">Macro-average over 25 graded topics · hover a bar for the exact score</div><div class="chart" id="ch-metrics"></div></div>` +
      `<div class="panel"><div class="panel-title">Precision–recall curves</div><div class="panel-sub">11-point interpolated precision · hover to compare models at a recall level</div><div class="chart" id="ch-pr"></div></div></div>` +
      `<div class="grid2 wide-left"><div class="panel"><div class="panel-title">Results table</div><div class="panel-sub">Evaluated with the Python engine · best value per column in bold</div>` +
      table(main, [["P@10", "P@10"], ["Recall@100", "R@100"], ["AP", "MAP"], ["nDCG@10", "nDCG@10"], ["RR", "MRR"]]) + `</div>` +
      `<div class="panel"><div class="panel-title">Keyword vs paraphrased queries</div><div class="panel-sub">Lexical models lose most when the query avoids the article's words</div><div class="chart" id="ch-style"></div></div></div>` +
      `<div class="panel-title" style="font-size:1.25rem;margin-top:8px">Topic explorer</div>` +
      `<div class="panel-sub">Re-runs any of the 25 test topics in your browser and grades every result against the judgments</div>` +
      `<div class="controls" style="grid-template-columns:minmax(0,2fr) auto;margin-top:8px"><div><label class="label" for="topic">Test topic</label>` +
      `<select class="topic-select" id="topic">${bench.queries.map((q) => `<option value="${q.qid}"${q.qid === state.topic ? " selected" : ""}>${q.qid} · ${esc(q.query)}</option>`).join("")}</select></div>` +
      `<div><span class="label">Model</span><div class="seg" id="topic-models">${MODELS.map((m) => `<button data-topic-model="${m}" aria-pressed="${m === state.topicModel}">${m}</button>`).join("")}</div></div></div>` +
      `<div id="explorer"></div>`;
    chartsDrawn = true;
    drawCharts();
  }
  document.querySelectorAll("[data-topic-model]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.topicModel === state.topicModel));
  await renderExplorer(seq);
}

function drawCharts() {
  if (typeof vegaEmbed === "undefined") { setTimeout(drawCharts, 200); return; }   // CDN scripts load deferred
  const opts = { actions: false, renderer: "svg" };
  const fit = (spec) => ({ ...spec, width: "container", autosize: { type: "fit", contains: "padding" } });
  vegaEmbed("#ch-metrics", fit(charts.metrics), opts);
  vegaEmbed("#ch-pr", fit(charts.pr), opts);
  vegaEmbed("#ch-style", fit(charts.style), opts);
}

async function renderExplorer(seq) {
  const topic = bench.queries.find((q) => q.qid === state.topic);
  const judg = bench.qrels[topic.qid];
  const grades = Object.values(judg);
  const { runs } = await engine.runAll(topic.query, 1000);
  if (seq !== searchSeq) return;
  const live = {};
  for (const m of MODELS) if (runs[m]) live[m] = evaluateRun(runs[m].ranking.map((i) => engine.ids[i]), judg);
  const model = runs[state.topicModel] ? state.topicModel : "BM25";
  const list = runs[model].ranking.slice(0, 10).map((i, n) => {
    const g = judg[engine.ids[i]];
    const [cls, label] = g === undefined ? ["g0", "Unjudged"] : GRADE[g];
    return `<div class="judged"><div class="n">${n + 1}</div><div class="t">${esc(engine.docs[i][2])} <span class="chip">${engine.ids[i]}</span></div><span class="grade ${cls}">${label}</span></div>`;
  }).join("");
  $("explorer").innerHTML =
    `<div class="topic"><div class="meta"><span class="desk">${topic.qid}</span><span class="desk">${DESKS[engine.meta.categories.indexOf(topic.category)]}</span>` +
    `<span class="desk">${topic.style === "keyword" ? "Keyword query" : "Natural-language query"}</span>` +
    `<span>${grades.filter((g) => g === 2).length} highly · ${grades.filter((g) => g === 1).length} partially relevant · ${grades.length} judged</span></div>` +
    `<div class="q">“${esc(topic.query)}”</div><p class="narr">${esc(topic.narrative)}</p></div>` +
    `<div class="grid2"><div class="panel"><div class="panel-title">Live metrics for this topic</div><div class="panel-sub">Computed in your browser from the graded judgments</div>` +
    table(live, [["P@10", "P@10"], ["AP", "AP"], ["nDCG@10", "nDCG@10"]], 3) +
    (dense === "ready" ? "" : `<p class="panel-sub" style="margin-top:10px">Neural models appear when the encoder has loaded.</p>`) + `</div>` +
    `<div class="panel"><div class="panel-title">${model} · top 10 with judgments</div><div class="panel-sub">Judged blind from depth-20 pools plus a keyword sweep</div>${list}</div></div>`;
}

// Test hook: evaluate every topic with every model (used by the parity check).
window.__bbc = {
  get engine() { return engine; },
  get dense() { return dense; },
  async evaluateTopics() {
    const out = [];
    for (const q of bench.queries) {
      const { runs } = await engine.runAll(q.query, 1000);
      for (const m of MODELS) if (runs[m]) out.push({ qid: q.qid, system: m, ...evaluateRun(runs[m].ranking.map((i) => engine.ids[i]), bench.qrels[q.qid]) });
    }
    return out;
  },
};

init().catch((err) => {
  $("panel-search").innerHTML = `<div class="empty"><b>Could not load the search index</b>${esc(err)}</div>`;
  console.error(err);
});
