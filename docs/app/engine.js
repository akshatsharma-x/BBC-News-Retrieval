// Browser port of the bbcir retrieval engine (preprocess, inverted index, TF-IDF, BM25,
// Sentence Transformer MaxP, Reciprocal Rank Fusion, metrics). Formulas, accumulation
// order and tie-breaking mirror the Python implementation so rankings match.

export const MODELS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"];
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

// ---------------------------------------------------------------- loading --
async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} loading ${url}`);
  return r;
}
export const webLoader = (base) => ({
  base: String(base),
  json: async (name) => (await fetchOk(new URL(name, base))).json(),
  bin: async (name, Type) => new Type(await (await fetchOk(new URL(name, base))).arrayBuffer()),
});

export async function loadEngine(loader) {
  const [meta, docs, terms, stems, stopwords, postDocs, postTfs, docLen] = await Promise.all([
    loader.json("meta.json"), loader.json("docs.json"), loader.json("terms.json"), loader.json("stems.json"),
    loader.json("stopwords.json"), loader.bin("post_docs.u16", Uint16Array), loader.bin("post_tfs.u8", Uint8Array),
    loader.bin("doc_len.u16", Uint16Array)]);
  return new Engine(meta, docs, terms, stems, stopwords, postDocs, postTfs, docLen, loader);
}

// ---------------------------------------------------------------- engine ---
export class Engine {
  constructor(meta, docs, terms, stems, stopwords, postDocs, postTfs, docLen, loader) {
    this.meta = meta;
    this.N = meta.n_docs;
    this.docs = docs;                       // [doc_id, category index, title, lead]
    this.ids = docs.map((d) => d[0]);
    this.termIndex = new Map(terms.terms.map((t, i) => [t, i]));
    this.offsets = terms.offsets;
    this.postDocs = postDocs;
    this.postTfs = postTfs;
    this.docLen = docLen;
    this.stems = new Map(Object.entries(stems));
    this.stop = new Set(stopwords);
    this.loader = loader;
    const { k1, b, avg_doc_len: avgdl } = meta;
    this.k1 = k1;
    this.norm = Float64Array.from(docLen, (l) => k1 * (1 - b + b * l / avgdl));
    const V = terms.terms.length;
    this.idfTfidf = new Float64Array(V);
    this.idfBm25 = new Float64Array(V);
    const sq = new Float64Array(this.N);
    for (let t = 0; t < V; t++) {
      const s = this.offsets[t], e = this.offsets[t + 1], df = e - s;
      this.idfTfidf[t] = Math.log((1 + this.N) / (1 + df)) + 1;          // sklearn smooth_idf
      this.idfBm25[t] = Math.log(1 + (this.N - df + 0.5) / (df + 0.5)); // Lucene BM25 idf
      for (let i = s; i < e; i++) {
        const w = (1 + Math.log(postTfs[i])) * this.idfTfidf[t];         // sublinear tf
        sq[postDocs[i]] += w * w;
      }
    }
    this.docNorm = sq.map(Math.sqrt);
    this.dense = null;
  }

  // ---- preprocessing (identical pipeline to bbcir/preprocess.py) ----
  clean(text) {
    let s = text.normalize("NFKD").replace(/[^\x00-\x7F]/g, "").toLowerCase();
    s = s.replace(/(?<=[a-z0-9])'s\b/g, "");
    return s.replace(/[^a-z0-9]+/g, " ").trim();
  }
  tokenize(text) { return text.match(/[a-z0-9]+/g) || []; }
  stem(word) { return this.stems.get(word) ?? porter(word); }
  preprocess(text) {
    return this.tokenize(this.clean(text)).filter((t) => t.length > 1 && !this.stop.has(t)).map((t) => this.stem(t));
  }

  // ---- lexical models ----
  bm25(qterms) {
    const scores = new Float64Array(this.N);
    for (const [term, qtf] of countInOrder(qterms)) {
      const ti = this.termIndex.get(term);
      if (ti === undefined) continue;
      const idf = this.idfBm25[ti];
      for (let i = this.offsets[ti]; i < this.offsets[ti + 1]; i++) {
        const d = this.postDocs[i], tf = this.postTfs[i];
        scores[d] += qtf * idf * tf * (this.k1 + 1) / (tf + this.norm[d]);
      }
    }
    return scores;
  }
  tfidf(qterms) {
    const scores = new Float64Array(this.N);
    const weights = [];
    let qn = 0;
    for (const [term, c] of countInOrder(qterms)) {
      const ti = this.termIndex.get(term);
      if (ti === undefined) continue;                                    // out-of-vocabulary terms are ignored
      const w = (1 + Math.log(c)) * this.idfTfidf[ti];
      weights.push([ti, w]);
      qn += w * w;
    }
    if (!qn) return scores;
    qn = Math.sqrt(qn);
    for (const [ti, w] of weights) {
      const wq = w / qn;
      for (let i = this.offsets[ti]; i < this.offsets[ti + 1]; i++) {
        const d = this.postDocs[i];
        scores[d] += wq * ((1 + Math.log(this.postTfs[i])) * this.idfTfidf[ti] / this.docNorm[d]);
      }
    }
    return scores;
  }

  // ---- dense model ----
  async loadDense(onProgress = () => {}) {
    if (this.dense) return this.dense;
    const [half, owner] = await Promise.all([
      this.loader.bin("passages.f16", Uint16Array), this.loader.bin("passage_doc.u16", Uint16Array)]);
    const table = halfToFloatTable();
    const vectors = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) vectors[i] = table[half[i]];
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = true;
    env.allowRemoteModels = false;                                       // model files are served with the site
    env.localModelPath = new URL("../models/", this.loader.base || location.href).href;
    const extractor = await pipeline("feature-extraction", this.meta.model,
      { dtype: this.meta.dtype || "q8", device: "wasm", progress_callback: onProgress });
    this.dense = { vectors, owner, dim: this.meta.dim, extractor };
    return this.dense;
  }
  async embed(query) {
    const out = await this.dense.extractor(query, { pooling: "mean", normalize: true });
    return out.data;
  }
  denseScores(qv) {
    const { vectors, owner, dim } = this.dense;
    const scores = new Float64Array(this.N).fill(-Infinity);
    for (let p = 0, o = 0; p < owner.length; p++, o += dim) {
      let dot = 0;
      for (let j = 0; j < dim; j++) dot += vectors[o + j] * qv[j];
      if (dot > scores[owner[p]]) scores[owner[p]] = dot;               // MaxP: best passage
    }
    return scores;
  }

  // ---- ranking ----
  rank(scores, depth = 1000, positiveOnly = false) {
    const idx = [];
    for (let i = 0; i < scores.length; i++) if (!positiveOnly || scores[i] > 0) idx.push(i);
    idx.sort((a, b) => (scores[b] - scores[a]) || (a - b));              // score desc, then index asc
    return idx.slice(0, depth);
  }
  rrf(rankings, k = this.meta.rrf_k, depth = 1000) {
    const fused = new Float64Array(this.N);
    for (const r of rankings) r.forEach((d, i) => { fused[d] += 1 / (k + i + 1); });
    return { ranking: this.rank(fused, depth, true), scores: fused };
  }

  /** Rank a query with every available model. qv: optional precomputed query vector. */
  async runAll(query, depth = 1000) {
    const q = this.preprocess(query);
    const sT = this.tfidf(q), sB = this.bm25(q);
    const runs = {
      "TF-IDF": { ranking: this.rank(sT, depth, true), scores: sT },
      "BM25": { ranking: this.rank(sB, depth, true), scores: sB },
    };
    if (this.dense) {
      const sD = this.denseScores(await this.embed(query));
      runs["Sentence Transformer"] = { ranking: this.rank(sD, depth), scores: sD };
      runs["Hybrid RRF"] = this.rrf([runs["BM25"].ranking, runs["Sentence Transformer"].ranking], this.meta.rrf_k, depth);
    }
    return { runs, qterms: q };
  }

  hasTerm(doc, term) {
    const ti = this.termIndex.get(term);
    if (ti === undefined) return false;
    let lo = this.offsets[ti], hi = this.offsets[ti + 1] - 1;
    while (lo <= hi) {                                                     // postings are sorted by doc id
      const mid = (lo + hi) >> 1, v = this.postDocs[mid];
      if (v === doc) return true;
      if (v < doc) lo = mid + 1; else hi = mid - 1;
    }
    return false;
  }
}

// ---------------------------------------------------------------- metrics --
const relevant = (qrels) => new Set(Object.entries(qrels).filter(([, g]) => g > 0).map(([d]) => d));
export function evaluateRun(ranking, qrels) {
  const rel = relevant(qrels);
  const p10 = ranking.slice(0, 10).filter((d) => rel.has(d)).length / 10;
  const r100 = rel.size ? ranking.slice(0, 100).filter((d) => rel.has(d)).length / rel.size : 0;
  let hits = 0, ap = 0, rr = 0;
  ranking.forEach((d, i) => {
    if (rel.has(d)) { hits += 1; ap += hits / (i + 1); if (!rr) rr = 1 / (i + 1); }
  });
  ap = rel.size ? ap / rel.size : 0;
  const gain = (g) => 2 ** g - 1;
  const dcg = ranking.slice(0, 10).reduce((s, d, i) => s + gain(qrels[d] || 0) / Math.log2(i + 2), 0);
  const ideal = Object.values(qrels).filter((g) => g > 0).sort((a, b) => b - a).slice(0, 10);
  const idcg = ideal.reduce((s, g, i) => s + gain(g) / Math.log2(i + 2), 0);
  return { "P@10": p10, "Recall@100": r100, AP: ap, "nDCG@10": idcg ? dcg / idcg : 0, RR: rr };
}

// ---------------------------------------------------------------- helpers --
function countInOrder(items) {
  const m = new Map();
  for (const x of items) m.set(x, (m.get(x) || 0) + 1);
  return m;
}

function halfToFloatTable() {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
}

// Porter stemmer (with the NLTK extensions that matter here). Only used for words
// that never occur in the corpus; corpus words use the exact NLTK stems shipped in stems.json.
const STEP2 = { ational: "ate", tional: "tion", enci: "ence", anci: "ance", izer: "ize", bli: "ble", alli: "al",
  entli: "ent", eli: "e", ousli: "ous", ization: "ize", ation: "ate", ator: "ate", alism: "al", iveness: "ive",
  fulness: "ful", ousness: "ous", aliti: "al", iviti: "ive", biliti: "ble", logi: "log", fulli: "ful", lessli: "less" };
const STEP3 = { icate: "ic", ative: "", alize: "al", iciti: "ic", ical: "ic", ful: "", ness: "" };
const c = "[^aeiou]", v = "[aeiouy]", C = c + "[^aeiouy]*", Vs = v + "[aeiou]*";
const MGR0 = new RegExp("^(" + C + ")?" + Vs + C), MEQ1 = new RegExp("^(" + C + ")?" + Vs + C + "(" + Vs + ")?$");
const MGR1 = new RegExp("^(" + C + ")?" + Vs + C + Vs + C), SV = new RegExp("^(" + C + ")?" + v);
const CVC = new RegExp("^" + C + v + "[^aeiouwxy]$");
export function porter(word) {
  let w = word;
  if (w.length < 3) return w;
  const firstY = w[0] === "y";
  if (firstY) w = "Y" + w.slice(1);
  let m;
  if ((m = /^(.+?)ies$/.exec(w))) w = w.length === 4 ? m[1] + "ie" : m[1] + "i";
  else if ((m = /^(.+?)(ss)es$/.exec(w))) w = m[1] + m[2];
  else if ((m = /^(.+?)([^s])s$/.exec(w))) w = m[1] + m[2];
  if ((m = /^(.+?)eed$/.exec(w))) { if (MGR0.test(m[1])) w = w.slice(0, -1); }
  else if ((m = /^(.+?)(ed|ing)$/.exec(w)) && SV.test(m[1])) {
    w = m[1];
    if (/(at|bl|iz)$/.test(w)) w += "e";
    else if (/([^aeiouylsz])\1$/.test(w)) w = w.slice(0, -1);
    else if (CVC.test(w)) w += "e";
  }
  if ((m = /^(.+?[^aeiou])y$/.exec(w)) && m[1].length > 1) w = m[1] + "i";
  if ((m = /^(.+?)(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti|logi|fulli|lessli)$/.exec(w)) && MGR0.test(m[1])) w = m[1] + STEP2[m[2]];
  if ((m = /^(.+?)(icate|ative|alize|iciti|ical|ful|ness)$/.exec(w)) && MGR0.test(m[1])) w = m[1] + STEP3[m[2]];
  if ((m = /^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$/.exec(w))) { if (MGR1.test(m[1])) w = m[1]; }
  else if ((m = /^(.+?)(s|t)(ion)$/.exec(w)) && MGR1.test(m[1] + m[2])) w = m[1] + m[2];
  if ((m = /^(.+?)e$/.exec(w)) && (MGR1.test(m[1]) || (MEQ1.test(m[1]) && !CVC.test(m[1])))) w = m[1];
  if (/ll$/.test(w) && MGR1.test(w)) w = w.slice(0, -1);
  if (firstY) w = "y" + w.slice(1);
  return w;
}
