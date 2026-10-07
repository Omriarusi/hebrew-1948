// Text generation with the exported model (onnxruntime-web), mirroring scripts/generate.py:
// same prompt normalization, EOS-prefixed context, repetition penalty on word tokens only,
// temperature + top-k sampling, and a KV cache so each new token costs one small step.
import { HebTokenizer } from "./tokenizer.js";

const INVISIBLE = /[​-‏‪-‮⁦-⁩﻿­]/g;
const NIKKUD = /[֑-ׇֽֿׁׂׅׄ׀׆]/g;
const CHAR_MAP = {
  "“": '"', "”": '"', "„": '"', "״": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‚": "'", "׳": "'",
  "־": "-", "‑": "-", "⁻": "-", "−": "-", "—": "–", "׃": ":", "|": "", " ": " ", "\t": " ",
};
const CHAR_RE = new RegExp("[" + Object.keys(CHAR_MAP).join("").replace(/[-\]\\^|]/g, "\\$&") + "]", "g");

// the subset of the corpus cleaning (scripts/02_filter_clean.py) that matters for typed input
export function normalize(text) {
  return text
    .replace(INVISIBLE, "")
    .replace(/[יִ-ﭏ]/g, (c) => c.normalize("NFKD"))
    .replace(NIKKUD, "")
    .replace(CHAR_RE, (c) => CHAR_MAP[c])
    .replace(/(?<=[א-ת])=(?=[א-ת])/g, "-")
    .replace(/ {2,}/g, " ")
    .replace(/ +([,.;:!?])/g, "$1");
}

export class Engine {
  constructor(ort, session, tokSpec, cfg) {
    this.ort = ort;
    this.session = session;
    this.tok = new HebTokenizer(tokSpec);
    this.cfg = cfg;
    this.hd = cfg.n_embd / cfg.n_head;
    // repetition penalty applies to tokens with letters/digits only, never to punctuation or <n>
    this.wordMask = Uint8Array.from(tokSpec.pieces, (p, i) =>
      tokSpec.types[i] === "normal" && /[א-תA-Za-z0-9]/.test(p) ? 1 : 0);
  }

  _empty() {
    const { n_layer: L, n_head: H } = this.cfg;
    const t = () => new this.ort.Tensor("float32", new Float32Array(0), [L, 1, H, 0, this.hd]);
    return { k: t(), v: t() };
  }

  async _run(ids, past) {
    const idx = new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
    const r = await this.session.run({ idx, past_k: past.k, past_v: past.v });
    return { logits: r.logits.data, k: r.present_k, v: r.present_v };
  }

  _sample(logits, ctx, { temperature, topK, repPenalty, repWindow }) {
    const l = Float32Array.from(logits);
    if (repPenalty !== 1) {
      for (const id of new Set(ctx.slice(-repWindow))) {
        if (this.wordMask[id]) l[id] = l[id] > 0 ? l[id] / repPenalty : l[id] * repPenalty;
      }
    }
    let best = 0;
    for (let i = 1; i < l.length; i++) if (l[i] > l[best]) best = i;
    if (temperature <= 0 || topK === 1) return best;                       // greedy
    const k = Math.min(topK, l.length);
    const kth = Float32Array.from(l).sort()[l.length - k];                // k-th largest logit
    let sum = 0;
    const cand = [], w = [];
    for (let i = 0; i < l.length; i++) {
      if (l[i] >= kth) { const e = Math.exp((l[i] - l[best]) / temperature); cand.push(i); w.push(e); sum += e; }
    }
    let r = Math.random() * sum;
    for (let j = 0; j < cand.length; j++) { r -= w[j]; if (r <= 0) return cand[j]; }
    return cand[cand.length - 1];
  }

  // Yields pieces of the continuation as they are produced. Stops at end-of-document,
  // after maxNewTokens, or when signal.aborted becomes true.
  async *generate(prompt, opts = {}) {
    const o = { maxNewTokens: 160, temperature: 0.8, topK: 100, repPenalty: 1.2, repWindow: 256, ...opts };
    const tok = this.tok, block = this.cfg.block_size;
    prompt = normalize(prompt).trim();
    const promptIds = prompt ? tok.encode(prompt) : [];
    let ctx = [tok.eos, ...promptIds];
    if (ctx.length > block - 64) ctx = ctx.slice(-(block - 64));   // very long text: keep the end
    let st = await this._run(ctx, this._empty());
    let cached = ctx.length;
    const out = [];
    let shown = tok.decode(promptIds);
    for (let step = 0; step < o.maxNewTokens; step++) {
      if (o.signal?.aborted) return;
      const next = this._sample(st.logits, ctx, o);
      if (next === tok.eos) return;
      out.push(next);
      ctx.push(next);
      const text = tok.decode(promptIds.concat(out));
      if (text.length > shown.length && text.startsWith(shown)) { yield text.slice(shown.length); shown = text; }
      if (cached + 1 >= block) {                       // context full: restart the cache on the recent window
        const win = ctx.slice(-(block / 2));
        st = await this._run(win, this._empty());
        cached = win.length;
      } else {
        st = await this._run([next], st);
        cached += 1;
      }
    }
  }
}
