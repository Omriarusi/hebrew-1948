// SentencePiece BPE tokenizer (port of the project's 16K Hebrew model) + the project's
// line handling: each line is encoded on its own and lines are joined with the <n> token.
// Verified token-for-token against Python SentencePiece by scripts/test_tokenizer.mjs.

const SPACE = "▁"; // ▁

export class HebTokenizer {
  constructor(spec) {
    this.pieces = spec.pieces;
    this.scores = spec.scores;
    this.types = spec.types;
    this.eos = spec.eos_id;
    this.nl = spec.nl_id;
    this.unk = spec.unk_id;
    this.vocab = new Map();   // mergeable pieces only
    this.byteId = new Array(256);
    spec.pieces.forEach((p, i) => {
      const t = spec.types[i];
      if (t === "normal") this.vocab.set(p, i);
      else if (t === "byte") this.byteId[parseInt(p.slice(3, 5), 16)] = i;
    });
    this.utf8 = new TextEncoder();
  }

  // SentencePiece normalization with the "identity" rule: collapse/strip spaces, add the
  // dummy-prefix ▁, and turn spaces into ▁.
  _normalize(text) {
    const t = text.replace(/ {2,}/g, " ").replace(/^ +| +$/g, "");
    return t ? SPACE + t.replaceAll(" ", SPACE) : "";
  }

  // BPE: repeatedly merge the adjacent pair whose merged piece has the highest score
  // (ties -> leftmost), exactly like sentencepiece's bpe_model.cc.
  _bpe(s) {
    const sym = Array.from(s);                 // unicode characters
    const n = sym.length;
    if (!n) return [];
    const prev = new Int32Array(n), next = new Int32Array(n), alive = new Uint8Array(n).fill(1);
    for (let i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
    const heap = [];                            // [score, left, right, mergedPiece]
    const better = (a, b) => a[0] > b[0] || (a[0] === b[0] && a[1] < b[1]);
    const push = (item) => {
      heap.push(item);
      let i = heap.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (!better(heap[i], heap[p])) break; [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; }
    };
    const pop = () => {
      const top = heap[0], last = heap.pop();
      if (heap.length) {
        heap[0] = last; let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1; let m = i;
          if (l < heap.length && better(heap[l], heap[m])) m = l;
          if (r < heap.length && better(heap[r], heap[m])) m = r;
          if (m === i) break; [heap[i], heap[m]] = [heap[m], heap[i]]; i = m;
        }
      }
      return top;
    };
    const tryPair = (l, r) => {
      if (l < 0 || r < 0) return;
      const merged = sym[l] + sym[r], id = this.vocab.get(merged);
      if (id !== undefined) push([this.scores[id], l, r, merged]);
    };
    for (let i = 0; i + 1 < n; i++) tryPair(i, i + 1);
    while (heap.length) {
      const [, l, r, merged] = pop();
      if (!alive[l] || !alive[r] || next[l] !== r || sym[l] + sym[r] !== merged) continue; // stale
      sym[l] = merged; alive[r] = 0;
      next[l] = next[r]; if (next[r] >= 0) prev[next[r]] = l;
      tryPair(prev[l], l); tryPair(l, next[l]);
    }
    const ids = [];
    for (let i = 0; i >= 0 && i < n; i = next[i]) {
      if (!alive[i]) continue;
      const id = this.vocab.get(sym[i]);
      if (id !== undefined) ids.push(id);
      else for (const b of this.utf8.encode(sym[i])) ids.push(this.byteId[b] ?? this.unk); // byte fallback
    }
    return ids;
  }

  encodeLine(line) { return this._bpe(this._normalize(line)); }

  // whole text: lines joined by <n> (same as scripts/hebtok.py)
  encode(text) {
    const out = [];
    text.split("\n").forEach((line, i) => { if (i) out.push(this.nl); out.push(...this.encodeLine(line.trim())); });
    return out;
  }

  _decodeLine(ids) {
    const bytes = [];
    let s = "";
    const flush = () => { if (bytes.length) { s += new TextDecoder().decode(new Uint8Array(bytes)); bytes.length = 0; } };
    for (const id of ids) {
      const t = this.types[id];
      if (t === "byte") bytes.push(parseInt(this.pieces[id].slice(3, 5), 16));
      else { flush(); if (t === "normal") s += this.pieces[id]; }
    }
    flush();
    s = s.replaceAll(SPACE, " ");
    return s.startsWith(" ") ? s.slice(1) : s;  // drop the dummy prefix
  }

  decode(ids) {
    const lines = [[]];
    for (const id of ids) { if (id === this.nl) lines.push([]); else if (id !== this.eos) lines[lines.length - 1].push(id); }
    return lines.map((l) => this._decodeLine(l)).join("\n");
  }
}
