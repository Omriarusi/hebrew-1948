// Runs the model off the main thread. Messages:
//   in : {type:"load"} | {type:"generate", id, prompt, opts} | {type:"stop"}
//   out: {type:"progress", loaded, total} | {type:"ready", info} | {type:"piece", id, text}
//        {type:"done", id, reason} | {type:"error", message}
import * as ort from "./vendor/ort.wasm.min.mjs";
import { Engine } from "./engine.js";

const base = new URL("./", import.meta.url);
ort.env.wasm.wasmPaths = new URL("vendor/", base).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let engine = null, controller = null;

async function fetchModel(url, version, size) {
  // keep the model in Cache Storage: the ~45MB download happens once per browser (per model version)
  let cache = null;
  try {
    const name = "heb1948-" + version;
    for (const k of await caches.keys()) if (k.startsWith("heb1948-") && k !== name) await caches.delete(k);
    cache = await caches.open(name);
  } catch { /* private mode etc. */ }
  const hit = cache && await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error("model download failed: " + res.status);
  // the host may gzip the file, so Content-Length can be the compressed size: prefer the real one
  const total = size || +res.headers.get("Content-Length") || 0;
  const reader = res.body.getReader(), chunks = [];
  let loaded = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    postMessage({ type: "progress", loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  if (cache) { try { await cache.put(url, new Response(buf)); } catch { /* quota: fine, just no cache */ } }
  return buf;
}

async function load() {
  const cfg = await fetch(new URL("model/config.json", base), { cache: "no-cache" }).then((r) => r.json());
  const ver = cfg.version || "dev";            // fingerprint of the model file (scripts/export_onnx.py)
  const spec = await fetch(new URL("model/tokenizer.json?" + ver, base)).then((r) => r.json());
  const bytes = await fetchModel(new URL("model/model_q8.onnx?" + ver, base).href, ver, cfg.model_bytes);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  engine = new Engine(ort, session, spec, cfg);
  postMessage({ type: "ready", info: { ...cfg, threads: ort.env.wasm.numThreads } });
}

onmessage = async ({ data: m }) => {
  try {
    if (m.type === "load") await load();
    else if (m.type === "stop") controller?.abort();
    else if (m.type === "generate") {
      controller = new AbortController();
      let reason = "end";
      for await (const text of engine.generate(m.prompt, { ...m.opts, signal: controller.signal })) {
        postMessage({ type: "piece", id: m.id, text });
      }
      if (controller.signal.aborted) reason = "stopped";
      postMessage({ type: "done", id: m.id, reason });
    }
  } catch (e) {
    postMessage({ type: "error", id: m.id, message: String(e?.message || e) });
  }
};
