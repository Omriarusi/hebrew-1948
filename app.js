const $ = (id) => document.getElementById(id);
const page = $("page"), paper = $("paper");
const EXAMPLES = ["ויהי בימים ההם", "ירושלים, עיר הקודש,", "אחינו בני ישראל!", "הלילה ירד על העיירה, ו", "יקירי, קבלתי את מכתבך"];
let ready = false, busy = false, genSpans = [], current = null, reqId = 0;

// plain-text editing where supported (Enter inserts a real line break, paste drops formatting)
page.contentEditable = "plaintext-only";
const EDIT_MODE = page.contentEditable === "plaintext-only" ? "plaintext-only" : "true";
page.contentEditable = EDIT_MODE;

// ---- settings, remembered per browser (works without storage too) ----
const store = {
  get(k, d) { try { const v = localStorage.getItem("heb1948." + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("heb1948." + k, v); } catch {} },
};
function bindRange(id, fmt) {
  const el = $(id), out = $(id + "Val");
  el.value = store.get(id, el.value);
  const show = () => { out.textContent = fmt(+el.value); store.set(id, el.value); };
  el.addEventListener("input", show); show();
}
bindRange("len", (v) => `· כ־${Math.round(v / 1.6)} מילים`);
bindRange("temp", (v) => `· ${v.toFixed(2)}`);
bindRange("rep", (v) => `· ${v.toFixed(2)}`);

// ---- model worker ----
const worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
worker.onmessage = ({ data: m }) => {
  if (m.type === "progress") {
    const mb = (b) => (b / 2 ** 20).toFixed(0);
    $("meter").style.width = m.total ? `${Math.min(100, (100 * m.loaded) / m.total)}%` : "50%";
    $("loadText").textContent = m.total ? `טוען את המודל… ${mb(m.loaded)} / ${mb(m.total)}MB` : `טוען את המודל… ${mb(m.loaded)}MB`;
  } else if (m.type === "ready") {
    ready = true; $("loading").hidden = true; $("aboutParams").textContent = Math.round(m.info.params_m);
    setBusy(false); caretToEnd();
  } else if (m.type === "piece" && current && m.id === current.id) {
    let t = m.text;
    if (current.first && /\s$/.test(current.prompt)) t = t.replace(/^ +/, "");   // you already typed the space
    current.first = false; current.chars += t.length;
    current.span.textContent += t; paper.scrollTop = paper.scrollHeight;
  } else if (m.type === "done" && current && m.id === current.id) {
    finish(m.reason === "stopped" ? "נעצר" : current.chars ? `נכתבו ${current.chars} תווים ב־${((performance.now() - current.t0) / 1000).toFixed(1)} שניות` : "המודל החליט שהטקסט נגמר כאן");
  } else if (m.type === "error") {
    if (!ready) { $("loadText").textContent = "לא הצלחתי לטעון את המודל: " + m.message; $("loadText").className = "err"; }
    else finish("שגיאה: " + m.message, true);
  }
};
worker.postMessage({ type: "load" });

// ---- generation ----
function setBusy(b) {
  busy = b;
  $("go").disabled = b || !ready; $("stop").disabled = !b; $("clear").disabled = b;
  $("undo").disabled = b || genSpans.length === 0;
  page.contentEditable = b ? "false" : EDIT_MODE;   // no typing while the model writes
}
function caretToEnd() {
  if (matchMedia("(hover: none)").matches) return;   // don't pop the phone keyboard by itself
  page.focus();
  const r = document.createRange(); r.selectNodeContents(page); r.collapse(false);
  const s = getSelection(); s.removeAllRanges(); s.addRange(r);
}
function go() {
  if (!ready || busy) return;
  const prompt = page.innerText.replace(/\n$/, "");
  const span = document.createElement("span"); span.className = "gen live";
  page.appendChild(span); genSpans.push(span);
  current = { id: ++reqId, span, prompt, first: true, chars: 0, t0: performance.now() };
  setBusy(true); $("status").textContent = "כותב…"; $("status").className = "";
  worker.postMessage({ type: "generate", id: current.id, prompt,
    opts: { maxNewTokens: +$("len").value, temperature: +$("temp").value, repPenalty: +$("rep").value } });
}
function finish(msg, isErr = false) {
  const { span } = current;
  span.classList.remove("live");
  if (!span.textContent) { span.remove(); genSpans.pop(); }
  current = null; setBusy(false); caretToEnd();
  $("status").textContent = msg; $("status").className = isErr ? "err" : "";
}
$("go").onclick = go;
$("stop").onclick = () => worker.postMessage({ type: "stop" });
$("undo").onclick = () => { const s = genSpans.pop(); if (s) s.remove(); setBusy(false); caretToEnd(); };
$("clear").onclick = () => { page.textContent = ""; genSpans = []; setBusy(false); caretToEnd(); };
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); go(); }
  if (e.key === "Escape" && busy) worker.postMessage({ type: "stop" });
});
for (const ex of EXAMPLES) {
  const b = document.createElement("button"); b.className = "chip"; b.textContent = ex;
  b.onclick = () => { if (busy) return; page.textContent = ex; genSpans = []; setBusy(false); if (ready) go(); };
  $("chips").appendChild(b);
}
setBusy(false);
