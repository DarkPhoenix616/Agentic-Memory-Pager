// AMP Chat front end: plain JavaScript, no build step, no external libraries.
// Talks to app/server.py: GET /api/state, POST /api/chat (streamed), /api/settings, /api/new,
// GET /api/sessions and POST /api/open (the chats that can be reopened).
"use strict";

const $ = (id) => document.getElementById(id);
const ICONS = { monitor: "📊", dispatch: "🔀", compaction: "🗜️", search: "🔎", code: "🐍", page_in: "📥", reply: "💬", error: "⚠️", resume: "📂", blocked: "🛡️" };
const fmt = (n) => Number(n).toLocaleString("en-US");
let state = null;
let busy = false;
let ready = false;
let pollTimer = null;

// ---------------------------------------------------------------------------
// Safe rendering
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Small Markdown subset: code blocks, inline code, bold, italics, headings, lists, paragraphs.
// The text is escaped first, so a reply can never inject HTML into the page.
function markdown(text) {
  const blocks = [];
  const source = escapeHtml(text).replace(/```[\w+#.-]*[^\n]*\n([\s\S]*?)(?:```|$)/g, (_, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const inline = (s) => s
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const line of source.split("\n")) {
    let m;
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else {
      closeList();
      if ((m = line.match(/^#{1,6}\s+(.*)$/))) out.push(`<h4>${inline(m[1])}</h4>`);
      else if (/^\u0000\d+\u0000$/.test(line.trim())) out.push(line.trim());
      else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join("\n").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

async function refresh() {
  try {
    state = await api("/api/state");
  } catch (err) {
    showError(`Cannot reach the AMP server: ${err.message}`);
    return;
  }
  render();
  clearTimeout(pollTimer);
  if (state.reclaiming) pollTimer = setTimeout(refresh, 1200);   // watch background compaction live
}

async function checkModel() {
  try {
    const status = await api("/api/status");
    ready = status.ready;
    $("model").textContent = status.message;
    $("model").className = `pill ${ready ? "ok" : "bad"}`;
    const vault = status.vault || {};
    $("vault-mode").textContent = vault.mode === "mcp" ? "Memory: over MCP" : "Memory: in-process";
    $("vault-mode").title = vault.message || "";
    $("vault-mode").className = `pill ${vault.ok ? "ok" : "bad"}`;
    if (!vault.ok) {
      ready = false;
      showError(vault.message);
    }
  } catch (err) {
    ready = false;
    $("model").textContent = "server not reachable";
    $("model").className = "pill bad";
  }
  setComposer();
  if (!ready) setTimeout(checkModel, 10000);
}

// ---------------------------------------------------------------------------
// Rendering the state
// ---------------------------------------------------------------------------

function messageNode(role, content, status) {
  const node = document.createElement("div");
  const paged = status && status !== "in context";
  node.className = `msg ${role}${paged ? " paged" : ""}`;
  node.innerHTML = `<div class="bubble">${markdown(content)}</div>` +
    (paged ? `<div class="tag">📦 ${escapeHtml(status)}: compressed into the vault</div>` : "");
  return node;
}

// A chat reopened from before chats saved their messages: show its saved summaries
// instead, oldest first, so the old conversation is still on screen.
function legacySummaries() {
  if ((state.restored_messages || 0) > 0 || (state.resumed || []).length === 0) return [];
  return [...(state.vault || [])].reverse();
}

function summaryNode(m) {
  const node = document.createElement("div");
  node.className = "msg assistant paged summary";
  node.innerHTML = `<div class="bubble">${markdown(m.text)}</div>` +
    `<div class="tag">📦 saved summary${m.engine === "AST" ? " of code" : ""}</div>`;
  return node;
}

function renderMessages() {
  const box = $("messages");
  box.querySelectorAll(".msg").forEach((n) => n.remove());
  const legacy = legacySummaries();
  // a new, empty chat shows only the greeting
  $("empty").hidden = state.transcript.length > 0 || legacy.length > 0 || (state.resumed || []).length > 0 ||
    state.restored > 0;
  for (const m of legacy) box.appendChild(summaryNode(m));
  for (const m of state.transcript) box.appendChild(messageNode(m.role, m.content, m.status));
  box.scrollTop = box.scrollHeight;
}

function setMeter(tokens) {
  const c = state.context;
  const pct = (x) => Math.min(100, (x / c.limit) * 100);
  $("meter-fill").style.width = `${pct(tokens)}%`;
  $("meter-fill").classList.toggle("over", tokens >= c.trigger);
  $("meter-text").textContent = `${fmt(tokens)} / ${fmt(c.limit)} tokens (${Math.round((tokens / c.limit) * 100)}%)`;
}

function renderMeter() {
  const c = state.context;
  const pct = (x) => Math.min(100, (x / c.limit) * 100);
  setMeter(c.tokens);
  $("meter-trigger").style.left = `${pct(c.trigger)}%`;
  $("meter-target").style.left = `${pct(c.reclaim_target)}%`;
  $("meter-note").textContent =
    `${c.chunks} chunks in the window (${c.chunks - c.code_chunks} text, ${c.code_chunks} code). ` +
    `Compaction starts at the dark line (${fmt(c.trigger)}); background compaction stops at the light one (${fmt(c.reclaim_target)}).`;
  $("reclaiming").hidden = !state.reclaiming;
  $("stat-compactions").textContent = fmt(state.stats.compactions);
  $("stat-out").textContent = fmt(state.stats.tokens_paged_out);
  $("stat-stored").textContent = fmt(state.stats.tokens_stored);
}

function memoryCard(memory, distance) {
  const card = document.createElement("details");
  card.className = "memory";
  const lists = ["decisions", "bugs", "constraints"]
    .filter((k) => memory[k] && memory[k].length)
    .map((k) => `<b>${k}</b><ul>${memory[k].map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`)
    .join("");
  const dist = distance === undefined || distance === null ? "" : ` · distance ${Number(distance).toFixed(2)}`;
  card.innerHTML =
    `<summary><span class="badge ${escapeHtml(memory.engine.toLowerCase())}">${escapeHtml(memory.engine)}</span> ` +
    `${escapeHtml(memory.label)} · ${fmt(memory.tokens)} tokens${dist}</summary>` +
    `<div class="memtext">${escapeHtml(memory.text)}</div>` + (lists ? `<div class="lists">${lists}</div>` : "");
  return card;
}

function fillList(id, items, emptyText, build) {
  const box = $(id);
  box.replaceChildren();
  if (!items.length) {
    box.innerHTML = `<div class="none">${escapeHtml(emptyText)}</div>`;
    return;
  }
  items.forEach((item) => box.appendChild(build(item)));
}

function eventNode(event) {
  const took = event.seconds >= 0.05 ? ` <span class="took">· ${event.seconds.toFixed(1)}s</span>` : "";
  if (event.kind === "compaction" && event.data && event.data.memory) {
    const node = document.createElement("details");
    node.className = "event";
    node.innerHTML = `<summary>${ICONS.compaction} ${escapeHtml(event.title)}${took}</summary>` +
      `<div class="detail">${escapeHtml(event.detail)}</div>`;
    return node;
  }
  const node = document.createElement("div");
  node.className = "event";
  node.innerHTML = `${ICONS[event.kind] || "•"} ${escapeHtml(event.title)}${took}` +
    (event.detail ? `<div class="detail">${escapeHtml(event.detail.slice(0, 300))}</div>` : "");
  return node;
}

function renderActivity() {
  const box = $("activity");
  box.replaceChildren();
  const turns = [...new Set(state.events.map((e) => e.turn))].sort((a, b) => b - a).slice(0, 8);
  if (!turns.length) {
    box.innerHTML = '<div class="none">Each step of a turn shows up here.</div>';
    return;
  }
  for (const turn of turns) {
    const head = document.createElement("div");
    head.className = "turn-head";
    head.textContent = `Turn ${turn}`;
    box.appendChild(head);
    state.events.filter((e) => e.turn === turn).forEach((e) => box.appendChild(eventNode(e)));
  }
}

function renderSettings() {
  const s = state.settings;
  $("limit").min = s.min_token_limit;
  $("limit").max = s.max_token_limit;
  if (document.activeElement !== $("limit")) $("limit").value = s.token_limit;
  if (document.activeElement !== $("threshold")) $("threshold").value = Math.round(s.threshold * 100);
  $("background").checked = s.background;
  $("limit-out").textContent = `${fmt($("limit").value)} tokens`;
  $("threshold-out").textContent = `${$("threshold").value}%`;
}

function render() {
  if (!busy) renderMessages();
  renderMeter();
  fillList("paged", state.last_paged, "Nothing yet: the vault fills once the window passes the trigger.",
    (m) => memoryCard(m, m.distance));
  renderActivity();
  $("vault-title").textContent = `Vault: ${state.vault.length} ${state.vault.length === 1 ? "memory" : "memories"} in this chat`;
  fillList("vault", state.vault, "Empty.", (m) => memoryCard(m));
  renderSettings();
}

// ---------------------------------------------------------------------------
// Sending a message (the reply streams back step by step)
// ---------------------------------------------------------------------------

function setComposer() {
  $("send").disabled = busy || !ready;
  $("input").disabled = !ready;
  $("new-chat").disabled = busy;
  document.querySelectorAll("#chats button").forEach((b) => { b.disabled = busy || b.classList.contains("current"); });
}

function showNotice(message) {
  $("notice").textContent = message || "";
  $("notice").hidden = !message;
}

// ---------------------------------------------------------------------------
// Chats that can be reopened
// ---------------------------------------------------------------------------

function when(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function chatNode(chat) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = `chat-item${chat.current ? " current" : ""}`;
  node.innerHTML = `<span class="chat-title">${escapeHtml(chat.title || chat.session_id)}</span>` +
    `<span class="chat-meta">` +
    (chat.messages ? `${fmt(chat.messages)} ${chat.messages === 1 ? "message" : "messages"} · ` : "") +
    `${fmt(chat.memories)} ${chat.memories === 1 ? "memory" : "memories"}` +
    (chat.unsaved ? " · unsaved part kept" : "") +
    (chat.last_time ? ` · ${escapeHtml(when(chat.last_time))}` : "") +
    (chat.current ? " · this chat" : "") + `</span>`;
  node.disabled = busy || chat.current;
  node.addEventListener("click", () => openChat(chat.session_id));
  return node;
}

async function loadChats() {
  try {
    const data = await api("/api/sessions");
    fillList("chats", data.chats, "No saved chats yet. A chat is kept after every message.", chatNode);
  } catch (err) {
    fillList("chats", [], `Cannot list the chats: ${err.message}`, chatNode);
  }
}

async function leave(path, body, message) {
  if (busy) return;
  busy = true;
  setComposer();
  $("error").hidden = true;
  showNotice(message);
  let next = null;
  try {
    next = await api(path, body);
  } catch (err) {
    showError(err.message);
  } finally {
    busy = false;               // first: render() skips the messages while busy
    showNotice("");
    if (next) {
      state = next;
      render();                 // the new or reopened chat's messages replace the old ones
    }
    setComposer();
    loadChats();
  }
}

function openChat(sessionId) {
  return leave("/api/open", { session_id: sessionId }, "Saving this chat to the vault, then opening the other one…");
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function addStep(steps, event) {
  const step = document.createElement("div");
  step.className = `step kind-${event.kind}`;
  const took = event.seconds >= 0.05 ? ` · ${event.seconds.toFixed(1)}s` : "";
  step.textContent = `${ICONS[event.kind] || "•"} ${event.title}${took}`;
  steps.insertBefore(step, steps.lastElementChild);
  if (event.kind === "monitor" && event.data && event.data.tokens !== undefined) setMeter(event.data.tokens);
}

async function send(text) {
  busy = true;
  setComposer();
  $("error").hidden = true;
  $("empty").hidden = true;
  const box = $("messages");
  box.appendChild(messageNode("user", text));
  const pending = document.createElement("div");
  pending.className = "msg assistant";
  pending.innerHTML = '<div class="bubble"><div class="steps"><div class="step"><span class="spinner"></span>Working…</div></div></div>';
  box.appendChild(pending);
  box.scrollTop = box.scrollHeight;
  const steps = pending.querySelector(".steps");

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `${res.status} ${res.statusText}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let failure = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const kind = (frame.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((frame.match(/^data: (.*)$/m) || [null, "{}"])[1]);
        if (kind === "event") {
          addStep(steps, data);
          box.scrollTop = box.scrollHeight;
        } else if (kind === "error") {
          failure = data.message;
        }
      }
    }
    if (failure) throw new Error(failure);
  } catch (err) {
    showError(err.message);
  } finally {
    busy = false;
    await refresh();
    setComposer();
    loadChats();
    $("input").focus();
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function autoSize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("input").value.trim();
  if (!text || busy || !ready) return;
  $("input").value = "";
  autoSize();
  send(text);
});

$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("input").addEventListener("input", autoSize);

$("new-chat").addEventListener("click", () => leave("/api/new", {}, "Saving this chat to the vault…"));

let settingsTimer = null;
function saveSettings() {
  $("limit-out").textContent = `${fmt($("limit").value)} tokens`;
  $("threshold-out").textContent = `${$("threshold").value}%`;
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(async () => {
    try {
      state = await api("/api/settings", {
        token_limit: Number($("limit").value),
        threshold: Number($("threshold").value) / 100,
        background: $("background").checked,
      });
      render();
    } catch (err) {
      showError(err.message);
    }
  }, 250);
}
["limit", "threshold"].forEach((id) => $(id).addEventListener("input", saveSettings));
$("background").addEventListener("change", saveSettings);

checkModel();
refresh();
loadChats();
