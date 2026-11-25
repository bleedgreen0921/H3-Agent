const state = {
  sessions: [], currentSession: null, currentInput: null,
  attachments: [], streamText: "", streamEl: null, source: null,
  planning: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (response.status === 401) { location.href = "/login"; throw new Error("unauthorized"); }
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addMessage(role, text, html, meta) {
  const item = el("div", `message ${role}`);
  item.appendChild(el("div", "role", role === "user" ? "你" : "助手"));
  const body = el("div", "body");
  if (html) body.innerHTML = html;
  else body.appendChild(el("pre", "", text || ""));
  item.appendChild(body);
  if (meta) item.appendChild(el("div", "meta", meta));
  $("messages").appendChild(item);
  $("messages").scrollTop = $("messages").scrollHeight;
  return body;
}

function statusLine(text) {
  const node = el("div", "tool-status", text);
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  const list = $("session-list");
  list.innerHTML = "";
  for (const session of state.sessions) {
    const node = el("div", "session-item", session.title || "新会话");
    node.title = session.id;
    if (session.id === state.currentSession) node.classList.add("active");
    node.onclick = () => selectSession(session.id);
    list.appendChild(node);
  }
  if (!state.currentSession && state.sessions.length) {
    await selectSession(state.sessions[0].id);
  }
}

async function selectSession(id) {
  state.currentSession = id;
  const session = state.sessions.find((item) => item.id === id);
  $("session-title").textContent = session ? (session.title || "新会话") : id;
  document.querySelectorAll(".session-item").forEach((node) => {
    node.classList.toggle("active", node.title === id);
  });
  $("messages").innerHTML = "";
  const rows = await api(`/api/sessions/${id}/messages`);
  for (const row of rows) {
    addMessage(row.role, row.text, row.html,
      row.created_at ? new Date(row.created_at).toLocaleString() : "");
  }
  await loadPlan();
}

function planStatusLabel(status) {
  return ({draft: "草稿", running: "执行中", completed: "已完成",
           failed: "失败", cancelled: "已取消"}[status] || status || "草稿");
}

async function loadPlan() {
  if (!state.currentSession) return;
  try {
    const data = await api(`/api/plan/${state.currentSession}`);
    state.planning = data.planning || null;
  } catch (_) {
    state.planning = null;
  }
  const planning = state.planning;
  const badge = $("mode-badge");
  const panel = $("plan-panel");
  if (!planning) {
    badge.textContent = "普通对话";
    badge.classList.remove("plan");
    panel.classList.add("hidden");
    $("input").placeholder = "输入消息";
    return;
  }
  badge.textContent = `规划模式 · ${planStatusLabel(planning.status)}`;
  badge.classList.add("plan");
  panel.classList.remove("hidden");
  $("plan-status").textContent = planStatusLabel(planning.status);
  $("input").placeholder = "补充计划，或输入新的规划目标";
  const content = $("plan-content");
  content.innerHTML = "";
  const plan = planning.plan || {};
  const goal = plan.goal || planning.goal || "";
  if (goal) content.appendChild(el("div", "plan-line", `目标：${goal}`));
  const questions = plan.questions || planning.questions || [];
  for (const question of questions) {
    content.appendChild(el("div", "plan-question", `• ${question}`));
  }
  const tasks = plan.tasks || [];
  if (tasks.length) {
    content.appendChild(el("div", "plan-line", `任务（${tasks.length}）：`));
    tasks.forEach((task, index) => {
      content.appendChild(el("div", "plan-question",
        `${index + 1}. [${task.id}] ${task.description}`));
    });
  }
  if (planning.status === "completed") {
    content.appendChild(el("div", "plan-line", "计划已结束；发送新目标可重新规划。"));
  } else if (planning.status === "failed" || planning.status === "cancelled") {
    content.appendChild(el("div", "plan-line", "计划已结束；发送新目标可重新规划。"));
  }
}

async function newSession() {
  const data = await api("/api/sessions", { method: "POST" });
  await loadSessions();
  await selectSession(data.id);
}

async function uploadFiles(files) {
  if (!state.currentSession) await newSession();
  for (const file of files) {
    const form = new FormData();
    form.append("session_id", state.currentSession);
    form.append("file", file);
    const data = await api("/api/attachments", { method: "POST", body: form });
    state.attachments.push(data);
    renderChips();
  }
  $("file-input").value = "";
}

function renderChips() {
  const box = $("attachment-chips");
  box.innerHTML = "";
  state.attachments.forEach((item, index) => {
    const chip = el("span", "chip");
    chip.appendChild(el("span", "", `${item.name} · ${Math.round(item.bytes / 1024)}KB`));
    const ingest = el("button", "mini", "入库");
    ingest.title = "显式提交到知识库";
    ingest.onclick = async (event) => {
      event.stopPropagation();
      try {
        const result = await api(`/api/attachments/${item.id}/ingest?session_id=${state.currentSession}`,
          { method: "POST" });
        alert(JSON.stringify(result, null, 2));
      } catch (error) { alert(error.message); }
    };
    chip.appendChild(ingest);
    chip.onclick = () => { state.attachments.splice(index, 1); renderChips(); };
    box.appendChild(chip);
  });
}

function startStream(inputId) {
  if (state.source) state.source.close();
  state.currentInput = inputId;
  state.streamText = "";
  state.streamEl = addMessage("assistant", "", null, "正在处理…");
  const source = new EventSource(`/api/turns/${inputId}/events`);
  state.source = source;

  source.addEventListener("model.delta", (event) => {
    const data = JSON.parse(event.data);
    state.streamText += data.text || "";
    state.streamEl.innerHTML = "";
    state.streamEl.appendChild(el("pre", "", state.streamText));
    $("messages").scrollTop = $("messages").scrollHeight;
  });
  source.addEventListener("tool.started", (event) => {
    const data = JSON.parse(event.data);
    statusLine(`调用工具：${data.label || data.name || "外部工具"}…`);
  });
  source.addEventListener("tool.finished", () => statusLine("工具调用完成。"));
  source.addEventListener("tool.failed", () => statusLine("工具调用失败。"));
  source.addEventListener("answer.final", async (event) => {
    const data = JSON.parse(event.data);
    if (data.sources && data.sources.length) {
      const meta = data.sources.map((s) => {
        const m = s.metadata || {};
        return m.source_name || m.document_id || m.source_uri || "来源";
      }).join("；");
      const metaNode = el("div", "meta", `来源：${meta}`);
      state.streamEl.parentElement.appendChild(metaNode);
    }
    await selectSession(state.currentSession);
  });
  source.addEventListener("turn.failed", (event) => {
    const data = JSON.parse(event.data);
    addMessage("assistant", data.message || "本轮失败。");
  });
  source.addEventListener("turn.cancelled", (event) => {
    const data = JSON.parse(event.data);
    addMessage("assistant", data.message || "已取消。");
  });
  source.addEventListener("done", () => {
    source.close();
    state.source = null;
    state.currentInput = null;
  });
  source.onerror = () => {
    // 浏览器会自动重连；轮次结果仍可通过持久消息读取。
  };
}

async function sendMessage(text, extra = {}) {
  if (!state.currentSession) await newSession();
  if (!text.trim()) return;
  const attachmentIds = state.attachments.map((item) => item.id);
  state.attachments = [];
  renderChips();
  const data = await api(`/api/sessions/${state.currentSession}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, attachment_ids: attachmentIds, ...extra }),
  });
  addMessage("user", text, null, "");
  if (attachmentIds.length) statusLine(`已附加 ${attachmentIds.length} 个文件。`);
  startStream(data.input_id);
  $("input").value = "";
}

async function loadKnowledge() {
  $("knowledge-sources").textContent = "加载中…";
  $("knowledge-documents").textContent = "加载中…";
  try {
    const [sources, documents] = await Promise.all([
      api("/api/knowledge/sources"), api("/api/knowledge/documents")]);
    $("knowledge-sources").textContent = JSON.stringify(sources, null, 2);
    $("knowledge-documents").textContent = JSON.stringify(documents, null, 2);
  } catch (error) {
    $("knowledge-sources").textContent = error.message;
    $("knowledge-documents").textContent = error.message;
  }
}

async function loadSettings() {
  try { $("settings").textContent = JSON.stringify(await api("/api/settings"), null, 2); }
  catch (error) { $("settings").textContent = error.message; }
}

function showTab(name) {
  for (const view of ["chat", "knowledge", "settings"]) {
    $(`${view}-view`).classList.toggle("hidden", view !== name);
  }
  document.querySelectorAll(".tabs button").forEach((node) => {
    node.classList.toggle("active", node.dataset.tab === name);
  });
  if (name === "knowledge") loadKnowledge();
  if (name === "settings") loadSettings();
}

$("new-session").onclick = () => newSession().catch((e) => alert(e.message));
$("composer").onsubmit = (event) => {
  event.preventDefault();
  sendMessage($("input").value).catch((e) => alert(e.message));
};
$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("file-input").onchange = (event) => uploadFiles([...event.target.files]).catch((e) => alert(e.message));
$("cancel-button").onclick = async () => {
  if (!state.currentInput) return;
  await api(`/api/turns/${state.currentInput}/cancel`, { method: "POST" });
};
$("plan-button").onclick = () => {
  const goal = prompt("规划目标");
  if (goal) sendMessage(goal, { planning_action: "create", planning_goal: goal })
    .catch((e) => alert(e.message));
};
$("plan-run").onclick = () => sendMessage("执行计划", { planning_action: "run" })
  .catch((e) => alert(e.message));
$("plan-exit").onclick = () => sendMessage("退出规划模式", { planning_action: "exit" })
  .catch((e) => alert(e.message));
document.querySelectorAll(".tabs button").forEach((node) => {
  node.onclick = () => showTab(node.dataset.tab);
});
$("sync-knowledge").onclick = async () => {
  try { alert(JSON.stringify(await api("/api/knowledge/sync", { method: "POST" }), null, 2)); }
  catch (e) { alert(e.message); }
};
$("knowledge-form").onsubmit = async (event) => {
  event.preventDefault();
  try {
    const result = await api("/api/knowledge/submit", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference: $("knowledge-reference").value }),
    });
    alert(JSON.stringify(result, null, 2));
  } catch (e) { alert(e.message); }
};

loadSessions().catch((e) => alert(e.message));
