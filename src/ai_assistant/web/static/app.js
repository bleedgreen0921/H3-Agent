const state = {
  sessions: [],
  currentSession: null,
  currentInput: null,
  attachments: [],
  streamText: "",
  streamItem: null,
  streamBody: null,
  streamHead: null,
  streamPre: null,
  streamState: null,
  streamStatus: null,
  source: null,
  planning: null,
  taskResults: null,
  pendingPlan: false,
  planOpen: null,
  planTimer: null,
  knowledgeTimer: null,
  knowledgeJob: null,
  activeTab: "chat",
  storage: null,
  storageThread: null,
  storageCheckpoints: null,
  storageDetail: null,
  storageFull: false,
  storageRequest: 0,
};

const $ = (id) => document.getElementById(id);
const PLAN_LABELS = {
  draft: "草稿", running: "执行中", completed: "已完成",
  failed: "失败", cancelled: "已取消",
};
const TASK_ICONS = {
  done: "✓", failed: "!", unknown: "!", partial: "!",
  blocked: "■", needs_input: "?", cancelled: "×",
  superseded: "↷", running: "◐", pending: "○",
};
const SOURCE_KINDS = {
  local_directory: "本地目录", feishu_wiki: "飞书知识库",
  feishu_doc: "飞书文档", feishu_sheet: "飞书表格",
  feishu_base: "飞书多维表格",
};

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

let toastTimer = null;
function showToast(message, kind = "error") {
  const node = $("toast");
  node.textContent = message || "";
  node.className = `toast ${kind}`;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.className = "toast hidden"; }, 4200);
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fullTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function relativeTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const diff = Date.now() - date.getTime();
  if (diff < 60_000) return "刚刚";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`;
  if (diff < 7 * 86_400_000) return `${Math.floor(diff / 86_400_000)} 天前`;
  return date.toLocaleDateString();
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MiB`;
}

function isNearBottom() {
  const node = $("messages");
  return node.scrollHeight - node.scrollTop - node.clientHeight < 120;
}

function jumpToBottom() {
  const node = $("messages");
  node.scrollTop = node.scrollHeight;
}

function renderEmptyState(title, text) {
  const box = el("div", "empty-state");
  box.appendChild(el("div", "empty-title", title));
  box.appendChild(el("div", "empty-text", text));
  $("messages").appendChild(box);
}

function addMessage(role, text, html, createdAt) {
  const stick = isNearBottom();
  // 空会话提示只在消息列表为空时存在；第一条消息到达后立即移除。
  const empty = $("messages").querySelector(".empty-state");
  if (empty) empty.remove();
  const labels = { user: "你", assistant: "助手", system: "系统" };
  const item = el("div", `message ${role}`);
  const head = el("div", "message-head");
  head.appendChild(el("span", "role", labels[role] || role));
  if (createdAt) {
    const stamp = el("time", "message-time", formatTime(createdAt));
    stamp.title = fullTime(createdAt);
    head.appendChild(stamp);
  }
  item.appendChild(head);

  const body = el("div", "body");
  if (html) body.innerHTML = html;
  else body.appendChild(el("pre", "plain", text || ""));
  item.appendChild(body);
  $("messages").appendChild(item);
  if (stick) jumpToBottom();
  return { item, body, head };
}

function clearToolStatus() {
  if (state.streamStatus) state.streamStatus.remove();
  state.streamStatus = null;
}

function setToolStatus(text) {
  if (!state.streamItem) return;
  const stick = isNearBottom();
  if (!state.streamStatus) {
    state.streamStatus = el("div", "tool-status", text);
    state.streamItem.insertAdjacentElement("afterend", state.streamStatus);
  } else {
    state.streamStatus.textContent = text;
  }
  if (stick) jumpToBottom();
}

function clearStreamState() {
  if (state.streamState) state.streamState.remove();
  state.streamState = null;
}

function resetStreamState() {
  state.streamText = "";
  state.streamItem = null;
  state.streamBody = null;
  state.streamHead = null;
  state.streamPre = null;
  clearStreamState();
  clearToolStatus();
}

function setBusy(busy) {
  $("send-button").disabled = busy;
  $("mode-chat").disabled = busy;
  $("mode-plan").disabled = busy;
  $("new-session").disabled = busy;
  $("plan-run").disabled = busy;
  $("cancel-button").hidden = !busy;
}

function setModeButton(button, active) {
  button.classList.toggle("active", active);
  button.setAttribute("aria-pressed", active ? "true" : "false");
}

function planQuestions(planning) {
  const plan = planning && planning.plan ? planning.plan : {};
  if (Array.isArray(plan.questions) && plan.questions.length) return plan.questions;
  if (Array.isArray(planning && planning.questions)) return planning.questions;
  return [];
}

function taskStatusOf(taskId) {
  const items = state.taskResults && state.taskResults.items;
  const result = items && items[taskId];
  return result && result.status ? result.status : "pending";
}

function renderMode() {
  const planning = state.planning;
  const inPlan = Boolean(planning) || state.pendingPlan;
  setModeButton($("mode-chat"), !inPlan);
  setModeButton($("mode-plan"), inPlan);
  $("composer-mode").textContent = inPlan ? "规划模式" : "普通对话";

  if (state.pendingPlan) {
    $("input").placeholder = "输入规划目标，Enter 开始规划";
    $("composer-hint").textContent = "输入规划目标，Enter 开始规划；Esc 取消";
    $("composer-hint").classList.remove("hidden");
  } else if (planning) {
    $("input").placeholder = "补充计划，或输入新的规划目标";
    $("composer-hint").classList.add("hidden");
  } else {
    $("input").placeholder = "输入消息；Enter 发送，Shift+Enter 换行";
    $("composer-hint").classList.add("hidden");
  }

  const plan = planning && planning.plan ? planning.plan : {};
  const tasks = Array.isArray(plan.tasks) ? plan.tasks : [];
  const runnable = Boolean(planning && planning.status === "draft"
    && !planQuestions(planning).length && tasks.length);
  $("plan-run").hidden = !runnable;
  $("plan-run").disabled = Boolean(state.currentInput);
  $("cancel-button").hidden = !state.currentInput;
}

function setPlanOpen(open) {
  $("plan-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  $("plan-body").classList.toggle("hidden", !open);
}

function renderPlanPanel() {
  const planning = state.planning;
  if (!planning) {
    $("plan-panel").classList.add("hidden");
    $("plan-goal").textContent = "";
    $("plan-questions").innerHTML = "";
    $("plan-tasks").innerHTML = "";
    $("plan-note").textContent = "";
    $("plan-progress").textContent = "";
    state.planOpen = null;
    return;
  }

  const plan = planning.plan || {};
  const status = planning.status || "draft";
  const tasks = Array.isArray(plan.tasks) ? plan.tasks : [];
  const questions = planQuestions(planning);
  const complete = tasks.filter((task) => taskStatusOf(task.id) === "done").length;

  $("plan-panel").classList.remove("hidden");
  $("plan-status").textContent = `规划 · ${PLAN_LABELS[status] || status}`;
  $("plan-status-dot").className = `status-dot ${status}`;
  $("plan-progress").textContent = tasks.length ? `${complete}/${tasks.length}` : "";
  const goal = plan.goal || planning.goal || "";
  $("plan-goal").textContent = goal ? `目标：${goal}` : "正在生成计划…";

  const questionBox = $("plan-questions");
  questionBox.innerHTML = "";
  questions.forEach((question, index) => {
    questionBox.appendChild(el("div", "plan-question", `${index + 1}. ${question}`));
  });

  const taskBox = $("plan-tasks");
  taskBox.innerHTML = "";
  tasks.forEach((task) => {
    const taskStatus = taskStatusOf(task.id);
    const row = el("li", "plan-task");
    const icon = el("span", `task-icon ${taskStatus}`, TASK_ICONS[taskStatus] || "○");
    icon.setAttribute("aria-hidden", "true");
    const ident = el("span", "plan-task-id", task.id);
    const text = el("span", "plan-task-text", task.description || task.intent || "");
    text.title = task.description || task.intent || "";
    row.append(icon, ident, text);
    taskBox.appendChild(row);
  });

  let note = "";
  if (status === "draft") {
    note = questions.length ? "请直接在下方补充信息；补充后会重新规划。"
      : tasks.length ? "计划已就绪；可点击“执行计划”，或直接回复“开始执行”。"
        : "正在生成计划…";
  } else if (status === "running") {
    note = "任务执行中；可取消当前轮次。";
  } else if (status === "completed") {
    note = "计划已结束；发送新目标可重新规划，或切换到普通对话。";
  } else if (status === "failed") {
    note = "计划未完成；发送新目标可重新规划，或切换到普通对话。";
  } else if (status === "cancelled") {
    note = "计划已取消；发送新目标可重新规划，或切换到普通对话。";
  }
  if (status === "failed" || status === "cancelled") {
    const failure = Object.values((state.taskResults && state.taskResults.items) || {})
      .find((item) => item && item.error && item.error.message);
    if (failure) note += `\n原因：${failure.error.message}`;
  }
  $("plan-note").textContent = note;

  const autoOpen = status === "draft" || status === "running";
  setPlanOpen(state.planOpen === null ? autoOpen : state.planOpen);
}

function applyPlan(data) {
  state.planning = data && data.planning ? data.planning : null;
  state.taskResults = data && data.task_results ? data.task_results : null;
}

async function loadPlan() {
  if (!state.currentSession) {
    state.planning = null;
    state.taskResults = null;
    renderMode();
    renderPlanPanel();
    return;
  }
  const sessionId = state.currentSession;
  try {
    const data = await api(`/api/plan/${sessionId}`);
    if (state.currentSession !== sessionId) return;
    applyPlan(data);
  } catch (_) {
    state.planning = null;
    state.taskResults = null;
  }
  renderMode();
  renderPlanPanel();
  schedulePlanPoll();
}

function clearPlanTimer() {
  if (state.planTimer) clearTimeout(state.planTimer);
  state.planTimer = null;
}

function schedulePlanPoll() {
  clearPlanTimer();
  if (!state.currentSession || state.activeTab !== "chat" || document.hidden) return;
  if (!state.planning || state.planning.status !== "running") return;
  state.planTimer = setTimeout(async () => {
    state.planTimer = null;
    if (!state.currentSession) return;
    try {
      const data = await api(`/api/plan/${state.currentSession}`);
      applyPlan(data);
      renderMode();
      renderPlanPanel();
    } catch (_) {}
    schedulePlanPoll();
  }, 2500);
}

function updateTitle() {
  const session = state.sessions.find((item) => item.id === state.currentSession);
  const title = session ? (session.title || "新会话") : "未选择会话";
  $("session-title").textContent = title;
  $("session-title").title = title;
}

function renderSessions() {
  const list = $("session-list");
  list.innerHTML = "";
  for (const session of state.sessions) {
    const title = session.title || "新会话";
    const item = el("div", "session-item");
    item.dataset.id = session.id;
    item.title = title;
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    if (session.id === state.currentSession) {
      item.classList.add("active");
      item.setAttribute("aria-current", "true");
    }
    item.appendChild(el("div", "session-title", title));
    item.appendChild(el("div", "session-meta", relativeTime(session.updated_at)));
    const remove = el("button", "session-delete", "×");
    remove.type = "button";
    remove.title = "删除会话";
    remove.setAttribute("aria-label", `删除会话：${title}`);
    remove.onclick = async (event) => {
      event.stopPropagation();
      await deleteSession(session.id, title);
    };
    item.appendChild(remove);
    item.onclick = () => selectSession(session.id);
    item.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectSession(session.id);
      }
    };
    list.appendChild(item);
  }
}

async function refreshSessions({ autoSelect = true } = {}) {
  state.sessions = await api("/api/sessions");
  renderSessions();
  updateTitle();
  if (autoSelect && !state.currentSession && state.sessions.length) {
    await selectSession(state.sessions[0].id);
  }
}

function clearChat() {
  clearPlanTimer();
  state.currentSession = null;
  state.currentInput = null;
  state.pendingPlan = false;
  state.planning = null;
  state.taskResults = null;
  state.planOpen = null;
  $("messages").innerHTML = "";
  showTab("chat");
  setBusy(false);
  renderSessions();
  updateTitle();
  renderMode();
  renderPlanPanel();
  renderEmptyState("开始一段新对话", "直接输入消息，或切换到规划模式先整理目标。");
}

async function deleteSession(id, title) {
  if (state.currentInput) {
    showToast("当前轮次仍在执行，请等待完成或取消。", "info");
    return;
  }
  if (!confirm(`删除会话「${title}」？删除后无法恢复。`)) return;
  try {
    await api(`/api/sessions/${id}`, { method: "DELETE" });
    if (state.currentSession === id) {
      state.currentSession = null;
      $("messages").innerHTML = "";
      state.planning = null;
      state.taskResults = null;
    }
    await refreshSessions();
    if (!state.currentSession) clearChat();
  } catch (error) {
    showToast(error.message);
  }
}

async function newSession() {
  if (state.currentInput) {
    showToast("当前轮次仍在执行，请等待完成或取消。", "info");
    return;
  }
  if (state.activeTab === "storage") showTab("chat");
  const data = await api("/api/sessions", { method: "POST" });
  await refreshSessions({ autoSelect: false });
  await selectSession(data.id);
}

async function selectSession(id) {
  if (state.currentInput) {
    showToast("当前轮次仍在执行，请等待完成或取消。", "info");
    return;
  }
  const session = state.sessions.find((item) => item.id === id);
  if (!session) return;
  if (state.activeTab !== "chat" && state.activeTab !== "storage") showTab("chat");
  state.currentSession = id;
  state.pendingPlan = false;
  state.planOpen = null;
  state.storageThread = null;
  state.storageDetail = null;
  state.storageFull = false;
  clearPlanTimer();
  clearKnowledgeTimer();
  renderSessions();
  updateTitle();
  $("messages").innerHTML = "";
  state.planning = null;
  state.taskResults = null;
  renderMode();
  renderPlanPanel();

  try {
    const rows = await api(`/api/sessions/${id}/messages`);
    if (state.currentSession !== id) return;
    for (const row of rows) {
      addMessage(row.role, row.text, row.html, row.created_at);
    }
    if (!rows.length) {
      renderEmptyState("这个会话还没有消息", "在下方输入消息，或切换到规划模式开始整理目标。");
    }
    jumpToBottom();
    await loadPlan();
    if (state.activeTab === "storage") await loadStorage();
  } catch (error) {
    showToast(error.message);
  }
}

function appendSources(body, sources) {
  if (!sources || !sources.length) return;
  const names = sources.map((source) => {
    const meta = source.metadata || {};
    return meta.source_name || meta.document_id || meta.source_uri || "来源";
  });
  body.appendChild(el("div", "message-sources", `来源：${names.join("；")}`));
}

function startStream(inputId) {
  if (state.source) state.source.close();
  state.currentInput = inputId;
  state.streamText = "";
  const node = addMessage("assistant", "", null, new Date().toISOString());
  state.streamItem = node.item;
  state.streamBody = node.body;
  state.streamHead = node.head;
  state.streamState = el("span", "stream-state", "生成中…");
  state.streamHead.appendChild(state.streamState);
  state.streamBody.classList.add("streaming");
  state.streamPre = el("pre", "plain");
  state.streamBody.appendChild(state.streamPre);
  setBusy(true);
  renderMode();

  const source = new EventSource(`/api/turns/${inputId}/events`);
  state.source = source;

  source.addEventListener("model.delta", (event) => {
    const data = JSON.parse(event.data);
    const stick = isNearBottom();
    state.streamText += data.text || "";
    if (state.streamPre) state.streamPre.textContent = state.streamText;
    if (stick) jumpToBottom();
  });
  source.addEventListener("tool.started", (event) => {
    const data = JSON.parse(event.data);
    setToolStatus(`正在调用工具：${data.label || data.name || "外部工具"}…`);
  });
  source.addEventListener("tool.finished", () => setToolStatus("工具调用完成。"));
  source.addEventListener("tool.failed", () => setToolStatus("工具调用失败。"));

  source.addEventListener("answer.final", async (event) => {
    const data = JSON.parse(event.data);
    clearToolStatus();
    const body = state.streamBody;
    const head = state.streamHead;
    const item = state.streamItem;
    if (body) {
      body.classList.remove("streaming");
      if (data.html) body.innerHTML = data.html;
      else body.textContent = data.text || "（无输出）";
      appendSources(body, data.sources);
    }
    if (head) {
      const stamp = head.querySelector(".message-time");
      if (stamp) stamp.textContent = formatTime(new Date().toISOString());
    }
    if (item) item.classList.remove("streaming");
    state.currentInput = null;
    setBusy(false);
    resetStreamState();
    try {
      await refreshSessions({ autoSelect: false });
    } catch (error) {
      showToast(error.message);
    }
    await loadPlan();
    if (state.activeTab === "storage") await loadStorage();
  });

  source.addEventListener("turn.failed", async (event) => {
    const data = JSON.parse(event.data);
    clearToolStatus();
    if (state.streamBody) {
      state.streamBody.classList.remove("streaming");
      state.streamBody.classList.add("error");
      state.streamBody.textContent = data.message || "本轮失败。";
    }
    if (state.streamHead) {
      const stamp = state.streamHead.querySelector(".message-time");
      if (stamp) stamp.textContent = formatTime(new Date().toISOString());
    }
    if (state.streamItem) state.streamItem.classList.add("failed");
    state.currentInput = null;
    setBusy(false);
    resetStreamState();
    showToast(data.message || "本轮失败。");
    await loadPlan();
  });

  source.addEventListener("turn.cancelled", (event) => {
    const data = JSON.parse(event.data);
    clearToolStatus();
    if (state.streamBody) {
      state.streamBody.classList.remove("streaming");
      state.streamBody.classList.add("cancelled");
      state.streamBody.textContent = data.message || "已取消当前执行。";
    }
    if (state.streamHead) {
      const stamp = state.streamHead.querySelector(".message-time");
      if (stamp) stamp.textContent = formatTime(new Date().toISOString());
    }
    if (state.streamItem) state.streamItem.classList.add("cancelled");
    state.currentInput = null;
    setBusy(false);
    resetStreamState();
    showToast(data.message || "已取消当前执行。", "info");
    loadPlan();
  });

  source.addEventListener("done", () => {
    source.close();
    if (state.source === source) state.source = null;
    state.currentInput = null;
    setBusy(false);
    renderMode();
  });
  source.onerror = () => {
    // EventSource 会自动重连；已经落库的轮次结果仍可从消息接口恢复。
  };
}

async function sendMessage(text, extra = {}) {
  const value = (text || "").trim();
  if (!value) return;
  if (state.currentInput) {
    showToast("当前轮次仍在执行，请等待完成或取消。", "info");
    return;
  }
  if (!state.currentSession) await newSession();
  if (!state.currentSession) return;

  const request = { ...extra };
  if (state.pendingPlan && request.planning_action === undefined
      && !value.startsWith("/plan")) {
    request.planning_action = "create";
    request.planning_goal = value;
  }
  const attachmentIds = state.attachments.map((item) => item.id);

  let data;
  try {
    data = await api(`/api/sessions/${state.currentSession}/turns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: value, attachment_ids: attachmentIds, ...request,
      }),
    });
  } catch (error) {
    showToast(error.message);
    return;
  }

  if (state.pendingPlan) state.pendingPlan = false;
  const terminalPlan = state.planning
    && ["completed", "failed", "cancelled"].includes(state.planning.status);
  const planTextCreate = value.startsWith("/plan")
    && !/^\/plan\s+(run|exit|status)(?:\s|$)/.test(value);
  const startingPlan = request.planning_action === "create"
    || (terminalPlan && (!value.startsWith("/plan") || planTextCreate))
    || (!state.planning && planTextCreate);
  if (startingPlan) {
    // 计划生成期间保持规划模式，避免界面短暂退回普通对话。
    state.planning = {
      active: true, status: "draft",
      goal: request.planning_goal || value.replace(/^\/plan\s*/, ""),
      plan: null, questions: [],
    };
    state.planOpen = null;
    renderPlanPanel();
  }
  if (request.planning_action === "run" && state.planning) {
    // 立即显示执行中，轮询会随后补上真实任务进度。
    state.planning = { ...state.planning, status: "running" };
    renderPlanPanel();
  }
  state.attachments = [];
  renderChips();
  addMessage("user", value, null, new Date().toISOString());
  $("input").value = "";
  resizeInput();
  startStream(data.input_id);
  schedulePlanPoll();
  $("input").focus();
}

function renderChips() {
  const box = $("attachment-chips");
  box.innerHTML = "";
  state.attachments.forEach((item, index) => {
    const chip = el("span", "chip");
    chip.appendChild(el("span", "", `${item.name} · ${formatBytes(item.bytes)}`));
    chip.title = "点击移除附件";
    chip.onclick = () => {
      state.attachments.splice(index, 1);
      renderChips();
    };
    box.appendChild(chip);
  });
}

async function uploadFiles(files) {
  if (!files.length) return;
  if (!state.currentSession) await newSession();
  if (!state.currentSession) return;
  for (const file of files) {
    const form = new FormData();
    form.append("session_id", state.currentSession);
    form.append("file", file);
    try {
      const data = await api("/api/attachments", { method: "POST", body: form });
      state.attachments.push(data);
    } catch (error) {
      showToast(error.message);
      break;
    }
  }
  renderChips();
  $("file-input").value = "";
}

function resizeInput() {
  const node = $("input");
  node.style.height = "auto";
  node.style.height = `${Math.min(node.scrollHeight, 180)}px`;
}

async function cancelTurn() {
  if (!state.currentInput) return;
  try {
    await api(`/api/turns/${state.currentInput}/cancel`, { method: "POST" });
    showToast("正在取消当前轮次…", "info");
  } catch (error) {
    showToast(error.message);
  }
}

function switchToChatMode() {
  if (state.currentInput) return;
  if (state.pendingPlan) {
    state.pendingPlan = false;
    renderMode();
    $("input").focus();
    return;
  }
  if (!state.planning) return;
  if (state.planning.status === "running") {
    showToast("计划正在执行，请先取消当前轮次。", "info");
    return;
  }
  sendMessage("退出规划模式", { planning_action: "exit" })
    .catch((error) => showToast(error.message));
}

function switchToPlanMode() {
  if (state.currentInput) return;
  if (state.planning) return;
  state.pendingPlan = true;
  renderMode();
  $("input").focus();
}

function clearKnowledgeTimer() {
  if (state.knowledgeTimer) clearTimeout(state.knowledgeTimer);
  state.knowledgeTimer = null;
  state.knowledgeJob = null;
}

function pollKnowledge(jobId, attempt = 0) {
  clearKnowledgeTimer();
  state.knowledgeJob = jobId;
  state.knowledgeTimer = setTimeout(async () => {
    if (state.knowledgeJob !== jobId) return;
    try {
      const data = await api(`/api/knowledge/status/${jobId}`);
      const progress = data.progress
        ? Object.entries(data.progress).map(([key, value]) => `${key} ${value}`).join(" · ")
        : "";
      $("knowledge-status").textContent = [data.state || "处理中", progress]
        .filter(Boolean).join(" · ");
      if (["ready", "failed", "cancelled"].includes(data.state) || attempt >= 30) {
        state.knowledgeJob = null;
        await loadKnowledge();
        return;
      }
    } catch (_) {}
    pollKnowledge(jobId, attempt + 1);
  }, 2000);
}

async function syncKnowledge() {
  const button = $("sync-knowledge");
  button.disabled = true;
  button.textContent = "同步中…";
  $("knowledge-status").textContent = "正在提交同步任务…";
  try {
    const data = await api("/api/knowledge/sync", { method: "POST" });
    $("knowledge-status").textContent = "已提交同步任务";
    pollKnowledge(data.sync_id);
  } catch (error) {
    $("knowledge-status").textContent = error.message;
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "同步来源";
  }
}

function sourceKind(kind) {
  return SOURCE_KINDS[kind] || kind || "未知来源";
}

function renderSources(sources) {
  const box = $("knowledge-sources");
  box.innerHTML = "";
  if (!sources.length) {
    box.appendChild(el("div", "muted", "暂无来源。"));
    return;
  }
  for (const source of sources) {
    const config = source.config || {};
    const card = el("div", "source-card");
    const head = el("div", "source-head");
    head.appendChild(el("strong", "", source.id));
    head.appendChild(el("span", `badge ${source.enabled ? "on" : "off"}`,
      source.enabled ? "已启用" : "未启用"));
    if (source.legacy) head.appendChild(el("span", "badge off", "旧来源"));
    card.appendChild(head);
    card.appendChild(el("div", "source-meta", sourceKind(config.kind)));
    card.appendChild(el("div", "source-target",
      config.path || config.token || config.node_token || "—"));
    if (source.error) card.appendChild(el("div", "source-error", source.error));
    card.appendChild(el("div", "source-meta",
      source.checked_at ? `最近检查：${fullTime(source.checked_at)}` : "尚未检查"));
    box.appendChild(card);
  }
}

function documentTitle(doc) {
  const metadata = doc.metadata || {};
  return metadata.title || metadata.name || doc.source_uri || doc.id;
}

function buildTable(headers, rows) {
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((label) => headRow.appendChild(el("th", "", label)));
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  rows.forEach((cells) => {
    const row = document.createElement("tr");
    cells.forEach((cell) => {
      const td = document.createElement("td");
      if (cell instanceof Node) td.appendChild(cell);
      else td.textContent = cell === undefined || cell === null ? "" : String(cell);
      row.appendChild(td);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  return table;
}

function renderDocuments(documents) {
  const box = $("knowledge-documents");
  box.innerHTML = "";
  $("knowledge-document-count").textContent = documents.length ? `${documents.length} 篇` : "";
  if (!documents.length) {
    box.appendChild(el("div", "muted", "暂无文档。"));
    return;
  }
  const rows = documents.map((document) => [
    documentTitle(document),
    (document.source_ids || []).join("、"),
    document.deleted ? "已删除" : (document.index_state || "未索引"),
    document.active_build || "",
  ]);
  box.appendChild(buildTable(["标题 / 路径", "来源", "状态", "当前构建"], rows));
}

async function loadKnowledge() {
  $("knowledge-status").textContent = "加载中…";
  $("knowledge-sources").innerHTML = "";
  $("knowledge-sources").appendChild(el("div", "muted", "加载中…"));
  $("knowledge-documents").innerHTML = "";
  try {
    const [sources, documents] = await Promise.all([
      api("/api/knowledge/sources"), api("/api/knowledge/documents")]);
    renderSources(sources);
    renderDocuments(documents);
    $("knowledge-status").textContent = "已加载";
  } catch (error) {
    $("knowledge-status").textContent = error.message;
    $("knowledge-sources").innerHTML = "";
    $("knowledge-sources").appendChild(el("div", "muted", error.message));
    $("knowledge-documents").innerHTML = "";
  }
}

function renderSettings(data) {
  const box = $("settings");
  box.innerHTML = "";
  const attachments = data.attachments || {};
  const rows = [
    ["默认模型", data.default_model || ""],
    ["可用模型", (data.models || []).join("、")],
    ["工作区", data.workspace || ""],
    ["知识库", data.knowledge ? "已启用" : "未启用"],
    ["飞书文档", data.feishu_docs ? "已启用" : "未启用"],
    ["附件", attachments.enabled
      ? `已启用（单文件最大 ${formatBytes(attachments.max_bytes)}）` : "未启用"],
    ["允许扩展名", (attachments.extensions || []).join(" ")],
    ["规划模式", data.planning ? "已启用" : "未启用"],
  ];
  rows.forEach(([key, value]) => {
    const row = el("div", "settings-row");
    row.appendChild(el("div", "settings-key", key));
    row.appendChild(el("div", "settings-value", String(value)));
    box.appendChild(row);
  });
}

async function loadSettings() {
  try {
    renderSettings(await api("/api/settings"));
  } catch (error) {
    $("settings").innerHTML = "";
    $("settings").appendChild(el("div", "muted", error.message));
    showToast(error.message);
  }
}

function storageBadge(status) {
  return el("span", `storage-badge ${status || ""}`, status || "未知");
}

function storageCard(label, value, mono = false) {
  const card = el("div", "storage-card");
  card.appendChild(el("div", "storage-card-label", label));
  card.appendChild(el("div", `storage-card-value${mono ? " mono" : ""}`,
    value === undefined || value === null || value === "" ? "-" : String(value)));
  return card;
}

async function copyText(text, label) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(String(text));
    showToast(`${label}已复制`, "info");
  } catch (_) {
    showToast("复制失败；请手动选择文本。");
  }
}

function storageEmpty(box, text) {
  box.innerHTML = "";
  box.appendChild(el("div", "storage-empty", text));
}

function renderStorageSummary(data) {
  const box = $("storage-summary");
  box.innerHTML = "";
  if (!data) return;
  const session = data.session || {};
  const counts = data.counts || {};
  box.appendChild(storageCard("会话", session.title || "新会话"));
  box.appendChild(storageCard("状态", session.status));
  box.appendChild(storageCard("线程", (data.threads || []).length));
  box.appendChild(storageCard("Checkpoint", counts.checkpoints));
  box.appendChild(storageCard("Web 消息", counts.web_messages));
  box.appendChild(storageCard("操作回执", counts.operations));
  const completeId = session.complete
    && session.complete.configurable
    && session.complete.configurable.checkpoint_id;
  box.appendChild(storageCard("完成点", completeId || "无", true));
  box.appendChild(storageCard("会话 ID", session.id, true));
}

function renderStorageHealth(data) {
  const box = $("storage-health");
  box.innerHTML = "";
  if (!data) return;
  const warnings = data.warnings || [];
  if (!warnings.length) {
    box.appendChild(el("div", "storage-health-ok", "未发现明显存储异常。"));
    return;
  }
  for (const warning of warnings) {
    box.appendChild(el("div", "storage-health-item", warning));
  }
}

function renderStorageTurns(data) {
  const box = $("storage-turns");
  box.innerHTML = "";
  const items = (data && data.items) || [];
  if (!items.length) {
    storageEmpty(box, "暂无轮次记录。");
    return;
  }
  for (const item of items) {
    const row = el("div", "storage-turn");
    const head = el("div", "storage-turn-head");
    head.appendChild(storageBadge(item.status));
    head.appendChild(el("span", "storage-turn-id", item.input_id || "unknown"));
    head.appendChild(el("span", "storage-turn-meta",
      fullTime(item.latest_checkpoint_ts) || ""));
    row.appendChild(head);
    row.appendChild(el("div", "storage-turn-meta",
      `checkpoint ${item.checkpoints || 0} · Web 消息 ${item.web_messages || 0} · ` +
      `操作 ${item.operations || 0} · ${item.assistant_phase || "-"}`));
    box.appendChild(row);
  }
}

function renderStorageThreads(data) {
  const select = $("storage-thread");
  select.innerHTML = "";
  const threads = (data && data.threads) || [];
  if (!threads.length) {
    select.appendChild(el("option", "", "无可用线程"));
    select.disabled = true;
    return;
  }
  select.disabled = false;
  const current = state.storageThread
    && threads.some((item) => item.thread_id === state.storageThread)
    ? state.storageThread : data.session.thread_id;
  state.storageThread = current;
  for (const item of threads) {
    const option = el("option", "", item.thread_id);
    option.value = item.thread_id;
    option.selected = item.thread_id === current;
    select.appendChild(option);
  }
}

function renderStorageCheckpoints(data) {
  const box = $("storage-checkpoints");
  box.innerHTML = "";
  const items = (data && data.items) || [];
  if (!items.length) {
    storageEmpty(box, "暂无 checkpoint。");
    return;
  }
  const list = el("div", "storage-checkpoints");
  for (const item of items) {
    const row = el("div", "storage-checkpoint");
    const button = el("button", "storage-checkpoint-main");
    button.type = "button";
    button.onclick = () => loadStorageCheckpoint(item.checkpoint_id);
    const head = el("div", "storage-turn-head");
    head.appendChild(storageBadge(item.turn_status));
    head.appendChild(el("span", "storage-checkpoint-id", item.checkpoint_id || ""));
    button.appendChild(head);
    button.appendChild(el("div", "storage-checkpoint-meta",
      `step ${item.step ?? "-"} · ${fullTime(item.ts) || "-"} · ` +
      `phase ${item.assistant_phase || "-"} · source ${item.source || "-"} · ` +
      `pending ${item.pending_writes || 0}`));
    const channels = (item.channels || []).map((channel) => channel.name).join(", ");
    if (channels) {
      button.appendChild(el("div", "storage-checkpoint-channels", channels));
    }
    row.appendChild(button);
    list.appendChild(row);
  }
  box.appendChild(list);
}

function storageTable(box, headers, rows) {
  box.innerHTML = "";
  if (!rows.length) {
    storageEmpty(box, "暂无数据。");
    return;
  }
  const wrap = el("div", "table-wrap");
  wrap.appendChild(buildTable(headers, rows));
  box.appendChild(wrap);
}

function renderStorageMessages(data) {
  if (!data) {
    storageEmpty($("storage-messages"), "Web 消息不可用。");
    return;
  }
  const rows = (data.items || []).map((item) => [
    fullTime(item.created_at) || "", item.role || "", item.status || "",
    item.input_id || "", item.text || "",
  ]);
  storageTable($("storage-messages"), ["时间", "角色", "状态", "input_id", "正文预览"], rows);
}

function renderStorageOperations(data) {
  if (!data) {
    storageEmpty($("storage-operations"), "操作回执不可用。");
    return;
  }
  const rows = (data.items || []).map((item) => [
    fullTime(item.created_at) || "", item.status || "", item.tool || "",
    item.input_id || "", item.source_key || "",
    JSON.stringify(item.result || {}).slice(0, 300),
  ]);
  storageTable($("storage-operations"),
    ["时间", "状态", "工具", "input_id", "source", "结果预览"], rows);
}

function renderStorageUsage(data) {
  const box = $("storage-usage");
  if (!data || !data.available) {
    storageEmpty(box, "用量结构不可用或尚未初始化。");
    return;
  }
  const rows = (data.items || []).map((item) => [
    item.role || "", item.profile || "", item.calls ?? "",
    item.input_tokens ?? "", item.output_tokens ?? "",
    item.total_tokens ?? "", item.unfinished ?? "", item.missing ?? "",
  ]);
  storageTable(box,
    ["角色", "配置", "调用", "输入", "输出", "合计", "未结束", "缺失"], rows);
}

async function loadStorageCheckpoints(before = null, append = false) {
  if (!state.currentSession || state.activeTab !== "storage") return;
  const sessionId = state.currentSession;
  const thread = state.storageThread
    || (state.storage && state.storage.summary.session.thread_id);
  if (!thread) return;
  try {
    const data = await api(`/api/storage/sessions/${sessionId}/checkpoints` +
      `?thread_id=${encodeURIComponent(thread)}&limit=20` +
      (before ? `&before=${encodeURIComponent(before)}` : ""));
    if (state.currentSession !== sessionId
        || (state.storageThread && state.storageThread !== thread)) return;
    if (append && state.storageCheckpoints
        && state.storageCheckpoints.thread_id === data.thread_id) {
      state.storageCheckpoints = {
        ...data,
        items: [...(state.storageCheckpoints.items || []), ...(data.items || [])],
      };
    } else {
      state.storageCheckpoints = data;
    }
    renderStorageCheckpoints(state.storageCheckpoints);
    $("storage-more").classList.toggle("hidden", !state.storageCheckpoints.next_before);
  } catch (error) {
    showToast(error.message);
    storageEmpty($("storage-checkpoints"), error.message);
    $("storage-more").classList.add("hidden");
  }
}

async function loadStorageCheckpoint(checkpointId) {
  if (!state.currentSession || !checkpointId) return;
  const sessionId = state.currentSession;
  const thread = state.storageThread
    || (state.storage && state.storage.summary.session.thread_id);
  try {
    const data = await api(`/api/storage/sessions/${sessionId}/checkpoints/` +
      `${encodeURIComponent(checkpointId)}?thread_id=${encodeURIComponent(thread || "")}` +
      `&full=${state.storageFull ? "true" : "false"}`);
    if (state.currentSession !== sessionId
        || (state.storageThread && thread && state.storageThread !== thread)) return;
    state.storageDetail = data;
    renderStorageDetail();
  } catch (error) {
    showToast(error.message);
  }
}

function renderStorageDetail() {
  const box = $("storage-detail");
  box.innerHTML = "";
  const data = state.storageDetail;
  if (!data) return;
  const head = el("div", "storage-detail-head");
  const title = el("div");
  title.appendChild(el("strong", "", `Checkpoint ${data.checkpoint_id || ""}`));
  title.appendChild(el("div", "storage-checkpoint-meta",
    `step ${data.step ?? "-"} · phase ${data.assistant_phase || "-"} · ` +
    `pending ${(data.pending_writes || []).length}`));
  head.appendChild(title);
  const toggle = el("button", "", state.storageFull ? "恢复预览" : "展开完整内容");
  toggle.type = "button";
  toggle.onclick = () => {
    state.storageFull = !state.storageFull;
    loadStorageCheckpoint(data.checkpoint_id);
  };
  head.appendChild(toggle);
  box.appendChild(head);
  const ids = el("div", "storage-id-actions");
  [["checkpoint_id", "复制 checkpoint"], ["input_id", "复制 input_id"],
   ["execution_thread_id", "复制 execution_thread"]].forEach(([key, label]) => {
    if (!data[key]) return;
    const button = el("button", "", label);
    button.type = "button";
    button.onclick = () => copyText(data[key], label.replace("复制 ", ""));
    ids.appendChild(button);
  });
  if (ids.childNodes.length) box.appendChild(ids);

  const body = el("div", "storage-detail-body");
  for (const [name, value] of Object.entries(data.channel_values || {})) {
    const details = document.createElement("details");
    details.className = "storage-channel";
    const summary = document.createElement("summary");
    summary.textContent = name;
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = typeof value === "string"
      ? value : JSON.stringify(value, null, 2);
    details.appendChild(pre);
    body.appendChild(details);
  }
  if (!Object.keys(data.channel_values || {}).length) {
    body.appendChild(el("div", "muted", "没有可展示的 channel 内容。"));
  }
  if ((data.pending_writes || []).length) {
    const details = document.createElement("details");
    details.className = "storage-channel";
    const summary = document.createElement("summary");
    summary.textContent = `pending_writes (${data.pending_writes.length})`;
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(data.pending_writes, null, 2);
    details.appendChild(pre);
    body.appendChild(details);
  }
  if ((data.blob_sizes || []).length) {
    const details = document.createElement("details");
    details.className = "storage-channel";
    const summary = document.createElement("summary");
    summary.textContent = `blob_sizes (${data.blob_sizes.length})`;
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(data.blob_sizes, null, 2);
    details.appendChild(pre);
    body.appendChild(details);
  }
  box.appendChild(body);
}

async function loadStorage() {
  if (!state.currentSession) {
    state.storage = null;
    storageEmpty($("storage-summary"), "请先从左侧选择一个会话。");
    storageEmpty($("storage-health"), "");
    storageEmpty($("storage-turns"), "");
    storageEmpty($("storage-checkpoints"), "");
    storageEmpty($("storage-messages"), "");
    storageEmpty($("storage-operations"), "");
    storageEmpty($("storage-usage"), "");
    $("storage-detail").innerHTML = "";
    $("storage-thread").innerHTML = "";
    return;
  }
  const sessionId = state.currentSession;
  const request = ++state.storageRequest;
  const base = `/api/storage/sessions/${sessionId}`;
  try {
    const [summary, turns, health, messages, operations, usage] = await Promise.all([
      api(base),
      api(`${base}/turns?limit=20`),
      api(`${base}/health`),
      api(`${base}/messages?limit=100`).catch(() => null),
      api(`${base}/operations?limit=100`).catch(() => null),
      api(`${base}/usage`).catch(() => null),
    ]);
    if (request !== state.storageRequest || state.currentSession !== sessionId) return;
    state.storage = { summary, turns, health, messages, operations, usage };
    if (!state.storageThread
        || !(summary.threads || []).some((item) => item.thread_id === state.storageThread)) {
      state.storageThread = summary.session.thread_id;
    }
    renderStorageSummary(summary);
    renderStorageHealth(health);
    renderStorageTurns(turns);
    renderStorageThreads(summary);
    renderStorageMessages(messages);
    renderStorageOperations(operations);
    renderStorageUsage(usage);
    state.storageDetail = null;
    state.storageFull = false;
    state.storageCheckpoints = null;
    $("storage-detail").innerHTML = "";
    $("storage-more").classList.add("hidden");
    await loadStorageCheckpoints();
  } catch (error) {
    showToast(error.message);
    storageEmpty($("storage-summary"), error.message);
    storageEmpty($("storage-health"), "");
    storageEmpty($("storage-turns"), "");
    storageEmpty($("storage-checkpoints"), "");
  }
}

function showTab(name) {
  state.activeTab = name;
  for (const view of ["chat", "knowledge", "settings", "storage"]) {
    $(`${view}-view`).classList.toggle("hidden", view !== name);
  }
  document.querySelectorAll(".tabs button").forEach((node) => {
    node.classList.toggle("active", node.dataset.tab === name);
  });
  if (name === "chat") {
    resizeInput();
    schedulePlanPoll();
  } else {
    clearPlanTimer();
  }
  if (name === "knowledge") loadKnowledge();
  else clearKnowledgeTimer();
  if (name === "settings") loadSettings();
  if (name === "storage") loadStorage();
}

function bind() {
  $("new-session").onclick = () => newSession().catch((error) => showToast(error.message));
  $("composer").onsubmit = (event) => {
    event.preventDefault();
    sendMessage($("input").value).catch((error) => showToast(error.message));
  };
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
    if (event.key === "Escape" && state.pendingPlan) {
      state.pendingPlan = false;
      renderMode();
      showToast("已取消规划输入。", "info");
    }
  });
  $("input").addEventListener("input", resizeInput);
  $("file-input").onchange = (event) => {
    uploadFiles([...event.target.files]).catch((error) => showToast(error.message));
  };
  $("cancel-button").onclick = cancelTurn;
  $("mode-chat").onclick = switchToChatMode;
  $("mode-plan").onclick = switchToPlanMode;
  $("plan-run").onclick = () => {
    sendMessage("执行计划", { planning_action: "run" })
      .catch((error) => showToast(error.message));
  };
  $("plan-toggle").onclick = () => {
    const open = $("plan-toggle").getAttribute("aria-expanded") === "true";
    state.planOpen = !open;
    setPlanOpen(!open);
  };
  $("sync-knowledge").onclick = syncKnowledge;
  $("storage-refresh").onclick = () => loadStorage()
    .catch((error) => showToast(error.message));
  $("storage-thread").onchange = (event) => {
    state.storageThread = event.target.value;
    state.storageDetail = null;
    state.storageFull = false;
    state.storageCheckpoints = null;
    $("storage-detail").innerHTML = "";
    $("storage-more").classList.add("hidden");
    loadStorageCheckpoints();
  };
  $("storage-more").onclick = () => {
    const before = state.storageCheckpoints && state.storageCheckpoints.next_before;
    if (before) loadStorageCheckpoints(before, true);
  };
  document.querySelectorAll(".tabs button").forEach((node) => {
    node.onclick = () => showTab(node.dataset.tab);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearPlanTimer();
    } else if (state.activeTab === "chat") {
      loadPlan();
    }
  });
  window.addEventListener("beforeunload", () => {
    clearPlanTimer();
    clearKnowledgeTimer();
    if (state.source) state.source.close();
  });
}

async function init() {
  bind();
  try {
    await refreshSessions();
  } catch (error) {
    showToast(error.message);
  }
  if (!state.currentSession) clearChat();
  renderMode();
  renderPlanPanel();
  resizeInput();
}

init();
