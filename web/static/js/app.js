const STATE = {
  LOADING: "loading",
  MODEL_MISSING: "model_missing",
  INTRO: "intro",
  ASKING_QUESTION: "asking_question",
  WAITING_ANSWER: "waiting_answer",
  INTERPRETING: "interpreting",
  NEEDS_CLARIFICATION: "needs_clarification",
  WAITING_CONFIRMATION: "waiting_confirmation",
  COMPLETED: "completed",
  SHOWING_RESULT: "showing_result",
  ERROR: "error",
};

const HELP_CHIPS = ["No entiendo", "Dame un ejemplo", "Explicar con palabras más simples"];

const state = {
  machine: STATE.LOADING,
  sessionId: `chat-${Date.now()}`,
  role: "caregiver",
  modelReady: false,
  targetColumn: "target_domain_conduct_final",
  targetIntro: "",
  questions: [],
  currentIndex: 0,
  pendingInterpretation: null,
  confirmedAnswers: {},
  waitingPrediction: false,
};

const el = {
  modelStatusText: document.getElementById("modelStatusText"),
  targetText: document.getElementById("targetText"),
  progressText: document.getElementById("progressText"),
  roleSelect: document.getElementById("roleSelect"),
  resetBtn: document.getElementById("resetBtn"),
  modelMissingCard: document.getElementById("modelMissingCard"),
  chatPanel: document.getElementById("chatPanel"),
  chatMessages: document.getElementById("chatMessages"),
  quickActions: document.getElementById("quickActions"),
  contextActions: document.getElementById("contextActions"),
  chatInputForm: document.getElementById("chatInputForm"),
  chatInput: document.getElementById("chatInput"),
  sendBtn: document.getElementById("sendBtn"),
  loadMetricsBtn: document.getElementById("loadMetricsBtn"),
  loadImportanceBtn: document.getElementById("loadImportanceBtn"),
  metricsBox: document.getElementById("metricsBox"),
  importanceBox: document.getElementById("importanceBox"),
};

function setState(nextState) {
  state.machine = nextState;
}

function updateProgressLabel() {
  const total = state.questions.length;
  if (total <= 0) {
    el.progressText.textContent = "Preparando preguntas...";
    return;
  }
  if (state.currentIndex >= total) {
    el.progressText.textContent = `Completado ${total} de ${total}`;
    return;
  }
  el.progressText.textContent = `Pregunta ${state.currentIndex + 1} de ${total}`;
}

function ensureVisible(element, visible) {
  if (!element) return;
  element.classList.toggle("hidden", !visible);
}

function scrollChatToBottom() {
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

function addMessage(author, text, extraClass = "") {
  const div = document.createElement("div");
  div.className = `chat-message ${author} ${extraClass}`.trim();
  div.textContent = text;
  el.chatMessages.appendChild(div);
  scrollChatToBottom();
}

function clearChat() {
  el.chatMessages.innerHTML = "";
}

function setInputEnabled(enabled) {
  el.chatInput.disabled = !enabled;
  el.sendBtn.disabled = !enabled;
}

function clearContextActions() {
  el.contextActions.innerHTML = "";
  ensureVisible(el.contextActions, false);
}

function renderContextActions(actions) {
  el.contextActions.innerHTML = "";
  if (!actions || actions.length === 0) {
    ensureVisible(el.contextActions, false);
    return;
  }
  actions.forEach((action) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `btn ${action.className || "btn-ghost"}`;
    btn.textContent = action.label;
    btn.addEventListener("click", action.onClick);
    el.contextActions.appendChild(btn);
  });
  ensureVisible(el.contextActions, true);
}

function normalizeText(value) {
  return (value || "")
    .toString()
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function confidenceText(confidence) {
  const value = Number(confidence || 0);
  if (value >= 0.85) return "alto";
  if (value >= 0.65) return "medio";
  return "bajo";
}

function currentQuestion() {
  return state.questions[state.currentIndex] || null;
}

async function emitAudit(event, payload = {}) {
  try {
    await fetch("/api/audit-event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, payload }),
    });
  } catch (_) {
    // Keep chat flow alive.
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "";
    try {
      const json = await response.json();
      detail = json.detail || json.message || JSON.stringify(json);
    } catch (_) {
      detail = `${response.status} ${response.statusText}`;
    }
    throw new Error(detail || "Solicitud fallida");
  }
  return response.json();
}

function uniqueChips(chips) {
  const out = [];
  const seen = new Set();
  chips.forEach((chip) => {
    const clean = `${chip || ""}`.trim();
    if (!clean) return;
    const key = normalizeText(clean);
    if (seen.has(key)) return;
    seen.add(key);
    out.push(clean);
  });
  return out;
}

function renderQuickChips(question = null, overrideChips = null) {
  el.quickActions.innerHTML = "";

  const questionChips = Array.isArray(overrideChips)
    ? overrideChips
    : Array.isArray(question?.quick_chips)
    ? question.quick_chips
    : [];

  const chips = uniqueChips([...questionChips, ...HELP_CHIPS]);
  chips.forEach((chipText) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = chipText;
    btn.dataset.chip = chipText;
    btn.addEventListener("click", async () => {
      await handleQuickChip(chipText);
    });
    el.quickActions.appendChild(btn);
  });
}

async function handleQuickChip(chipText) {
  const normalized = normalizeText(chipText);
  if (normalized.includes("no entiendo") || normalized.includes("ejemplo") || normalized.includes("explicar")) {
    addMessage("user", chipText);
    const mode = normalized.includes("ejemplo") ? "example" : "simple";
    await explainCurrentQuestion(mode);
    return;
  }

  if (![STATE.WAITING_ANSWER, STATE.NEEDS_CLARIFICATION, STATE.WAITING_CONFIRMATION].includes(state.machine)) {
    return;
  }

  if (state.machine === STATE.WAITING_CONFIRMATION) {
    correctionFlow();
  }

  el.chatInput.value = "";
  await processUserAnswer(chipText);
}

async function loadModelStatus() {
  const payload = await fetchJson("/api/model-status");
  state.modelReady = Boolean(payload.model_trained);
  state.targetColumn = payload.target_column || state.targetColumn;
  state.targetIntro = payload.target_intro || "";

  el.targetText.textContent = state.targetColumn;
  el.modelStatusText.textContent = payload.message || "Estado no disponible";
  return payload;
}

async function loadQuestions() {
  state.role = el.roleSelect.value;
  const data = await fetchJson(
    `/api/questions?role=${encodeURIComponent(state.role)}&session_id=${encodeURIComponent(state.sessionId)}`
  );
  if (!Array.isArray(data.questions) || data.questions.length === 0) {
    throw new Error(
      "No se encontraron preguntas para el target seleccionado. Revisa artifacts/feature_schema.json o ejecuta python train.py."
    );
  }
  state.questions = data.questions;
  state.currentIndex = 0;
  state.pendingInterpretation = null;
  state.confirmedAnswers = {};
  if (data.target_column) state.targetColumn = data.target_column;
  if (data.intro_text) state.targetIntro = data.intro_text;
  updateProgressLabel();
}

function introMessages() {
  addMessage(
    "assistant",
    "Hola. Soy un asistente local de apoyo preliminar. Te haré preguntas sencillas y puedes responder con tus palabras. Si no entiendes una pregunta, escribe 'no entiendo' o usa los botones de ayuda."
  );
  addMessage(
    "assistant",
    state.targetIntro || "Te haré preguntas relacionadas con el dominio seleccionado."
  );
  addMessage("assistant", "Avanzaremos paso a paso. Solo continuaré cuando confirmes cada respuesta.");
}

function askCurrentQuestion() {
  const q = currentQuestion();
  if (!q) {
    setState(STATE.COMPLETED);
    updateProgressLabel();
    renderQuickChips(null, []);
    addMessage("assistant", "Listo, ya tengo la información necesaria.");
    renderContextActions([
      {
        label: "Generar estimación",
        className: "btn-primary",
        onClick: () => generatePrediction(),
      },
    ]);
    return;
  }

  setState(STATE.ASKING_QUESTION);
  updateProgressLabel();

  addMessage("assistant", `Pregunta ${q.progress_index} de ${q.progress_total}: ${q.question}`);
  if (q.help_text) addMessage("system", q.help_text);
  if (q.human_options_text) addMessage("system", q.human_options_text);

  renderQuickChips(q);
  clearContextActions();
  setState(STATE.WAITING_ANSWER);
  setInputEnabled(true);
  el.chatInput.focus();
}

async function explainCurrentQuestion(mode = "simple") {
  const q = currentQuestion();
  if (!q) return;

  const explain = await fetchJson("/api/chat/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feature: q.feature, mode }),
  });

  const lines = [
    explain.simple_explanation || "Te explico la pregunta con palabras más simples.",
    Array.isArray(explain.examples) && explain.examples.length
      ? `Ejemplos: ${explain.examples.join(" | ")}`
      : "",
    explain.expected_answer || explain.human_options_text || "",
  ]
    .filter(Boolean)
    .join("\n");

  addMessage("assistant", lines);
  renderQuickChips(q, explain.quick_chips || q.quick_chips || []);
  await emitAudit("frontend_help_requested", {
    session_id: state.sessionId,
    feature: q.feature,
    mode,
  });
  setState(STATE.WAITING_ANSWER);
  setInputEnabled(true);
}

function buildInterpretationMessage(interpreted) {
  const confidenceLevel = confidenceText(interpreted.confidence);
  const lines = [
    interpreted.user_friendly_interpretation || "Interpreté tu respuesta.",
    interpreted.value_explanation || "",
    `Nivel de seguridad: ${confidenceLevel}.`,
  ];

  if (interpreted.reasoning_summary && !normalizeText(interpreted.reasoning_summary).includes("detecto")) {
    lines.push(interpreted.reasoning_summary);
  }
  return lines.filter(Boolean).join("\n");
}

async function interpretAnswer(rawAnswer) {
  const q = currentQuestion();
  if (!q) return;

  setState(STATE.INTERPRETING);
  setInputEnabled(false);

  const payload = await fetchJson("/api/chat/interpret", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      feature: q.feature,
      answer: rawAnswer,
      role: state.role,
      session_id: state.sessionId,
      question_metadata: {
        scale_type: q.scale_type,
        human_options_text: q.human_options_text,
        response_options: q.response_options,
      },
    }),
  });

  if (payload.needs_explanation) {
    const explanation = payload.explanation || {};
    const explainText = [
      "Claro, te ayudo con esta pregunta.",
      explanation.simple_explanation || "",
      Array.isArray(explanation.examples) && explanation.examples.length
        ? `Ejemplos: ${explanation.examples.join(" | ")}`
        : "",
      explanation.expected_answer || explanation.human_options_text || "",
    ]
      .filter(Boolean)
      .join("\n");

    addMessage("assistant", explainText);
    renderQuickChips(q, explanation.quick_chips || q.quick_chips || []);
    setState(STATE.NEEDS_CLARIFICATION);
    setInputEnabled(true);
    return;
  }

  const interpreted = payload.interpreted || null;
  state.pendingInterpretation = interpreted;

  if (!interpreted) {
    addMessage("assistant", "No pude entender bien esta respuesta. Necesito una respuesta más clara para continuar.");
    setState(STATE.NEEDS_CLARIFICATION);
    setInputEnabled(true);
    return;
  }

  addMessage("assistant", buildInterpretationMessage(interpreted));

  if (interpreted.needs_clarification || interpreted.parsed_value === null || interpreted.parsed_value === undefined) {
    addMessage(
      "assistant",
      interpreted.clarification_question ||
        "No estoy completamente seguro de haber entendido. ¿Puedes responder de forma más directa?"
    );
    if (interpreted.human_options_text) {
      addMessage("system", interpreted.human_options_text);
    }
    renderQuickChips(q, interpreted.quick_chips || q.quick_chips || []);
    clearContextActions();
    setState(STATE.NEEDS_CLARIFICATION);
    setInputEnabled(true);
    return;
  }

  addMessage("assistant", "¿Deseas continuar o corregir esta interpretación?");
  setState(STATE.WAITING_CONFIRMATION);
  setInputEnabled(false);
  renderContextActions([
    {
      label: "Continuar",
      className: "btn-success",
      onClick: () => confirmCurrentInterpretation(false),
    },
    {
      label: "Corregir",
      className: "btn-warning",
      onClick: () => correctionFlow(),
    },
  ]);
}

async function confirmCurrentInterpretation(useMissingStrategy) {
  const q = currentQuestion();
  const interpreted = state.pendingInterpretation;
  if (!q || !interpreted) {
    addMessage("assistant", "Aún no tengo una interpretación para confirmar.");
    return;
  }

  if (interpreted.parsed_value === null || interpreted.parsed_value === undefined) {
    addMessage("assistant", "Necesito una respuesta más clara antes de continuar con esta pregunta.");
    setState(STATE.NEEDS_CLARIFICATION);
    setInputEnabled(true);
    return;
  }

  await fetchJson("/api/chat/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      feature: q.feature,
      parsed_value: interpreted.parsed_value,
      raw_answer: interpreted.raw_answer || "",
      confidence: interpreted.confidence || 0,
      session_id: state.sessionId,
      used_missing_strategy: Boolean(useMissingStrategy),
    }),
  });

  state.confirmedAnswers[q.feature] = interpreted.parsed_value;
  state.pendingInterpretation = null;
  clearContextActions();
  setInputEnabled(true);

  state.currentIndex += 1;
  await emitAudit("frontend_answer_confirmed", {
    session_id: state.sessionId,
    feature: q.feature,
    value: interpreted.parsed_value,
    index: state.currentIndex,
    total: state.questions.length,
  });
  askCurrentQuestion();
}

function correctionFlow() {
  state.pendingInterpretation = null;
  clearContextActions();
  setState(STATE.WAITING_ANSWER);
  setInputEnabled(true);
  addMessage("assistant", "Perfecto. Corrige tu respuesta y la vuelvo a interpretar.");
  el.chatInput.focus();
}

async function generatePrediction() {
  if (state.waitingPrediction) return;
  if (Object.keys(state.confirmedAnswers).length === 0) {
    addMessage("assistant", "Todavía no hay respuestas confirmadas para estimar.");
    return;
  }

  state.waitingPrediction = true;
  setInputEnabled(false);
  clearContextActions();
  renderQuickChips(null, []);
  addMessage("assistant", "Estoy generando la estimación con las respuestas confirmadas...");

  const payload = await fetchJson("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      answers: state.confirmedAnswers,
    }),
  });

  state.waitingPrediction = false;
  if (!payload.ok) {
    setState(STATE.ERROR);
    addMessage("assistant", payload.message || "No fue posible generar la estimación.", "error");
    setInputEnabled(true);
    return;
  }

  const p = payload.prediction;
  const probability = Number(p.probability_estimated || 0) * 100;
  const text = [
    "Resultado final",
    `Probabilidad estimada: ${probability.toFixed(2)}%`,
    `Nivel de riesgo: ${p.risk_text || "No disponible"}`,
    `Threshold usado: ${p.threshold_used ?? "No disponible"}`,
    p.explanation || "",
    p.medical_disclaimer || "Este resultado no es un diagnóstico médico y debe ser revisado por un profesional calificado.",
  ]
    .filter(Boolean)
    .join("\n");

  addMessage("assistant", text);
  setState(STATE.SHOWING_RESULT);
  renderContextActions([
    {
      label: "Iniciar nueva evaluación",
      className: "btn-primary",
      onClick: () => restartSession(),
    },
  ]);
}

async function processUserAnswer(answer) {
  const text = (answer || "").trim();
  if (!text) return;

  addMessage("user", text);
  await emitAudit("frontend_user_message", {
    session_id: state.sessionId,
    state: state.machine,
    text,
  });

  if (normalizeText(text).includes("no entiendo") || normalizeText(text).includes("dame un ejemplo") || normalizeText(text).includes("explicar")) {
    await explainCurrentQuestion("simple");
    setInputEnabled(true);
    return;
  }

  await interpretAnswer(text);
}

async function restartSession() {
  await fetchJson("/api/reset-session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  });

  state.currentIndex = 0;
  state.pendingInterpretation = null;
  state.confirmedAnswers = {};
  state.questions = [];
  state.waitingPrediction = false;
  clearContextActions();
  clearChat();
  renderQuickChips(null, []);
  updateProgressLabel();
  setInputEnabled(true);
  await initializeChatFlow();
}

async function initializeChatFlow() {
  setState(STATE.LOADING);
  ensureVisible(el.modelMissingCard, false);
  ensureVisible(el.chatPanel, true);
  setInputEnabled(false);
  updateProgressLabel();
  clearContextActions();
  clearChat();
  renderQuickChips(null, []);

  try {
    const modelStatus = await loadModelStatus();
    await emitAudit("frontend_model_status_loaded", {
      session_id: state.sessionId,
      model_trained: modelStatus.model_trained,
      target: modelStatus.target_column,
    });

    if (!modelStatus.model_trained) {
      setState(STATE.MODEL_MISSING);
      ensureVisible(el.modelMissingCard, true);
      ensureVisible(el.chatPanel, false);
      return;
    }

    await loadQuestions();
    setState(STATE.INTRO);
    introMessages();
    askCurrentQuestion();
  } catch (error) {
    setState(STATE.ERROR);
    ensureVisible(el.chatPanel, true);
    addMessage(
      "assistant",
      error.message ||
        "Ocurrió un problema al preparar el chat. Revisa artifacts/feature_schema.json o ejecuta python train.py.",
      "error"
    );
    setInputEnabled(false);
    await emitAudit("frontend_error", {
      session_id: state.sessionId,
      message: error.message || "Error desconocido",
      stage: "initialize_chat_flow",
    });
  }
}

async function onSubmit(event) {
  event.preventDefault();
  const answer = el.chatInput.value.trim();
  if (!answer) return;

  const allowedStates = [STATE.WAITING_ANSWER, STATE.NEEDS_CLARIFICATION, STATE.WAITING_CONFIRMATION];
  if (!allowedStates.includes(state.machine)) return;

  if (state.machine === STATE.WAITING_CONFIRMATION) {
    correctionFlow();
  }

  el.chatInput.value = "";
  await processUserAnswer(answer);
}

async function loadMetrics() {
  const data = await fetchJson("/api/metrics");
  ensureVisible(el.metricsBox, true);
  if (!data.ok) {
    el.metricsBox.textContent = data.message || "No hay métricas disponibles.";
    return;
  }
  const finalMetrics = data.threshold_final || {};
  el.metricsBox.textContent = [
    `Target: ${data.target_column || "N/A"}`,
    `Features: ${data.features_count ?? "N/A"}`,
    `F1 final: ${finalMetrics.f1 ?? "N/A"}`,
    `Recall final: ${finalMetrics.recall ?? "N/A"}`,
    `Precision final: ${finalMetrics.precision ?? "N/A"}`,
  ].join("\n");
}

async function loadImportance() {
  const data = await fetchJson("/api/feature-importance");
  ensureVisible(el.importanceBox, true);
  if (!data.ok) {
    el.importanceBox.textContent = data.message || "No hay importancia disponible.";
    return;
  }
  const lines = (data.feature_importance_aggregated || []).slice(0, 12).map((row) => {
    const label = row.label || row.feature;
    return `${label}: ${(Number(row.importance || 0) * 100).toFixed(2)}%`;
  });
  el.importanceBox.textContent = lines.length ? lines.join("\n") : "Sin datos de importancia.";
}

function attachEvents() {
  el.chatInputForm.addEventListener("submit", onSubmit);
  el.resetBtn.addEventListener("click", () => restartSession());
  el.roleSelect.addEventListener("change", () => restartSession());

  el.loadMetricsBtn.addEventListener("click", async () => {
    try {
      await loadMetrics();
      await emitAudit("frontend_metrics_viewed", { session_id: state.sessionId });
    } catch (error) {
      addMessage("assistant", `No pude cargar métricas: ${error.message}`, "error");
    }
  });

  el.loadImportanceBtn.addEventListener("click", async () => {
    try {
      await loadImportance();
      await emitAudit("frontend_importance_viewed", { session_id: state.sessionId });
    } catch (error) {
      addMessage("assistant", `No pude cargar importancia: ${error.message}`, "error");
    }
  });
}

async function bootstrap() {
  attachEvents();
  await emitAudit("frontend_loaded", { session_id: state.sessionId });
  await initializeChatFlow();
}

bootstrap();
