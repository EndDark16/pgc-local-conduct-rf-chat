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
  RESULT_QA: "result_qa",
  RESTARTING: "restarting",
  ERROR: "error",
};

const HELP_TERMS = [
  "no entiendo",
  "no comprendo",
  "explicame",
  "explicar",
  "que significa",
  "que quiere decir",
  "dame un ejemplo",
  "ejemplo",
  "ayuda",
  "no se que responder",
  "explicame con palabras simples",
];

const CONFIRM_TERMS = ["si", "de acuerdo", "correcto", "continuar", "ok", "dale", "confirmo"];
const DENY_TERMS = ["no", "corregir", "cambiar", "no es correcto", "no exactamente"];

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
  lastPrediction: null,
  waitingPrediction: false,
  autoPredictTriggered: false,
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
  typingIndicator: document.getElementById("typingIndicator"),
  quickActions: document.getElementById("quickActions"),
  contextActions: document.getElementById("contextActions"),
  chatInputForm: document.getElementById("chatInputForm"),
  chatInput: document.getElementById("chatInput"),
  sendBtn: document.getElementById("sendBtn"),
  loadMetricsBtn: document.getElementById("loadMetricsBtn"),
  loadImportanceBtn: document.getElementById("loadImportanceBtn"),
  metricCards: document.getElementById("metricCards"),
  metricsChart: document.getElementById("metricsChart"),
  confusionChart: document.getElementById("confusionChart"),
  importanceChart: document.getElementById("importanceChart"),
  importanceList: document.getElementById("importanceList"),
  overfitNote: document.getElementById("overfitNote"),
};

function normalizeText(value) {
  return (value || "")
    .toString()
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function looksLikeHelpIntent(text) {
  const norm = normalizeText(text);
  if (!norm) return false;
  return HELP_TERMS.some((term) => norm.includes(term));
}

function isAffirmative(text) {
  const norm = normalizeText(text);
  return CONFIRM_TERMS.some((term) => norm === term || norm.startsWith(`${term} `));
}

function isNegative(text) {
  const norm = normalizeText(text);
  return DENY_TERMS.some((term) => norm === term || norm.startsWith(`${term} `));
}

function confidenceLabel(value) {
  const n = Number(value || 0);
  if (n >= 0.85) return "alto";
  if (n >= 0.65) return "medio";
  return "bajo";
}

function setState(nextState) {
  state.machine = nextState;
}

function ensureVisible(element, visible) {
  if (!element) return;
  element.classList.toggle("hidden", !visible);
}

function scrollChatToBottom() {
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

function addMessage(author, text, extraClass = "", asHtml = false) {
  const div = document.createElement("div");
  div.className = `chat-message ${author} ${extraClass}`.trim();
  if (asHtml) {
    div.innerHTML = text;
  } else {
    div.textContent = text;
  }
  el.chatMessages.appendChild(div);
  scrollChatToBottom();
}

function clearChat() {
  el.chatMessages.innerHTML = "";
}

function clearContextActions() {
  el.contextActions.innerHTML = "";
  ensureVisible(el.contextActions, false);
}

function renderContextActions(actions = []) {
  clearContextActions();
  if (!actions.length) return;
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `btn ${action.className || "btn-ghost"}`;
    button.textContent = action.label;
    button.disabled = Boolean(action.disabled);
    button.addEventListener("click", action.onClick);
    el.contextActions.appendChild(button);
  });
  ensureVisible(el.contextActions, true);
}

function setTyping(visible) {
  ensureVisible(el.typingIndicator, visible);
}

function setInputEnabled(enabled) {
  el.chatInput.disabled = !enabled;
  el.sendBtn.disabled = !enabled;
}

function currentQuestion() {
  return state.questions[state.currentIndex] || null;
}

function updateProgressLabel() {
  const total = state.questions.length;
  if (total === 0) {
    el.progressText.textContent = "Preparando preguntas...";
    return;
  }
  if (state.currentIndex >= total) {
    el.progressText.textContent = `Completado ${total} de ${total}`;
    return;
  }
  el.progressText.textContent = `Pregunta ${state.currentIndex + 1} de ${total}`;
}

async function emitAudit(event, payload = {}) {
  try {
    await fetch("/api/audit-event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, payload }),
    });
  } catch (_) {
    // Keep flow alive.
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const json = await response.json();
      message = json.detail || json.message || message;
    } catch (_) {
      // keep default
    }
    throw new Error(message);
  }
  return response.json();
}

function dedupeChips(chips = []) {
  const out = [];
  const seen = new Set();
  chips.forEach((chip) => {
    const text = `${chip || ""}`.trim();
    if (!text) return;
    const key = normalizeText(text);
    if (seen.has(key)) return;
    seen.add(key);
    out.push(text);
  });
  return out;
}

function renderQuickChips(chips = []) {
  el.quickActions.innerHTML = "";
  dedupeChips(chips).forEach((chipText) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = chipText;
    btn.addEventListener("click", async () => {
      await onChipSelected(chipText);
    });
    el.quickActions.appendChild(btn);
  });
}

function questionChips(question, interpreted = null) {
  if (interpreted && Array.isArray(interpreted.quick_chips) && interpreted.quick_chips.length) {
    return interpreted.quick_chips;
  }
  if (question && Array.isArray(question.quick_chips) && question.quick_chips.length) {
    return question.quick_chips;
  }
  return ["No entiendo", "Dame un ejemplo"];
}

async function onChipSelected(chipText) {
  const text = `${chipText || ""}`.trim();
  if (!text) return;

  if (state.machine === STATE.RESULT_QA) {
    if (normalizeText(text).includes("repetir")) {
      await restartSession();
      return;
    }
    await processResultQuestion(text);
    return;
  }

  if ([STATE.WAITING_ANSWER, STATE.NEEDS_CLARIFICATION, STATE.WAITING_CONFIRMATION].includes(state.machine)) {
    el.chatInput.value = "";
    await processUserMessage(text);
  }
}

async function loadModelStatus() {
  const payload = await fetchJson("/api/model-status");
  state.modelReady = Boolean(payload.model_trained);
  state.targetColumn = payload.target_column || state.targetColumn;
  state.targetIntro = payload.target_intro || "";

  el.modelStatusText.textContent = payload.message || "Estado no disponible";
  el.targetText.textContent = state.targetColumn;
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
  state.lastPrediction = null;
  state.autoPredictTriggered = false;
  if (data.target_column) state.targetColumn = data.target_column;
  if (data.intro_text) state.targetIntro = data.intro_text;
  updateProgressLabel();
}

function introMessages() {
  addMessage(
    "assistant",
    "Hola. Soy un asistente local de apoyo preliminar. Te haré preguntas sencillas y puedes responder con tus palabras. Si no entiendes una pregunta, escribe 'no entiendo' o usa los chips de ayuda."
  );
  addMessage("assistant", state.targetIntro || "Te haré preguntas relacionadas con el dominio seleccionado.");
  addMessage("assistant", "Avanzaremos paso a paso y confirmaré cada respuesta antes de continuar.");
}

function askCurrentQuestion() {
  const question = currentQuestion();
  if (!question) {
    onQuestionnaireCompleted();
    return;
  }

  setState(STATE.ASKING_QUESTION);
  updateProgressLabel();

  addMessage("assistant", `Pregunta ${question.progress_index} de ${question.progress_total}: ${question.question}`);
  if (question.help_text) addMessage("system", question.help_text);
  if (question.human_options_text) addMessage("system", question.human_options_text);

  renderQuickChips(questionChips(question));
  clearContextActions();
  setState(STATE.WAITING_ANSWER);
  setInputEnabled(true);
  el.chatInput.focus();
}

function onQuestionnaireCompleted() {
  setState(STATE.COMPLETED);
  updateProgressLabel();
  clearContextActions();
  renderQuickChips([]);
  addMessage("assistant", "Listo, ya tengo la información necesaria.");

  renderContextActions([
    {
      label: "Generar impresión orientativa",
      className: "btn-primary",
      onClick: () => generatePrediction(),
    },
  ]);

  if (!state.autoPredictTriggered) {
    state.autoPredictTriggered = true;
    setTimeout(() => {
      if (state.machine === STATE.COMPLETED) {
        generatePrediction();
      }
    }, 600);
  }
}

async function explainCurrentQuestion(mode = "simple") {
  const question = currentQuestion();
  if (!question) return;

  setTyping(true);
  try {
    const explain = await fetchJson("/api/chat/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature: question.feature, mode }),
    });

    const text = [
      explain.simple_explanation || "Te explico la pregunta con palabras más simples.",
      Array.isArray(explain.examples) && explain.examples.length ? `Ejemplos: ${explain.examples.join(" | ")}` : "",
      explain.expected_answer || explain.human_options_text || "",
    ]
      .filter(Boolean)
      .join("\n");

    addMessage("assistant", text);
    renderQuickChips(questionChips(question, explain));
    setState(STATE.WAITING_ANSWER);
    setInputEnabled(true);
    await emitAudit("frontend_help_requested", {
      session_id: state.sessionId,
      feature: question.feature,
      mode,
    });
  } finally {
    setTyping(false);
  }
}

function buildInterpretationMessage(interpreted) {
  const lines = [
    interpreted.user_friendly_interpretation || "Interpreté tu respuesta.",
    interpreted.value_explanation || "",
    `Nivel de seguridad: ${confidenceLabel(interpreted.confidence)}.`,
  ];
  return lines.filter(Boolean).join("\n");
}

async function interpretCurrentAnswer(answerText) {
  const question = currentQuestion();
  if (!question) return;

  setState(STATE.INTERPRETING);
  setTyping(true);
  setInputEnabled(false);

  try {
    const payload = await fetchJson("/api/chat/interpret", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feature: question.feature,
        answer: answerText,
        role: state.role,
        session_id: state.sessionId,
        question_metadata: {
          response_type: question.response_type,
          response_options: question.response_options,
          scale_guidance: question.scale_guidance,
          help_text: question.help_text,
          question: question.question,
          min_value: question.min_value,
          max_value: question.max_value,
        },
      }),
    });

    if (payload.needs_explanation) {
      const explanation = payload.explanation || {};
      const text = [
        "Claro, te ayudo con esta pregunta.",
        explanation.simple_explanation || "",
        Array.isArray(explanation.examples) && explanation.examples.length ? `Ejemplos: ${explanation.examples.join(" | ")}` : "",
        explanation.expected_answer || explanation.human_options_text || "",
      ]
        .filter(Boolean)
        .join("\n");
      addMessage("assistant", text);
      renderQuickChips(questionChips(question, explanation));
      setState(STATE.NEEDS_CLARIFICATION);
      setInputEnabled(true);
      return;
    }

    const interpreted = payload.interpreted;
    state.pendingInterpretation = interpreted;

    if (!interpreted) {
      addMessage("assistant", "No pude entender bien esta respuesta. Necesito una respuesta un poco más clara para continuar.", "error");
      setState(STATE.NEEDS_CLARIFICATION);
      setInputEnabled(true);
      return;
    }

    addMessage("assistant", buildInterpretationMessage(interpreted));

    const parsedMissing = interpreted.parsed_value === null || interpreted.parsed_value === undefined;
    if (interpreted.needs_clarification || parsedMissing) {
      addMessage(
        "assistant",
        interpreted.clarification_question || "No estoy completamente seguro de haber entendido. ¿Puedes responder de forma más directa?"
      );
      if (interpreted.human_options_text) {
        addMessage("system", interpreted.human_options_text);
      }

      const actions = [];
      if (interpreted.allow_missing_value) {
        actions.push({
          label: "Omitir esta pregunta",
          className: "btn-warning",
          onClick: () => confirmCurrentInterpretation(true),
        });
      }
      renderContextActions(actions);
      renderQuickChips(questionChips(question, interpreted));
      setState(STATE.NEEDS_CLARIFICATION);
      setInputEnabled(true);
      return;
    }

    addMessage("assistant", "¿Deseas continuar o corregir esta interpretación?");
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
    renderQuickChips(questionChips(question, interpreted));
    setState(STATE.WAITING_CONFIRMATION);
    setInputEnabled(true);
  } finally {
    setTyping(false);
  }
}

async function confirmCurrentInterpretation(useMissingStrategy = false) {
  const question = currentQuestion();
  const interpreted = state.pendingInterpretation;
  if (!question || !interpreted) {
    addMessage("assistant", "Aún no tengo una interpretación para confirmar.", "error");
    return;
  }

  let parsedValue = interpreted.parsed_value;
  if ((parsedValue === null || parsedValue === undefined) && !useMissingStrategy) {
    addMessage("assistant", "Necesito una respuesta más clara antes de continuar con esta pregunta.", "error");
    setState(STATE.NEEDS_CLARIFICATION);
    setInputEnabled(true);
    return;
  }

  if ((parsedValue === null || parsedValue === undefined) && useMissingStrategy) {
    parsedValue = null;
  }

  await fetchJson("/api/chat/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      feature: question.feature,
      parsed_value: parsedValue,
      raw_answer: interpreted.raw_answer || "",
      confidence: interpreted.confidence || 0,
      session_id: state.sessionId,
      used_missing_strategy: Boolean(useMissingStrategy),
    }),
  });

  state.confirmedAnswers[question.feature] = parsedValue;
  state.pendingInterpretation = null;
  clearContextActions();

  addMessage("assistant", "Respuesta guardada. Continuemos.", "success");

  state.currentIndex += 1;
  updateProgressLabel();
  await emitAudit("frontend_answer_confirmed", {
    session_id: state.sessionId,
    feature: question.feature,
    value: parsedValue,
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

function reportHtml(report, prediction) {
  const indicators = Array.isArray(report.observed_indicators) ? report.observed_indicators : [];
  const indicatorList = indicators.length
    ? `<ul>${indicators.map((item) => `<li>${item}</li>`).join("")}</ul>`
    : "<p>No se registraron indicadores destacados.</p>";

  const probability = Number(prediction.probability_estimated || 0) * 100;
  const threshold = prediction.threshold_used;

  return `
    <strong>${report.title || "Impresión orientativa"}</strong><br />
    <span>${report.clinical_style_summary || ""}</span><br /><br />
    <strong>Nivel:</strong> ${report.compatibility_level || "No disponible"}<br />
    <strong>Probabilidad estimada:</strong> ${probability.toFixed(2)}%<br />
    <strong>Threshold usado:</strong> ${threshold ?? "No disponible"}<br /><br />
    <strong>Indicadores observados:</strong>
    ${indicatorList}
    <strong>Impacto funcional:</strong> ${report.functional_impact || ""}<br /><br />
    <strong>Recomendación profesional:</strong> ${report.professional_recommendation || ""}<br /><br />
    <strong>Aclaración importante:</strong> ${report.important_clarification || ""}
  `;
}

async function generatePrediction() {
  if (state.waitingPrediction) return;
  if (!Object.keys(state.confirmedAnswers).length) {
    addMessage("assistant", "Todavía no hay respuestas confirmadas para estimar.", "error");
    return;
  }

  state.waitingPrediction = true;
  setState(STATE.SHOWING_RESULT);
  clearContextActions();
  renderQuickChips([]);
  setTyping(true);
  setInputEnabled(false);

  try {
    const payload = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        answers: state.confirmedAnswers,
      }),
    });

    if (!payload.ok) {
      setState(STATE.ERROR);
      addMessage("assistant", payload.message || "No fue posible generar la estimación.", "error");
      setInputEnabled(true);
      return;
    }

    const prediction = payload.prediction;
    state.lastPrediction = prediction;

    const report = prediction.orientative_report || {};
    addMessage("assistant", reportHtml(report, prediction), "", true);

    if (prediction.overfit_warning) {
      addMessage("system", prediction.overfit_warning);
    }

    setState(STATE.RESULT_QA);
    addMessage(
      "assistant",
      "Si quieres, ahora puedes preguntarme sobre este resultado: significado, indicadores, métricas o próximos pasos."
    );

    renderQuickChips(prediction.result_qa_chips || []);
    renderContextActions([
      {
        label: "Iniciar nueva evaluación",
        className: "btn-primary",
        onClick: () => restartSession(),
      },
    ]);

    setInputEnabled(true);
    await refreshTechnicalPanels();
  } finally {
    state.waitingPrediction = false;
    setTyping(false);
  }
}

async function processResultQuestion(text) {
  setTyping(true);
  try {
    const payload = await fetchJson("/api/chat/result-question", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        session_id: state.sessionId,
      }),
    });

    if (!payload.ok) {
      addMessage("assistant", "No pude responder esa pregunta por ahora.", "error");
      return;
    }

    addMessage("assistant", payload.answer || "Puedo ayudarte con más detalles del resultado.");
    if (Array.isArray(payload.chips) && payload.chips.length) {
      renderQuickChips(payload.chips);
    }
  } finally {
    setTyping(false);
  }
}

async function processUserMessage(rawText) {
  const text = (rawText || "").trim();
  if (!text) return;

  addMessage("user", text);
  await emitAudit("frontend_user_message", {
    session_id: state.sessionId,
    state: state.machine,
    text,
  });

  if (state.machine === STATE.RESULT_QA) {
    await processResultQuestion(text);
    return;
  }

  if (state.machine === STATE.WAITING_CONFIRMATION) {
    const norm = normalizeText(text);
    const pendingRaw = normalizeText(state.pendingInterpretation?.raw_answer || "");

    if (isAffirmative(norm) || (pendingRaw && norm === pendingRaw)) {
      await confirmCurrentInterpretation(false);
      return;
    }

    if (isNegative(norm)) {
      correctionFlow();
      return;
    }

    // If user typed something else, treat as correction answer (no loops).
    correctionFlow();
    await interpretCurrentAnswer(text);
    return;
  }

  if (looksLikeHelpIntent(text)) {
    await explainCurrentQuestion("simple");
    return;
  }

  await interpretCurrentAnswer(text);
}

async function restartSession() {
  setState(STATE.RESTARTING);
  setInputEnabled(false);
  await fetchJson("/api/reset-session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  });

  state.currentIndex = 0;
  state.pendingInterpretation = null;
  state.confirmedAnswers = {};
  state.questions = [];
  state.lastPrediction = null;
  state.waitingPrediction = false;
  state.autoPredictTriggered = false;

  clearContextActions();
  clearChat();
  renderQuickChips([]);
  updateProgressLabel();

  await emitAudit("frontend_session_restarted", { session_id: state.sessionId });
  await initializeChatFlow();
}

function renderMetricCards(metricsPayload) {
  const metrics = metricsPayload.threshold_final || {};
  const rows = [
    { label: "F1", value: metrics.f1 },
    { label: "Recall", value: metrics.recall },
    { label: "Precision", value: metrics.precision },
    { label: "Accuracy", value: metrics.accuracy },
    { label: "Threshold", value: metrics.threshold },
  ];

  el.metricCards.innerHTML = "";
  rows.forEach((row) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `<div class="label">${row.label}</div><div class="value">${row.value ?? "N/A"}</div>`;
    el.metricCards.appendChild(card);
  });
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function drawMetricBars(metricsPayload) {
  const canvas = el.metricsChart;
  const ctx = fitCanvas(canvas);
  const metrics = metricsPayload.threshold_final || {};
  const values = [
    { key: "F1", value: Number(metrics.f1 || 0) },
    { key: "Recall", value: Number(metrics.recall || 0) },
    { key: "Precision", value: Number(metrics.precision || 0) },
    { key: "Accuracy", value: Number(metrics.accuracy || 0) },
  ];

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1628";
  ctx.fillRect(0, 0, w, h);

  const barW = Math.max(36, (w - 70) / values.length - 14);
  const baseY = h - 26;
  values.forEach((item, idx) => {
    const x = 24 + idx * (barW + 14);
    const barH = Math.max(2, item.value * (h - 58));
    const y = baseY - barH;

    const gradient = ctx.createLinearGradient(0, y, 0, baseY);
    gradient.addColorStop(0, "#7c5cff");
    gradient.addColorStop(1, "#22d3ee");

    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, barW, barH);

    ctx.fillStyle = "#f4f7fb";
    ctx.font = "12px Segoe UI";
    ctx.fillText(item.key, x, h - 8);
    ctx.fillText(item.value.toFixed(3), x, y - 6);
  });
}

function drawConfusionMatrix(metricsPayload) {
  const canvas = el.confusionChart;
  const ctx = fitCanvas(canvas);
  const matrix = metricsPayload.confusion_matrix || [[0, 0], [0, 0]];

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1628";
  ctx.fillRect(0, 0, w, h);

  const boxW = (w - 100) / 2;
  const boxH = (h - 80) / 2;
  const startX = 50;
  const startY = 28;

  const flat = matrix.flat().map((v) => Number(v || 0));
  const max = Math.max(1, ...flat);

  for (let r = 0; r < 2; r += 1) {
    for (let c = 0; c < 2; c += 1) {
      const x = startX + c * boxW;
      const y = startY + r * boxH;
      const value = Number(matrix[r]?.[c] || 0);
      const alpha = 0.2 + (value / max) * 0.65;
      ctx.fillStyle = `rgba(124, 92, 255, ${alpha.toFixed(3)})`;
      ctx.fillRect(x, y, boxW - 4, boxH - 4);
      ctx.strokeStyle = "#27324a";
      ctx.strokeRect(x, y, boxW - 4, boxH - 4);

      ctx.fillStyle = "#f4f7fb";
      ctx.font = "13px Segoe UI";
      ctx.fillText(String(value), x + 14, y + boxH / 2);
    }
  }

  ctx.fillStyle = "#aab4c5";
  ctx.font = "12px Segoe UI";
  ctx.fillText("Pred 0", startX + 10, h - 10);
  ctx.fillText("Pred 1", startX + boxW + 10, h - 10);
  ctx.fillText("Real 0", 4, startY + 18);
  ctx.fillText("Real 1", 4, startY + boxH + 18);
}

function drawImportance(importancePayload) {
  const rows = (importancePayload.feature_importance_aggregated || []).slice(0, 8);
  const canvas = el.importanceChart;
  const ctx = fitCanvas(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1628";
  ctx.fillRect(0, 0, w, h);

  if (!rows.length) {
    ctx.fillStyle = "#aab4c5";
    ctx.font = "13px Segoe UI";
    ctx.fillText("Sin datos de importancia.", 12, 22);
    el.importanceList.innerHTML = "";
    return;
  }

  const max = Math.max(...rows.map((row) => Number(row.importance || 0)), 0.0001);
  const barAreaW = w - 210;
  const rowH = Math.max(24, Math.floor((h - 22) / rows.length));

  rows.forEach((row, idx) => {
    const y = 12 + idx * rowH;
    const pct = Number(row.importance || 0) / max;
    const barW = Math.max(2, pct * barAreaW);

    ctx.fillStyle = "rgba(34, 211, 238, 0.2)";
    ctx.fillRect(170, y, barAreaW, rowH - 7);

    ctx.fillStyle = "#22d3ee";
    ctx.fillRect(170, y, barW, rowH - 7);

    ctx.fillStyle = "#dbe5fa";
    ctx.font = "12px Segoe UI";
    const label = String(row.label || row.feature || "").slice(0, 22);
    ctx.fillText(label, 8, y + 14);
    ctx.fillText(`${(Number(row.importance || 0) * 100).toFixed(2)}%`, 174 + barW, y + 14);
  });

  el.importanceList.innerHTML = rows
    .map(
      (row) =>
        `<div class="importance-item"><span>${row.label || row.feature}</span><strong>${(Number(
          row.importance || 0
        ) * 100).toFixed(2)}%</strong></div>`
    )
    .join("");
}

async function loadMetricsPanel() {
  const data = await fetchJson("/api/metrics");
  if (!data.ok) {
    el.metricCards.innerHTML = `<article class="metric-card"><div class="label">Estado</div><div class="value">${
      data.message || "Sin métricas"
    }</div></article>`;
    clearCanvas(el.metricsChart);
    clearCanvas(el.confusionChart);
    ensureVisible(el.overfitNote, false);
    return null;
  }

  renderMetricCards(data);
  drawMetricBars(data);
  drawConfusionMatrix(data);

  if (data.overfit_warning) {
    el.overfitNote.textContent = data.overfit_warning;
    ensureVisible(el.overfitNote, true);
  } else {
    ensureVisible(el.overfitNote, false);
  }
  return data;
}

async function loadImportancePanel() {
  const data = await fetchJson("/api/feature-importance");
  if (!data.ok) {
    el.importanceList.innerHTML = `<div class="importance-item"><span>${data.message || "Sin datos"}</span><strong></strong></div>`;
    clearCanvas(el.importanceChart);
    return null;
  }

  drawImportance(data);
  return data;
}

async function refreshTechnicalPanels() {
  try {
    await loadMetricsPanel();
    await loadImportancePanel();
  } catch (error) {
    addMessage("assistant", `No pude actualizar detalles técnicos: ${error.message}`, "error");
  }
}

async function initializeChatFlow() {
  setState(STATE.LOADING);
  ensureVisible(el.modelMissingCard, false);
  ensureVisible(el.chatPanel, true);
  clearContextActions();
  clearChat();
  renderQuickChips([]);
  setInputEnabled(false);
  updateProgressLabel();

  try {
    const status = await loadModelStatus();
    await emitAudit("frontend_model_status_loaded", {
      session_id: state.sessionId,
      model_trained: status.model_trained,
      target: status.target_column,
    });

    if (!status.model_trained) {
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
    addMessage(
      "assistant",
      error.message || "Ocurrió un problema al preparar el chat. Revisa el esquema y vuelve a entrenar.",
      "error"
    );
    setInputEnabled(false);
    await emitAudit("frontend_error", {
      session_id: state.sessionId,
      message: error.message || "Error desconocido",
      stage: "initialize",
    });
  }
}

async function onSubmit(event) {
  event.preventDefault();
  const text = el.chatInput.value.trim();
  if (!text) return;

  const allowed = [
    STATE.WAITING_ANSWER,
    STATE.NEEDS_CLARIFICATION,
    STATE.WAITING_CONFIRMATION,
    STATE.RESULT_QA,
  ];
  if (!allowed.includes(state.machine)) return;

  el.chatInput.value = "";
  await processUserMessage(text);
}

function attachEvents() {
  el.chatInputForm.addEventListener("submit", onSubmit);
  el.resetBtn.addEventListener("click", () => restartSession());
  el.roleSelect.addEventListener("change", () => restartSession());

  el.loadMetricsBtn.addEventListener("click", async () => {
    await loadMetricsPanel();
    await emitAudit("frontend_metrics_viewed", { session_id: state.sessionId });
  });

  el.loadImportanceBtn.addEventListener("click", async () => {
    await loadImportancePanel();
    await emitAudit("frontend_importance_viewed", { session_id: state.sessionId });
  });

  window.addEventListener("resize", () => {
    if (state.lastPrediction) {
      refreshTechnicalPanels();
    }
  });
}

async function bootstrap() {
  attachEvents();
  await emitAudit("frontend_loaded", { session_id: state.sessionId });
  await initializeChatFlow();
}

bootstrap();
