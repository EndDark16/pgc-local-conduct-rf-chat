"use strict";

const fs = require("fs");
const path = require("path");

const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};

const BLOCKED_HEADERS = new Set([
  "host",
  "connection",
  "x-forwarded-for",
  "x-forwarded-host",
  "x-forwarded-proto",
  "x-nf-client-connection-ip",
  "x-nf-request-id",
  "content-length",
]);

const TARGET_LABELS = {
  target_domain_conduct_final: "Conducta y convivencia",
  target_domain_adhd_final: "Atención e hiperactividad (ADHD)",
  target_domain_anxiety_final: "Ansiedad",
  target_domain_depression_final: "Estado de ánimo (depresión)",
  target_domain_elimination_final: "Eliminación y control de esfínteres",
};

const TARGET_INTROS = {
  target_domain_conduct_final:
    "Te haré preguntas sobre comportamientos observables relacionados con normas, convivencia y conducta.",
  target_domain_adhd_final: "Te haré preguntas sobre atención, inquietud e impulsividad.",
  target_domain_anxiety_final: "Te haré preguntas sobre preocupaciones, miedos o señales de ansiedad.",
  target_domain_depression_final: "Te haré preguntas sobre estado de ánimo, interés y energía.",
  target_domain_elimination_final:
    "Te haré preguntas relacionadas con control de esfínteres y situaciones asociadas.",
};

const MEDICAL_DISCLAIMER =
  "Este resultado no es un diagnóstico médico y debe ser revisado por un profesional calificado.";

const HELP_EXACT = new Set([
  "no entiendo",
  "no comprendo",
  "explicame",
  "explica",
  "explicar",
  "que significa",
  "que quiere decir",
  "dame un ejemplo",
  "ejemplo",
  "ayuda",
  "no se que responder",
  "explicar con palabras mas simples",
  "explicame con palabras mas simples",
]);

const HELP_PARTIAL = [
  "no entiendo",
  "no comprendo",
  "que significa",
  "que quiere decir",
  "dame un ejemplo",
  "explicar con palabras mas simples",
  "explicame con palabras mas simples",
  "no se que responder",
];

const CONDUCT_FALLBACK_LABELS = {
  age_years: "edad en años cumplidos",
  sex_assigned_at_birth: "sexo asignado al nacer",
  conduct_impairment_global: "afectación global en la vida diaria",
  conduct_onset_before_10: "inicio temprano de conductas problemáticas (antes de los 10 años)",
  conduct_01_bullies_threatens_intimidates: "acoso o intimidación a otras personas",
  conduct_02_initiates_fights: "inicio de peleas físicas",
  conduct_03_weapon_use: "uso de objetos o armas que podrían causar daño grave",
  conduct_04_physical_cruelty_people: "crueldad física hacia otras personas",
  conduct_05_physical_cruelty_animals: "crueldad física hacia animales",
  conduct_06_steals_confronting_victim: "robo con amenaza, fuerza o intimidación",
  conduct_07_forced_sex: "conducta sexual forzada",
  conduct_08_fire_setting: "provocar incendios de manera deliberada",
  conduct_09_property_destruction: "destrucción deliberada de propiedad",
  conduct_10_breaks_into_house_building_car: "entrar por la fuerza a casa, edificio o vehículo",
  conduct_11_lies_to_obtain_or_avoid: "mentiras repetidas para obtener beneficios o evitar consecuencias",
  conduct_12_steals_without_confrontation: "robo sin confrontación directa",
  conduct_13_stays_out_at_night_before_13: "permanecer fuera de casa por la noche antes de los 13 años",
  conduct_14_runs_away_overnight: "escaparse de casa durante la noche",
  conduct_15_truancy_before_13: "ausencias injustificadas a la escuela antes de los 13 años",
  conduct_lpe_01_lack_remorse_guilt: "dificultad para mostrar remordimiento o culpa",
  conduct_lpe_02_callous_lack_empathy: "baja empatía hacia otras personas",
  conduct_lpe_03_unconcerned_performance: "poca preocupación por desempeño o consecuencias",
  conduct_lpe_04_shallow_deficient_affect: "expresión emocional superficial o limitada",
};

const TECH_PATTERNS = [
  /\binput for\b/i,
  /\btarget_domain_\w+\b/i,
  /\bresponse_options_json\b/i,
  /\bfeature_name\b/i,
  /\b[a-z]+_[a-z0-9_]+\b/,
  /\b(adhd|conduct|anxiety|depression|elimination)\s*\d{1,2}\b/i,
  /\blpe\s*\d{1,2}\b/i,
];

const sessions = new Map();
const cache = {
  loaded: false,
  metadata: {},
  metrics: {},
  featureImportance: {},
  schema: {},
  targetColumn: "target_domain_conduct_final",
};

function json(statusCode, payload) {
  return {
    statusCode,
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  };
}

function normalizeBaseUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  if (!/^https?:\/\//i.test(raw)) return "";
  return raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

function resolveSplatPath(event) {
  const rawPath = String(event.path || "");
  const prefix = "/.netlify/functions/api-proxy/";
  if (rawPath.startsWith(prefix)) return rawPath.slice(prefix.length);
  const match = rawPath.match(/\/api\/(.*)$/);
  return match && match[1] ? match[1] : "";
}

function buildTargetUrl(event, backendBase) {
  const splat = resolveSplatPath(event);
  const apiPath = splat ? `/api/${splat}` : "/api";
  const target = new URL(apiPath, `${backendBase}/`);
  const query = event.rawQuery || "";
  if (query) target.search = query;
  return target.toString();
}

function buildForwardHeaders(eventHeaders) {
  const out = {};
  for (const [rawKey, rawValue] of Object.entries(eventHeaders || {})) {
    const key = String(rawKey || "").toLowerCase();
    if (!key || BLOCKED_HEADERS.has(key)) continue;
    if (rawValue === undefined || rawValue === null || rawValue === "") continue;
    out[key] = String(rawValue);
  }
  return out;
}

function normalizeText(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function decodeMojibake(text) {
  if (typeof text !== "string") return text;
  if (!/[ÃÂ]/.test(text)) return text;
  try {
    const decoded = Buffer.from(text, "latin1").toString("utf8");
    if (decoded && decoded.length >= Math.ceil(text.length * 0.8)) return decoded;
  } catch (_) {
    // no-op
  }
  return text;
}

function cleanText(value) {
  const text = decodeMojibake(String(value || ""));
  return text.replace(/\s+/g, " ").trim();
}

function looksTechnical(text) {
  const raw = cleanText(text);
  if (!raw) return false;
  const norm = normalizeText(raw);
  if (norm.startsWith("input for ")) return true;
  return TECH_PATTERNS.some((pattern) => pattern.test(raw) || pattern.test(norm));
}

function safeHumanText(candidates, fallback = "") {
  for (const candidate of candidates) {
    const text = cleanText(candidate || "");
    if (!text) continue;
    if (looksTechnical(text)) continue;
    return text;
  }
  return fallback;
}

function parseJsonBody(event) {
  if (!event.body) return {};
  try {
    const raw = event.isBase64Encoded ? Buffer.from(event.body, "base64").toString("utf8") : event.body;
    return JSON.parse(raw);
  } catch (_) {
    return {};
  }
}

function getQueryParam(event, key, defaultValue = "") {
  const qp = event.queryStringParameters || {};
  return qp[key] ?? defaultValue;
}

function getSession(sessionId) {
  const key = String(sessionId || "default");
  if (!sessions.has(key)) {
    sessions.set(key, { role: "caregiver", answers_confirmed: {}, latest_prediction: null, attempts_by_feature: {} });
  }
  return sessions.get(key);
}

function findReadablePath(relativeFile) {
  const candidates = [
    path.resolve(process.cwd(), relativeFile),
    path.resolve(__dirname, "..", "..", relativeFile),
    path.resolve(__dirname, "..", relativeFile),
    path.resolve(__dirname, relativeFile),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return "";
}

function loadJsonFile(relativeFile, fallbackValue = {}) {
  const p = findReadablePath(relativeFile);
  if (!p) return fallbackValue;
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch (_) {
    return fallbackValue;
  }
}

function ensureCacheLoaded() {
  if (cache.loaded) return;
  cache.metadata = loadJsonFile("models/metadata.json", {});
  cache.metrics = loadJsonFile("artifacts/metrics.json", {});
  cache.featureImportance = loadJsonFile("artifacts/feature_importance.json", {});
  cache.schema = loadJsonFile("artifacts/feature_schema.json", {});
  cache.targetColumn = String(cache.metadata.target_column || "target_domain_conduct_final");
  cache.loaded = true;
}

function targetPrefix(targetColumn) {
  if (targetColumn.includes("conduct")) return "conduct_";
  if (targetColumn.includes("adhd")) return "adhd_";
  if (targetColumn.includes("anxiety")) return "anxiety_";
  if (targetColumn.includes("depression")) return "depression_";
  if (targetColumn.includes("elimination")) return "elimination_";
  return "conduct_";
}

function featureBelongsToTarget(meta, targetColumn) {
  const feature = String(meta.feature || "").trim();
  if (!feature) return false;
  const normFeature = normalizeText(feature);
  const prefix = targetPrefix(targetColumn);
  const domain = targetColumn.replace("target_domain_", "").replace("_final", "");
  const domains = normalizeText(meta.domains_final || "");
  const section = normalizeText(meta.section_name || "");
  if (["age_years", "sex_assigned_at_birth"].includes(normFeature)) return true;
  if (normFeature.startsWith("target_domain_")) return false;
  if (/^(adhd|anxiety|depression|elimination)_/.test(normFeature) && !normFeature.startsWith(prefix)) return false;
  if (normFeature.startsWith(prefix)) return true;
  if (domains && domains.includes(domain)) return true;
  if (targetColumn === "target_domain_conduct_final") {
    if (section.includes("comportamiento") || section.includes("convivencia") || section.includes("conducta")) return true;
  }
  return false;
}

function parseOptions(meta) {
  let options = meta.response_options;
  if (!options && meta.response_options_json) options = meta.response_options_json;
  if (typeof options === "string") {
    try {
      options = JSON.parse(options);
    } catch (_) {
      options = [];
    }
  }
  if (!Array.isArray(options)) options = [];
  const normalized = [];
  const seen = new Set();
  for (const item of options) {
    if (!item) continue;
    const value = item.value ?? item.id ?? null;
    const labelRaw = cleanText(item.label ?? item.text ?? item.name ?? "");
    if (!labelRaw || labelRaw.toLowerCase() === "nan") continue;
    const key = `${normalizeText(labelRaw)}::${String(value)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    normalized.push({ value, label: labelRaw });
  }
  return normalized;
}

function inferScaleType(meta, optionsList) {
  const responseType = normalizeText(meta.response_type || "");
  const labelsText = optionsList.map((o) => normalizeText(o.label || "")).join(" ");
  const ctx = [
    responseType,
    normalizeText(meta.scale_guidance || ""),
    normalizeText(meta.help_text || ""),
    normalizeText(meta.question_text_primary || meta.caregiver_question || meta.question || ""),
    labelsText,
  ].join(" ");

  const values = new Set(
    optionsList.map((o) => (typeof o.value === "number" ? o.value : Number(o.value))).filter((v) => Number.isFinite(v) && Number.isInteger(v))
  );

  if (responseType.includes("boolean") || responseType.includes("yes_no") || responseType.includes("binaria")) return "binary";
  if (values.size === 2 && values.has(0) && values.has(1) && /\b(si|no|verdadero|falso)\b/.test(ctx)) return "binary";
  if (values.size === 3 && values.has(0) && values.has(1) && values.has(2)) {
    if (/(6 meses|12 meses|ocurrio|reciente|antes)/.test(ctx)) return "temporal_0_2";
    if (/(se observa|claramente|duda|persistente)/.test(ctx)) return "observation_0_2";
  }
  if (values.size === 4 && values.has(0) && values.has(1) && values.has(2) && values.has(3)) {
    if (/(impacto|afecta|moderado|marcado|grave|interfiere)/.test(ctx)) return "impact_0_3";
    if (/(nunca|ocasional|frecuente|casi siempre|todo el tiempo)/.test(ctx)) return "frequency_0_3";
    return "frequency_0_3";
  }

  const minNum = Number(meta.min_value);
  const maxNum = Number(meta.max_value);
  if (Number.isFinite(minNum) && Number.isFinite(maxNum) && /\b(integer|numeric|number|edad|range)\b/.test(responseType || "")) return "numeric_range";
  if (Number.isFinite(minNum) && Number.isFinite(maxNum) && optionsList.length === 0) return "numeric_range";
  if (optionsList.length) return "categorical";
  return "unknown";
}

function dedupeStrings(values) {
  const out = [];
  const seen = new Set();
  for (const v of values || []) {
    const text = cleanText(v || "");
    if (!text) continue;
    const key = normalizeText(text);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(text);
  }
  return out;
}

function defaultScalePayload(scaleType) {
  if (scaleType === "binary") return { human_options_text: "Puedes responder: No o Sí.", quick_chips: ["Sí", "No", "No entiendo"] };
  if (scaleType === "temporal_0_2") return { human_options_text: "Puedes responder: no ocurrió, ocurrió antes, u ocurrió recientemente.", quick_chips: ["No ocurrió", "Ocurrió antes", "Ocurrió recientemente", "No entiendo"] };
  if (scaleType === "observation_0_2") return { human_options_text: "Puedes responder: no se observa, a veces, o claramente.", quick_chips: ["No se observa", "A veces", "Claramente", "No entiendo"] };
  if (scaleType === "frequency_0_3") return { human_options_text: "Puedes responder: nunca, leve u ocasional, frecuente, o casi siempre.", quick_chips: ["Nunca", "Ocasional", "Frecuente", "Casi siempre", "No entiendo"] };
  if (scaleType === "impact_0_3") return { human_options_text: "Puedes responder: no afecta, leve, moderado, o marcado.", quick_chips: ["No afecta", "Leve", "Moderado", "Marcado", "No entiendo"] };
  if (scaleType === "numeric_range") return { human_options_text: "Responde con un número dentro del rango esperado.", quick_chips: ["No entiendo"] };
  return { human_options_text: "Puedes responder con tus palabras.", quick_chips: ["No entiendo", "Dame un ejemplo"] };
}

function formatOptions(meta) {
  const optionsList = parseOptions(meta);
  const scaleType = inferScaleType(meta, optionsList);
  const payload = defaultScalePayload(scaleType);

  if (scaleType === "categorical" && optionsList.length) {
    const labels = optionsList.map((o) => cleanText(o.label)).filter(Boolean);
    if (labels.length === 1) payload.human_options_text = `Puedes responder: ${labels[0]}.`;
    else if (labels.length === 2) payload.human_options_text = `Puedes responder: ${labels[0]} o ${labels[1]}.`;
    else if (labels.length > 2) payload.human_options_text = `Puedes responder: ${labels.slice(0, -1).join(", ")}, o ${labels[labels.length - 1]}.`;
    payload.quick_chips = dedupeStrings(labels.slice(0, 4).concat(["No entiendo"]));
  }

  return {
    options_list: optionsList,
    scale_type: scaleType,
    human_options_text: payload.human_options_text,
    quick_chips: dedupeStrings(payload.quick_chips),
  };
}

function humanizeFeatureName(feature, meta) {
  const safeQuestion = safeHumanText([meta.caregiver_question, meta.question_text_primary, meta.feature_label_human], "");
  if (safeQuestion) {
    const q = safeQuestion.replace(/[?¿]/g, "").trim();
    if (q) return q.charAt(0).toLowerCase() + q.slice(1);
  }
  const safeLabel = safeHumanText([meta.feature_label_human], "");
  if (safeLabel) return safeLabel;
  const normalizedFeature = normalizeText(feature).replace(/\s+/g, "_");
  if (CONDUCT_FALLBACK_LABELS[feature]) return CONDUCT_FALLBACK_LABELS[feature];
  if (CONDUCT_FALLBACK_LABELS[normalizedFeature]) return CONDUCT_FALLBACK_LABELS[normalizedFeature];
  let clean = String(feature || "").replace(/^(conduct|adhd|anxiety|depression|elimination)_/i, "").replace(/^\d+_/, "").replace(/_/g, " ").trim();
  clean = clean.replace(/\s+/g, " ");
  return clean || "indicador observado";
}

function explainQuestion(meta, role = "caregiver") {
  const question = safeHumanText(
    role === "psychologist" ? [meta.psychologist_question, meta.question_text_primary, meta.caregiver_question] : [meta.caregiver_question, meta.question_text_primary, meta.psychologist_question],
    "¿Puedes contarme qué has observado en esta situación?"
  );
  const options = formatOptions(meta);
  const simple = safeHumanText([meta.help_text, meta.term_explanation, meta.feature_description], "Esta pregunta busca entender lo que observas en situaciones cotidianas, sin juzgar ni etiquetar.");
  const examplesByScale = {
    binary: ["Marca Sí cuando esto se observa de forma repetida en la vida diaria.", "Marca No cuando no se observa o es algo excepcional."],
    temporal_0_2: ["No ocurrió: no ha pasado.", "Ocurrió antes: pasó hace tiempo, pero no recientemente.", "Ocurrió recientemente: ha pasado en los últimos meses o sigue pasando."],
    frequency_0_3: ["Nunca: prácticamente no sucede.", "Ocasional: pasa de vez en cuando.", "Frecuente/Casi siempre: se repite de forma clara en la rutina."],
    observation_0_2: ["No se observa: no lo has notado.", "A veces: aparece en algunas situaciones o hay duda.", "Claramente: se observa de forma persistente."],
    impact_0_3: ["Sin impacto: no afecta el funcionamiento diario.", "Leve/Moderado/Marcado: aumenta según cuánto interfiere en casa, escuela o convivencia."],
  };
  return {
    feature_name: String(meta.feature || ""),
    question,
    simple_explanation: cleanText(simple),
    examples: examplesByScale[options.scale_type] || ["Puedes responder con lo que observas normalmente en casa, escuela o convivencia."],
    not_about: ["No se trata de juzgar al niño o a la familia.", "No significa que exista un diagnóstico confirmado.", "Es una estimación preliminar de apoyo."],
    expected_answer: options.human_options_text,
    human_options_text: options.human_options_text,
    quick_chips: options.quick_chips,
    scale_type: options.scale_type,
  };
}

function isHelpRequest(rawText) {
  const norm = normalizeText(rawText);
  if (!norm) return false;
  if (HELP_EXACT.has(norm)) return true;
  return HELP_PARTIAL.some((term) => norm.includes(term));
}

function hasAny(text, terms) {
  return terms.some((term) => text.includes(term));
}

function parseBinary(norm, meta) {
  const onset = normalizeText(meta.question || meta.question_text_primary || "").includes("antes de los 10");
  if (hasAny(norm, ["no se observa", "no ocurre", "no afecta", "no ha pasado", "para nada", "nunca"]) || /^no\b/.test(norm) || norm === "no") {
    return {
      parsed_value: 0,
      confidence: hasAny(norm, ["creo", "quizas", "tal vez"]) ? 0.78 : 0.93,
      needs_clarification: false,
      reasoning_summary: "Se detectó una negación clara para una escala binaria.",
      user_friendly_interpretation: onset ? "Lo entendí como: no, no empezó antes de los 10 años." : "Lo entendí como: no.",
      value_explanation: "Para esta escala equivale al valor 0.",
      answer_category: "respuesta clara",
    };
  }
  if (hasAny(norm, ["si se observa", "si ocurre", "si afecta", "si ha pasado", "afirmativo", "verdadero"]) || /^si\b/.test(norm) || norm === "si" || norm === "sí") {
    return {
      parsed_value: 1,
      confidence: hasAny(norm, ["creo", "quizas", "tal vez"]) ? 0.78 : 0.93,
      needs_clarification: false,
      reasoning_summary: "Se detectó una afirmación clara para una escala binaria.",
      user_friendly_interpretation: onset ? "Lo entendí como: sí, empezó antes de los 10 años." : "Lo entendí como: sí.",
      value_explanation: "Para esta escala equivale al valor 1.",
      answer_category: "respuesta clara",
    };
  }
  if (hasAny(norm, ["a veces", "depende", "tal vez", "quizas", "no estoy seguro", "no estoy segura"])) {
    return {
      parsed_value: null,
      confidence: 0.45,
      needs_clarification: true,
      clarification_question: "Entiendo que ocurre algunas veces. Para esta pregunta necesito una respuesta más directa: ¿dirías que sí se observa de forma repetida en la vida diaria, o que no?",
      reasoning_summary: "La respuesta expresa duda para una pregunta binaria.",
      user_friendly_interpretation: "Entiendo que ocurre en algunas situaciones, pero necesito confirmarlo.",
      value_explanation: "",
      answer_category: "respuesta parcialmente clara",
    };
  }
  return {
    parsed_value: null,
    confidence: 0.35,
    needs_clarification: true,
    clarification_question: "Para esta pregunta necesito confirmar si la respuesta es sí o no.",
    reasoning_summary: "No se detectó una respuesta binaria clara.",
    user_friendly_interpretation: "Todavía no estoy seguro de si corresponde a sí o no.",
    value_explanation: "",
    answer_category: "respuesta ambigua",
  };
}

function parseTemporal(norm) {
  if (hasAny(norm, ["no ocurrio", "nunca", "jamas", "no ha pasado", "no se ha presentado"])) return { parsed_value: 0, confidence: 0.9, needs_clarification: false, reasoning_summary: "Se detectó que no ocurrió.", user_friendly_interpretation: "Lo entendí como: no ocurrió.", value_explanation: "Para esta escala equivale al valor 0.", answer_category: "respuesta clara" };
  if (hasAny(norm, ["ocurrio antes", "antes pasaba", "hace tiempo", "hace mas de seis meses", "hace mas de 6 meses", "el ano pasado", "paso pero ya no", "no recientemente", "paso hace tiempo"])) return { parsed_value: 1, confidence: 0.9, needs_clarification: false, reasoning_summary: "Se detectó ocurrencia anterior no reciente.", user_friendly_interpretation: "Lo entendí como: ocurrió antes, pero no recientemente.", value_explanation: "Para esta escala equivale al valor 1.", answer_category: "respuesta clara" };
  if (hasAny(norm, ["ocurrio recientemente", "recientemente", "actualmente", "pasa ahora", "ultimos 6 meses", "ultimos seis meses", "hace poco", "sigue pasando", "una vez hace poco"])) return { parsed_value: 2, confidence: 0.92, needs_clarification: false, reasoning_summary: "Se detectó ocurrencia reciente.", user_friendly_interpretation: "Lo entendí como: ocurrió recientemente.", value_explanation: "Para esta escala equivale al valor 2.", answer_category: "respuesta clara" };
  if (hasAny(norm, ["ocurrio", "paso", "ha pasado", "sucede", "se presento"])) return { parsed_value: null, confidence: 0.62, needs_clarification: true, clarification_question: "Entiendo que sí ocurrió. Para ubicarlo bien, necesito saber si fue algo anterior o si ocurrió recientemente.", reasoning_summary: "Se detectó ocurrencia, pero falta temporalidad.", user_friendly_interpretation: "Entiendo que sí ocurrió, pero aún necesito cuándo.", value_explanation: "", answer_category: "respuesta parcialmente clara" };
  return { parsed_value: null, confidence: 0.35, needs_clarification: true, clarification_question: "Necesito saber cuándo ocurrió: no ocurrió, ocurrió antes pero no recientemente, u ocurrió en los últimos 6 meses.", reasoning_summary: "No se logró identificar la temporalidad.", user_friendly_interpretation: "Todavía no puedo ubicar cuándo ocurrió.", value_explanation: "", answer_category: "respuesta ambigua" };
}

function parseFrequency(norm) {
  if (hasAny(norm, ["ocurrio recientemente", "recientemente", "hace poco", "ultimos 6 meses"])) {
    return { parsed_value: null, confidence: 0.35, needs_clarification: true, clarification_question: "Eso me dice cuándo ocurrió, pero esta pregunta necesita saber qué tan frecuente o marcado es. ¿Dirías que nunca, leve u ocasional, frecuente, o casi siempre?", reasoning_summary: "La respuesta describe temporalidad y no frecuencia.", user_friendly_interpretation: "Entiendo cuándo ocurrió, pero aquí necesito qué tan frecuente es.", value_explanation: "", answer_category: "respuesta parcialmente clara" };
  }
  const table = [
    { v: 3, t: ["casi siempre", "siempre", "todo el tiempo", "constantemente", "muy frecuente", "demasiado", "muchisimo", "casi todo el tiempo"], c: 0.92, l: "casi siempre" },
    { v: 2, t: ["frecuente", "frecuentemente", "varias veces", "muchas veces", "a menudo", "se nota", "bastante"], c: 0.84, l: "frecuente" },
    { v: 1, t: ["leve", "ocasional", "rara vez", "pocas veces", "muy poco", "de vez en cuando", "a veces"], c: 0.72, l: "ocasional" },
    { v: 0, t: ["nunca", "no ocurre", "ausente", "para nada", "casi nunca"], c: 0.88, l: "nunca" },
  ];
  for (const row of table) {
    if (hasAny(norm, row.t)) return { parsed_value: row.v, confidence: row.c, needs_clarification: false, reasoning_summary: "Se detectó una expresión compatible con escala de frecuencia.", user_friendly_interpretation: `Lo entendí como: ${row.l}.`, value_explanation: `Para esta escala equivale al nivel ${row.v}.`, answer_category: row.c >= 0.8 ? "respuesta clara" : "respuesta parcialmente clara" };
  }
  return { parsed_value: null, confidence: 0.34, needs_clarification: true, clarification_question: "Necesito ubicarlo en una escala de frecuencia. ¿Se parece más a nunca, leve/ocasional, frecuente o casi siempre?", reasoning_summary: "No se detectó frecuencia clara.", user_friendly_interpretation: "Todavía no puedo ubicar esta respuesta en frecuencia.", value_explanation: "", answer_category: "respuesta ambigua" };
}

function parseObservation(norm) {
  const table = [
    { v: 0, t: ["no se observa", "no pasa", "no lo he visto", "nunca"], c: 0.9, l: "no se observa" },
    { v: 1, t: ["a veces", "algunas veces", "hay duda", "no estoy seguro", "ocasionalmente", "puede ser"], c: 0.72, l: "se observa a veces" },
    { v: 2, t: ["claramente", "si se observa", "frecuente", "persistente", "casi siempre", "muy claro", "sin duda"], c: 0.9, l: "se observa claramente" },
  ];
  for (const row of table) {
    if (hasAny(norm, row.t)) return { parsed_value: row.v, confidence: row.c, needs_clarification: false, reasoning_summary: "Se detectó una expresión compatible con escala de observación.", user_friendly_interpretation: `Lo entendí como: ${row.l}.`, value_explanation: `Para esta escala equivale al valor ${row.v}.`, answer_category: row.c >= 0.8 ? "respuesta clara" : "respuesta parcialmente clara" };
  }
  return { parsed_value: null, confidence: 0.33, needs_clarification: true, clarification_question: "Necesito ubicarlo como: no se observa, a veces, o claramente.", reasoning_summary: "No se detectó nivel de observación claro.", user_friendly_interpretation: "Todavía no puedo ubicar esta respuesta en observación.", value_explanation: "", answer_category: "respuesta ambigua" };
}

function parseImpact(norm) {
  const table = [
    { v: 3, t: ["marcado", "fuerte", "grave", "severo", "afecta mucho", "impide actividades"], c: 0.9, l: "impacto marcado" },
    { v: 2, t: ["moderado", "medio", "afecta bastante", "interfiere", "bastante"], c: 0.84, l: "impacto moderado" },
    { v: 1, t: ["leve", "poco", "afecta un poco", "manejable", "no mucho"], c: 0.82, l: "impacto leve" },
    { v: 0, t: ["sin impacto", "no afecta", "nada", "no hay problema"], c: 0.9, l: "sin impacto" },
  ];
  for (const row of table) {
    if (hasAny(norm, row.t)) return { parsed_value: row.v, confidence: row.c, needs_clarification: false, reasoning_summary: "Se detectó una expresión compatible con escala de impacto.", user_friendly_interpretation: `Lo entendí como: ${row.l}.`, value_explanation: `Para esta escala equivale al nivel ${row.v}.`, answer_category: row.c >= 0.8 ? "respuesta clara" : "respuesta parcialmente clara" };
  }
  return { parsed_value: null, confidence: 0.33, needs_clarification: true, clarification_question: "Necesito saber cuánto afecta: no afecta, leve, moderado o marcado.", reasoning_summary: "No se detectó nivel de impacto claro.", user_friendly_interpretation: "Todavía no puedo ubicar cuánto impacto describe la respuesta.", value_explanation: "", answer_category: "respuesta ambigua" };
}

function parseNumeric(norm, meta) {
  const m = norm.match(/-?\d+(?:\.\d+)?/);
  if (!m) return { parsed_value: null, confidence: 0.3, needs_clarification: true, clarification_question: "Para esta pregunta necesito un número dentro del rango esperado.", reasoning_summary: "No se detectó un número en la respuesta.", user_friendly_interpretation: "Necesito un número para registrar esta respuesta.", value_explanation: "", answer_category: "respuesta insuficiente" };
  const value = Number(m[0]);
  const minValue = Number(meta.min_value);
  const maxValue = Number(meta.max_value);
  if (Number.isFinite(minValue) && value < minValue) return { parsed_value: null, confidence: 0.3, needs_clarification: true, clarification_question: `El valor parece menor al mínimo permitido (${minValue}).`, reasoning_summary: "El valor está por debajo del rango permitido.", user_friendly_interpretation: "El número parece menor al mínimo permitido.", value_explanation: "", answer_category: "respuesta ambigua", validation_error: `El valor está por debajo del mínimo permitido (${minValue}).` };
  if (Number.isFinite(maxValue) && value > maxValue) return { parsed_value: null, confidence: 0.3, needs_clarification: true, clarification_question: `El valor parece mayor al máximo permitido (${maxValue}).`, reasoning_summary: "El valor está por encima del rango permitido.", user_friendly_interpretation: "El número parece mayor al máximo permitido.", value_explanation: "", answer_category: "respuesta ambigua", validation_error: `El valor está por encima del máximo permitido (${maxValue}).` };
  return { parsed_value: Number.isInteger(value) ? value : Number(value.toFixed(2)), confidence: 0.92, needs_clarification: false, reasoning_summary: "Se detectó un valor numérico válido.", user_friendly_interpretation: `Lo entendí como: ${value}.`, value_explanation: "Guardaré ese valor numérico.", answer_category: "respuesta clara" };
}

function parseCategorical(norm, optionsList) {
  if (!optionsList.length) return { parsed_value: null, confidence: 0.3, needs_clarification: true, clarification_question: "Necesito una respuesta más concreta para esta pregunta.", reasoning_summary: "No hay opciones disponibles para mapear.", user_friendly_interpretation: "Necesito una respuesta más concreta.", value_explanation: "", answer_category: "respuesta ambigua" };
  let best = null;
  let bestScore = 0;
  for (const opt of optionsList) {
    const labelNorm = normalizeText(opt.label);
    let score = 0;
    if (norm === labelNorm) score = 100;
    else if (norm.includes(labelNorm) || labelNorm.includes(norm)) score = 86;
    if (score > bestScore) {
      bestScore = score;
      best = opt;
    }
  }
  if (!best || bestScore < 72) return { parsed_value: null, confidence: 0.35, needs_clarification: true, clarification_question: "Necesito una respuesta más cercana a las opciones sugeridas.", reasoning_summary: "No hubo coincidencia clara con las opciones disponibles.", user_friendly_interpretation: "No encontré una opción clara en tu respuesta.", value_explanation: "", answer_category: "respuesta ambigua" };
  const parsedValue = best.value !== null && best.value !== undefined ? best.value : best.label;
  return { parsed_value: parsedValue, confidence: Math.min(0.92, bestScore / 100), needs_clarification: false, reasoning_summary: "Se detectó una coincidencia con una opción válida.", user_friendly_interpretation: `Lo entendí como: ${best.label}.`, value_explanation: "Guardaré esa opción para el modelo.", answer_category: bestScore >= 85 ? "respuesta clara" : "respuesta parcialmente clara" };
}

function interpretAnswer(feature, rawAnswer, meta, attempt = 1, maxAttempts = 3) {
  const raw = cleanText(rawAnswer || "");
  const norm = normalizeText(raw);
  const options = formatOptions(meta);
  const scaleType = options.scale_type;
  const result = {
    feature_name: feature,
    raw_answer: raw,
    parsed_value: null,
    expected_type: scaleType,
    confidence: 0,
    needs_clarification: true,
    clarification_question: "",
    reasoning_summary: "",
    user_friendly_interpretation: "",
    value_explanation: "",
    validation_error: "",
    answer_category: "respuesta ambigua",
    human_options_text: options.human_options_text,
    quick_chips: options.quick_chips,
    scale_type: scaleType,
  };
  if (!norm) {
    result.confidence = 0.2;
    result.needs_clarification = true;
    result.clarification_question = scaleType === "binary" ? "Para esta pregunta necesito confirmar si la respuesta es sí o no." : "Necesito una respuesta para continuar.";
    result.reasoning_summary = "La respuesta está vacía.";
    result.user_friendly_interpretation = "Necesito una respuesta para continuar.";
    result.answer_category = "respuesta insuficiente";
    return result;
  }
  let parsed;
  if (scaleType === "binary") parsed = parseBinary(norm, meta);
  else if (scaleType === "temporal_0_2") parsed = parseTemporal(norm);
  else if (scaleType === "observation_0_2") parsed = parseObservation(norm);
  else if (scaleType === "frequency_0_3") parsed = parseFrequency(norm);
  else if (scaleType === "impact_0_3") parsed = parseImpact(norm);
  else if (scaleType === "numeric_range") parsed = parseNumeric(norm, meta);
  else parsed = parseCategorical(norm, options.options_list);
  Object.assign(result, parsed);
  result.confidence = Number(Number(result.confidence || 0).toFixed(4));
  if (result.parsed_value !== null && result.parsed_value !== undefined && result.confidence < 0.65) {
    result.needs_clarification = true;
    if (result.answer_category === "respuesta clara") result.answer_category = "respuesta parcialmente clara";
  }
  if (result.needs_clarification && !result.clarification_question) result.clarification_question = "Necesito una respuesta un poco más clara para continuar.";
  if (result.needs_clarification && attempt >= maxAttempts) result.clarification_question = "No he podido interpretar esta respuesta con suficiente seguridad. Por favor responde de forma más directa usando una de las opciones sugeridas.";
  return result;
}

function buildQuestionPayload(meta, role, index, total, targetColumn) {
  const explanation = explainQuestion(meta, role);
  const options = formatOptions(meta);
  return {
    feature: String(meta.feature || ""),
    question: explanation.question,
    help_text: safeHumanText([meta.help_text, explanation.simple_explanation], explanation.simple_explanation),
    scale_guidance: safeHumanText([meta.scale_guidance], ""),
    response_options: options.options_list,
    response_type: meta.response_type || "",
    min_value: meta.min_value ?? null,
    max_value: meta.max_value ?? null,
    feature_label_human: safeHumanText([meta.feature_label_human], ""),
    term_explanation: safeHumanText([meta.term_explanation], ""),
    examples: explanation.examples,
    simple_explanation: explanation.simple_explanation,
    human_options_text: options.human_options_text,
    quick_chips: options.quick_chips,
    scale_type: options.scale_type,
    progress_index: index,
    progress_total: total,
    target_column: targetColumn,
    is_required: meta.is_required !== false,
  };
}

function buildCompatibilityLevel(probability, threshold) {
  const lowCut = Math.max(0.1, threshold * 0.75);
  const highCut = Math.min(0.9, threshold + 0.2);
  if (probability < lowCut) return "compatibilidad baja";
  if (probability < threshold) return "compatibilidad intermedia o zona de observación";
  if (probability < highCut) return "compatibilidad relevante";
  return "compatibilidad alta";
}

function humanValue(meta, value) {
  const options = formatOptions(meta).options_list;
  for (const option of options) if (String(option.value) === String(value)) return option.label;
  const scaleType = inferScaleType(meta, options);
  if (scaleType === "binary") return Number(value) === 1 ? "Sí" : "No";
  if (scaleType === "temporal_0_2") return Number(value) === 2 ? "Ocurrió recientemente" : Number(value) === 1 ? "Ocurrió antes" : "No ocurrió";
  if (scaleType === "frequency_0_3") return ({ 0: "Nunca", 1: "Ocasional", 2: "Frecuente", 3: "Casi siempre" })[Number(value)] || String(value);
  if (scaleType === "observation_0_2") return ({ 0: "No se observa", 1: "A veces", 2: "Claramente" })[Number(value)] || String(value);
  if (scaleType === "impact_0_3") return ({ 0: "Sin impacto", 1: "Leve", 2: "Moderado", 3: "Marcado" })[Number(value)] || String(value);
  return String(value);
}

function computeFallbackPrediction(answers) {
  ensureCacheLoaded();
  const threshold = Number(cache.metadata?.thresholds?.final ?? 0.41);
  const schemaFeatures = Array.isArray(cache.schema.features) ? cache.schema.features : [];
  const schemaMap = new Map(schemaFeatures.map((item) => [String(item.feature || ""), item]));
  const importanceRows = Array.isArray(cache.featureImportance.feature_importance_aggregated) ? cache.featureImportance.feature_importance_aggregated : [];
  const importanceMap = new Map();
  for (const row of importanceRows) {
    const feature = String(row.feature || "");
    const score = Number(row.importance || 0);
    if (feature && Number.isFinite(score) && score > 0) importanceMap.set(feature, score);
  }
  let weighted = 0;
  let totalWeight = 0;
  const indicators = [];
  for (const [feature, rawValue] of Object.entries(answers || {})) {
    const meta = schemaMap.get(feature) || { feature };
    const options = formatOptions(meta);
    const scaleType = options.scale_type;
    const weight = importanceMap.get(feature) ?? 0.03;
    let normalizedValue = 0;
    const num = Number(rawValue);
    if (Number.isFinite(num)) {
      if (scaleType === "binary") normalizedValue = num >= 1 ? 1 : 0;
      else if (["temporal_0_2", "observation_0_2"].includes(scaleType)) normalizedValue = Math.max(0, Math.min(1, num / 2));
      else if (["frequency_0_3", "impact_0_3"].includes(scaleType)) normalizedValue = Math.max(0, Math.min(1, num / 3));
      else if (scaleType === "numeric_range") {
        const minV = Number(meta.min_value);
        const maxV = Number(meta.max_value);
        if (Number.isFinite(minV) && Number.isFinite(maxV) && maxV > minV) normalizedValue = Math.max(0, Math.min(1, (num - minV) / (maxV - minV)));
      } else normalizedValue = Math.max(0, Math.min(1, num));
    }
    weighted += weight * normalizedValue;
    totalWeight += weight;
    if (normalizedValue > 0.2) indicators.push({ feature, importance: weight, value: rawValue, label: humanizeFeatureName(feature, meta), value_text: humanValue(meta, rawValue), score: weight * normalizedValue });
  }
  const rawScore = totalWeight > 0 ? weighted / totalWeight : 0;
  const probability = Math.max(0.02, Math.min(0.98, 0.08 + rawScore * 0.84));
  const internalClass = probability >= threshold ? 1 : 0;
  const compatibility = buildCompatibilityLevel(probability, threshold);
  indicators.sort((a, b) => b.score - a.score);
  const observedIndicators = indicators.slice(0, 6).map((item) => `${item.label}: ${item.value_text}`);
  const synthesisByLevel = {
    "compatibilidad baja": "La impresión orientativa sugiere baja compatibilidad con el dominio evaluado. En las respuestas registradas no predominan indicadores conductuales de alta intensidad.",
    "compatibilidad intermedia o zona de observación": "La impresión orientativa se ubica en una zona intermedia. Se observan algunas señales que conviene monitorear con seguimiento estructurado.",
    "compatibilidad relevante": "La impresión orientativa muestra compatibilidad relevante con el dominio evaluado. Existen indicadores consistentes que justifican una valoración profesional.",
    "compatibilidad alta": "La impresión orientativa muestra compatibilidad alta con el dominio evaluado. Las respuestas describen un patrón conductual de alta presencia e impacto potencial.",
  };
  const orientativeReport = {
    title: "Impresión psicológica orientativa",
    general_synthesis: synthesisByLevel[compatibility] || synthesisByLevel["compatibilidad intermedia o zona de observación"],
    compatibility_level: compatibility,
    observed_indicators: observedIndicators,
    functional_impact: "La información sugiere que el comportamiento podría afectar convivencia, normas o funcionamiento cotidiano. Es recomendable contrastar estos hallazgos con entrevistas y observación clínica.",
    professional_recommendation: "Se recomienda una valoración por psicología clínica o neuropsicología infantil, incluyendo entrevista con cuidadores y contexto escolar, para confirmar o descartar hipótesis diagnósticas.",
    important_clarification: "Esta interpretación es una estimación preliminar generada por un sistema de apoyo. No constituye un dictamen psicológico definitivo, no reemplaza una valoración clínica y no debe usarse como diagnóstico médico o psicológico preciso.",
    suggested_questions: ["¿Qué situaciones concretas activan estas conductas?", "¿Con qué frecuencia ocurren en casa y escuela?", "¿Qué estrategias de manejo se han intentado y con qué resultado?"],
    technical_summary: { probability_estimated: probability, threshold_used: threshold, compatibility_level: compatibility, internal_class: internalClass },
  };
  const metricsBlock = cache.metrics.test_metrics_threshold_final || {};
  const overfitWarning = cache.metadata.overfit_warning || ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"].some((k) => Number(metricsBlock[k] || 0) > 0.98)
    ? "Las métricas son muy altas. Esto puede indicar que el dataset contiene señales muy directas del target o que se requiere validación externa adicional."
    : null;
  return {
    probability_estimated: Number(probability.toFixed(6)),
    threshold_used: threshold,
    internal_class: internalClass,
    compatibility_text: internalClass === 1 ? "patrón compatible" : "patrón no compatible",
    risk_text: probability < 0.35 ? "riesgo estimado bajo" : probability < 0.65 ? "riesgo estimado intermedio" : "riesgo estimado alto",
    explanation: "La estimación se basa en respuestas interpretadas y confirmadas por el usuario. Este resultado expresa probabilidad estimada, no una conclusión clínica.",
    medical_disclaimer: MEDICAL_DISCLAIMER,
    orientative_report: orientativeReport,
    result_qa_chips: ["¿Qué significa este resultado?", "¿Por qué salió así?", "¿Qué debo hacer ahora?", "¿Esto es un diagnóstico?", "Ver indicadores principales", "Repetir encuesta"],
    overfit_warning: overfitWarning,
    metrics_snapshot: { f1: metricsBlock.f1, recall: metricsBlock.recall, precision: metricsBlock.precision, accuracy: metricsBlock.accuracy, roc_auc: metricsBlock.roc_auc, pr_auc: metricsBlock.pr_auc },
  };
}

function answerResultQuestion(question, report) {
  const q = normalizeText(question || "");
  const indicators = Array.isArray(report.observed_indicators) ? report.observed_indicators : [];
  const level = cleanText(report.compatibility_level || "compatibilidad no disponible");
  if (q.includes("que significa") || q.includes("compatibilidad")) return `Este resultado indica ${level}. Compatibilidad significa qué tan parecidas son las respuestas al patrón que el modelo aprendió para este dominio, pero no equivale a un diagnóstico clínico.`;
  if (q.includes("por que") || q.includes("porque") || q.includes("indicadores") || q.includes("variables")) return indicators.length ? `El resultado se apoyó principalmente en estos indicadores: ${indicators.slice(0, 5).join("; ")}. Estas variables son relevantes para el modelo, pero no implican causalidad directa.` : "El resultado se basó en el patrón global de respuestas confirmadas durante la evaluación.";
  if (q.includes("que debo hacer") || q.includes("que sigue") || q.includes("recomendacion") || q.includes("ahora que")) return `${cleanText(report.professional_recommendation || "")} Si el nivel de compatibilidad es relevante o alto, conviene priorizar una evaluación profesional pronta.`.trim();
  if (q.includes("diagnostico") || q.includes("dictamen")) return "No. Esta salida no es un diagnóstico. Es una impresión orientativa automatizada que sirve como apoyo y siempre debe ser revisada por un profesional calificado.";
  if (q.includes("ver indicadores") || q.includes("indicadores principales")) return indicators.length ? `Indicadores principales observados: ${indicators.slice(0, 6).join("; ")}` : "No hay indicadores destacados para esta sesión.";
  if (q.includes("threshold")) return `El threshold es el punto de corte del modelo. En esta evaluación fue ${report.technical_summary?.threshold_used}. Cuando la probabilidad supera ese valor, el sistema marca mayor compatibilidad.`;
  if (q.includes("recall") || q.includes("f1") || q.includes("precision") || q.includes("metric")) {
    const finalM = cache.metrics.test_metrics_threshold_final || {};
    return `El modelo se evalúa priorizando F1 y recall. Valores actuales: F1=${finalM.f1}, Recall=${finalM.recall}, Precision=${finalM.precision}.`;
  }
  if (q.includes("repetir") || q.includes("reiniciar")) return "Puedes usar el botón 'Iniciar nueva evaluación' para reiniciar la sesión desde cero.";
  return "Puedo ayudarte a explicar el nivel de compatibilidad, los indicadores que influyeron, las métricas del modelo o los pasos recomendados.";
}

function resolveMetricsBlock(metrics) {
  const candidates = [metrics.test_metrics_threshold_final, metrics.test_metrics, metrics.metrics_test, metrics];
  return candidates.find((block) => block && typeof block === "object" && ("f1" in block || "recall" in block || "precision" in block || "accuracy" in block)) || {};
}

function resolveConfusionMatrix(metrics, block) {
  if (block && block.confusion_matrix) return block.confusion_matrix;
  if (metrics.confusion_matrix) return metrics.confusion_matrix;
  if (metrics.test_metrics_threshold_final?.confusion_matrix) return metrics.test_metrics_threshold_final.confusion_matrix;
  if (metrics.test_metrics?.confusion_matrix) return metrics.test_metrics.confusion_matrix;
  if (metrics.metrics_test?.confusion_matrix) return metrics.metrics_test.confusion_matrix;
  return null;
}

function confusionPayload(matrix) {
  if (Array.isArray(matrix) && matrix.length >= 2 && Array.isArray(matrix[0]) && Array.isArray(matrix[1])) {
    const tn = Number(matrix[0][0] ?? 0);
    const fp = Number(matrix[0][1] ?? 0);
    const fn = Number(matrix[1][0] ?? 0);
    const tp = Number(matrix[1][1] ?? 0);
    return { tn, fp, fn, tp, matrix: [[tn, fp], [fn, tp]] };
  }
  return { tn: null, fp: null, fn: null, tp: null, matrix: null };
}

async function handleLocalApi(event) {
  ensureCacheLoaded();
  const method = String(event.httpMethod || "GET").toUpperCase();
  const route = `/${resolveSplatPath(event)}`.replace(/\/+$/, "") || "/";

  if (method === "GET" && route === "/model-status") {
    const modelTrained = Boolean(cache.metadata && Object.keys(cache.metadata).length);
    const targetColumn = cache.targetColumn || "target_domain_conduct_final";
    return json(200, {
      model_trained: modelTrained,
      target_column: targetColumn,
      target_label: TARGET_LABELS[targetColumn] || "Dominio evaluado",
      target_intro: TARGET_INTROS[targetColumn] || TARGET_INTROS.target_domain_conduct_final,
      threshold_final: cache.metadata?.thresholds?.final ?? 0.41,
      features_count: cache.metadata?.n_features_used ?? (cache.schema.features || []).length,
      model_variant_selected: cache.metadata?.model_variant_selected || "fallback_local_netlify",
      overfit_guard_applied: Boolean(cache.metadata?.overfit_guard_applied),
      overfit_warning: Boolean(cache.metadata?.overfit_warning),
      medical_disclaimer: MEDICAL_DISCLAIMER,
      message: modelTrained ? "Modelo listo para predicción." : "Modelo no entrenado. Ejecuta python train.py",
      mode: "netlify_local_fallback",
    });
  }

  if (method === "GET" && route === "/questions") {
    const role = String(getQueryParam(event, "role", "caregiver") || "caregiver").toLowerCase();
    const sessionId = String(getQueryParam(event, "session_id", "default") || "default");
    const targetColumn = cache.targetColumn || "target_domain_conduct_final";
    const selected = (Array.isArray(cache.schema.features) ? cache.schema.features : []).filter((meta) => featureBelongsToTarget(meta, targetColumn));
    if (!selected.length) return json(500, { detail: "No se encontraron preguntas para el target seleccionado. Revisa artifacts/feature_schema.json o ejecuta python train.py." });
    const questions = selected.map((meta, idx) => buildQuestionPayload(meta, role, idx + 1, selected.length, targetColumn));
    if (targetColumn === "target_domain_conduct_final" && questions.some((q) => String(q.feature || "").startsWith("adhd_"))) {
      return json(500, { detail: "Error de validación: se detectaron preguntas ADHD en target de conducta." });
    }
    getSession(sessionId).role = role;
    return json(200, {
      role,
      target_column: targetColumn,
      target_label: TARGET_LABELS[targetColumn] || "Dominio evaluado",
      intro_text: TARGET_INTROS[targetColumn] || TARGET_INTROS.target_domain_conduct_final,
      total: questions.length,
      questions,
      mode: "netlify_local_fallback",
    });
  }

  if (method === "POST" && route === "/chat/explain") {
    const payload = parseJsonBody(event);
    const feature = String(payload.feature || payload.feature_name || "").trim();
    if (!feature) return json(400, { detail: "Feature no especificada." });
    const meta = (cache.schema.features || []).find((item) => String(item.feature || "") === feature);
    if (!meta) return json(404, { detail: "No se encontró la pregunta solicitada." });
    return json(200, explainQuestion(meta, "caregiver"));
  }

  if (method === "POST" && route === "/chat/interpret") {
    const payload = parseJsonBody(event);
    const feature = String(payload.feature || payload.feature_name || "").trim();
    const answer = String(payload.answer || "");
    const sessionId = String(payload.session_id || "default");
    if (!feature) return json(400, { detail: "Feature no especificada para interpretar." });
    const schemaMeta = (cache.schema.features || []).find((item) => String(item.feature || "") === feature) || {};
    const mergedMeta = { ...schemaMeta, ...(payload.question_metadata || {}), feature };
    const session = getSession(sessionId);
    session.attempts_by_feature[feature] = (session.attempts_by_feature[feature] || 0) + 1;
    const attempt = session.attempts_by_feature[feature];
    if (isHelpRequest(answer)) {
      return json(200, { ok: true, interpreted: null, needs_explanation: true, explanation: explainQuestion(mergedMeta, payload.role || "caregiver"), message: "Te explico la pregunta en palabras más simples." });
    }
    const interpreted = interpretAnswer(feature, answer, mergedMeta, attempt, 3);
    interpreted.is_required = mergedMeta.is_required !== false;
    interpreted.attempt = attempt;
    interpreted.allow_missing_value = Boolean(interpreted.needs_clarification && attempt >= 3 && !interpreted.is_required);
    interpreted.max_attempts_reached = attempt >= 3;
    return json(200, { ok: true, interpreted, needs_explanation: false, mode: "netlify_local_fallback" });
  }

  if (method === "POST" && route === "/chat/confirm") {
    const payload = parseJsonBody(event);
    const feature = String(payload.feature || payload.feature_name || "").trim();
    const sessionId = String(payload.session_id || "default");
    if (!feature) return json(400, { detail: "Feature no especificada para confirmar." });
    const session = getSession(sessionId);
    const schemaMeta = (cache.schema.features || []).find((item) => String(item.feature || "") === feature) || {};
    const isRequired = schemaMeta.is_required !== false;
    const parsedValue = payload.parsed_value;
    if ((parsedValue === null || parsedValue === undefined) && isRequired) return json(400, { detail: "Este dato es importante para hacer la estimación. Necesito una respuesta más clara." });
    session.answers_confirmed[feature] = parsedValue;
    session.attempts_by_feature[feature] = 0;
    return json(200, { ok: true, confirmed_answers_count: Object.keys(session.answers_confirmed).length, mode: "netlify_local_fallback" });
  }

  if (method === "POST" && route === "/predict") {
    const payload = parseJsonBody(event);
    const sessionId = String(payload.session_id || "default");
    const session = getSession(sessionId);
    const answers = payload.answers && typeof payload.answers === "object" ? payload.answers : session.answers_confirmed;
    if (!answers || Object.keys(answers).length === 0) return json(400, { detail: "Aún no hay respuestas confirmadas." });
    const missingRequired = [];
    for (const item of cache.schema.features || []) {
      const feature = String(item.feature || "");
      if (!feature) continue;
      if (item.is_required === false) continue;
      if (!(feature in answers) || answers[feature] === null || answers[feature] === undefined) {
        missingRequired.push(safeHumanText([item.caregiver_question, item.question_text_primary, item.feature_label_human], "Pregunta obligatoria pendiente"));
      }
    }
    if (missingRequired.length) {
      return json(200, { ok: false, message: "Faltan algunos datos importantes para generar la impresión orientativa.", missing_required_questions: missingRequired.slice(0, 12), medical_disclaimer: MEDICAL_DISCLAIMER });
    }
    const prediction = computeFallbackPrediction(answers);
    session.latest_prediction = prediction;
    return json(200, { ok: true, prediction, mode: "netlify_local_fallback" });
  }

  if (method === "POST" && route === "/chat/result-question") {
    const payload = parseJsonBody(event);
    const session = getSession(String(payload.session_id || "default"));
    if (!session.latest_prediction) return json(400, { detail: "Aún no hay resultado final para explicar." });
    const report = session.latest_prediction.orientative_report || {};
    return json(200, { ok: true, answer: answerResultQuestion(payload.question || "", report), chips: session.latest_prediction.result_qa_chips || [], mode: "netlify_local_fallback" });
  }

  if (method === "POST" && route === "/reset-session") {
    const payload = parseJsonBody(event);
    sessions.set(String(payload.session_id || "default"), { role: "caregiver", answers_confirmed: {}, latest_prediction: null, attempts_by_feature: {} });
    return json(200, { ok: true, message: "Sesión reiniciada", mode: "netlify_local_fallback" });
  }

  if (method === "POST" && route === "/audit-event") return json(200, { ok: true, mode: "netlify_local_fallback" });

  if (method === "GET" && route === "/metrics") {
    const metrics = cache.metrics || {};
    if (!Object.keys(metrics).length) return json(200, { ok: false, message: "No existen métricas aún. Ejecuta python train.py" });
    const metricsBlock = resolveMetricsBlock(metrics);
    const cm = confusionPayload(resolveConfusionMatrix(metrics, metricsBlock));
    const overfitWarning = cache.metadata.overfit_warning || ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"].some((key) => Number(metricsBlock[key] || 0) > 0.98)
      ? "Las métricas son muy altas. Esto puede indicar que el dataset contiene señales muy directas del target o que se requiere validación externa adicional."
      : null;
    return json(200, {
      ok: true,
      target_column: metrics.target_column || cache.targetColumn,
      features_count: metrics.features_count,
      threshold_0_5: metrics.test_metrics_threshold_0_5 || {},
      threshold_final: metricsBlock,
      confusion_matrix: cm.matrix,
      confusion_matrix_detail: cm,
      overfit_warning: overfitWarning,
      overfit_guard_applied: Boolean(cache.metadata.overfit_guard_applied),
      model_variant_selected: cache.metadata.model_variant_selected || "fallback_local_netlify",
      metrics_above_limit: cache.metadata.metrics_above_limit || [],
      selected_model_reason: cache.metadata.selected_model_reason || "",
      leakage_audit_summary: cache.metadata.leakage_audit_summary || {},
      mode: "netlify_local_fallback",
    });
  }

  if (method === "GET" && route === "/feature-importance") {
    const rows = Array.isArray(cache.featureImportance.feature_importance_aggregated) ? cache.featureImportance.feature_importance_aggregated.slice(0, 20) : [];
    const schemaMap = new Map((cache.schema.features || []).map((item) => [String(item.feature || ""), item]));
    const normalizedRows = rows.map((row) => {
      const feature = String(row.feature || "");
      const meta = schemaMap.get(feature) || { feature };
      const label = humanizeFeatureName(feature, meta);
      const explanation = safeHumanText([meta.feature_description, meta.help_text, meta.term_explanation], `Esta variable resume señales observables relacionadas con: ${label}.`);
      return { feature, technical_name: feature, label, plain_explanation: explanation, importance: row.importance };
    });
    return json(200, { ok: true, feature_importance_aggregated: normalizedRows, note: "Estas variables fueron relevantes para el modelo, pero no significan causa directa.", mode: "netlify_local_fallback" });
  }

  return json(404, { ok: false, detail: `Endpoint no soportado en fallback local: ${method} /api${route}`, mode: "netlify_local_fallback" });
}

exports.handler = async (event) => {
  const headerBackend = normalizeBaseUrl((event.headers && (event.headers["x-backend-url"] || event.headers["X-Backend-Url"])) || "");
  const envBackend = normalizeBaseUrl(process.env.BACKEND_API_URL);
  const backendBase = envBackend || headerBackend;

  if (!backendBase) {
    try {
      return await handleLocalApi(event);
    } catch (error) {
      return json(500, { ok: false, detail: "Error en modo fallback local de Netlify.", error: String(error && error.message ? error.message : error) });
    }
  }

  const method = String(event.httpMethod || "GET").toUpperCase();
  const targetUrl = buildTargetUrl(event, backendBase);
  const headers = buildForwardHeaders(event.headers);
  const hasBody = !["GET", "HEAD"].includes(method);
  let body;
  if (hasBody && event.body) body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;

  try {
    const response = await fetch(targetUrl, { method, headers, body, redirect: "manual" });
    const responseText = await response.text();
    const contentType = response.headers.get("content-type") || "application/json; charset=utf-8";
    return {
      statusCode: response.status,
      headers: { "content-type": contentType, "cache-control": "no-store" },
      body: responseText,
    };
  } catch (error) {
    return json(502, { ok: false, detail: "No fue posible conectar con BACKEND_API_URL desde el proxy de Netlify.", error: String(error && error.message ? error.message : error), target_url: targetUrl });
  }
};
