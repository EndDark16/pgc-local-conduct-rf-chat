"use strict";

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
  if (rawPath.startsWith(prefix)) {
    return rawPath.slice(prefix.length);
  }

  const match = rawPath.match(/\/api\/(.*)$/);
  if (match && match[1]) return match[1];
  return "";
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

exports.handler = async (event) => {
  const headerBackend = normalizeBaseUrl(
    (event.headers && (event.headers["x-backend-url"] || event.headers["X-Backend-Url"])) || ""
  );
  const envBackend = normalizeBaseUrl(process.env.BACKEND_API_URL);
  const backendBase = envBackend || headerBackend;
  if (!backendBase) {
    return json(503, {
      ok: false,
      detail:
        "Backend no configurado. Define BACKEND_API_URL en Netlify o envía x-backend-url con la URL pública del backend FastAPI (sin slash final).",
      example: "https://tu-backend-ejemplo.onrender.com",
    });
  }

  const method = String(event.httpMethod || "GET").toUpperCase();
  const targetUrl = buildTargetUrl(event, backendBase);
  const headers = buildForwardHeaders(event.headers);

  const hasBody = !["GET", "HEAD"].includes(method);
  let body;
  if (hasBody && event.body) {
    body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;
  }

  try {
    const response = await fetch(targetUrl, {
      method,
      headers,
      body,
      redirect: "manual",
    });

    const responseText = await response.text();
    const contentType = response.headers.get("content-type") || "application/json; charset=utf-8";

    return {
      statusCode: response.status,
      headers: {
        "content-type": contentType,
        "cache-control": "no-store",
      },
      body: responseText,
    };
  } catch (error) {
    return json(502, {
      ok: false,
      detail: "No fue posible conectar con BACKEND_API_URL desde el proxy de Netlify.",
      error: String(error && error.message ? error.message : error),
      target_url: targetUrl,
    });
  }
};
