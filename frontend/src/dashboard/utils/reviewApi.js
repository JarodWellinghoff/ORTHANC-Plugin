const apiBase = (import.meta.env?.VITE_API_URL ?? "").replace(/\/$/, "");

export async function requestReview(path = "", options = {}) {
  const response = await fetch(`${apiBase}/cho-result-reviews${path}`, {
    credentials: "include", cache: "no-store", ...options,
    headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers },
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    const error = new Error(`Invalid review response (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  if (!response.ok) {
    const error = new Error(payload?.error || `Review request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return payload;
}
