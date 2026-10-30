// One candidate, one in-flight decision. Failed requests leave it unsaved.
export function createReviewSession(id, request) {
  if (!id || typeof request !== "function") {
    throw new TypeError("A review ID and request function are required.");
  }
  let resolved = false;
  let pendingDecision = null;

  return {
    get resolved() { return resolved; },
    get busy() { return pendingDecision !== null; },
    async decide(decision, { keepalive = false } = {}) {
      if (!["overwrite", "discard"].includes(decision)) {
        throw new TypeError("Invalid review decision.");
      }
      if (resolved || pendingDecision) return false;
      pendingDecision = decision;
      try {
        try {
          await request(`/${encodeURIComponent(id)}`, {
            method: "POST",
            body: JSON.stringify({ decision }),
            keepalive,
          });
        } catch (error) {
          // Another tab may already have resolved it. A missing draft is safe
          // to leave, but must never be reported as a successful save.
          if (decision !== "discard" || error.status !== 404) throw error;
        }
        resolved = true;
        return true;
      } finally {
        pendingDecision = null;
      }
    },
  };
}
