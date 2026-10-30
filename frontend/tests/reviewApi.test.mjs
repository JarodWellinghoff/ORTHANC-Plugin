import test from "node:test";
import assert from "node:assert/strict";
import { requestReview } from "../src/dashboard/utils/reviewApi.js";

test("review requests include credentials, avoid cache, and encode decisions as JSON", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/cho-result-reviews/review-1");
    assert.equal(options.credentials, "include");
    assert.equal(options.cache, "no-store");
    assert.equal(options.headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(options.body), { decision: "discard" });
    return { ok: true, status: 200, json: async () => ({ decision: "discard" }) };
  });
  assert.deepEqual(await requestReview("/review-1", { method: "POST", body: JSON.stringify({ decision: "discard" }) }), { decision: "discard" });
});
test("save conflicts retain the server's status and message", async (t) => {
  t.mock.method(globalThis, "fetch", async () => ({ ok: false, status: 409, json: async () => ({ error: "Saved results changed" }) }));
  await assert.rejects(requestReview("/review-1"), { status: 409, message: "Saved results changed" });
});
test("malformed server responses fail rather than resolving a review", async (t) => {
  t.mock.method(globalThis, "fetch", async () => ({ ok: true, status: 200, json: async () => { throw new Error("bad json"); } }));
  await assert.rejects(requestReview("/review-1"), { message: "Invalid review response (200)." });
});
test("GET polling passes its abort signal through", async (t) => {
  const controller = new AbortController();
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/cho-result-reviews");
    assert.equal(options.signal, controller.signal);
    assert.equal(options.headers["Content-Type"], undefined);
    return { ok: true, status: 200, json: async () => ({ reviews: [] }) };
  });
  assert.deepEqual(await requestReview("", { signal: controller.signal }), { reviews: [] });
});
