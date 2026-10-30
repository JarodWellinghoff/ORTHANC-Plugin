import test from "node:test";
import assert from "node:assert/strict";
import { createReviewSession } from "../src/dashboard/utils/reviewSession.js";
import { installReviewUnloadGuard, leavesReviewPage } from "../src/dashboard/utils/reviewNavigation.js";

const location = (pathname = "/results/series-1", search = "?review=draft-1", hash = "") => ({ pathname, search, hash });
const transition = (next, current = location()) => ({ currentLocation: current, nextLocation: next });
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};

class FakeWindow {
  listeners = new Map();
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  removeEventListener(type, callback) {
    if (this.listeners.get(type) === callback) this.listeners.delete(type);
  }
  emit(type) {
    const event = { defaultPrevented: false, returnValue: undefined,
      preventDefault() { this.defaultPrevented = true; } };
    this.listeners.get(type)?.(event);
    return event;
  }
}

test("route changes, series changes, and review exit/replacement need confirmation", () => {
  assert.equal(leavesReviewPage(transition(location("/results", ""))), true);
  assert.equal(leavesReviewPage(transition(location("/plotting", ""))), true);
  assert.equal(leavesReviewPage(transition(location("/results/series-2"))), true);
  assert.equal(leavesReviewPage(transition(location(undefined, ""))), true);
  assert.equal(leavesReviewPage(transition(location(undefined, "?review=draft-2"))), true);
});

test("same-page hash links and unrelated query changes do not discard a review", () => {
  assert.equal(leavesReviewPage(transition(location(undefined, undefined, "#analysis-results"))), false);
  assert.equal(leavesReviewPage(transition(location(undefined, "?tab=plots&review=draft-1"))), false);
  assert.equal(leavesReviewPage(transition(location("/results/series-1/"))), false);
});

test("creating a review session neither saves nor discards", () => {
  const session = createReviewSession("draft-1", () => { assert.fail("Unexpected request"); });
  assert.equal(session.resolved, false);
  assert.equal(session.busy, false);
});

test("save sends overwrite explicitly and only clears dirty state after acknowledgement", async () => {
  const response = deferred();
  const calls = [];
  const session = createReviewSession("draft/1", (...args) => { calls.push(args); return response.promise; });
  const saving = session.decide("overwrite", { keepalive: true });
  assert.equal(session.busy, true);
  assert.equal(session.resolved, false);
  assert.deepEqual(calls, [["/draft%2F1", { method: "POST", body: '{"decision":"overwrite"}', keepalive: true }]]);
  response.resolve({});
  assert.equal(await saving, true);
  assert.equal(session.resolved, true);
  assert.equal(session.busy, false);
  assert.equal(await session.decide("overwrite"), false);
  assert.equal(calls.length, 1);
});

test("double-clicks and leaving during save cannot send competing decisions", async () => {
  const response = deferred();
  let calls = 0;
  const session = createReviewSession("draft-1", () => { calls++; return response.promise; });
  const saving = session.decide("overwrite");
  assert.equal(await session.decide("overwrite"), false);
  assert.equal(await session.decide("discard"), false);
  assert.equal(calls, 1);
  response.resolve({});
  await saving;
});

test("failed save keeps results unsaved and permits retry", async () => {
  let attempts = 0;
  const session = createReviewSession("draft-1", async () => {
    if (++attempts === 1) throw new Error("Network unavailable");
    return {};
  });
  await assert.rejects(session.decide("overwrite"), /Network unavailable/);
  assert.equal(session.resolved, false);
  assert.equal(session.busy, false);
  assert.equal(await session.decide("overwrite"), true);
});

test("failed discard must not authorize navigation; retry can complete it", async () => {
  let attempts = 0;
  const session = createReviewSession("draft-1", async (_path, options) => {
    assert.equal(JSON.parse(options.body).decision, "discard");
    if (++attempts === 1) throw Object.assign(new Error("Busy"), { status: 409 });
    return {};
  });
  await assert.rejects(session.decide("discard"), /Busy/);
  assert.equal(session.resolved, false);
  assert.equal(await session.decide("discard"), true);
  assert.equal(session.resolved, true);
});

test("a missing draft can be left but is never reported as a successful save", async () => {
  const missing = async () => { throw Object.assign(new Error("Missing"), { status: 404 }); };
  const save = createReviewSession("draft-1", missing);
  await assert.rejects(save.decide("overwrite"), /Missing/);
  assert.equal(save.resolved, false);
  const discard = createReviewSession("draft-1", missing);
  assert.equal(await discard.decide("discard"), true);
});

test("invalid decisions cannot reach the API", async () => {
  const session = createReviewSession("draft-1", () => { assert.fail("Unexpected request"); });
  await assert.rejects(session.decide("save"), TypeError);
  assert.equal(session.resolved, false);
});

test("browser confirmation does not discard when the user cancels", () => {
  const target = new FakeWindow();
  let discarded = 0;
  const cleanup = installReviewUnloadGuard(target, {
    isDirty: () => true, isBusy: () => false,
    discard: () => { discarded++; }, onError: assert.fail,
  });
  const event = target.emit("beforeunload");
  assert.equal(event.defaultPrevented, true);
  assert.equal(event.returnValue, true);
  assert.equal(discarded, 0);
  // StrictMode/unmount cleanup must also never discard a review.
  cleanup();
  assert.equal(discarded, 0);
  assert.equal(target.listeners.size, 0);
});

test("confirmed document departure dispatches a keepalive discard", () => {
  const target = new FakeWindow();
  const calls = [];
  installReviewUnloadGuard(target, {
    isDirty: () => true, isBusy: () => false,
    discard: (options) => { calls.push(options); return Promise.resolve(true); }, onError: assert.fail,
  });
  target.emit("visibilitychange");
  assert.equal(calls.length, 0);
  target.emit("pagehide");
  assert.deepEqual(calls, [{ keepalive: true }]);
});

test("pagehide never races an active save or re-discards resolved results", () => {
  for (const [dirty, busy] of [[true, true], [false, false]]) {
    const target = new FakeWindow();
    installReviewUnloadGuard(target, {
      isDirty: () => dirty, isBusy: () => busy,
      discard: () => { assert.fail("Unexpected discard"); }, onError: assert.fail,
    });
    assert.equal(target.emit("beforeunload").defaultPrevented, dirty);
    target.emit("pagehide");
  }
});

test("unload request failures are handled without claiming the draft was removed", async () => {
  const target = new FakeWindow();
  const errors = [];
  const session = createReviewSession("draft-1", async () => { throw new Error("Offline"); });
  installReviewUnloadGuard(target, {
    isDirty: () => !session.resolved, isBusy: () => session.busy,
    discard: (options) => session.decide("discard", options), onError: (error) => errors.push(error),
  });
  target.emit("pagehide");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(errors[0].message, "Offline");
  assert.equal(session.resolved, false);
});
