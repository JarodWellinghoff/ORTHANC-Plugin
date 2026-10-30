import test from "node:test";
import assert from "node:assert/strict";
import { applyChoProgress, backgroundChoRun, candidatePlotData, reviewMatchesSeries,
  reviewResultsUrl, selectReviewId, shouldNotifyReview } from "../src/dashboard/utils/choResultFlow.js";

const saved = { average_noise_level: 8 };
const candidate = { results: { average_noise_level: 9 }, series: { series_instance_uid: "1.2.3" } };
const series = { open: true, seriesId: "1.2.3", seriesUuid: "uuid-1", stage: "running",
  runMode: "foreground", results: null, runId: "run-new", previousRunId: "run-old",
  storedResults: { data: saved }, progress: { value: 25 } };
const review = { id: "review-1", series_uid: "1.2.3", series_uuid: "uuid-1" };
const completion = { series_id: "uuid-1", eventType: "completed", results: candidate,
  metadata: { run_id: "run-new", review_required: false } };

test("foreground first analysis renders its completed result in the page", () => {
  const next = applyChoProgress(series, completion);
  assert.equal(next.stage, "results");
  assert.equal(next.results, candidate);
  assert.equal(next.storedResults.data, saved);
});
test("foreground rerun waits for the staged review, never treats the SSE payload as saved", () => {
  const next = applyChoProgress(series, { ...completion, metadata: { run_id: "run-new", review_required: true } });
  assert.equal(next.results, null);
  assert.equal(next.reviewRequired, true);
  assert.equal(next.storedResults.data, saved);
});
test("background first completion does not replace the displayed saved data", () => {
  const next = applyChoProgress({ ...series, runMode: "background" }, completion);
  assert.equal(next.stage, "config");
  assert.equal(next.results, null);
  assert.equal(next.storedResults.data, saved);
});
test("background rerun keeps accepted data and is eligible for a review toast", () => {
  const next = applyChoProgress({ ...series, runMode: "background" }, {
    ...completion, metadata: { run_id: "run-new", review_required: true },
  });
  assert.equal(next.results, null);
  assert.equal(next.storedResults.data, saved);
  assert.equal(shouldNotifyReview(review, next, null), true);
});
test("external or batch work cannot update an open series with no local run", () => {
  const previous = { ...series, runMode: null };
  assert.equal(applyChoProgress(previous, completion), previous);
});
test("completion for another series is ignored", () => {
  assert.equal(applyChoProgress(series, { ...completion, series_id: "uuid-2" }), series);
});
test("closed analysis is not reopened by completion", () => {
  const previous = { ...series, open: false };
  assert.equal(applyChoProgress(previous, completion), previous);
});
test("previous run completion cannot complete a newly started run", () => {
  assert.equal(applyChoProgress(series, { ...completion, metadata: { run_id: "run-old" } }), series);
});
test("a different run ID cannot replace the active run", () => {
  assert.equal(applyChoProgress(series, { ...completion, metadata: { run_id: "other" } }), series);
});
test("duplicate completion after review does not resurrect discarded results", () => {
  const previous = { ...series, stage: "config", runMode: null };
  assert.equal(applyChoProgress(previous, completion), previous);
});
test("cleanup events do not restart progress", () => {
  assert.equal(applyChoProgress(series, { ...completion, eventType: "cleanup" }), series);
});
test("SSE reconnect snapshots obey foreground/background policy", () => {
  const snapshot = { ...completion, eventType: undefined, status: "completed" };
  assert.equal(applyChoProgress(series, snapshot).results, candidate);
  assert.equal(applyChoProgress({ ...series, runMode: "background" }, snapshot).results, null);
});
test("errors remain visible without displaying results", () => {
  const next = applyChoProgress(series, { ...completion, eventType: "failed", error: "Failed" });
  assert.equal(next.stage, "config");
  assert.equal(next.runMode, null);
  assert.equal(next.pollError, "Failed");
  assert.equal(next.results, null);
});
test("progress binds a fresh server run ID", () => {
  const next = applyChoProgress({ ...series, runId: null }, {
    series_id: "uuid-1", eventType: "start", progress: 5,
    metadata: { run_id: "newest", review_required: true },
  });
  assert.equal(next.runId, "newest");
  assert.equal(next.reviewRequired, true);
  assert.equal(next.progress.value, 5);
});
test("switching to background clears results without cancelling progress", () => {
  const next = backgroundChoRun({ ...series, results: candidate });
  assert.equal(next.runMode, "background");
  assert.equal(next.stage, "running");
  assert.equal(next.results, null);
  assert.equal(next.progress, series.progress);
});
test("background click racing completion still hides the results", () => {
  const next = backgroundChoRun({ ...series, stage: "results", results: candidate });
  assert.equal(next.stage, "config");
  assert.equal(next.results, null);
});
test("foreground reruns select the matching review, not the first unrelated one", () => {
  assert.equal(selectReviewId({ pending: [{ ...review, id: "other", series_uid: "2" }, review], series }), review.id);
});
test("background reruns require an explicit request to show the candidate", () => {
  assert.equal(selectReviewId({ pending: [review], series: { ...series, runMode: "background" } }), null);
  assert.equal(selectReviewId({ pending: [review], series: { ...series, runMode: "background" }, requestedId: review.id }), review.id);
});
test("empty review list is silent, including first-time saves", () => {
  assert.equal(selectReviewId({ pending: [], series }), null);
  assert.equal([].filter((item) => shouldNotifyReview(item, series, null)).length, 0);
});
test("foreground and explicitly requested reviews do not generate redundant notifications", () => {
  assert.equal(shouldNotifyReview(review, series, null), false);
  assert.equal(shouldNotifyReview(review, { ...series, runMode: null }, review.id), false);
  assert.equal(shouldNotifyReview(review, { ...series, open: false }, null), true);
});
test("UID mismatches and missing identifiers never match by accident", () => {
  assert.equal(reviewMatchesSeries(review, { seriesId: "other", seriesUuid: "uuid-1" }), false);
  assert.equal(reviewMatchesSeries({}, {}), false);
  assert.equal(reviewMatchesSeries(review, { seriesUuid: "uuid-1" }), true);
});
test("review links contain both encoded series and review IDs", () => {
  assert.equal(reviewResultsUrl({ id: "a&b", series_uid: "1/2" }), "/results/1%2F2?review=a%26b#analysis-results");
  assert.throws(() => reviewResultsUrl({ id: "missing-series" }));
});
test("candidate plots cannot load the accepted MinIO image, without mutating metadata", () => {
  const plot = candidatePlotData(candidate);
  assert.equal(plot.series.series_instance_uid, undefined);
  assert.equal(candidate.series.series_instance_uid, "1.2.3");
  assert.equal(plot.results, candidate.results);
});
