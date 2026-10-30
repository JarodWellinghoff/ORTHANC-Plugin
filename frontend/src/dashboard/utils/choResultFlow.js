// Presentation policy only. The server remains authoritative for saving/review.
const present = (value) => value !== null && value !== undefined && value !== "";
const same = (left, right) => present(left) && present(right) && String(left) === String(right);

export const reviewMatchesSeries = (review, series) => {
  if (!review || !series) return false;
  // Prefer the DICOM UID when both sides have one; do not match conflicting UIDs.
  if (present(review.series_uid) && present(series.seriesId)) {
    return same(review.series_uid, series.seriesId);
  }
  return same(review.series_uuid, series.seriesUuid);
};

export const reviewResultsUrl = (review) => {
  if (!present(review?.series_uid) || !present(review?.id)) {
    throw new Error("The review has no series UID or review ID.");
  }
  return `/results/${encodeURIComponent(review.series_uid)}?review=${encodeURIComponent(review.id)}#analysis-results`;
};

export const selectReviewId = ({ pending, series, requestedId }) => {
  if (requestedId) return requestedId;
  if (series?.runMode !== "foreground") return null;
  return pending.find((review) => reviewMatchesSeries(review, series))?.id ?? null;
};

export const shouldNotifyReview = (review, series, requestedId) => {
  if (!series?.open || !reviewMatchesSeries(review, series)) return true;
  return series.runMode !== "foreground" && requestedId !== review.id;
};

export const candidatePlotData = (candidate) => candidate ? ({
  ...candidate,
  // The accepted MinIO image must not be represented as a candidate image.
  series_instance_uid: undefined,
  series: { ...candidate.series, series_instance_uid: undefined },
}) : null;

export const applyChoProgress = (previous, entry) => {
  if (!previous.open || !previous.runMode || previous.stage !== "running" || !entry) {
    return previous;
  }
  const id = entry.series_id ?? entry.series_uuid ?? entry.seriesId;
  if (!same(id, previous.seriesUuid) && !same(id, previous.seriesId)) return previous;
  const event = entry.eventType ?? (
    entry.status === "completed" ? "completed" :
    ["failed", "error", "cancelled"].includes(entry.status) ? "failed" : "progress"
  );
  if (!["start", "progress", "completed", "failed"].includes(event)) return previous;

  const runId = entry.metadata?.run_id;
  if (runId && (runId === previous.previousRunId || (previous.runId && runId !== previous.runId))) {
    return previous;
  }
  const reviewRequired = entry.metadata?.review_required ?? previous.reviewRequired ?? false;
  const common = { ...previous, runId: runId ?? previous.runId, reviewRequired };
  if (event === "completed") {
    return {
      ...common,
      stage: previous.runMode === "foreground" ? "results" : "config",
      // Reruns are read from the review endpoint only, after staging completes.
      // Never inject a background completion's payload into the visible page.
      results: previous.runMode === "foreground" && !reviewRequired ? entry.results ?? null : null,
      progress: { value: 100, message: entry.message ?? "Calculation completed", stage: "completed" },
      pollError: null,
    };
  }
  if (event === "failed") {
    return {
      ...common, stage: "config", runMode: null, results: null,
      pollError: entry.error ?? entry.message ?? "Calculation failed",
      progress: { ...previous.progress, message: entry.message ?? "Calculation failed", stage: "error" },
    };
  }
  return {
    ...common,
    progress: {
      value: typeof entry.progress === "number" ? entry.progress : previous.progress?.value ?? 0,
      message: entry.message ?? previous.progress?.message ?? "Processing...",
      stage: entry.current_stage ?? previous.progress?.stage ?? "initialization",
    },
  };
};

export const backgroundChoRun = (previous) => previous.runMode === "foreground" ? ({
  ...previous, runMode: "background", results: null,
  stage: previous.stage === "running" ? "running" : "config",
}) : previous;
