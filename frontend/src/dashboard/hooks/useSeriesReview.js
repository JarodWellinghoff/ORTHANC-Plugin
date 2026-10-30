import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useSnackbar } from "notistack";
import { useDashboard } from "../context/DashboardContext";
import { useRerunReviewQueue } from "../context/RerunReviewContext";
import { requestReview } from "../utils/reviewApi";
import { reviewMatchesSeries, selectReviewId } from "../utils/choResultFlow";
import { createReviewSession } from "../utils/reviewSession";
import useUnsavedReviewGuard from "./useUnsavedReviewGuard";

export default function useSeriesReview() {
  const { choModal, actions } = useDashboard();
  const { pending, pollError, resolveReview } = useRerunReviewQueue();
  const { enqueueSnackbar } = useSnackbar();
  const [searchParams, setSearchParams] = useSearchParams();
  const resolvedIds = useRef(new Set());
  const pendingReview = pending.find((item) =>
    !resolvedIds.current.has(item.id) && reviewMatchesSeries(item, choModal)) ?? null;
  const requestedId = searchParams.get("review");
  const selectedId = selectReviewId({ pending, series: choModal, requestedId });
  const id = resolvedIds.current.has(selectedId) ? null : selectedId;
  const key = JSON.stringify([choModal.seriesId, choModal.seriesUuid, id]);
  const [state, setState] = useState({ key: null, review: null, error: "" });
  const [reload, setReload] = useState(0);
  const [operation, setOperation] = useState(null);
  const mounted = useRef(false);
  const selection = useRef(key);
  selection.current = key;
  const matchingState = state.key === key ? state : null;
  const review = matchingState?.review ?? null;
  const error = matchingState?.error ?? "";
  const loading = Boolean(id && !review && !error);
  const reviewId = review?.id;
  const session = useMemo(() => reviewId
    ? createReviewSession(reviewId, requestReview) : null, [reviewId]);
  const busy = operation !== null;
  const saving = operation === "overwrite";

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    if (!id || !choModal.seriesId) return undefined;
    const controller = new AbortController();
    const load = async () => {
      setState({ key, review: null, error: "" });
      try {
        const data = await requestReview(`/${encodeURIComponent(id)}`, { signal: controller.signal });
        if (data?.id !== id || data.state !== "ready" || !data.candidate?.results || !data.baseline) {
          throw new Error("The rerun is not ready for review.");
        }
        if (!reviewMatchesSeries(data, { seriesId: choModal.seriesId, seriesUuid: choModal.seriesUuid })) {
          throw new Error("This review does not belong to the selected series.");
        }
        if (!controller.signal.aborted) setState({ key, review: data, error: "" });
      } catch (failure) {
        if (!controller.signal.aborted) setState({ key, review: null, error: failure.message });
      }
    };
    load();
    return () => controller.abort();
  }, [id, key, choModal.seriesId, choModal.seriesUuid, reload]);

  const openReview = useCallback(() => {
    if (!pendingReview) return;
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      next.set("review", pendingReview.id);
      return next;
    });
  }, [pendingReview, setSearchParams]);

  const applyDecision = useCallback(async (decision, { keepalive = false } = {}) => {
    if (!review || !session || session.busy || session.resolved) return false;
    setOperation(decision);
    try {
      const applied = await session.decide(decision, {
        // Only the small decision payload is sent, never imaging data.
        keepalive: keepalive || decision === "overwrite",
      });
      if (!applied) return false;
      // Mark clean synchronously, before URL cleanup can invoke the blocker.
      resolvedIds.current.add(review.id);
      resolveReview(review.id);
      window.dispatchEvent(new CustomEvent("ctpro-review-resolved", {
        detail: { seriesUuid: review.series_uuid, decision },
      }));
      if (decision === "overwrite") {
        enqueueSnackbar("Reviewed results saved.", { variant: "success" });
      }
      // A completed request must not reset a different exam.
      if (mounted.current && selection.current === key) {
        setState({ key: null, review: null, error: "" });
        if (decision === "overwrite" || keepalive) {
          // An in-app departure uses blocker.proceed() instead; do not replace
          // its intended destination with our own query-string navigation.
          setSearchParams((previous) => {
            const next = new URLSearchParams(previous);
            if (next.get("review") === review.id) next.delete("review");
            return next;
          }, { replace: true });
        }
        // Clear foreground/review flags for both decisions, including leaving
        // a review by removing its query parameter without unmounting the page.
        try {
          const refreshed = await actions.reloadStoredChoResults({ clearResults: true });
          if (refreshed === false) throw new Error("Saved results could not be refreshed.");
        } catch {
          if (mounted.current && decision === "overwrite") {
            enqueueSnackbar("Results were saved, but the display could not refresh. Reload the page.", { variant: "warning" });
          }
        }
      }
      return true;
    } finally {
      if (mounted.current) setOperation(null);
    }
  }, [review, session, key, resolveReview, enqueueSnackbar, setSearchParams, actions]);

  const save = useCallback(async () => {
    if (!review || session?.busy || session?.resolved) return;
    setState({ key, review, error: "" });
    try {
      await applyDecision("overwrite");
    } catch (failure) {
      // No reload/reset on failure: the same candidate remains unsaved and
      // guarded, and the user can press Save results again.
      if (mounted.current && selection.current === key) {
        setState({ key, review, error: failure.message || "Unable to save results. Please try again." });
      }
    }
  }, [review, session, key, applyDecision]);

  const discardOnLeave = useCallback((options) => applyDecision("discard", options), [applyDecision]);
  const navigation = useUnsavedReviewGuard({ session, busy, discard: discardOnLeave });

  return {
    review, pendingReview, loading, error, saving, busy, pollError, navigation,
    requested: Boolean(id), openReview, save,
    retry: () => { if (!review && !busy) setReload((value) => value + 1); },
  };
}
