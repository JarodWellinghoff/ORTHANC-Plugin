import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useSnackbar } from "notistack";
import { useDashboard } from "../context/DashboardContext";
import { useRerunReviewQueue } from "../context/RerunReviewContext";
import { requestReview } from "../utils/reviewApi";
import { reviewMatchesSeries, selectReviewId } from "../utils/choResultFlow";

export default function useSeriesReview() {
  const { choModal, actions } = useDashboard();
  const { pending, pollError, resolveReview } = useRerunReviewQueue();
  const { enqueueSnackbar } = useSnackbar();
  const [searchParams, setSearchParams] = useSearchParams();
  const pendingReview = pending.find((item) => reviewMatchesSeries(item, choModal)) ?? null;
  const requestedId = searchParams.get("review");
  const id = selectReviewId({ pending, series: choModal, requestedId });
  const key = JSON.stringify([choModal.seriesId, choModal.seriesUuid, id]);
  const [state, setState] = useState({ key: null, review: null, error: "" });
  const [reload, setReload] = useState(0);
  const [saving, setSaving] = useState(false);
  const lock = useRef(false);
  const mounted = useRef(false);
  const selection = useRef(key);
  selection.current = key;
  const matchingState = state.key === key ? state : null;
  const review = matchingState?.review ?? null;
  const error = matchingState?.error ?? "";
  const loading = Boolean(id && !review && !error);

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

  const decide = useCallback(async (decision) => {
    if (!review || lock.current || !["overwrite", "discard"].includes(decision)) return;
    lock.current = true;
    setSaving(true);
    setState({ key, review, error: "" });
    try {
      await requestReview(`/${encodeURIComponent(review.id)}`, {
        method: "POST", body: JSON.stringify({ decision }),
      });
    } catch (failure) {
      // Keep the candidate available on conflicts, network errors, and save failures.
      if (mounted.current && selection.current === key) setState({ key, review, error: failure.message });
      lock.current = false;
      if (mounted.current) setSaving(false);
      return;
    }

    resolveReview(review.id);
    window.dispatchEvent(new CustomEvent("ctpro-review-resolved", {
      detail: { seriesUuid: review.series_uuid, decision },
    }));
    enqueueSnackbar(decision === "overwrite" ? "Reviewed results saved." : "New results discarded. Saved results were kept.", { variant: "success" });
    try {
      // A POST continues if the user leaves; its completion must not reset another exam.
      if (mounted.current && selection.current === key) {
        setState({ key: null, review: null, error: "" });
        setSearchParams((previous) => {
          const next = new URLSearchParams(previous);
          if (next.get("review") === review.id) next.delete("review");
          return next;
        }, { replace: true });
        const refreshed = await actions.reloadStoredChoResults({ clearResults: true });
        if (refreshed === false && mounted.current) {
          enqueueSnackbar("Decision applied. Reload the saved results to refresh this page.", { variant: "warning" });
        }
      }
    } finally {
      lock.current = false;
      if (mounted.current) setSaving(false);
    }
  }, [review, key, resolveReview, enqueueSnackbar, setSearchParams, actions]);

  return {
    review, pendingReview, loading, error, saving, pollError,
    requested: Boolean(id), openReview, decide,
    retry: () => setReload((value) => value + 1),
  };
}
