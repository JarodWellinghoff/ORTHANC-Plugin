import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Stack } from "@mui/material";
import { Link, useLocation } from "react-router-dom";
import { useSnackbar } from "notistack";
import { useDashboard } from "../context/DashboardContext";
import { RerunReviewContext } from "../context/RerunReviewContext";
import { requestReview } from "../utils/reviewApi";
import { reviewResultsUrl, shouldNotifyReview } from "../utils/choResultFlow";

// A queue and notification provider, not a modal or automatic navigation host.
export default function RerunReviewHost({ children }) {
  const { choModal } = useDashboard();
  const location = useLocation();
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const [pending, setPending] = useState([]);
  const [pollError, setPollError] = useState("");
  const resolved = useRef(new Set());
  const notified = useRef(new Map());
  const errorNotified = useRef(false);
  const current = useRef(null);
  current.current = { choModal, search: location.search, enqueueSnackbar, closeSnackbar };

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    const poll = async () => {
      try {
        const payload = await requestReview("", { signal: controller.signal });
        if (controller.signal.aborted) return;
        if (!Array.isArray(payload?.reviews)) throw new Error("Invalid pending-review response.");
        const reviews = payload.reviews.filter((item) => !resolved.current.has(item.id));
        setPending(reviews);
        setPollError("");
        const active = new Set(reviews.map((review) => review.id));
        const ui = current.current;
        if (errorNotified.current) {
          ui.closeSnackbar("ctpro-review-unavailable");
          errorNotified.current = false;
        }
        for (const [id, key] of notified.current) {
          if (!active.has(id)) {
            ui.closeSnackbar(key);
            notified.current.delete(id);
          }
        }
        const requestedId = new URLSearchParams(ui.search).get("review");
        for (const review of reviews) {
          if (notified.current.has(review.id) || !shouldNotifyReview(review, ui.choModal, requestedId)) continue;
          const href = reviewResultsUrl(review);
          const key = `ctpro-review-${review.id}`;
          notified.current.set(review.id, key);
          ui.enqueueSnackbar("New results are ready. Review them before replacing the saved results.", {
            key, variant: "info", persist: true, preventDuplicate: true,
            action: (snackbarKey) => (
              <Stack direction='row' spacing={1}>
                <Button color='inherit' component={Link} to={href}
                  onClick={() => ui.closeSnackbar(snackbarKey)}>Review results</Button>
                <Button color='inherit' aria-label='Dismiss review notification'
                  onClick={() => ui.closeSnackbar(snackbarKey)}>Dismiss</Button>
              </Stack>
            ),
          });
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setPollError(error.message);
          if (!errorNotified.current) {
            errorNotified.current = true;
            current.current.enqueueSnackbar("Result review is temporarily unavailable. Retrying automatically.", {
              key: "ctpro-review-unavailable", variant: "warning", preventDuplicate: true,
            });
          }
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 4000);
      }
    };
    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
      for (const key of notified.current.values()) current.current.closeSnackbar(key);
      notified.current.clear();
      current.current.closeSnackbar("ctpro-review-unavailable");
    };
  }, []);

  const resolveReview = useCallback((id) => {
    // A poll started before the decision must not resurrect the resolved review.
    resolved.current.add(id);
    setPending((reviews) => reviews.filter((review) => review.id !== id));
    const key = notified.current.get(id);
    if (key) current.current.closeSnackbar(key);
    notified.current.delete(id);
  }, []);

  const value = useMemo(() => ({ pending, pollError, resolveReview }), [pending, pollError, resolveReview]);
  return <RerunReviewContext.Provider value={value}>{children}</RerunReviewContext.Provider>;
}
