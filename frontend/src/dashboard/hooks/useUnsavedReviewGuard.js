import { useCallback, useEffect, useRef, useState } from "react";
import { useBlocker } from "react-router-dom";
import { installReviewUnloadGuard, leavesReviewPage } from "../utils/reviewNavigation";

export default function useUnsavedReviewGuard({ session, busy, discard }) {
  const current = useRef({ session, discard });
  current.current = { session, discard };
  const leaveLock = useRef(false);
  const mounted = useRef(false);
  const [leaving, setLeaving] = useState(false);
  const [error, setError] = useState("");
  const dirty = Boolean(session && !session.resolved);
  const blocker = useBlocker(useCallback((transition) => {
    const active = current.current.session;
    return Boolean(active && !active.resolved && leavesReviewPage(transition));
  }, []));
  const blockerRef = useRef(blocker);
  blockerRef.current = blocker;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    if (!dirty) return undefined;
    return installReviewUnloadGuard(window, {
      isDirty: () => Boolean(current.current.session && !current.current.session.resolved),
      isBusy: () => Boolean(current.current.session?.busy || leaveLock.current),
      discard: (options) => current.current.discard(options),
      // A failed unload request retains the server draft; do not claim it was
      // discarded. This also preserves the candidate if restored from bfcache.
      onError: () => {},
    });
  }, [dirty]);

  useEffect(() => {
    // A save may finish while a navigation prompt is open. Stay on the saved
    // results rather than replaying a previously blocked navigation silently.
    if (!dirty && !leaveLock.current && blocker.state === "blocked") blocker.reset();
  }, [dirty, blocker]);

  const stay = useCallback(() => {
    if (leaveLock.current) return;
    setError("");
    if (blockerRef.current.state === "blocked") blockerRef.current.reset();
  }, []);

  const leave = useCallback(async () => {
    if (leaveLock.current || current.current.session?.busy || blockerRef.current.state !== "blocked") return;
    leaveLock.current = true;
    setLeaving(true);
    setError("");
    try {
      const discarded = await current.current.discard();
      if (!discarded) throw new Error("The results are still being processed. Please try again.");
      // Wait for the server acknowledgement before changing route. Failed
      // discards keep the page and its unsaved candidate available for retry.
      if (mounted.current && blockerRef.current.state === "blocked") blockerRef.current.proceed();
    } catch (failure) {
      if (mounted.current) setError(failure.message || "Unable to leave without saving. Please try again.");
    } finally {
      leaveLock.current = false;
      if (mounted.current) setLeaving(false);
    }
  }, []);

  return { open: blocker.state === "blocked", leaving, busy, error, stay, leave };
}
