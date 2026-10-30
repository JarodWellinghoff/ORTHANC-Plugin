export function leavesReviewPage({ currentLocation, nextLocation }) {
  const pathname = (location) => location.pathname.replace(/\/+$/, "") || "/";
  const reviewId = (location) => new URLSearchParams(location.search).get("review");
  // Hash links and unrelated query parameters do not leave the current review.
  return pathname(currentLocation) !== pathname(nextLocation) ||
    reviewId(currentLocation) !== reviewId(nextLocation);
}

export function installReviewUnloadGuard(target, { isDirty, isBusy, discard, onError }) {
  const beforeUnload = (event) => {
    if (!isDirty()) return;
    // Never discard here: the user may cancel the browser's confirmation.
    event.preventDefault();
    event.returnValue = true;
  };
  const pageHide = () => {
    if (!isDirty() || isBusy()) return;
    // pagehide follows a confirmed document departure, not tab visibility
    // changes. Do not race an in-flight save with a second decision.
    try {
      Promise.resolve(discard({ keepalive: true })).catch(onError);
    } catch (error) {
      onError(error);
    }
  };
  target.addEventListener("beforeunload", beforeUnload);
  target.addEventListener("pagehide", pageHide);
  return () => {
    target.removeEventListener("beforeunload", beforeUnload);
    target.removeEventListener("pagehide", pageHide);
  };
}
