# Save-only result review

Patch baseline: `e5c64f42b11c92bab48f40e482420138e49ee5d7`.
The touched existing files are unchanged from its parent `1fb1ac2`.

## Behavior

Rerun candidates stay in the existing inline results layout. Structured metric
labels/units render as text with subscript/superscript elements, and comparison
rows use metric IDs as React keys. Precision follows `unit.decimals` when set.

The only results decision button on the page is **Save results**. It explicitly
posts the existing `overwrite` decision. Failed requests retain the candidate,
keep the unsaved-results guard active, and permit another save attempt.

Internal navigation away from the series or selected review opens a confirmation
with **Stay** and **Leave without saving**. Stay preserves the view. Leave posts
the existing `discard` decision and waits for acknowledgement before proceeding.
A failed discard keeps the page open with an error. Discard never calls overwrite
and does not delete the accepted results. No request is sent during React effect
cleanup, including StrictMode's development cleanup cycle.

The page's Back button no longer clears the dashboard state before navigation
is accepted. Hash links and unrelated query parameters are not departures.
Navigation during an active save is blocked until the save finishes; duplicate
or competing decisions cannot be submitted concurrently.

The root uses `createBrowserRouter` because React Router's `useBlocker` requires
a data router. The existing Dashboard routes, URLs and providers are retained.
There are no dependency, database schema, Orthanc endpoint or Docker changes.

Background analysis behavior is unchanged: background reruns remain in the
review queue until explicitly opened; initial analyses with no saved results
continue to follow the existing automatic-save policy.

## Refresh and tab/window close

While a candidate is unsaved, `beforeunload` requests the browser's native
confirmation. Nothing is discarded in that handler because the user may cancel.
After a document departure reaches `pagehide`, a small authenticated discard
POST uses `fetch` keepalive. It never races an active save, and tab visibility
changes alone do not discard anything.

Browsers control native-dialog wording and may suppress lifecycle events,
particularly during process termination or on mobile. Keepalive delivery is
best effort, not a guarantee. An unacknowledged discard leaves the server draft
recoverable rather than claiming it was removed. In-app navigation, unlike a
forced document shutdown, waits for a successful server response.

## Automated regression checks

From `frontend`:

```sh
node --test tests/resultComparisonRendering.test.mjs tests/reviewSaveOnly.test.mjs
npm test
npm run build
```

The two new dependency-free test files cover formatting and row identity,
precision/missing values, navigation classification, decision concurrency,
save/discard failures, missing drafts, native confirmation cancellation,
keepalive dispatch and cleanup safety. These are focused unit tests, not an
end-to-end browser test of React Router or the live Orthanc service.

## Browser verification

1. Rerun a series with saved results; inspect inline curves/comparison and the
   browser console. Confirm there are no object-child or duplicate-key errors.
2. Confirm the page offers Save results and no standalone discard button.
3. Try the sidebar, Back to Dashboard, browser Back/Forward, another series,
   and removing/changing the review query parameter. Choose Stay each time;
   confirm the candidate and saved baseline remain visible and no decision POST
   was sent. Same-page anchors should not prompt.
4. Choose Leave without saving. Confirm the discard POST completes before the
   route changes. Reopen the series and verify the previously saved results.
5. Force save and discard requests to fail. Confirm the candidate remains
   available and the action can be retried. Repeat with rapid double-clicks.
6. Save successfully. Confirm the accepted display refreshes and navigation no
   longer prompts. Block a navigation during a delayed save and verify the
   Leave button is disabled until that request settles.
7. Refresh or close while unsaved. Cancel the native prompt and confirm there
   was no discard. Repeat and confirm departure; inspect server-side draft
   removal. Verify browser back/forward cache restoration shows accepted data
   after the acknowledged discard.
8. Run in the background. Confirm no results open automatically and no unsaved
   warning appears until the candidate has actually been opened for review.
