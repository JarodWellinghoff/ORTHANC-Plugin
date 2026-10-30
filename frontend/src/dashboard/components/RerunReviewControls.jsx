import { Alert, Button, LinearProgress, Stack, Typography } from "@mui/material";

export default function RerunReviewControls({ flow, awaitingReview, running, background, onBackground }) {
  const { review, pendingReview, loading, error, saving, pollError, requested, openReview, decide, retry } = flow;
  return (
    <Stack spacing={2}>
      {running && (
        <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
          <Typography variant='body2' color='text.secondary'>
            {background ? "Analysis is running in the background. Results will not open automatically." : "Analysis is running."}
          </Typography>
          {!background && <Button onClick={onBackground}>Run in background</Button>}
        </Stack>
      )}
      {pendingReview && !requested && (
        <Alert severity='info' action={<Button color='inherit' onClick={openReview}>Review results</Button>}>
          New results are awaiting review. The saved results below are unchanged.
        </Alert>
      )}
      {(loading || (awaitingReview && !requested && !pollError)) && (
        <Stack spacing={1} role='status' aria-live='polite'>
          <LinearProgress />
          <Typography variant='body2'>Loading new results for inline review...</Typography>
        </Stack>
      )}
      {pollError && (awaitingReview || requested) && (
        <Alert severity='warning'>Review availability could not be checked. Retrying automatically.</Alert>
      )}
      {error && (
        <Alert severity='error' action={<Button color='inherit' disabled={saving} onClick={retry}>Retry</Button>}>
          {error}
        </Alert>
      )}
      {review && (
        <Stack spacing={1.5}>
          <Alert severity='info'>
            You are reviewing new results alongside the saved results. Nothing is replaced until you choose Overwrite saved results.
            The saved coronal image is not displayed behind the new curves.
          </Alert>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
            <Button variant='contained' disabled={saving} onClick={() => decide("overwrite")}>
              {saving ? "Applying decision..." : "Overwrite saved results"}
            </Button>
            <Button variant='outlined' disabled={saving} onClick={() => decide("discard")}>Discard new results</Button>
          </Stack>
        </Stack>
      )}
    </Stack>
  );
}
