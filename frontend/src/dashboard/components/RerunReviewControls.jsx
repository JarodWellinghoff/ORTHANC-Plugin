import {
  Alert, Button, Dialog, DialogActions, DialogContent, DialogContentText,
  DialogTitle, LinearProgress, Stack, Typography,
} from "@mui/material";

export default function RerunReviewControls({ flow, awaitingReview, running, background, onBackground }) {
  const { review, pendingReview, loading, error, saving, busy, pollError, requested, openReview, save, retry, navigation } = flow;
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
        <Alert severity='error' action={!review && <Button color='inherit' disabled={busy} onClick={retry}>Retry</Button>}>
          {error}
        </Alert>
      )}
      {review && (
        <Stack spacing={1.5}>
          <Alert severity='info'>
            You are reviewing unsaved results alongside the saved results. Choose Save results to keep the new results,
            or leave this page without saving to keep the previous results.
            The saved coronal image is not displayed behind the new curves.
          </Alert>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
            <Button variant='contained' disabled={busy} onClick={save}>
              {saving ? "Saving results..." : "Save results"}
            </Button>
          </Stack>
        </Stack>
      )}
      <Dialog
        open={navigation.open}
        onClose={navigation.leaving ? undefined : navigation.stay}
        disableEscapeKeyDown={navigation.leaving}
        aria-labelledby='unsaved-review-title'
        aria-describedby='unsaved-review-description'>
        <DialogTitle id='unsaved-review-title'>Leave with unsaved results?</DialogTitle>
        <DialogContent>
          <DialogContentText id='unsaved-review-description'>
            {saving
              ? "Your results are being saved. Wait for the save to finish before leaving."
              : "Leaving without saving will discard the new results. Your previously saved results will remain unchanged."}
          </DialogContentText>
          {navigation.error && <Alert severity='error' sx={{ mt: 2 }}>{navigation.error}</Alert>}
        </DialogContent>
        <DialogActions>
          <Button autoFocus disabled={navigation.leaving} onClick={navigation.stay}>Stay</Button>
          <Button color='warning' disabled={busy || navigation.leaving} onClick={navigation.leave}>
            {navigation.leaving ? "Leaving..." : "Leave without saving"}
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
