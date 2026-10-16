import { useEffect, useMemo, useRef, useState } from "react";
import { Alert, Box, Button, CircularProgress, Dialog, DialogActions, DialogContent,
  DialogTitle, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from "@mui/material";
import { useSnackbar } from "notistack";
import { useDashboard } from "../context/DashboardContext";
import { SCALAR_METRICS } from "../utils/aggregateMetrics";
import { finiteNumber, getReportMetrics, PATIENT_COUNT_KEY } from "./plots/protocolReport";
import ChoPlots from "./plots/ChoPlots";
const apiBase = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
const request = async (path, options = {}) => {
  const response = await fetch(`${apiBase}/cho-result-reviews${path}`, {
    credentials: "include", cache: "no-store", ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  let payload;
  try { payload = await response.json(); } catch { throw new Error(`Invalid review response (${response.status}).`); }
  if (!response.ok) throw new Error(payload.error || `Review request failed (${response.status}).`);
  return payload;
};
const metrics = getReportMetrics(SCALAR_METRICS).filter((item) => item.key !== PATIENT_COUNT_KEY);
const display = (value, decimals) => {
  const number = finiteNumber(value);
  return number === null ? "Not available" : number.toFixed(decimals ?? 2);
};
export default function RerunReviewHost() {
  const dashboard = useDashboard();
  const dashboardRef = useRef(dashboard);
  dashboardRef.current = dashboard;
  const { enqueueSnackbar } = useSnackbar();
  const [pending, setPending] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [review, setReview] = useState(null);
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const [saving, setSaving] = useState(false);
  const [reload, setReload] = useState(0);
  const deferred = useRef(new Set());
  const resolved = useRef(new Set());
  const decisionLock = useRef(false);
  useEffect(() => {
    const controller = new AbortController();
    let timer;
    const poll = async () => {
      try {
        const data = await request("", { signal: controller.signal });
        if (controller.signal.aborted) return;
        const reviews = (data.reviews ?? []).filter((item) => !resolved.current.has(item.id));
        setPending(reviews); setPollError("");
        setSelectedId((current) => current ?? reviews.find((item) => !deferred.current.has(item.id))?.id ?? null);
      } catch (err) { if (!controller.signal.aborted) setPollError(err.message); }
      finally { if (!controller.signal.aborted) timer = setTimeout(poll, 4000); }
    };
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);
  useEffect(() => {
    if (!selectedId) { setReview(null); setError(""); return undefined; }
    const controller = new AbortController();
    setReview(null); setError("");
    request(`/${selectedId}`, { signal: controller.signal }).then((data) => {
      if (!data.candidate?.results || !data.baseline || data.state !== "ready") throw new Error("The rerun is not ready for review.");
      if (!controller.signal.aborted) setReview(data);
    }).catch((err) => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [selectedId, reload]);
  const close = () => {
    if (decisionLock.current) return;
    deferred.current.add(selectedId); setSelectedId(null);
  };
  const decide = async (decision) => {
    if (!review || decisionLock.current) return;
    decisionLock.current = true;
    setSaving(true); setError("");
    try {
      await request(`/${review.id}`, { method: "POST", body: JSON.stringify({ decision }) });
      resolved.current.add(review.id);
      setPending((items) => items.filter((item) => item.id !== review.id));
      setSelectedId(null);
      enqueueSnackbar(decision === "overwrite" ? "Saved results replaced with the reviewed rerun." : "New results discarded. Saved results were kept.", { variant: "success" });
      window.dispatchEvent(new CustomEvent("ctpro-review-resolved", { detail: { seriesUuid: review.series_uuid, decision } }));
      const { choModal, actions } = dashboardRef.current;
      if (choModal.seriesUuid === review.series_uuid || choModal.seriesId === review.series_uid) {
        // Clear the unsaved in-page result as well, especially after discard.
        actions.openChoModal({ ...choModal.seriesSummary, series_uuid: review.series_uuid, series_id: review.series_uid });
        try { await actions.reloadStoredChoResults(); }
        catch { enqueueSnackbar("Decision applied. Refresh the exam to reload its saved results.", { variant: "warning" }); }
      }
    } catch (err) { setError(err.message); }
    finally { decisionLock.current = false; setSaving(false); }
  };
  const newPlotData = useMemo(() => review ? ({ ...review.candidate,
    // The stored MinIO image still belongs to the accepted result. Do not put
    // it behind an unsaved candidate's curves and imply it is a new image.
    series: { ...review.candidate.series, series_instance_uid: undefined },
  }) : null, [review]);
  return <>
    {pollError && <Alert severity='warning' sx={{ width: "100%" }}>Rerun review is unavailable: {pollError}</Alert>}
    {pending.length > 0 && <Alert severity='info' sx={{ width: "100%" }}
      action={<Button color='inherit' onClick={() => setSelectedId(pending[0].id)}>Review</Button>}>
      {pending.length} rerun{pending.length === 1 ? "" : "s"} awaiting review.
    </Alert>}
    <Dialog open={Boolean(selectedId)} maxWidth='lg' fullWidth onClose={close}>
      <DialogTitle>Review rerun results</DialogTitle>
      <DialogContent dividers>
        {error && <Alert severity='error' action={!review && <Button onClick={() => setReload((value) => value + 1)}>Retry</Button>}>{error}</Alert>}
        {!review && !error && <Box sx={{ p: 4, textAlign: "center" }}><CircularProgress aria-label='Loading new results' /></Box>}
        {review && <Stack spacing={2}>
          <Typography>Would you like to overwrite the saved results with this rerun, or discard the new results?</Typography>
          <Typography variant='body2' color='text.secondary'>
            {review.baseline.patient_name || "Patient"} / {review.baseline.protocol_name || "Protocol unspecified"}
            {" "}(series {review.series_uid}). Saved results remain unchanged until you choose Overwrite.
          </Typography>
          <TableContainer sx={{ maxHeight: 360, overflow: "auto" }}><Table size='small' stickyHeader aria-label='Saved and new analysis results'>
            <TableHead><TableRow><TableCell>Metric</TableCell><TableCell align='right'>Saved</TableCell><TableCell align='right'>New</TableCell></TableRow></TableHead>
            <TableBody>{metrics.map((metric) => <TableRow key={metric.key}>
              <TableCell>{metric.label}{metric.unit ? ` (${metric.unit})` : ""}</TableCell>
              <TableCell align='right'>{display(review.baseline[metric.key], metric.decimals)}</TableCell>
              <TableCell align='right'>{display(review.candidate.results[metric.key] ?? review.candidate.series[metric.key], metric.decimals)}</TableCell>
            </TableRow>)}</TableBody>
          </Table></TableContainer>
          <ChoPlots data={newPlotData} comparison={review.baseline} />
        </Stack>}
      </DialogContent>
      <DialogActions>
        <Button disabled={saving} onClick={close}>Review later</Button>
        <Button disabled={!review || saving} onClick={() => decide("discard")}>Discard new results</Button>
        <Button variant='contained' disabled={!review || saving} onClick={() => decide("overwrite")}>{saving ? "Applying decision..." : "Overwrite saved results"}</Button>
      </DialogActions>
    </Dialog>
  </>;
}
