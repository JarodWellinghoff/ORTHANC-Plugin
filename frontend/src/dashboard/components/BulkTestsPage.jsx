import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Button, Chip, CircularProgress, FormControl, IconButton, InputLabel,
  MenuItem, Select, Stack, Tooltip, Typography } from "@mui/material";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import { DataGrid, Toolbar, ToolbarButton, ColumnsPanelTrigger } from "@mui/x-data-grid";
import { useSnackbar } from "notistack";
import { useDashboard } from "../context/DashboardContext";
import FiltersPanel from "./FiltersPanel";
import { useFilters } from "../../hooks/useFilters";
import { defaultChoParams, formatDateTime, fetchJson, normalizeChoRow,
  resolveSeriesKey, sleep, statusColorMap, statusLabelMap } from "../utils/choResultsShared";
const GridToolbar = () => <Toolbar><Tooltip title='Columns'>
  <ColumnsPanelTrigger render={<ToolbarButton />}><ViewColumnIcon fontSize='small' /></ColumnsPanelTrigger>
</Tooltip></Toolbar>;
export default function BulkTestsPage() {
  const { enqueueSnackbar } = useSnackbar();
  const { summary, calculationStates, actions } = useDashboard();
  const { filters, updateFilter, resetFilters } = useFilters();
  const { items, pagination } = summary;
  const [sortModel, setSortModel] = useState([{ field: "patientName", sort: "asc" }]);
  const [loading, setLoading] = useState(true);
  const [availableSeries, setAvailableSeries] = useState([]);
  const [modalities, setModalities] = useState([]);
  const [selectedModality, setSelectedModality] = useState("");
  const [selection, setSelection] = useState({ type: "include", ids: new Set() });
  const [progress, setProgress] = useState({});
  const [recovering, setRecovering] = useState({});
  const [runningBulk, setRunningBulk] = useState(false);
  const [bulkRun, setBulkRun] = useState({ total: 0, done: 0 });
  const [activeCount, setActiveCount] = useState(0);
  const statesRef = useRef(calculationStates);
  const mounted = useRef(true);
  const activeRows = useRef(new Set());
  const bulkLock = useRef(false);
  statesRef.current = calculationStates;
  const loadAvailableSeries = useCallback(async () => {
    const data = await fetchJson("/series/");
    if (mounted.current) setAvailableSeries(Array.isArray(data) ? data : []);
  }, []);
  const load = (page = 1, sort = sortModel) => actions.loadSummary({ ...filters, page, limit: pagination.limit ?? 25,
    sort_by: sort[0]?.field, sort_order: sort[0]?.sort ?? "asc" });
  useEffect(() => {
    mounted.current = true;
    Promise.all([actions.loadSummary({ ...filters, page: 1 }), loadAvailableSeries(),
      fetchJson("/dicom-modalities").then((data) => {
        if (!mounted.current) return;
        const list = Array.isArray(data?.modalities) ? data.modalities : [];
        setModalities(list); setSelectedModality(list[0]?.id ?? "");
      }),
    ]).catch((error) => enqueueSnackbar(error.message, { variant: "error" }))
      .finally(() => { if (mounted.current) setLoading(false); });
    return () => { mounted.current = false; };
    // Queries after mounting are explicit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!runningBulk && activeCount === 0) return undefined;
    const warn = (event) => { event.preventDefault(); event.returnValue = "Analyses are still running."; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [runningBulk, activeCount]);
  const available = useMemo(() => new Set(availableSeries.map(String)), [availableSeries]);
  const rows = useMemo(() => items.map((item, index) => normalizeChoRow(item, index, available)), [items, available]);
  const selectedRows = useMemo(() => rows.filter((row) => selection.type === "exclude" ? !selection.ids.has(row.id) : selection.ids.has(row.id)), [rows, selection]);
  useEffect(() => {
    const onReviewed = (event) => {
      const row = rows.find((item) => item.seriesUuid === event.detail?.seriesUuid);
      if (!row) return;
      setProgress((old) => ({ ...old, [row.id]: { status: "reviewed", message: event.detail.decision === "overwrite" ? "Saved" : "Saved result kept" } }));
      actions.loadSummary({ ...filters, page: pagination.page ?? 1, limit: pagination.limit ?? 25,
        sort_by: sortModel[0]?.field, sort_order: sortModel[0]?.sort ?? "asc" });
    };
    window.addEventListener("ctpro-review-resolved", onReviewed);
    return () => window.removeEventListener("ctpro-review-resolved", onReviewed);
  }, [actions, filters, pagination.page, pagination.limit, rows, sortModel]);
  const updateProgress = (id, value) => { if (mounted.current) setProgress((old) => ({ ...old, [id]: value })); };
  const runSingle = async (row) => {
    if (activeRows.current.has(row.id) || !row.hasDicom || !row.seriesUuid) return false;
    activeRows.current.add(row.id);
    setActiveCount((count) => count + 1);
    const key = resolveSeriesKey(row);
    const previous = statesRef.current[key];
    updateProgress(row.id, { status: "running", message: "Starting..." });
    try {
      await fetchJson("/cho-analysis-modal", { method: "POST", body: JSON.stringify({
        series_uuid: row.seriesUuid, testType: "full", ...defaultChoParams, saveResults: true,
      }) });
      const started = Date.now();
      while (Date.now() - started < 300000) {
        if (!mounted.current) return false; // The server continues the current calculation.
        const state = statesRef.current[key];
        const fresh = state && (state.metadata?.run_id ? state.metadata.run_id !== previous?.metadata?.run_id : state.start_time !== previous?.start_time);
        if (fresh && ["failed", "error", "cancelled"].includes(state.status)) throw new Error(state.error || "Analysis failed or was cancelled.");
        if (fresh && state.status === "completed") {
          setProgress((old) => old[row.id]?.status === "reviewed" ? old : ({ ...old, [row.id]: {
            status: "done", message: state.metadata?.review_required ? "Awaiting review" : "Completed" } }));
          return true;
        }
        await sleep(1000);
      }
      throw new Error("Analysis has not reported completion. It may still be running; check its progress before retrying.");
    } catch (error) {
      updateProgress(row.id, { status: "error", message: error.message });
      if (mounted.current) enqueueSnackbar(error.message, { variant: "error" });
      return false;
    } finally {
      activeRows.current.delete(row.id);
      if (mounted.current) setActiveCount((count) => Math.max(0, count - 1));
    }
  };
  const runSelected = async () => {
    if (bulkLock.current) return;
    const runnable = selectedRows.filter((row) => row.hasDicom && row.seriesUuid && !activeRows.current.has(row.id));
    if (!runnable.length) { enqueueSnackbar("Select series with DICOM available. Pull missing DICOM first.", { variant: "warning" }); return; }
    if (runnable.length < selectedRows.length) enqueueSnackbar("Unavailable or already-running series will be skipped.", { variant: "warning" });
    bulkLock.current = true; setRunningBulk(true); setBulkRun({ total: runnable.length, done: 0 });
    runnable.forEach((row) => updateProgress(row.id, { status: "queued", message: "Waiting to run" }));
    let completed = 0;
    try {
      for (const row of runnable) {
        if (!mounted.current) break;
        if (await runSingle(row)) completed += 1;
        if (mounted.current) setBulkRun((old) => ({ ...old, done: old.done + 1 }));
      }
      if (mounted.current) enqueueSnackbar(`${completed} of ${runnable.length} analyses completed. Reruns require review before saving.`, { variant: completed === runnable.length ? "success" : "warning" });
    } finally { bulkLock.current = false; if (mounted.current) setRunningBulk(false); }
  };
  const recover = async (row) => {
    if (!selectedModality || !row.seriesInstanceUid) return;
    setRecovering((old) => ({ ...old, [row.id]: true }));
    try {
      await fetchJson("/dicom-pull/recover", { method: "POST", body: JSON.stringify({
        modality: selectedModality, seriesInstanceUID: row.seriesInstanceUid,
        studyInstanceUID: row.studyInstanceUid, patientId: row.raw?.patient_id ?? null,
      }) });
      updateProgress(row.id, { status: "pending", message: "Recovery requested" });
      await loadAvailableSeries();
      if (mounted.current) await load(pagination.page ?? 1);
    } catch (error) { updateProgress(row.id, { status: "error", message: error.message }); }
    finally { if (mounted.current) setRecovering((old) => ({ ...old, [row.id]: false })); }
  };
  const columns = [
    ...[["patientId", "Patient ID"], ["patientName", "Patient"], ["institutionName", "Institute"],
      ["protocolName", "Protocol"], ["scannerModel", "Scanner Model"], ["stationName", "Scanner Station"]]
      .map(([field, headerName]) => ({ field, headerName, flex: 1, minWidth: 150 })),
    { field: "studyDate", headerName: "Study Date", minWidth: 120, valueFormatter: (value) => value ? String(value).replace(/^(\d{4})-(\d{2})-(\d{2})$/, "$2/$3/$1") : "--" },
    { field: "pullScheduleName", headerName: "Pull Schedule", minWidth: 160, valueFormatter: (value) => value ?? "--" },
    { field: "testStatus", headerName: "Status", width: 165, renderCell: ({ row }) => {
      const live = calculationStates[resolveSeriesKey(row)];
      const local = progress[row.id];
      let label = statusLabelMap[row.testStatus] ?? row.testStatus;
      let color = statusColorMap[row.testStatus] ?? "default";
      let message = label;
      if (local?.status === "reviewed") { label = local.message; color = "success"; }
      else if (["queued", "error", "pending"].includes(local?.status)) {
        label = { queued: "Queued", error: "Failed", pending: "Recovery" }[local.status];
        color = local.status === "error" ? "error" : "info"; message = local.message;
      } else if (live?.status === "running") { label = `Running ${Math.round(live.progress ?? 0)}%`; color = "info"; message = live.message; }
      else if (["failed", "error"].includes(live?.status)) { label = "Failed"; color = "error"; message = live.error || live.message; }
      else if (live?.status === "completed") { label = live.metadata?.review_required ? "Review required" : "Complete"; color = live.metadata?.review_required ? "warning" : "success"; }
      else if (local?.status === "running") { label = "Starting..."; color = "info"; }
      else if (local?.status === "done") { label = local.message; color = "success"; }
      return <Tooltip title={message ?? ""}><Chip size='small' color={color} label={label} /></Tooltip>;
    } },
    { field: "latestAnalysis", headerName: "Latest Analysis", minWidth: 180, valueFormatter: formatDateTime },
    { field: "actions", headerName: "Actions", width: 150, sortable: false, filterable: false,
      renderCell: ({ row }) => row.hasDicom ? <Tooltip title='Run analysis'><span>
        <IconButton aria-label='Run analysis' disabled={runningBulk || activeRows.current.has(row.id)} onClick={() => runSingle(row)}><PlayArrowRoundedIcon /></IconButton>
      </span></Tooltip> : <Button size='small' startIcon={recovering[row.id] ? <CircularProgress size={14} /> : <CloudDownloadRoundedIcon />}
        disabled={recovering[row.id] || !selectedModality} onClick={() => recover(row)}>Pull DICOM</Button> },
  ];
  return <Stack spacing={3}>
    <Typography variant='h4' component='h1'>Batch Analysis</Typography>
    <FiltersPanel filters={filters} onChange={updateFilter} onQuery={() => load()} onReset={resetFilters} />
    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ sm: "center" }} justifyContent='space-between'>
      <Stack direction='row' spacing={2} alignItems='center'>
        <Button variant='contained' disabled={runningBulk || !selectedRows.length} onClick={runSelected}
          startIcon={runningBulk ? <CircularProgress size={16} color='inherit' /> : <PlayArrowRoundedIcon />}>
          {runningBulk ? "Running..." : `Run Selected (${selectedRows.length})`}
        </Button>
        {runningBulk && <Typography variant='body2'>Running {Math.min(bulkRun.done + 1, bulkRun.total)} of {bulkRun.total}</Typography>}
      </Stack>
      <FormControl size='small' sx={{ minWidth: 220 }} disabled={!modalities.length}>
        <InputLabel id='recovery-server-label'>Recovery server</InputLabel>
        <Select labelId='recovery-server-label' label='Recovery server' value={selectedModality} onChange={(event) => setSelectedModality(event.target.value)}>
          {modalities.map((modality) => <MenuItem key={modality.id} value={modality.id}>{modality.name ?? modality.id}</MenuItem>)}
        </Select>
      </FormControl>
    </Stack>
    <Box sx={{ height: 560, minWidth: 0 }}>
      <DataGrid rows={rows} columns={columns} getRowId={(row) => row.id} rowCount={pagination.total ?? rows.length}
        loading={loading || summary.loading} autoHeight={false} paginationMode='server'
        paginationModel={{ page: Math.max(0, (pagination.page ?? 1) - 1), pageSize: pagination.limit ?? 25 }}
        pageSizeOptions={[]} onPaginationModelChange={(model) => { if (model.page !== (pagination.page ?? 1) - 1) load(model.page + 1); }}
        sortingMode='server' sortModel={sortModel} onSortModelChange={(model) => { setSortModel(model); load(1, model); }}
        checkboxSelection disableRowSelectionOnClick disableColumnFilter rowSelectionModel={selection} onRowSelectionModelChange={setSelection}
        slots={{ toolbar: GridToolbar }} showToolbar initialState={{ columns: { columnVisibilityModel: {
          latestAnalysis: false, pullScheduleName: false, studyDate: false, stationName: false, patientId: false,
        } } }} getRowClassName={({ row }) => row.hasDicom ? "" : "missing-dicom"}
        sx={{ "& .missing-dicom": { bgcolor: "action.hover" } }} />
    </Box>
  </Stack>;
}
