import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Stack from "@mui/material/Stack";
import Divider from "@mui/material/Divider";
import Tooltip from "@mui/material/Tooltip";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import { useDashboard } from "../context/DashboardContext";
import { useSnackbar } from "notistack";
import {
  DataGrid,
  Toolbar,
  ToolbarButton,
  ColumnsPanelTrigger,
  ExportCsv,
  ExportPrint,
} from "@mui/x-data-grid";
import FiltersPanel from "./FiltersPanel";
import { useFilters } from "../../hooks/useFilters";
import {
  defaultChoParams,
  fetchJson,
  normalizeChoRow,
  resolveSeriesKey,
  sleep,
  statusColorMap,
  statusLabelMap,
} from "../utils/choResultsShared";

// ─────────────────────────────────────────────────────────────────────────────
// GridToolbar
// ─────────────────────────────────────────────────────────────────────────────
const GridToolbar = () => {
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const exportMenuTriggerRef = useRef(null);

  return (
    <Toolbar>
      <Tooltip title='Columns'>
        <ColumnsPanelTrigger render={<ToolbarButton />}>
          <ViewColumnIcon fontSize='small' />
        </ColumnsPanelTrigger>
      </Tooltip>
      <Tooltip title='Filters'></Tooltip>
      <Divider
        orientation='vertical'
        variant='middle'
        flexItem
        sx={{ mx: 0.5 }}
      />
      <Tooltip title='Export'>
        <ToolbarButton
          ref={exportMenuTriggerRef}
          id='export-menu-trigger'
          aria-controls='export-menu'
          aria-haspopup='true'
          aria-expanded={exportMenuOpen ? "true" : undefined}
          onClick={() => setExportMenuOpen(true)}>
          <FileDownloadIcon fontSize='small' />
        </ToolbarButton>
      </Tooltip>
      <Menu
        id='export-menu'
        anchorEl={exportMenuTriggerRef.current}
        open={exportMenuOpen}
        onClose={() => setExportMenuOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          list: {
            "aria-labelledby": "export-menu-trigger",
          },
        }}>
        <ExportPrint
          render={<MenuItem />}
          onClick={() => setExportMenuOpen(false)}>
          Print
        </ExportPrint>
        <ExportCsv
          render={<MenuItem />}
          onClick={() => setExportMenuOpen(false)}>
          Download as CSV
        </ExportCsv>
      </Menu>
    </Toolbar>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// BulkTestsPage
//
// Testing-focused counterpart to ResultsPage. Owns everything related to
// kicking off CHO analyses: modality selection for DICOM recovery, test type,
// per-row Pull DICOM and Run buttons, the calculationStates wiring that
// surfaces in-progress chips in the Status column, and the beforeunload guard
// that warns when navigating away while tests are still running.
//
// The "View Results" action used to live in this page's actions column; that
// has moved to ResultsPage so the two pages have a clean responsibility split.
// ─────────────────────────────────────────────────────────────────────────────
const BulkTestsPage = () => {
  const { enqueueSnackbar } = useSnackbar();
  const { summary, calculationStates, actions } = useDashboard();
  const { filters, updateFilter, resetFilters } = useFilters();
  const { items, pagination } = summary;

  const [filterModel, setFilterModel] = useState({ items: [] });
  const [sortModel, setSortModel] = useState([
    { field: "patientName", sort: "asc" },
  ]);
  const sortRef = useRef(sortModel);

  const [loading, setLoading] = useState(true);
  const [availableSeries, setAvailableSeries] = useState([]);
  const [modalities, setModalities] = useState([]);
  const [selectedModality, setSelectedModality] = useState("");
  const [loadingModalities, setLoadingModalities] = useState(false);
  const [selectionModel, setSelectionModel] = useState({
    type: "include",
    ids: new Set(),
  });
  const [testType] = useState("full");
  const [bulkProgress, setBulkProgress] = useState({});
  const [runningBulk, setRunningBulk] = useState(false);
  const [recoveringMap, setRecoveringMap] = useState({});
  const [activeRunCount, setActiveRunCount] = useState(0);

  const calculationStatesRef = useRef(calculationStates);
  useEffect(() => {
    calculationStatesRef.current = calculationStates;
  }, [calculationStates]);

  const handleQuery = () => actions.loadSummary(filters);

  const shouldWarnOnLeave = runningBulk || activeRunCount > 0;

  const handleSortModelChange = useCallback(
    (model) => {
      setSortModel(model);
      sortRef.current = model;
      const sort = model[0];
      actions.loadSummary({
        ...filters,
        page: 1,
        sort_by: sort?.field,
        sort_order: sort?.sort ?? "asc",
      });
    },
    [actions, filters],
  );

  // Warn the user before they navigate away mid-run. Without this, closing the
  // tab silently abandons in-flight analyses and the progress chips never
  // resolve.
  useEffect(() => {
    if (!shouldWarnOnLeave) {
      return undefined;
    }

    const handleBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue =
        "Tests are still running. Are you sure you want to leave this page?";
      return event.returnValue;
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [shouldWarnOnLeave]);

  const loadModalities = useCallback(async () => {
    setLoadingModalities(true);
    try {
      const data = await fetchJson("/dicom-modalities");
      const list = Array.isArray(data?.modalities) ? data.modalities : [];
      setModalities(list);
      if (!selectedModality && list.length > 0) {
        setSelectedModality(list[0].id);
      }
    } catch (err) {
      console.error("Failed to load modalities", err);
      enqueueSnackbar("Failed to load modalities", {
        variant: "error",
      });
    } finally {
      setLoadingModalities(false);
    }
  }, [selectedModality, enqueueSnackbar]);

  const loadAvailableSeries = useCallback(async () => {
    try {
      const data = await fetchJson("/series/");
      if (Array.isArray(data)) {
        setAvailableSeries(data);
      } else {
        console.warn("Unexpected response for /series/", data);
        setAvailableSeries([]);
      }
    } catch (err) {
      console.error("Failed to load available series", err);
    }
  }, []);

  // Initial load: summary (drives the grid), Orthanc availability (for the
  // hasDicom badge / Pull DICOM gating), and the modality list (for recovery).
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      try {
        await actions.loadSummary(filters);
      } catch (err) {
        if (!cancelled) {
          console.error("Failed to load summary", err);
          enqueueSnackbar("Failed to load results", { variant: "error" });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    loadAvailableSeries();
    loadModalities();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const availableSet = useMemo(
    () => new Set(availableSeries.filter(Boolean).map((v) => String(v))),
    [availableSeries],
  );

  const normalizedResults = useMemo(
    () =>
      items.map((item, index) => normalizeChoRow(item, index, availableSet)),
    [items, availableSet],
  );

  // ── Test running ──────────────────────────────────────────────────────────

  const updateBulkProgress = useCallback((id, next) => {
    setBulkProgress((prev) => ({
      ...prev,
      [id]: { ...(prev[id] ?? {}), ...next },
    }));
  }, []);

  const waitForSeriesCompletion = useCallback(async (row, options = {}) => {
    const seriesKey = resolveSeriesKey(row);
    if (!seriesKey) {
      return "unknown";
    }
    const pollIntervalMs = options.pollIntervalMs ?? 2000;
    const startTimeoutMs = options.startTimeoutMs ?? 15000;
    const timeoutMs = options.timeoutMs ?? 300000;

    const startTime = Date.now();
    let finalStatus = null;

    while (Date.now() - startTime < timeoutMs) {
      const state = calculationStatesRef.current[seriesKey];
      const status = state?.status ?? state?.eventType;

      if (!state && Date.now() - startTime > startTimeoutMs) {
        const timeoutError = new Error(
          "Analysis did not start within the expected time. It may still be running.",
        );
        timeoutError.code = "WAIT_TIMEOUT";
        throw timeoutError;
      }

      if (status === "completed" || status === "failed" || status === "error") {
        finalStatus = status;
        break;
      }

      await sleep(pollIntervalMs);
    }

    return finalStatus ?? "completed";
  }, []);

  const runAnalysisForSeries = useCallback(
    async (row) => {
      if (!row.seriesUuid) {
        throw new Error("Series UUID not available for this row.");
      }
      if (!row.hasDicom) {
        throw new Error("DICOM series is not currently available.");
      }
      const payload = {
        series_uuid: row.seriesUuid,
        testType,
        ...defaultChoParams,
        saveResults: true,
      };
      await fetchJson("/cho-analysis-modal", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
    [testType],
  );

  const handleRunSingle = useCallback(
    async (row) => {
      setActiveRunCount((count) => count + 1);
      updateBulkProgress(row.id, { status: "running", message: "Starting…" });
      try {
        await runAnalysisForSeries(row);
        const result = await waitForSeriesCompletion(row);
        updateBulkProgress(row.id, {
          status: result === "completed" ? "done" : "error",
          message: result === "completed" ? "Completed" : "Failed",
        });
      } catch (err) {
        updateBulkProgress(row.id, {
          status: "error",
          message: err.message ?? "Failed to start",
        });
        if (err?.code === "WAIT_TIMEOUT") {
          enqueueSnackbar(
            `Analysis for series ${row.seriesInstanceUid} did not complete within the expected time. It may still be running.`,
            { variant: "warning" },
          );
        }
      } finally {
        setActiveRunCount((count) => Math.max(0, count - 1));
      }
    },
    [
      enqueueSnackbar,
      runAnalysisForSeries,
      updateBulkProgress,
      waitForSeriesCompletion,
    ],
  );

  const handleRunBulk = useCallback(async () => {
    const type = selectionModel.type;
    const rowsById = new Map(normalizedResults.map((row) => [row.id, row]));
    let selectedIds = null;
    if (type === "include") {
      if (selectionModel.ids.size === 0) {
        enqueueSnackbar("Select at least one row to start bulk testing.", {
          variant: "warning",
        });
        return;
      }
      selectedIds = selectionModel.ids.intersection
        ? selectionModel.ids.intersection(rowsById)
        : new Set([...selectionModel.ids].filter((id) => rowsById.has(id)));
      if (selectedIds.size === 0) {
        enqueueSnackbar(
          "Selected rows are no longer available in the current data set.",
          { variant: "error" },
        );
        return;
      }
    } else if (type === "exclude") {
      if (selectionModel.ids.size === rowsById.size) {
        enqueueSnackbar("Select at least one row to start bulk testing.", {
          variant: "warning",
        });
        return;
      }
      selectedIds = new Set(rowsById.keys()).difference
        ? new Set(rowsById.keys()).difference(selectionModel.ids)
        : new Set(
            [...rowsById.keys()].filter((id) => !selectionModel.ids.has(id)),
          );
    }

    const queue = [];
    if (selectedIds) {
      for (const value of selectedIds) {
        if (rowsById.has(value)) queue.push(rowsById.get(value));
      }
    }
    if (queue.length === 0) {
      enqueueSnackbar("No valid selections remain to process.", {
        variant: "error",
      });
      return;
    }

    setRunningBulk(true);
    try {
      for (const row of queue) {
        await handleRunSingle(row);
      }
    } finally {
      setRunningBulk(false);
    }
  }, [enqueueSnackbar, handleRunSingle, normalizedResults, selectionModel]);

  const handleRecoverDicom = useCallback(
    async (row) => {
      if (!selectedModality) {
        enqueueSnackbar("Select a server to use for DICOM recovery.", {
          variant: "warning",
        });
        return;
      }
      if (!row.seriesInstanceUid) {
        enqueueSnackbar(
          "Series Instance UID is required to recover DICOM for this entry.",
          { variant: "error" },
        );
        return;
      }

      setRecoveringMap((prev) => ({ ...prev, [row.id]: true }));
      try {
        await fetchJson("/dicom-pull/recover", {
          method: "POST",
          body: JSON.stringify({
            modality: selectedModality,
            seriesInstanceUID: row.seriesInstanceUid,
            studyInstanceUID: row.studyInstanceUid,
            patientId: row.raw?.patient_id ?? null,
          }),
        });
        updateBulkProgress(row.id, {
          status: "pending",
          message: "Recovery requested",
        });
        await loadAvailableSeries();
      } catch (err) {
        updateBulkProgress(row.id, {
          status: "error",
          message: err.message ?? "Recovery failed",
        });
      } finally {
        setRecoveringMap((prev) => ({ ...prev, [row.id]: false }));
      }
    },
    [
      selectedModality,
      enqueueSnackbar,
      updateBulkProgress,
      loadAvailableSeries,
    ],
  );

  // ── Grid config ───────────────────────────────────────────────────────────

  const columns = useMemo(() => {
    return [
      {
        field: "patientId",
        headerName: "Patient ID",
        flex: 0.8,
        minWidth: 120,
      },
      {
        field: "patientName",
        headerName: "Patient",
        flex: 1.1,
        minWidth: 160,
      },
      {
        field: "institutionName",
        headerName: "Institute",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "protocolName",
        headerName: "Protocol",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "pullScheduleName",
        headerName: "Pull Schedule",
        flex: 1,
        minWidth: 160,
        valueFormatter: (value) => value ?? "—",
      },
      {
        field: "testStatus",
        headerName: "Status",
        width: 140,
        renderCell: (params) => {
          const value = params.row.testStatus ?? "none";
          const chipColor = statusColorMap[value] ?? "default";
          const label = statusLabelMap[value] ?? value;
          const seriesKey = params.row.seriesUuid
            ? String(params.row.seriesUuid)
            : params.row.seriesInstanceUid
              ? String(params.row.seriesInstanceUid)
              : null;
          const calculationState = seriesKey
            ? calculationStates[seriesKey]
            : null;
          const localProgress = bulkProgress[params.row.id];

          if (calculationState) {
            const status =
              calculationState.status ?? calculationState.eventType;
            const message =
              calculationState.message ??
              calculationState.error ??
              (status === "completed" ? "Completed" : status);
            return (
              <Tooltip title={message ?? ""}>
                <Chip
                  size='small'
                  color={
                    status === "completed"
                      ? "success"
                      : status === "failed" || status === "error"
                        ? "error"
                        : "info"
                  }
                  label={
                    status === "completed"
                      ? "Done"
                      : status === "failed" || status === "error"
                        ? "Failed"
                        : "Running"
                  }
                />
              </Tooltip>
            );
          }

          if (localProgress?.status === "done") {
            return (
              <Tooltip title='Completed'>
                <Chip
                  size='small'
                  color='success'
                  icon={<CloudDoneRoundedIcon fontSize='small' />}
                  label='Finished'
                />
              </Tooltip>
            );
          }
          if (localProgress?.status === "pending") {
            return (
              <Tooltip title={localProgress.message ?? "Requested"}>
                <Chip
                  size='small'
                  color='info'
                  icon={<CloudDownloadRoundedIcon fontSize='small' />}
                  label='Recovery'
                />
              </Tooltip>
            );
          }
          return (
            <Chip
              size='small'
              label={label}
              color={chipColor}
              variant={chipColor === "default" ? "outlined" : "filled"}
            />
          );
        },
      },
      {
        field: "actions",
        headerName: "Actions",
        width: 160,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params) => {
          const row = params.row;
          const isRecovering = recoveringMap[row.id] ?? false;
          return (
            <Stack direction='row' spacing={0.5} alignItems='center'>
              {!row.hasDicom ? (
                <Tooltip title='Recover DICOM from the selected modality'>
                  <span>
                    <Button
                      size='small'
                      variant='outlined'
                      startIcon={
                        isRecovering ? (
                          <CircularProgress size={14} />
                        ) : (
                          <CloudDownloadRoundedIcon fontSize='small' />
                        )
                      }
                      disabled={isRecovering || !selectedModality}
                      onClick={(event) => {
                        event.stopPropagation();
                        handleRecoverDicom(row);
                      }}>
                      {isRecovering ? "Recovering" : "Pull DICOM"}
                    </Button>
                  </span>
                </Tooltip>
              ) : (
                <Tooltip title='Run test in background'>
                  <span>
                    <IconButton
                      size='small'
                      color='primary'
                      disabled={runningBulk || !row.hasDicom}
                      onClick={(event) => {
                        event.stopPropagation();
                        handleRunSingle(row);
                      }}>
                      <PlayArrowRoundedIcon fontSize='small' />
                    </IconButton>
                  </span>
                </Tooltip>
              )}
            </Stack>
          );
        },
      },
    ];
  }, [
    calculationStates,
    bulkProgress,
    recoveringMap,
    selectedModality,
    runningBulk,
    handleRecoverDicom,
    handleRunSingle,
  ]);

  const paginationModel = useMemo(
    () => ({
      page: Math.max(0, (pagination.page ?? 1) - 1),
      pageSize: pagination.limit ?? 25,
    }),
    [pagination.page, pagination.limit],
  );

  const getRowId = useCallback((row) => {
    const baseId =
      row.series_id ??
      row.series_uuid ??
      row.series_instance_uid ??
      row.seriesId ??
      row.seriesUuid ??
      row.study_id ??
      null;
    if (baseId !== null && baseId !== undefined && baseId !== "") {
      return String(baseId);
    }
    return `${row.patient_name ?? "patient"}-${
      row.latest_analysis_date ?? "na"
    }`;
  }, []);

  const handlePaginationModelChange = useCallback(
    (model) => {
      const sort = sortRef.current[0];
      const sortExtras = sort
        ? { sort_by: sort.field, sort_order: sort.sort ?? "asc" }
        : {};

      if (model.page !== paginationModel.page) {
        actions.changePage(model.page + 1, sortExtras);
      }
      if (model.pageSize !== paginationModel.pageSize) {
        actions.changePageSize(model.pageSize, sortExtras);
      }
    },
    [actions, paginationModel.page, paginationModel.pageSize],
  );

  // `handleRunBulk`, `modalities`, and `loadingModalities` are kept in scope for
  // a future bulk-action toolbar (the page used to surface them inline); they
  // remain unread by the current render, matching the file's pre-split state.

  return (
    <Stack spacing={3}>
      <FiltersPanel
        filters={filters}
        onChange={updateFilter}
        onQuery={handleQuery}
        onReset={resetFilters}
      />

      <DataGrid
        rows={loading ? [] : (normalizedResults ?? [])}
        columns={columns}
        getRowId={getRowId}
        rowCount={pagination.total ?? normalizedResults?.length ?? 0}
        paginationMode='server'
        filterMode='client'
        filterModel={filterModel}
        onFilterModelChange={setFilterModel}
        paginationModel={paginationModel}
        onPaginationModelChange={handlePaginationModelChange}
        sortingMode='server'
        sortModel={sortModel}
        onSortModelChange={handleSortModelChange}
        checkboxSelection
        disableRowSelectionOnClick
        loading={loading}
        pageSizeOptions={[25, 50, 100]}
        slots={{ toolbar: GridToolbar }}
        showToolbar
        initialState={{
          pagination: { paginationModel: paginationModel },
          columns: {
            columnVisibilityModel: {
              latestAnalysis: false,
            },
          },
        }}
        onRowSelectionModelChange={(model) => setSelectionModel(model)}
        rowSelectionModel={selectionModel}
        slotProps={{
          toolbar: {
            showQuickFilter: true,
            quickFilterProps: { debounceMs: 500 },
          },
        }}
        getRowClassName={(params) =>
          params.row.hasDicom ? "" : "missing-dicom"
        }
        sx={{
          "& .missing-dicom": {
            bgcolor: (theme) =>
              theme.palette.mode === "light"
                ? "rgba(255, 214, 0, 0.08)"
                : "rgba(255, 214, 0, 0.16)",
          },
        }}
      />
    </Stack>
  );
};

export default BulkTestsPage;
