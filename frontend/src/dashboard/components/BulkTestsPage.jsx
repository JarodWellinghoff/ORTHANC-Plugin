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
import ContentPasteSearchIcon from "@mui/icons-material/ContentPasteSearch";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import CloudOffRoundedIcon from "@mui/icons-material/CloudOffRounded";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import { useNavigate } from "react-router-dom";
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

const apiBase = import.meta.env.VITE_API_URL;
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
const defaultChoParams = {
  resamples: 500,
  internalNoise: 2.25,
  resamplingMethod: "Bootstrap",
  roiSize: 6,
  thresholdLow: 0,
  thresholdHigh: 150,
  windowLength: 15,
  stepSize: 5,
  channelType: "Gabor",
  lesionSet: "standard",
};

const statusColorMap = {
  full: "success",
  partial: "warning",
  error: "error",
  none: "default",
  pending: "default",
};

const statusLabelMap = {
  full: "Full",
  partial: "Global Noise",
  error: "Error",
  none: "Unknown",
  pending: "Pending",
};

const fetchJson = async (path, options = {}) => {
  const response = await fetch(`${apiBase}${path}`, {
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    ...options,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || response.statusText || "Request failed");
  }
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const formatDateTime = (value) => {
  if (!value) return "--";
  try {
    return new Date(value).toLocaleString();
  } catch (error) {
    console.debug("Failed to format date", error);
    return value;
  }
};

const deriveRowId = (item, index) => {
  return (
    item.series_uuid ??
    item.series_id ??
    item.series_instance_uid ??
    item.seriesId ??
    item.study_id ??
    item.study_instance_uid ??
    `${item.patient_id ?? "row"}-${index}`
  );
};

const resolveSeriesKey = (row) => {
  if (!row) return null;
  if (row.seriesUuid) return String(row.seriesUuid);
  if (row.seriesInstanceUid) return String(row.seriesInstanceUid);
  if (row.studyInstanceUid) return String(row.studyInstanceUid);
  if (row.raw?.series_id) return String(row.raw.series_id);
  if (row.raw?.series_uuid) return String(row.raw.series_uuid);
  return null;
};

const BulkTestsPage = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [filterModel, setFilterModel] = useState({ items: [] });
  const { summary, calculationStates, actions } = useDashboard();

  const handleQuery = () => actions.loadSummary(filters);
  const { filters, updateFilter, resetFilters } = useFilters();
  //   console.log(actions.loadSummary(filters));
  const { items, pagination } = summary;
  //   const { items, pagination } = {
  //     items: [],
  //     pagination: { page: 1, totalPages: 1, totalItems: 0 },
  //   };
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [availableSeries, setAvailableSeries] = useState([]);
  const [modalities, setModalities] = useState([]);
  const [selectedModality, setSelectedModality] = useState("");
  const [loadingModalities, setLoadingModalities] = useState(false);
  const [selectionModel, setSelectionModel] = useState({
    type: "include",
    ids: new Set(),
  });
  const [testType, setTestType] = useState("full");
  const [bulkProgress, setBulkProgress] = useState({});
  const [runningBulk, setRunningBulk] = useState(false);
  const [recoveringMap, setRecoveringMap] = useState({});
  const [activeRunCount, setActiveRunCount] = useState(0);
  const calculationStatesRef = useRef(calculationStates);
  useEffect(() => {
    calculationStatesRef.current = calculationStates;
  }, [calculationStates]);

  const shouldWarnOnLeave = runningBulk || activeRunCount > 0;

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
  }, [selectedModality]);

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

  const loadResults = useCallback(async (overrides = {}) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: String(overrides.limit ?? 250),
        page: String(overrides.page ?? 1),
      });
      if (overrides.patientSearch) {
        params.append("patient_search", overrides.patientSearch);
      }
      if (overrides.protocolName) {
        params.append("protocol_name", overrides.protocolName);
      }
      if (overrides.status && overrides.status !== "all") {
        params.append("test_status", overrides.status);
      }

      const response = await fetchJson(`/cho-results?${params.toString()}`);
      const items = Array.isArray(response)
        ? response
        : Array.isArray(response?.data)
          ? response.data
          : [];
      setResults(items);
    } catch (err) {
      console.error("Failed to load results", err);
      enqueueSnackbar("Failed to load results", {
        variant: "error",
      });
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadResults();
    loadAvailableSeries();
    loadModalities();
  }, [loadResults, loadAvailableSeries, loadModalities]);

  const availableSet = useMemo(
    () => new Set(availableSeries.filter(Boolean)),
    [availableSeries],
  );
  const normalizedResults = useMemo(() => {
    return items.map((item, index) => {
      const id = deriveRowId(item, index);
      const seriesUuid =
        item.series_uuid ?? item.seriesUuid ?? item.series_id ?? null;
      const seriesInstanceUid =
        item.series_id ??
        item.series_instance_uid ??
        item.seriesInstanceUid ??
        null;
      const studyInstanceUid =
        item.study_id ??
        item.study_instance_uid ??
        item.studyInstanceUid ??
        null;
      const statusRaw = (item.test_status ?? "").toLowerCase();
      const status =
        statusRaw === "full" || statusRaw === "partial" || statusRaw === "error"
          ? statusRaw
          : statusRaw || "none";
      const hasDicom =
        Boolean(seriesUuid) && availableSet.has(String(seriesUuid));
      console.log(
        "Row",
        id,
        "has DICOM:",
        hasDicom,
        "seriesUuid:",
        seriesUuid,
        "availableSet:",
        availableSet,
      );

      return {
        id,
        raw: item,
        seriesUuid,
        seriesInstanceUid,
        studyInstanceUid,
        patientName: item.patient_name ?? "N/A",
        institutionName: item.institution_name ?? "N/A",
        protocolName: item.protocol_name ?? "N/A",
        scannerModel: item.scanner_model ?? "N/A",
        stationName: item.station_name ?? "N/A",
        latestAnalysis: item.latest_analysis_date ?? null,
        testStatus: status,
        hasDicom,
        // CHANGED: map pull_schedule_name from the API response
        pullScheduleName: item.pull_schedule_name ?? null,
      };
    });
  }, [items, availableSet]);

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

  const numSelected = useMemo(() => {
    const type = selectionModel.type;
    if (type === "include") {
      return selectionModel.ids.size;
    } else if (type === "exclude") {
      return normalizedResults.length;
    }
  }, [selectionModel, normalizedResults]);

  const handleRunBulk = async () => {
    setActiveRunCount((count) => count + 1);
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

      selectedIds = selectionModel.ids.intersection(rowsById);
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
      selectedIds = new Set(rowsById.keys()).difference(selectionModel.ids);
    }

    const queue = [];
    const enqueueRow = (rowId) => {
      if (rowsById.has(rowId)) {
        queue.push(rowsById.get(rowId));
      }
    };
    if (selectedIds) {
      if (Array.isArray(selectedIds)) {
        selectedIds.forEach(enqueueRow);
      } else if (typeof selectedIds.forEach === "function") {
        selectedIds.forEach(enqueueRow);
      } else if (typeof selectedIds[Symbol.iterator] === "function") {
        for (const value of selectedIds) {
          enqueueRow(value);
        }
      }
    }
    if (queue.length === 0) {
      enqueueSnackbar("No valid selections remain to process.", {
        variant: "error",
      });
      return;
    }

    setRunningBulk(true);
    for (const row of queue) {
      await handleRunSingle(row);
    }
    setRunningBulk(false);
    setActiveRunCount((count) => Math.max(0, count - 1));
  };

  const handleRunSingle = useCallback(
    async (row) => {
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
        field: "scannerModel",
        headerName: "Scanner Model",
        minWidth: 190,
        flex: 0.8,
      },
      {
        field: "stationName",
        headerName: "Station Name",
        minWidth: 190,
        flex: 0.8,
      },
      {
        field: "latestAnalysis",
        headerName: "Last Analysis",
        minWidth: 190,
        flex: 0.8,
        valueFormatter: (value) => formatDateTime(value),
      },
      {
        field: "hasDicom",
        headerName: "DICOM",
        width: 140,
        renderCell: (params) => {
          if (params.row.hasDicom) {
            return (
              <Chip
                size='small'
                color='success'
                icon={<CloudDoneRoundedIcon fontSize='small' />}
                label='Available'
                variant='outlined'
              />
            );
          }
          return (
            <Chip
              size='small'
              color='warning'
              icon={<CloudOffRoundedIcon fontSize='small' />}
              label='Missing'
              variant='outlined'
            />
          );
        },
      },
      {
        field: "actions",
        headerName: "Actions",
        width: 140,
        sortable: false,
        filterable: false,
        renderCell: (params) => {
          const row = params.row;
          const { seriesInstanceUid } = row;
          const isRecovering = recoveringMap[row.id] ?? false;
          return (
            <Stack direction='row' spacing={1} alignItems='center'>
              {!row.hasDicom ? (
                <Tooltip
                  title={
                    selectedModality
                      ? `Pull from server ${selectedModality}`
                      : "Select a server first"
                  }>
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
                <Tooltip title='Run analysis with current settings'>
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

              <Tooltip title='View results for this series'>
                <span>
                  <IconButton
                    size='small'
                    color='primary'
                    disabled={runningBulk || !row.hasDicom}
                    onClick={(event) => {
                      event.stopPropagation();
                      navigate(
                        `/results/${encodeURIComponent(seriesInstanceUid)}`,
                      );
                    }}>
                    <ContentPasteSearchIcon fontSize='small' />
                  </IconButton>
                </span>
              </Tooltip>
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
    navigate,
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
      if (model.page !== paginationModel.page) {
        actions.changePage(model.page + 1);
      }
      if (model.pageSize !== paginationModel.pageSize) {
        actions.changePageSize(model.pageSize);
      }
    },
    [actions, paginationModel.page, paginationModel.pageSize],
  );

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
        checkboxSelection
        disableRowSelectionOnClick
        loading={loading}
        pageSizeOptions={[25, 50, 100]}
        slots={{ toolbar: GridToolbar }}
        showToolbar
        initialState={{
          pagination: { paginationModel: paginationModel },
          sorting: {
            sortModel: [{ field: "patientName", sort: "asc" }],
          },
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
      {/* </Paper> */}
    </Stack>
  );
};

export default BulkTestsPage;
