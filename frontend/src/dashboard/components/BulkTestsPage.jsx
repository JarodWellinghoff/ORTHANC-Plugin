import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import FiltersPanel from "./FiltersPanel";
import { useNavigate } from "react-router-dom";
import ContentPasteSearchIcon from "@mui/icons-material/ContentPasteSearch";
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import CloudOffRoundedIcon from "@mui/icons-material/CloudOffRounded";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import ErrorOutlineRoundedIcon from "@mui/icons-material/ErrorOutlineRounded";
import ScheduleRoundedIcon from "@mui/icons-material/ScheduleRounded";
import { useDashboard } from "../context/DashboardContext";

const apiBase = import.meta.env.VITE_API_URL;

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

// CHANGED: added pullScheduleName to defaultFilters
const defaultFilters = {
  patient: "",
  protocol: "",
  pullScheduleName: "",
  status: "all",
  dicom: "all",
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
  const { summary, calculationStates, actions } = useDashboard();
  const { items, pagination } = summary;
  console.log("items:", items);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState(defaultFilters);
  const [availableSeries, setAvailableSeries] = useState([]);
  const [modalities, setModalities] = useState([]);
  const [modalitiesError, setModalitiesError] = useState(null);
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
    setModalitiesError(null);
    try {
      const data = await fetchJson("/dicom-modalities");
      const list = Array.isArray(data?.modalities) ? data.modalities : [];
      setModalities(list);
      if (!selectedModality && list.length > 0) {
        setSelectedModality(list[0].id);
      }
    } catch (err) {
      console.error("Failed to load modalities", err);
      setModalitiesError(err.message);
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
    setError(null);
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
      setError(err.message);
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

  const filteredRows = useMemo(() => {
    return normalizedResults.filter((row) => {
      if (
        filters.patient &&
        !row.patientName.toLowerCase().includes(filters.patient.toLowerCase())
      ) {
        return false;
      }
      if (
        filters.protocol &&
        !row.protocolName.toLowerCase().includes(filters.protocol.toLowerCase())
      ) {
        return false;
      }
      // CHANGED: filter by pull schedule name
      if (
        filters.pullScheduleName &&
        !String(row.pullScheduleName ?? "")
          .toLowerCase()
          .includes(filters.pullScheduleName.toLowerCase())
      ) {
        return false;
      }
      if (filters.status !== "all" && row.testStatus !== filters.status) {
        return false;
      }
      if (filters.dicom === "available" && !row.hasDicom) {
        return false;
      }
      if (filters.dicom === "missing" && row.hasDicom) {
        return false;
      }
      return true;
    });
  }, [normalizedResults, filters]);

  const statusOptions = useMemo(() => {
    const unique = new Set(
      normalizedResults
        .map((row) => row.testStatus)
        .filter((value) => value && value !== "none"),
    );
    return Array.from(unique);
  }, [normalizedResults]);

  const handleFilterChange = (field) => (event) => {
    const value = event?.target?.value ?? "";
    setFilters((prev) => ({ ...prev, [field]: value }));
  };

  const handleResetFilters = () => {
    setFilters(defaultFilters);
  };

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
        setError("Select at least one row to start bulk testing.");
        return;
      }

      selectedIds = selectionModel.ids.intersection(rowsById);
      if (selectedIds.size === 0) {
        setError(
          "Selected rows are no longer available in the current data set.",
        );
        return;
      }
    } else if (type === "exclude") {
      if (selectionModel.ids.size === rowsById.size) {
        setError("Select at least one row to start bulk testing.");
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
      setError("No valid selections remain to process.");
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
          setError(err.message);
        }
      } finally {
        setActiveRunCount((count) => Math.max(0, count - 1));
      }
    },
    [runAnalysisForSeries, updateBulkProgress, waitForSeriesCompletion],
  );

  const handleRecoverDicom = useCallback(
    async (row) => {
      if (!selectedModality) {
        setError("Select a server to use for DICOM recovery.");
        return;
      }
      if (!row.seriesInstanceUid) {
        setError("Series Instance UID is not available for this entry.");
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
    [selectedModality, updateBulkProgress, loadAvailableSeries],
  );

  const columns = useMemo(() => {
    return [
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
      // CHANGED: added Pull Schedule Name column
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
        width: 200,
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
    runningBulk,
    selectedModality,
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
      <Stack direction='row' justifyContent='space-between' alignItems='center'>
        <Typography variant='h4'>Bulk Test Runner</Typography>
        <Stack direction='row' spacing={1}>
          <Button
            variant='outlined'
            startIcon={<RefreshRoundedIcon />}
            onClick={() => {
              loadResults();
              loadAvailableSeries();
            }}
            disabled={loading}>
            Refresh
          </Button>
        </Stack>
      </Stack>

      <FiltersPanel
        actionItems={[
          "advancedFilters",
          "clearFilters",
          "refresh",
          <FormControl size='small' sx={{ minWidth: 200 }}>
            <InputLabel id='modality-label'>
              Preferred Server{loadingModalities ? " (loading…)" : ""}
            </InputLabel>
            <Select
              labelId='modality-label'
              label='Preferred Modality'
              value={selectedModality}
              onChange={(event) => setSelectedModality(event.target.value)}
              disabled={loadingModalities || modalities.length === 0}>
              {modalities.map((item) => (
                <MenuItem key={item.id} value={item.id}>
                  {item.title ?? item.id} ({item.aet ?? "AET"})
                </MenuItem>
              ))}
            </Select>
          </FormControl>,
          "exportCsv",
          <Tooltip
            title={
              numSelected
                ? `Run ${numSelected} selected tests`
                : "Select rows to start bulk testing"
            }>
            <span>
              <Button
                variant='contained'
                sx={{ whiteSpace: "nowrap" }}
                startIcon={
                  runningBulk ? (
                    <CircularProgress size={18} />
                  ) : (
                    <PlayArrowRoundedIcon />
                  )
                }
                disabled={runningBulk || numSelected === 0}
                onClick={handleRunBulk}>
                {runningBulk ? "Starting..." : `Run ${numSelected || ""} Tests`}
              </Button>
            </span>
          </Tooltip>,
        ]}
      />
      {error && (
        <Alert severity='error' onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {modalitiesError && (
        <Alert severity='warning' onClose={() => setModalitiesError(null)}>
          Failed to load modalities: {modalitiesError}
        </Alert>
      )}

      <Paper variant='outlined' sx={{ p: 2 }}>
        <Stack
          spacing={2}
          direction={{ xs: "column", md: "row" }}
          alignItems={{ xs: "stretch", md: "center" }}
          justifyContent='space-between'
          sx={{ mb: 2 }}>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={2}
            flexWrap='wrap'>
            <TextField
              value={filters.patient}
              onChange={handleFilterChange("patient")}
              label='Patient'
              size='small'
            />
            <TextField
              value={filters.protocol}
              onChange={handleFilterChange("protocol")}
              label='Protocol'
              size='small'
            />
            {/* CHANGED: Pull Schedule Name filter field */}
            <TextField
              value={filters.pullScheduleName}
              onChange={handleFilterChange("pullScheduleName")}
              label='Pull Schedule Name'
              size='small'
            />
            <FormControl size='small' sx={{ minWidth: 140 }}>
              <InputLabel id='status-filter-label'>Status</InputLabel>
              <Select
                labelId='status-filter-label'
                label='Status'
                value={filters.status}
                onChange={handleFilterChange("status")}>
                <MenuItem value='all'>All</MenuItem>
                {statusOptions.map((status) => (
                  <MenuItem key={status} value={status}>
                    {statusLabelMap[status] ?? status}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size='small' sx={{ minWidth: 160 }}>
              <InputLabel id='dicom-filter-label'>DICOM</InputLabel>
              <Select
                labelId='dicom-filter-label'
                label='DICOM'
                value={filters.dicom}
                onChange={handleFilterChange("dicom")}>
                <MenuItem value='all'>Show All</MenuItem>
                <MenuItem value='available'>Only Available</MenuItem>
                <MenuItem value='missing'>Only Missing</MenuItem>
              </Select>
            </FormControl>
          </Stack>
          <Button variant='text' onClick={handleResetFilters}>
            Reset filters
          </Button>
        </Stack>

        <Box sx={{ width: "100%" }}>
          {console.log("Filtered Rows:", filteredRows)}
          <DataGrid
            rows={loading ? [] : (filteredRows ?? [])}
            columns={columns}
            getRowId={getRowId}
            rowCount={pagination.total ?? filteredRows?.length ?? 0}
            paginationMode='server'
            paginationModel={paginationModel}
            onPaginationModelChange={handlePaginationModelChange}
            checkboxSelection
            disableRowSelectionOnClick
            loading={loading}
            pageSizeOptions={[25, 50, 100]}
            initialState={{
              pagination: { paginationModel: paginationModel },
              sorting: {
                sortModel: [{ field: "latestAnalysis", sort: "desc" }],
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
        </Box>
      </Paper>
    </Stack>
  );
};

export default BulkTestsPage;
