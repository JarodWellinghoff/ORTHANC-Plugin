import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Modal,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import GridToolbar from "./GridToolbar";
import { DataGrid } from "@mui/x-data-grid";
import dayjs from "dayjs";
import CloudDoneRoundedIcon from "@mui/icons-material/CloudDoneRounded";
import CloudOffRoundedIcon from "@mui/icons-material/CloudOffRounded";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import ScheduleRoundedIcon from "@mui/icons-material/ScheduleRounded";
import { DateTimePicker } from "@mui/x-date-pickers/DateTimePicker";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";

import FiltersPanel, { FILTER_FIELDS } from "./FiltersPanel";
import { useFilters } from "../../hooks/useFilters";
import { useSnackbar } from "notistack";

const apiBase = import.meta.env.VITE_API_URL;

// ─────────────────────────────────────────────────────────────────────────────
// Constants & helpers
// ─────────────────────────────────────────────────────────────────────────────

// Cap on how many permutations we're willing to fire at a remote modality.
// Remote C-FIND is expensive; explosive cartesian products are a footgun.
const MAX_QUERIES = 25;

// Only the filter fields that correspond to standard DICOM C-FIND matching
// keys show up on this page. pullSchedule is an internal concept that the
// remote server knows nothing about, and age ranges aren't a real C-FIND key
// (they'd require deriving age from PatientBirthDate client-side anyway).
const DICOM_VISIBLE_FIELDS = [
  FILTER_FIELDS.patientId,
  FILTER_FIELDS.patientName,
  FILTER_FIELDS.institute,
  FILTER_FIELDS.protocolName,
  FILTER_FIELDS.scannerModel,
  FILTER_FIELDS.scannerStation,
  FILTER_FIELDS.studyDate,
  FILTER_FIELDS.age,
];

const statusColorMap = {
  pending: "default",
  in_progress: "primary",
  paused: "warning",
  completed: "success",
  failed: "error",
  cancelled: "default",
  expired: "warning",
};

const defaultWindow = {
  start: null,
  end: null,
  displayName: "",
  notes: "",
};

const apiRequest = async (path, options = {}) => {
  const response = await fetch(`${apiBase}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || response.statusText || "Request failed");
  }

  if (response.status === 204) return null;

  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
};

const toIsoOrNull = (value) => {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString();
};

const toArr = (v) => {
  if (v === null || v === undefined || v === "") return [];
  if (Array.isArray(v)) return v;
  return [v];
};

const getSelectionKey = (item) =>
  item.seriesInstanceUID ||
  item.seriesInstanceUid ||
  item.studyInstanceUID ||
  item.studyInstanceUid;

const sumEstimatedSeconds = (items) =>
  items.reduce((acc, item) => acc + (item.estimatedSeconds || 0), 0);

const formatDateTime = (value) => {
  if (!value) return "--";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
};

const formatDuration = (seconds) => {
  if (!seconds || Number.isNaN(seconds)) return "--";
  const rounded = Math.round(seconds);
  if (rounded < 60) return `${rounded}s`;
  const minutes = Math.floor(rounded / 60);
  const rem = rounded % 60;
  if (minutes < 60) return rem ? `${minutes}m ${rem}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return `${hours}h ${remMinutes}m`;
};

const formatStudyDateRange = (start, end) => {
  const s = start ? dayjs(start).format("YYYYMMDD") : "";
  const e = end ? dayjs(end).format("YYYYMMDD") : "";
  if (!s && !e) return "";
  if (s && e) return `${s}-${e}`;
  if (s) return `${s}-`;
  return `-${e}`;
};

/**
 * Builds an array of DICOM C-FIND query objects from the current filter state.
 *
 * DICOM C-FIND doesn't natively support "IN (a, b, c)"-style matching on a
 * key — each match is single-valued (wildcards aside). So when a filter like
 * "Protocol Name" has multiple entries, we take the cartesian product across
 * all multi-valued fields and emit one query per combination, then merge the
 * results on the client. A MAX_QUERIES cap protects us from combinatorial
 * explosions (e.g. 5 patients × 5 protocols × 5 stations = 125 queries).
 *
 * Empty strings are used as C-FIND wildcards rather than omitting the keys
 * entirely — that's the DICOM convention for "return this tag, any value".
 */
const buildQueriesFromFilters = (filters) => {
  const base = {
    PatientID: "",
    PatientName: "",
    PatientAge: "",
    InstitutionName: "",
    StationName: "",
    ProtocolName: "",
    ManufacturerModelName: "",
    StudyDate: "",
    SeriesDate: "",
    SeriesDescription: "",
    Modality: "",
    NumberOfSeriesRelatedInstances: "",
  };

  const studyDate = formatStudyDateRange(
    filters.studyDateStartSearch,
    filters.studyDateEndSearch,
  );
  if (studyDate) base.StudyDate = dayjs(studyDate);

  // Axes of the cartesian product. A missing filter becomes [null] so it
  // still contributes exactly one iteration (with no matching constraint).
  const axes = {
    PatientID: toArr(filters.patientIdSearch),
    PatientName: toArr(filters.patientNameSearch),
    InstitutionName: toArr(filters.instituteSearch),
    StationName: toArr(filters.scannerStationSearch),
    ProtocolName: toArr(filters.protocolNameSearch),
    ManufacturerModelName: toArr(filters.scannerModelSearch),
  };
  const keys = Object.keys(axes);
  const values = keys.map((k) =>
    axes[k].length ? axes[k].map((v) => `*${v}*`) : [null],
  );
  console.log("values:", values);

  const queries = [];
  const combine = (index, acc) => {
    if (queries.length >= MAX_QUERIES) return;
    if (index === keys.length) {
      queries.push({ ...base, ...acc });
      return;
    }
    for (const v of values[index]) {
      if (queries.length >= MAX_QUERIES) return;
      combine(index + 1, {
        ...acc,
        [keys[index]]: v ? String(v).trim() : "",
      });
    }
  };
  combine(0, {});

  return queries.length ? queries : [base];
};

/**
 * Normalize a DICOM query response item to the same row shape that
 * BulkTestsPage feeds into its DataGrid. This keeps the column definitions
 * portable between the two pages.
 */
const normalizeDicomItem = (item) => {
  const seriesInstanceUid =
    item.seriesInstanceUID ?? item.seriesInstanceUid ?? null;
  const studyInstanceUid =
    item.studyInstanceUID ?? item.studyInstanceUid ?? null;

  // Use series UID when present, study UID as fallback for Study-level
  // queries, with a last-ditch synthetic key so the grid always has an id.
  const id =
    seriesInstanceUid ??
    studyInstanceUid ??
    `${item.patientId ?? "row"}-${item.studyDate ?? ""}-${
      item.seriesDate ?? ""
    }`;

  return {
    id,
    studyInstanceUid,
    seriesInstanceUid,
    seriesUuid: null, // remote DICOM items don't have an Orthanc UUID yet
    patientName: item.patientName ?? "N/A",
    patientId: item.patientId ?? "N/A",
    institutionName: item.institutionName ?? item.InstitutionName ?? "N/A",
    protocolName: item.protocolName ?? item.ProtocolName ?? "N/A",
    scannerModel:
      item.manufacturerModelName ?? item.ManufacturerModelName ?? "N/A",
    stationName: item.stationName ?? item.StationName ?? "N/A",
    modality: item.modality ?? "N/A",
    description: item.description ?? "",
    studyDate: item.studyDate ?? null,
    seriesDate: item.seriesDate ?? null,
    numberOfInstances: item.numberOfInstances ?? null,
    estimatedSeconds: item.estimatedSeconds ?? 0,
    age: item.age ?? "N/A",
    hasDicom: false, // these are remote — by definition not local yet
    raw: item,
  };
};

/**
 * Pick out unique values across a list of normalized rows so we can feed
 * them back into FiltersPanel as autocomplete suggestions. C-FIND doesn't
 * give us a list of valid values up front, so the best we can do is learn
 * from results.
 */
const accumulateOptions = (rows, previous = {}) => {
  const add = (set, value) => {
    if (value === null || value === undefined || value === "") return;
    const str = String(value).trim();
    if (str && str !== "N/A") set.add(str);
  };

  const patientIds = new Set(previous.patient_ids ?? []);
  const patientNames = new Set(previous.patient_names ?? []);
  const institutes = new Set(previous.institutes ?? []);
  const protocols = new Set(previous.protocol_names ?? []);
  const models = new Set(previous.scanner_models ?? []);
  const stations = new Set(previous.scanner_stations ?? []);

  for (const row of rows) {
    add(patientIds, row.patientId);
    add(patientNames, row.patientName);
    add(institutes, row.institutionName);
    add(protocols, row.protocolName);
    add(models, row.scannerModel);
    add(stations, row.stationName);
  }

  const sortedArray = (set) => Array.from(set).sort();

  return {
    patient_ids: sortedArray(patientIds),
    patient_names: sortedArray(patientNames),
    institutes: sortedArray(institutes),
    protocol_names: sortedArray(protocols),
    scanner_models: sortedArray(models),
    scanner_stations: sortedArray(stations),
  };
};

// ─────────────────────────────────────────────────────────────────────────────
// Subcomponents
// ─────────────────────────────────────────────────────────────────────────────

const BatchStatusChip = ({ status }) => {
  const color = statusColorMap[status] || "default";
  return <Chip size='small' color={color} label={status.replace("_", " ")} />;
};

const BatchesTable = ({ batches }) => {
  if (!batches.length) {
    return (
      <Typography variant='body2' color='text.secondary' sx={{ py: 4 }}>
        No scheduled pulls yet.
      </Typography>
    );
  }

  return (
    <Table size='small'>
      <TableHead>
        <TableRow>
          <TableCell>Name</TableCell>
          <TableCell>Modality</TableCell>
          <TableCell>Window</TableCell>
          <TableCell>Status</TableCell>
          <TableCell align='right'>Progress</TableCell>
          <TableCell align='right'>Est. Duration</TableCell>
          <TableCell align='right'>Updated</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {batches.map((batch) => {
          const total = batch.total_items || 0;
          const completed = batch.completed_items || 0;
          const failed = batch.failed_items || 0;
          const progressLabel = `${completed}/${total}${
            failed ? ` (${failed} failed)` : ""
          }`;

          return (
            <TableRow key={batch.id} hover>
              <TableCell>
                <Stack spacing={0.5}>
                  <Typography variant='body2' sx={{ fontWeight: 600 }}>
                    {batch.display_name || `Batch #${batch.id}`}
                  </Typography>
                  <Typography variant='caption' color='text.secondary'>
                    ID: {batch.id}
                  </Typography>
                </Stack>
              </TableCell>
              <TableCell>{batch.remote_modality}</TableCell>
              <TableCell>
                <Typography variant='caption' color='text.secondary'>
                  {formatDateTime(batch.start_time)}
                </Typography>
                <Typography variant='caption' color='text.secondary'>
                  {" → "}
                  {formatDateTime(batch.end_time)}
                </Typography>
              </TableCell>
              <TableCell>
                <BatchStatusChip status={batch.status} />
              </TableCell>
              <TableCell align='right'>
                <Typography variant='body2'>{progressLabel}</Typography>
              </TableCell>
              <TableCell align='right'>
                {formatDuration(batch.estimated_total_seconds)}
              </TableCell>
              <TableCell align='right'>
                <Typography variant='caption' color='text.secondary'>
                  {formatDateTime(batch.updated_at)}
                </Typography>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// DicomPullsPage
// ─────────────────────────────────────────────────────────────────────────────

const DicomPullsPage = () => {
  // Local filter state — intentionally not sharing with DashboardContext,
  // since that context's filters drive BulkTestsPage results and conflating
  // the two means one page's filters start steering the other's queries.
  const { filters, updateFilter, resetFilters } = useFilters();
  const { enqueueSnackbar } = useSnackbar();

  const [modalities, setModalities] = useState([]);
  const [selectedModality, setSelectedModality] = useState("");
  const [loadingModalities, setLoadingModalities] = useState(false);

  const [results, setResults] = useState([]);
  const [loadingResults, setLoadingResults] = useState(false);
  const [selectionModel, setSelectionModel] = useState({
    type: "include",
    ids: new Set(),
  });

  // Options discovered from past query results, fed back into FiltersPanel.
  const [dicomFilterOptions, setDicomFilterOptions] = useState({});

  const [schedule, setSchedule] = useState(defaultWindow);
  const [batches, setBatches] = useState([]);
  const [loadingBatches, setLoadingBatches] = useState(false);
  const [creatingBatch, setCreatingBatch] = useState(false);

  // ── Derived data ────────────────────────────────────────────────────────

  // Currently-checked rows in the DataGrid → items we'll actually schedule.
  const selectedItems = useMemo(() => {
    if (!selectionModel?.ids) return [];
    const ids = Array.from(selectionModel.ids);
    return results.filter((row) => ids.includes(row.id)).map((row) => row.raw);
  }, [selectionModel, results]);

  const estimatedSeconds = useMemo(
    () => sumEstimatedSeconds(selectedItems),
    [selectedItems],
  );

  // ── Data loading ────────────────────────────────────────────────────────

  const loadModalities = useCallback(async () => {
    setLoadingModalities(true);
    try {
      const data = await apiRequest("/dicom-modalities");
      const mods = data?.modalities ?? [];
      setModalities(mods);
      // Auto-pick the first server so the page is usable without extra clicks,
      // but only if nothing has been chosen yet.
      setSelectedModality((prev) => prev || (mods.length ? mods[0].id : ""));
    } catch (error) {
      enqueueSnackbar(`Failed to load modalities: ${error.message}`, {
        variant: "error",
      });
    } finally {
      setLoadingModalities(false);
    }
  }, []);

  const loadBatches = useCallback(async () => {
    setLoadingBatches(true);
    try {
      const data = await apiRequest("/dicom-pull/batches");
      setBatches(data?.batches ?? []);
    } catch (error) {
      enqueueSnackbar(`Failed to load scheduled pulls: ${error.message}`, {
        variant: "error",
      });
    } finally {
      setLoadingBatches(false);
    }
  }, []);

  useEffect(() => {
    loadModalities();
    loadBatches();
  }, [loadModalities, loadBatches]);

  // ── Query ───────────────────────────────────────────────────────────────

  const handleRunQuery = useCallback(async () => {
    if (!selectedModality) {
      enqueueSnackbar("Select a server before querying.", {
        variant: "warning",
      });
      return;
    }

    const queries = buildQueriesFromFilters(filters);
    const limit = 50;

    setLoadingResults(true);

    try {
      const responses = await Promise.all(
        queries.map((query) =>
          apiRequest("/dicom-store/query", {
            method: "POST",
            body: JSON.stringify({
              modality: selectedModality,
              level: "Series",
              query,
              limit,
            }),
          }),
        ),
      );

      // Merge and dedupe by series/study UID.
      const seen = new Set();
      const normalized = [];
      for (const data of responses) {
        for (const item of data?.results ?? []) {
          const key = getSelectionKey(item) ?? JSON.stringify(item);
          if (!seen.has(key)) {
            seen.add(key);
            normalized.push(normalizeDicomItem(item));
          }
        }
      }

      setResults(normalized);
      setSelectionModel({ type: "include", ids: new Set() });

      // Feed discovered values back into the filter panel as suggestions.
      setDicomFilterOptions((prev) => accumulateOptions(normalized, prev));

      if (!normalized.length) {
        enqueueSnackbar("No series matched the query.", { variant: "info" });
      } else if (queries.length > 1) {
        enqueueSnackbar(
          `Ran ${queries.length} queries — ${normalized.length} unique series found.`,
          { variant: "info" },
        );
      } else {
        enqueueSnackbar(`Found ${normalized.length} series.`, {
          variant: "success",
        });
      }
    } catch (error) {
      enqueueSnackbar(`Query failed: ${error.message}`, { variant: "error" });
    } finally {
      setLoadingResults(false);
    }
  }, [filters, selectedModality]);

  const handleResetFilters = useCallback(() => {
    resetFilters();
    setResults([]);
    setSelectionModel({ type: "include", ids: new Set() });
  }, [resetFilters]);

  // ── Schedule ────────────────────────────────────────────────────────────

  const handleScheduleChange = (event) => {
    const { name, value } = event.target;
    setSchedule((prev) => ({ ...prev, [name]: value }));
  };

  const handleCreateBatch = async () => {
    if (!selectedModality || !selectedItems.length) {
      enqueueSnackbar("Select a server and at least one series to schedule.", {
        variant: "warning",
      });
      return;
    }

    setCreatingBatch(true);

    try {
      await apiRequest("/dicom-pull/batches", {
        method: "POST",
        body: JSON.stringify({
          modality: selectedModality,
          items: selectedItems,
          start_time: toIsoOrNull(schedule.start),
          end_time: toIsoOrNull(schedule.end),
          display_name: schedule.displayName || null,
          notes: schedule.notes || null,
        }),
      });

      enqueueSnackbar(`Scheduled ${selectedItems.length} series for pull.`, {
        variant: "success",
      });
      setSelectionModel({ type: "include", ids: new Set() });
      await loadBatches();
    } catch (error) {
      enqueueSnackbar(`Failed to create batch: ${error.message}`, {
        variant: "error",
      });
    } finally {
      setCreatingBatch(false);
    }
  };

  // ── Grid config ─────────────────────────────────────────────────────────

  // Mirrors BulkTestsPage's column definitions for consistency. The key
  // difference: no test status / last-analysis columns since remote results
  // haven't been ingested yet.
  const columns = useMemo(
    () => [
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
        field: "scannerModel",
        headerName: "Scanner",
        flex: 1,
        minWidth: 140,
      },
      {
        field: "stationName",
        headerName: "Station",
        flex: 0.8,
        minWidth: 120,
      },
      {
        field: "modality",
        headerName: "Modality",
        width: 100,
      },
      {
        field: "studyDate",
        headerName: "Study Date",
        width: 120,
      },
      {
        field: "age",
        headerName: "Age",
        width: 75,
      },
      {
        field: "numberOfInstances",
        headerName: "Images",
        width: 75,
        type: "number",
        valueFormatter: (value) =>
          value === null || value === undefined ? "—" : value,
      },
      {
        field: "estimatedSeconds",
        headerName: "Est. Pull",
        width: 110,
        valueFormatter: (value) => formatDuration(value),
      },
      {
        field: "hasDicom",
        headerName: "DICOM",
        width: 140,
        renderCell: (params) =>
          params.row.hasDicom ? (
            <Chip
              size='small'
              color='success'
              icon={<CloudDoneRoundedIcon fontSize='small' />}
              label='Available'
              variant='outlined'
            />
          ) : (
            <Chip
              size='small'
              color='warning'
              icon={<CloudOffRoundedIcon fontSize='small' />}
              label='Remote'
              variant='outlined'
            />
          ),
      },
    ],
    [],
  );

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <Stack spacing={3}>
      {/* Modality picker — lives right above the filters so it's clear the
          filters are scoped to whatever server is selected. */}
      <Paper variant='outlined' sx={{ p: 2 }}>
        <Stack
          direction={{ xs: "column", sm: "row" }}
          spacing={2}
          alignItems={{ xs: "stretch", sm: "center" }}>
          <FormControl sx={{ minWidth: 280 }} size='small'>
            <InputLabel id='dicom-pulls-modality-label'>
              DICOM server
            </InputLabel>
            <Select
              labelId='dicom-pulls-modality-label'
              label='DICOM server'
              value={selectedModality}
              onChange={(event) => setSelectedModality(event.target.value)}
              disabled={loadingModalities || modalities.length === 0}>
              {modalities.map((item) => (
                <MenuItem key={item.id} value={item.id}>
                  {item.title || item.aet || item.id}
                  {item.host ? (
                    <Typography
                      component='span'
                      variant='caption'
                      color='text.secondary'
                      sx={{ ml: 1 }}>
                      ({item.host}:{item.port})
                    </Typography>
                  ) : null}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            size='small'
            variant='outlined'
            startIcon={<RefreshRoundedIcon />}
            onClick={loadModalities}
            disabled={loadingModalities}>
            Refresh servers
          </Button>
          {loadingResults && (
            <Stack direction='row' spacing={1} alignItems='center'>
              <CircularProgress size={16} />
              <Typography variant='body2' color='text.secondary'>
                Querying remote server…
              </Typography>
            </Stack>
          )}
        </Stack>
      </Paper>

      {/* Filters — FiltersPanel is driven off local filter state and fed a
          dynamically-discovered options dictionary (since C-FIND doesn't
          expose option lists up front). pullSchedule + age filters are
          hidden since they don't map to real DICOM query keys. */}
      <FiltersPanel
        filters={filters}
        onChange={updateFilter}
        onQuery={handleRunQuery}
        onReset={handleResetFilters}
        filterOptions={dicomFilterOptions}
        visibleFields={DICOM_VISIBLE_FIELDS}
      />

      {/* Results grid — mirrors BulkTestsPage's grid configuration so the
          look and feel is identical. */}
      <Paper variant='outlined' sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Stack
            direction='row'
            alignItems='center'
            justifyContent='space-between'>
            <Typography variant='h6'>Query Results</Typography>
            <Typography variant='body2' color='text.secondary'>
              {results.length} series
              {selectedItems.length > 0
                ? ` · ${selectedItems.length} selected`
                : ""}
            </Typography>
          </Stack>
          <Box sx={{ minHeight: 420 }}>
            <DataGrid
              rows={loadingResults ? [] : results}
              columns={columns}
              getRowId={(row) => row.id}
              loading={loadingResults}
              checkboxSelection
              disableRowSelectionOnClick
              rowSelectionModel={selectionModel}
              onRowSelectionModelChange={(model) => setSelectionModel(model)}
              pageSizeOptions={[25, 50, 100]}
              slots={{ toolbar: GridToolbar }}
              showToolbar
              initialState={{
                pagination: { paginationModel: { pageSize: 25, page: 0 } },
                sorting: {
                  sortModel: [{ field: "patientName", sort: "asc" }],
                },
              }}
              slotProps={{
                toolbar: {
                  showQuickFilter: true,
                  quickFilterProps: { debounceMs: 500 },
                },
              }}
            />
          </Box>
        </Stack>
      </Paper>

      {/* Schedule pull panel — unchanged in behaviour from before, just
          cleaned up visually. */}
      <Paper variant='outlined' sx={{ p: 3 }}>
        <Stack spacing={2}>
          <Typography variant='h6'>Schedule Pull</Typography>
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <Grid container spacing={2}>
              <Grid item size={{ xs: 12, sm: 6, md: 3 }}>
                <TextField
                  fullWidth
                  size='small'
                  label='Schedule Name'
                  name='displayName'
                  value={schedule.displayName}
                  onChange={handleScheduleChange}
                />
              </Grid>
              <Grid item size={{ xs: 12, sm: 6, md: 3 }}>
                <DateTimePicker
                  label='Start'
                  value={schedule.start ? dayjs(schedule.start) : null}
                  onChange={(value) =>
                    setSchedule((prev) => ({ ...prev, start: value }))
                  }
                  slotProps={{ textField: { size: "small", fullWidth: true } }}
                />
              </Grid>
              <Grid item size={{ xs: 12, sm: 6, md: 3 }}>
                <DateTimePicker
                  label='End'
                  value={schedule.end ? dayjs(schedule.end) : null}
                  onChange={(value) =>
                    setSchedule((prev) => ({ ...prev, end: value }))
                  }
                  slotProps={{ textField: { size: "small", fullWidth: true } }}
                />
              </Grid>
              <Grid item size={{ xs: 12, sm: 6, md: 3 }}>
                <TextField
                  fullWidth
                  size='small'
                  label='Notes'
                  name='notes'
                  value={schedule.notes}
                  onChange={handleScheduleChange}
                />
              </Grid>
            </Grid>
          </LocalizationProvider>

          <Stack
            direction='row'
            spacing={2}
            alignItems='center'
            justifyContent='space-between'>
            <Typography variant='body2' color='text.secondary'>
              {selectedItems.length} selected · est.{" "}
              {formatDuration(estimatedSeconds)}
            </Typography>
            <Button
              variant='contained'
              startIcon={
                creatingBatch ? (
                  <CircularProgress size={16} color='inherit' />
                ) : (
                  <ScheduleRoundedIcon />
                )
              }
              onClick={handleCreateBatch}
              disabled={
                creatingBatch || !selectedItems.length || !selectedModality
              }>
              {creatingBatch
                ? "Scheduling…"
                : schedule.start === null
                  ? "Pull now"
                  : "Schedule pull"}
            </Button>
          </Stack>
        </Stack>
      </Paper>

      {/* Scheduled pulls table. */}
      <Paper variant='outlined' sx={{ p: 3 }}>
        <Stack
          direction='row'
          alignItems='center'
          justifyContent='space-between'
          sx={{ mb: 2 }}>
          <Typography variant='h6'>Scheduled pulls</Typography>
          <Button
            size='small'
            variant='outlined'
            startIcon={<RefreshRoundedIcon />}
            onClick={loadBatches}
            disabled={loadingBatches}>
            Refresh
          </Button>
        </Stack>
        <BatchesTable batches={batches} loading={loadingBatches} />
      </Paper>
    </Stack>
  );
};

export default DicomPullsPage;
