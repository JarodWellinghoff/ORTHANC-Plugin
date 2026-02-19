import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Checkbox,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  IconButton,
  InputAdornment,
  InputLabel,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import dayjs from "dayjs";
import Collapse from "@mui/material/Collapse";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import SearchIcon from "@mui/icons-material/Search";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";
import ScheduleRoundedIcon from "@mui/icons-material/ScheduleRounded";
import { DateTimePicker } from "@mui/x-date-pickers/DateTimePicker";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import CustomActionBar from "./CustomActionBar";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";

import { useDashboard } from "../context/DashboardContext";
const apiBase = import.meta.env.VITE_API_URL;

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

  if (response.status === 204) {
    return null;
  }

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

const sanitizeStudyDate = (value) => {
  if (!value) return "";
  return value.replaceAll("-", "");
};

// Normalize a filter value to an array (handles legacy string values from context)
const toArr = (v) => {
  if (!v) return [];
  if (Array.isArray(v)) return v;
  return [v];
};

const getSelectionKey = (item) =>
  item.seriesInstanceUID || item.studyInstanceUID || item.studyInstanceUid;

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
  if (rounded < 60) {
    return `${rounded}s`;
  }
  const minutes = Math.floor(rounded / 60);
  const rem = rounded % 60;
  if (minutes < 60) {
    return rem ? `${minutes}m ${rem}s` : `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return `${hours}h ${remMinutes}m`;
};

/**
 * Builds an array of DICOM query objects from the current filters.
 * When a filter field has multiple values, separate queries are generated
 * for each combination so every selection is queried against the server.
 * The result set is capped at MAX_QUERIES to prevent runaway requests.
 */
const MAX_QUERIES = 25;

const buildQueriesFromFilters = (filters) => {
  const base = {
    PatientName: "",
    InstitutionName: "",
    StationName: "",
    ProtocolName: "",
    ManufacturerModelName: "",
    StudyDate: "",
    SeriesDescription: "",
  };

  // Study date range
  if (filters.studyDateStart || filters.studyDateEnd) {
    const start = filters.studyDateStart
      ? dayjs(filters.studyDateStart).format("YYYYMMDD")
      : "";
    const end = filters.studyDateEnd
      ? dayjs(filters.studyDateEnd).format("YYYYMMDD")
      : "";
    if (start && end) base.StudyDate = `${start}-${end}`;
    else if (start) base.StudyDate = `${start}-`;
    else base.StudyDate = `-${end}`;
  }

  // Multi-value filter axes
  const patientValues = toArr(filters.patientSearch);
  const instituteValues = toArr(filters.institute);
  const stationValues = toArr(filters.scannerStation);
  const protocolValues = toArr(filters.protocolName);
  const modelValues = toArr(filters.scannerModel);

  // Use sentinel [null] so a missing filter still participates in the product
  const axes = [
    patientValues.length ? patientValues : [null],
    instituteValues.length ? instituteValues : [null],
    stationValues.length ? stationValues : [null],
    protocolValues.length ? protocolValues : [null],
    modelValues.length ? modelValues : [null],
  ];

  const queries = [];

  for (const patient of axes[0]) {
    for (const institute of axes[1]) {
      for (const station of axes[2]) {
        for (const protocol of axes[3]) {
          for (const model of axes[4]) {
            if (queries.length >= MAX_QUERIES) break;
            const q = {
              ...base,
              PatientName: patient ? patient.trim() : "",
              InstitutionName: institute ? institute.trim() : "",
              StationName: station ? station.trim() : "",
              ProtocolName: protocol ? protocol.trim() : "",
              ManufacturerModelName: model ? model.trim() : "",
            };
            queries.push(q);
          }
          if (queries.length >= MAX_QUERIES) break;
        }
        if (queries.length >= MAX_QUERIES) break;
      }
      if (queries.length >= MAX_QUERIES) break;
    }
    if (queries.length >= MAX_QUERIES) break;
  }

  return queries.length ? queries : [base];
};

const BatchStatusChip = ({ status }) => {
  const color = statusColorMap[status] || "default";
  return <Chip size='small' color={color} label={status.replace("_", " ")} />;
};

const ResultsList = ({
  results,
  loading,
  selectedMap,
  onToggle,
  onSelectAll,
  disabled,
}) => {
  if (loading) {
    return (
      <Stack alignItems='center' justifyContent='center' sx={{ py: 6, gap: 2 }}>
        <CircularProgress size={32} />
        <Typography variant='body2' color='text.secondary'>
          Querying remote server…
        </Typography>
      </Stack>
    );
  }

  if (!results.length) {
    return (
      <Typography variant='body2' color='text.secondary' sx={{ py: 4 }}>
        No results yet. Submit a query to see available series.
      </Typography>
    );
  }

  const allSelected = results.every(
    (item) => !!selectedMap[getSelectionKey(item)],
  );

  return (
    <Box>
      <Stack direction='row' alignItems='center' justifyContent='space-between'>
        <Typography variant='subtitle2'>
          {results.length} series found
        </Typography>
        <Button size='small' onClick={onSelectAll} disabled={disabled}>
          {allSelected ? "Clear selection" : "Select all"}
        </Button>
      </Stack>
      <List dense>
        {results.map((item) => {
          const key = getSelectionKey(item);
          const checked = !!selectedMap[key];
          console.log("Rendering item", key, { item });
          return (
            <ListItem
              key={key}
              disablePadding
              secondaryAction={
                <Checkbox
                  edge='end'
                  checked={checked}
                  disabled={disabled}
                  onChange={() => onToggle(item)}
                />
              }>
              <ListItemIcon sx={{ minWidth: 32 }}>
                <CloudDownloadRoundedIcon fontSize='small' />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography variant='body2' sx={{ fontWeight: 600 }}>
                    {item.description || "Unnamed Series"}
                  </Typography>
                }
                secondary={
                  <Typography variant='caption' color='text.secondary'>
                    {item.patientName || "Unknown patient"} ·{" "}
                    {item.studyInstanceUID}
                    {item.seriesInstanceUID
                      ? ` · ${item.seriesInstanceUID}`
                      : ""}
                  </Typography>
                }
              />
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
};

const BatchesTable = ({ batches, loading, onRefresh }) => {
  if (loading) {
    return (
      <Stack alignItems='center' sx={{ py: 6 }}>
        <CircularProgress size={32} />
      </Stack>
    );
  }

  if (!batches.length) {
    return (
      <Stack alignItems='center' spacing={2} sx={{ py: 4 }}>
        <Typography variant='body2' color='text.secondary'>
          No scheduled pulls yet.
        </Typography>
        <Button
          variant='outlined'
          size='small'
          startIcon={<RefreshRoundedIcon />}
          onClick={onRefresh}>
          Refresh
        </Button>
      </Stack>
    );
  }

  return (
    <Table size='small'>
      <TableHead>
        <TableRow>
          <TableCell>Name</TableCell>
          <TableCell>Server</TableCell>
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

const DicomPullsPage = () => {
  const { filters, filterOptions, advancedFiltersOpen, actions } =
    useDashboard();
  const [modalities, setModalities] = useState([]);
  const [schedule, setSchedule] = useState(defaultWindow);
  const [selectedModality, setSelectedModality] = useState("");
  const [results, setResults] = useState([]);
  const [selectedMap, setSelectedMap] = useState({});
  const [batches, setBatches] = useState([]);
  const [statusMessage, setStatusMessage] = useState(null);

  const [loadingModalities, setLoadingModalities] = useState(false);
  const [loadingResults, setLoadingResults] = useState(false);
  const [loadingBatches, setLoadingBatches] = useState(false);
  const [creatingBatch, setCreatingBatch] = useState(false);

  const { date_range: dateRange, age_range: ageRange } = filterOptions;

  const examDateFromValue = useMemo(() => {
    if (!filters.examDateFrom) return null;
    const parsed = dayjs(filters.examDateFrom);
    return parsed.isValid() ? parsed : null;
  }, [filters.examDateFrom]);

  const minExamDate = useMemo(() => {
    if (!dateRange?.min) return undefined;
    const parsed = dayjs(dateRange.min);
    return parsed.isValid() ? parsed : undefined;
  }, [dateRange]);

  const maxExamDate = useMemo(() => {
    if (!dateRange?.max) return undefined;
    const parsed = dayjs(dateRange.max);
    return parsed.isValid() ? parsed : undefined;
  }, [dateRange]);

  const examDateToValue = useMemo(() => {
    if (!filters.examDateTo) return null;
    const parsed = dayjs(filters.examDateTo);
    return parsed.isValid() ? parsed : null;
  }, [filters.examDateTo]);

  const selectedItems = useMemo(
    () => Object.values(selectedMap),
    [selectedMap],
  );
  const estimatedSeconds = useMemo(
    () => sumEstimatedSeconds(selectedItems),
    [selectedItems],
  );

  const loadModalities = useCallback(async () => {
    setLoadingModalities(true);
    try {
      const data = await apiRequest("/dicom-modalities");
      const mods = data?.modalities ?? [];
      setModalities(mods);
      if (!selectedModality && mods.length) {
        setSelectedModality(mods[0].id);
      }
    } catch (error) {
      setStatusMessage({
        severity: "error",
        text: `Failed to load modalities: ${error.message}`,
      });
    } finally {
      setLoadingModalities(false);
    }
  }, [selectedModality]);

  const loadBatches = useCallback(async () => {
    setLoadingBatches(true);
    try {
      const data = await apiRequest("/dicom-pull/batches");
      setBatches(data?.batches ?? []);
    } catch (error) {
      setStatusMessage({
        severity: "error",
        text: `Failed to load scheduled pulls: ${error.message}`,
      });
    } finally {
      setLoadingBatches(false);
    }
  }, []);

  useEffect(() => {
    loadModalities();
    loadBatches();
    // Load filter options so Autocomplete dropdowns are populated
    actions.loadFilterOptions?.();
  }, [loadModalities, loadBatches]);

  // Generic handler for Autocomplete multi-select fields
  const handleAutocompleteChange = (name) => (_event, newValue) => {
    actions.updateFilter(name, newValue);
  };

  const handleScheduleChange = (event) => {
    const { name, value } = event.target;
    setSchedule((prev) => ({ ...prev, [name]: value }));
  };

  const handleStartDatePickerChange = (value) => {
    setSchedule((prev) => ({ ...prev, start: value }));
  };

  const handleEndDatePickerChange = (value) => {
    setSchedule((prev) => ({ ...prev, end: value }));
  };

  const handleStudyDateChange = (name) => (value) => {
    actions.updateFilter(name, value);
  };

  const handleRunQuery = async () => {
    if (!selectedModality) {
      setStatusMessage({
        severity: "warning",
        text: "Select a server before querying.",
      });
      return;
    }

    const queries = buildQueriesFromFilters(filters);
    const limit = 50;

    setLoadingResults(true);
    setStatusMessage(null);

    try {
      // Run all queries in parallel and merge results
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

      // Merge and deduplicate by series instance UID
      const seen = new Set();
      const merged = [];
      for (const data of responses) {
        for (const item of data?.results ?? []) {
          const key = getSelectionKey(item);
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(item);
          }
        }
      }

      setResults(merged);
      setSelectedMap({});

      if (!merged.length) {
        setStatusMessage({
          severity: "info",
          text: "No series matched the query.",
        });
      } else if (queries.length > 1) {
        setStatusMessage({
          severity: "info",
          text: `Ran ${queries.length} queries — ${merged.length} unique series found.`,
        });
      }
    } catch (error) {
      setStatusMessage({
        severity: "error",
        text: `Query failed: ${error.message}`,
      });
    } finally {
      setLoadingResults(false);
    }
  };

  const handleToggleSelection = (item) => {
    const key = getSelectionKey(item);
    setSelectedMap((prev) => {
      const next = { ...prev };
      if (next[key]) {
        delete next[key];
      } else {
        next[key] = item;
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    if (!results.length) return;
    const allSelected = results.every(
      (item) => !!selectedMap[getSelectionKey(item)],
    );
    if (allSelected) {
      setSelectedMap({});
      return;
    }
    const next = {};
    results.forEach((item) => {
      next[getSelectionKey(item)] = item;
    });
    setSelectedMap(next);
  };

  const handleCreateBatch = async () => {
    if (!selectedModality || !selectedItems.length) {
      setStatusMessage({
        severity: "warning",
        text: "Select a server and at least one series to schedule.",
      });
      return;
    }

    setCreatingBatch(true);
    setStatusMessage(null);

    try {
      const startIso = toIsoOrNull(schedule.start);
      const endIso = toIsoOrNull(schedule.end);

      await apiRequest("/dicom-pull/batches", {
        method: "POST",
        body: JSON.stringify({
          modality: selectedModality,
          items: selectedItems,
          start_time: startIso,
          end_time: endIso,
          display_name: schedule.displayName || null,
          notes: schedule.notes || null,
        }),
      });

      setStatusMessage({
        severity: "success",
        text: `Scheduled ${selectedItems.length} series for pull.`,
      });
      setSelectedMap({});
      await loadBatches();
    } catch (error) {
      setStatusMessage({
        severity: "error",
        text: `Failed to create batch: ${error.message}`,
      });
    } finally {
      setCreatingBatch(false);
    }
  };

  // Normalize filter values to arrays for Autocomplete
  const patientSearchValue = toArr(filters.patientSearch);
  const instituteValue = toArr(filters.institute);
  const scannerStationValue = toArr(filters.scannerStation);
  const protocolNameValue = toArr(filters.protocolName);
  const scannerModelValue = toArr(filters.scannerModel);

  const pickerButtonSx = {
    "&&": {
      border: "none",
      boxShadow: "none",
      bgcolor: "transparent",
      width: 36,
      height: 36,
      p: 0.5,
      "&:hover": { bgcolor: "transparent", borderColor: "transparent" },
      "&:active": { bgcolor: "transparent" },
      "& .MuiSvgIcon-root": { fontSize: 18 },
    },
  };

  const pickerSlotProps = {
    actionBar: { actions: ["clear", "today", "accept"] },
    openPickerButton: { sx: pickerButtonSx },
    textField: { placeholder: "", InputLabelProps: { shrink: true } },
  };

  return (
    <Stack spacing={3}>
      {statusMessage ? (
        <Alert
          severity={statusMessage.severity}
          onClose={() => setStatusMessage(null)}>
          {statusMessage.text}
        </Alert>
      ) : null}

      <Stack spacing={3}>
        <Grid item size={{ xs: 12, md: 6 }}>
          <Paper elevation={0} variant='outlined' sx={{ p: 3 }}>
            <Stack spacing={2}>
              <Typography variant='h6'>Remote Query</Typography>

              <Stack spacing={3}>
                {/* ── Primary search + server selector ── */}
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={2}
                  alignItems={{ xs: "stretch", md: "center" }}>
                  <Autocomplete
                    multiple
                    freeSolo
                    filterSelectedOptions
                    options={[]}
                    value={patientSearchValue}
                    onChange={handleAutocompleteChange("patientSearch")}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip
                          size='small'
                          label={option}
                          {...getTagProps({ index })}
                        />
                      ))
                    }
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        placeholder={
                          patientSearchValue.length
                            ? ""
                            : "Search by patient name or ID"
                        }
                        slotProps={{
                          input: {
                            ...params.InputProps,
                            startAdornment: (
                              <>
                                <InputAdornment position='start'>
                                  <SearchIcon fontSize='small' />
                                </InputAdornment>
                                {params.InputProps.startAdornment}
                              </>
                            ),
                          },
                        }}
                      />
                    )}
                    fullWidth
                  />

                  <FormControl size='small'>
                    <InputLabel id='modality-label'>Server</InputLabel>
                    <Select
                      labelId='modality-label'
                      label='Server'
                      value={selectedModality}
                      sx={{ minWidth: 150 }}
                      onChange={(event) =>
                        setSelectedModality(event.target.value)
                      }
                      disabled={loadingModalities}>
                      {modalities.map((item) => (
                        <MenuItem key={item.id} value={item.id}>
                          {item.title || item.id}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Stack>

                {/* ── Secondary filters ── */}
                <Grid container spacing={2} columns={15}>
                  <Grid size={{ sm: 5, md: 3 }}>
                    <Autocomplete
                      multiple
                      freeSolo
                      filterSelectedOptions
                      options={filterOptions.institutes ?? []}
                      value={instituteValue}
                      onChange={handleAutocompleteChange("institute")}
                      renderTags={(value, getTagProps) =>
                        value.map((option, index) => (
                          <Chip
                            size='small'
                            label={option}
                            {...getTagProps({ index })}
                          />
                        ))
                      }
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label='Institute'
                          placeholder={instituteValue.length ? "" : "Any"}
                        />
                      )}
                      fullWidth
                    />
                  </Grid>

                  <Grid size={{ sm: 5, md: 3 }}>
                    <Autocomplete
                      multiple
                      freeSolo
                      filterSelectedOptions
                      options={filterOptions.scanner_stations ?? []}
                      value={scannerStationValue}
                      onChange={handleAutocompleteChange("scannerStation")}
                      renderTags={(value, getTagProps) =>
                        value.map((option, index) => (
                          <Chip
                            size='small'
                            label={option}
                            {...getTagProps({ index })}
                          />
                        ))
                      }
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label='Scanner Name'
                          placeholder={scannerStationValue.length ? "" : "Any"}
                        />
                      )}
                      fullWidth
                    />
                  </Grid>

                  <Grid size={{ sm: 5, md: 3 }}>
                    <Autocomplete
                      multiple
                      freeSolo
                      filterSelectedOptions
                      options={filterOptions.protocol_names ?? []}
                      value={protocolNameValue}
                      onChange={handleAutocompleteChange("protocolName")}
                      renderTags={(value, getTagProps) =>
                        value.map((option, index) => (
                          <Chip
                            size='small'
                            label={option}
                            {...getTagProps({ index })}
                          />
                        ))
                      }
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label='Protocol Name'
                          placeholder={protocolNameValue.length ? "" : "Any"}
                        />
                      )}
                      fullWidth
                    />
                  </Grid>

                  <Grid size={{ sm: 5, md: 3 }}>
                    <Autocomplete
                      multiple
                      freeSolo
                      filterSelectedOptions
                      options={filterOptions.scanner_models ?? []}
                      value={scannerModelValue}
                      onChange={handleAutocompleteChange("scannerModel")}
                      renderTags={(value, getTagProps) =>
                        value.map((option, index) => (
                          <Chip
                            size='small'
                            label={option}
                            {...getTagProps({ index })}
                          />
                        ))
                      }
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label='Scanner Model'
                          placeholder={scannerModelValue.length ? "" : "Any"}
                        />
                      )}
                      fullWidth
                    />
                  </Grid>

                  {/* ── Study date range (DatePickers — kept as-is) ── */}
                  <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <Grid size={{ sm: 5, md: 3 }}>
                      <DatePicker
                        enableAccessibleFieldDOMStructure={false}
                        slots={{ textField: TextField }}
                        slotProps={pickerSlotProps}
                        label='Study Date Start'
                        name='studyDateStart'
                        value={filters.studyDateStart ?? null}
                        onChange={handleStudyDateChange("studyDateStart")}
                        sx={{ width: "100%" }}
                      />
                    </Grid>
                    <Grid size={{ sm: 5, md: 3 }}>
                      <DatePicker
                        enableAccessibleFieldDOMStructure={false}
                        slots={{ textField: TextField }}
                        slotProps={pickerSlotProps}
                        label='Study Date End'
                        name='studyDateEnd'
                        value={filters.studyDateEnd ?? null}
                        onChange={handleStudyDateChange("studyDateEnd")}
                        sx={{ width: "100%" }}
                      />
                    </Grid>
                  </LocalizationProvider>
                </Grid>
              </Stack>

              <Box>
                <Button
                  variant='contained'
                  startIcon={<SearchRoundedIcon />}
                  onClick={handleRunQuery}
                  disabled={
                    loadingResults || loadingModalities || !selectedModality
                  }>
                  Run query
                </Button>
              </Box>
              <Divider />
              <ResultsList
                results={results}
                loading={loadingResults}
                selectedMap={selectedMap}
                onToggle={handleToggleSelection}
                onSelectAll={handleSelectAll}
                disabled={creatingBatch}
              />
            </Stack>
          </Paper>
        </Grid>

        <Grid item size={{ xs: 12, md: 6 }}>
          <Paper elevation={0} variant='outlined' sx={{ p: 3 }}>
            <Stack spacing={2}>
              <Typography variant='h6'>Schedule Pull</Typography>
              <Grid container spacing={2}>
                <Grid item size={{ xs: 12, sm: 3 }}>
                  <TextField
                    fullWidth
                    label='Schedule Name'
                    name='displayName'
                    value={schedule.displayName}
                    onChange={handleScheduleChange}
                    placeholder='Optional label for this batch'
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 3 }}>
                  <TextField
                    fullWidth
                    label='Notes'
                    name='notes'
                    value={schedule.notes}
                    onChange={handleScheduleChange}
                    placeholder='Optional context or reminders'
                    multiline
                    minRows={2}
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 3 }}>
                  <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <DateTimePicker
                      enableAccessibleFieldDOMStructure={false}
                      slots={{ textField: TextField }}
                      slotProps={{
                        actionBar: { actions: ["clear", "today", "accept"] },
                        openPickerButton: { sx: pickerButtonSx },
                        textField: {
                          placeholder: "",
                          InputLabelProps: { shrink: true },
                        },
                      }}
                      label='Start Time'
                      value={schedule.start}
                      onChange={handleStartDatePickerChange}
                      sx={{ width: "100%" }}
                    />
                  </LocalizationProvider>
                </Grid>
                <Grid item size={{ xs: 12, sm: 3 }}>
                  <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <DateTimePicker
                      enableAccessibleFieldDOMStructure={false}
                      slots={{ textField: TextField }}
                      slotProps={{
                        actionBar: { actions: ["clear", "today", "accept"] },
                        openPickerButton: { sx: pickerButtonSx },
                        textField: {
                          placeholder: "",
                          InputLabelProps: { shrink: true },
                        },
                      }}
                      label='End Time'
                      value={schedule.end}
                      onChange={handleEndDatePickerChange}
                      sx={{ width: "100%" }}
                    />
                  </LocalizationProvider>
                </Grid>
              </Grid>

              {selectedItems.length > 0 && (
                <Typography variant='body2' color='text.secondary'>
                  {selectedItems.length} series selected · est.{" "}
                  {formatDuration(estimatedSeconds)}
                </Typography>
              )}

              <Stack direction='row' spacing={2}>
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
        </Grid>
      </Stack>

      <Paper elevation={0} variant='outlined' sx={{ p: 3 }}>
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
        <BatchesTable
          batches={batches}
          loading={loadingBatches}
          onRefresh={loadBatches}
        />
      </Paper>
    </Stack>
  );
};

export default DicomPullsPage;
