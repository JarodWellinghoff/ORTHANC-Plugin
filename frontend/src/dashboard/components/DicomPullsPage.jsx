import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  IconButton,
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
  Autocomplete,
  Tooltip,
  Typography,
} from "@mui/material";
import dayjs from "dayjs";
import Collapse from "@mui/material/Collapse";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import SearchIcon from "@mui/icons-material/Search";
import InputAdornment from "@mui/material/InputAdornment";
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

const defaultFilters = {
  patientId: "",
  studyDate: null,
  seriesDescription: "",
  seriesInstanceUid: "",
  limit: 50,
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

const buildQueryFromFilters = (filters) => {
  const query = {};
  if (filters.patientId) {
    query.PatientID = filters.patientId.trim();
  }
  if (filters.studyDate) {
    query.StudyDate = sanitizeStudyDate(filters.studyDate);
  }
  if (filters.seriesDescription) {
    query.SeriesDescription = filters.seriesDescription.trim();
  }
  if (filters.seriesInstanceUid) {
    query.SeriesInstanceUID = filters.seriesInstanceUid.trim();
  }
  return query;
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
                    {item.seriesDescription || "Unnamed Series"}
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
  //   const [filters, setFilters] = useState(defaultFilters);
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
    if (!filters.examDateFrom) {
      return null;
    }
    const parsed = dayjs(filters.examDateFrom);
    return parsed.isValid() ? parsed : null;
  }, [filters.examDateFrom]);
  const minExamDate = useMemo(() => {
    if (!dateRange?.min) {
      return undefined;
    }
    const parsed = dayjs(dateRange.min);
    return parsed.isValid() ? parsed : undefined;
  }, [dateRange]);

  const maxExamDate = useMemo(() => {
    if (!dateRange?.max) {
      return undefined;
    }
    const parsed = dayjs(dateRange.max);
    return parsed.isValid() ? parsed : undefined;
  }, [dateRange]);
  const examDateToValue = useMemo(() => {
    if (!filters.examDateTo) {
      return null;
    }
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
  }, [loadModalities, loadBatches]);

  const handleFilterChange = (event) => {
    const { name, value } = event.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
  };

  const handleScheduleChange = (event) => {
    console.log("event:", event);
    console.log("event.target:", event.target);
    const { name, value } = event.target;
    setSchedule((prev) => ({ ...prev, [name]: value }));
  };

  const handleStartDatePickerChange = (value) => {
    setSchedule((prev) => ({ ...prev, start: value }));
  };

  const handleEndDatePickerChange = (value) => {
    setSchedule((prev) => ({ ...prev, end: value }));
  };

  const handleStudyDateChange = (value) => {
    setFilters((prev) => ({ ...prev, studyDate: value }));
  };

  const handleRunQuery = async () => {
    if (!selectedModality) {
      setStatusMessage({
        severity: "warning",
        text: "Select a server before querying.",
      });
      return;
    }

    const query = buildQueryFromFilters(filters);
    setLoadingResults(true);
    setStatusMessage(null);
    try {
      const payload = {
        modality: selectedModality,
        level: "Series",
        query,
        limit: Number(filters.limit) || 50,
      };
      const data = await apiRequest("/dicom-store/query", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const series = data?.results ?? [];
      setResults(series);
      setSelectedMap({});
      if (!series.length) {
        setStatusMessage({
          severity: "info",
          text: "No series matched the query.",
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
      const payload = {
        modality: selectedModality,
        startTime: schedule.start,
        endTime: schedule.end,
        displayName: schedule.displayName || undefined,
        notes: schedule.notes || undefined,
        items: selectedItems.map((item) => ({
          studyInstanceUID: item.studyInstanceUID || item.studyInstanceUid,
          seriesInstanceUID: item.seriesInstanceUID,
          patientId: item.patientId,
          patientName: item.patientName,
          description: item.seriesDescription || item.description,
          modality: item.modality,
          bodyPart: item.bodyPart,
          studyDate: item.studyDate,
          seriesDate: item.seriesDate,
          numberOfInstances: item.numberOfInstances,
          estimatedSeconds: item.estimatedSeconds,
        })),
      };

      const data = await apiRequest("/dicom-pull/batches", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      setStatusMessage({
        severity: "success",
        text: `Scheduled pull batch #${data.id} successfully.`,
      });
      setSelectedMap({});
      setSchedule(defaultWindow);
      loadBatches();
    } catch (error) {
      setStatusMessage({
        severity: "error",
        text: `Failed to create pull batch: ${error.message}`,
      });
    } finally {
      setCreatingBatch(false);
    }
  };

  const disableScheduling =
    creatingBatch || !selectedItems.length || !schedule.start || !schedule.end;

  return (
    <Stack spacing={3}>
      <Stack direction='row' justifyContent='space-between' alignItems='center'>
        <Typography variant='h4' component='h2'>
          DICOM Pull Scheduler
        </Typography>
        <Stack direction='row' spacing={1}>
          <Tooltip title='Refresh scheduled pulls'>
            <Button
              variant='outlined'
              size='small'
              startIcon={<RefreshRoundedIcon />}
              onClick={loadBatches}
              disabled={loadingBatches}>
              Refresh pulls
            </Button>
          </Tooltip>
          <Tooltip title='Reload modalities'>
            <Button
              variant='outlined'
              size='small'
              onClick={loadModalities}
              disabled={loadingModalities}>
              Modalities
            </Button>
          </Tooltip>
        </Stack>
      </Stack>

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
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={2}
                  alignItems={{ xs: "stretch", md: "center" }}>
                  <TextField
                    name='patientSearch'
                    value={filters.patientSearch}
                    //   onChange={handleChange}
                    placeholder='Search by patient name or ID'
                    // sx={{
                    //   width: "50%",
                    // }}
                    fullWidth
                    slotProps={{
                      input: {
                        startAdornment: (
                          <InputAdornment position='start'>
                            <SearchIcon fontSize='small' />
                          </InputAdornment>
                        ),
                      },
                    }}
                  />
                  <FormControl size='small'>
                    <InputLabel id='modality-label'>Server</InputLabel>
                    <Select
                      labelId='modality-label'
                      label='Server'
                      value={selectedModality}
                      sx={{
                        minWidth: 150,
                      }}
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

                <Grid container spacing={2} columns={15}>
                  <Grid size={{ sm: 5, md: 3 }}>
                    <TextField
                      name='institute'
                      value={filters.institute}
                      //   onChange={handleChange}
                      placeholder='Institute'
                      fullWidth
                    />
                  </Grid>
                  {/* <FormControl sx={{ m: 1, width: 300 }}>
                      <InputLabel id='institute-label'>Institute</InputLabel>
                      <Select
                        labelId='institute-label'
                        name='institute'
                        value={filters.institute}
                        // onChange={handleChange}
                        fullWidth>
                        {filterOptions.institutes?.map((option) => (
                          <MenuItem key={option} value={option}>
                            {option}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl> */}
                  <Grid size={{ sm: 5, md: 3 }}>
                    <TextField
                      name='scannerStation'
                      value={filters.scannerStation}
                      //   onChange={handleChange}
                      placeholder='Scanner Name'
                      fullWidth
                    />
                  </Grid>
                  {/* <FormControl sx={{ m: 1, width: 300 }}>
                      <InputLabel id='scannerStation-label'>
                        Scanner Name
                      </InputLabel>
                      <Select
                        labelId='scannerStation-label'
                        name='scannerStation'
                        value={filters.scannerStation}
                        // onChange={handleChange}
                        fullWidth>
                        {filterOptions.scanner_stations?.map((option) => (
                          <MenuItem key={option} value={option}>
                            {option}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl> */}
                  <Grid size={{ sm: 5, md: 3 }}>
                    <TextField
                      name='protocolName'
                      value={filters.protocolName}
                      //   onChange={handleChange}
                      placeholder='Protocol Name'
                      fullWidth
                    />
                  </Grid>
                  {/* <FormControl sx={{ m: 1, width: 300 }}>
                      <InputLabel id='protocolName-label'>
                        Protocol Name
                      </InputLabel>
                      <Select
                        labelId='protocolName-label'
                        name='protocolName'
                        value={filters.protocolName}
                        // onChange={handleChange}
                        fullWidth>
                        {filterOptions.protocol_names?.map((option) => (
                          <MenuItem key={option} value={option}>
                            {option}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl> */}
                  <Grid size={{ sm: 5, md: 3 }}>
                    <TextField
                      name='scannerModel'
                      value={filters.scannerModel}
                      //   onChange={handleChange}
                      placeholder='Scanner Model'
                      fullWidth
                    />
                  </Grid>
                  {/* <FormControl sx={{ m: 1, width: 300 }}>
                      <InputLabel id='scannerModel-label'>
                        Scanner Model
                      </InputLabel>
                      <Select
                        labelId='scannerModel-label'
                        name='scannerModel'
                        value={filters.scannerModel}
                        // onChange={handleChange}
                        fullWidth>
                        {filterOptions.scanner_models?.map((option) => (
                          <MenuItem key={option} value={option}>
                            {option}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl> */}
                  <Grid size={{ sm: 5, md: 3 }}>
                    <TextField
                      name='seriesDescription'
                      value={filters.seriesDescription}
                      //   onChange={handleChange}
                      placeholder='Series Description'
                      fullWidth
                    />
                    <Autocomplete
                      multiple
                      id='seriesDescription'
                      freeSolo
                      options={[]}
                      defaultValue={[]}
                      renderValue={(value, getItemProps) =>
                        value.map((option, index) => {
                          const { key, ...itemProps } = getItemProps({ index });
                          return (
                            <Chip
                              variant='outlined'
                              label={option}
                              key={key}
                              {...itemProps}
                            />
                          );
                        })
                      }
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          label='Series Description'
                          placeholder='Series Description'
                        />
                      )}
                    />
                  </Grid>
                </Grid>
                <Grid container spacing={2} columns={15}>
                  <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <Grid size={{ sm: 5, md: 3 }}>
                      <DatePicker
                        enableAccessibleFieldDOMStructure={false}
                        slots={{
                          textField: TextField,
                        }}
                        slotProps={{
                          actionBar: {
                            actions: ["clear", "today", "accept"],
                          },
                          openPickerButton: {
                            sx: {
                              "&&": {
                                border: "none",
                                boxShadow: "none",
                                bgcolor: "transparent",
                                width: 36,
                                height: 36,
                                p: 0.5,
                                "&:hover": {
                                  bgcolor: "transparent",
                                  borderColor: "transparent",
                                },
                                "&:active": { bgcolor: "transparent" },
                                "& .MuiSvgIcon-root": { fontSize: 18 },
                              },
                            },
                          },
                          textField: {
                            placeholder: "",
                            InputLabelProps: { shrink: true },
                          },
                        }}
                        label='Study Date Start'
                        name='studyDateStart'
                        value={filters.studyDateStart}
                        onChange={handleStudyDateChange}
                        sx={{
                          width: "100%",
                        }}
                      />
                    </Grid>
                    <Grid size={{ sm: 5, md: 3 }}>
                      <DatePicker
                        enableAccessibleFieldDOMStructure={false}
                        slots={{
                          textField: TextField,
                        }}
                        slotProps={{
                          actionBar: {
                            actions: ["clear", "today", "accept"],
                          },
                          openPickerButton: {
                            sx: {
                              "&&": {
                                border: "none",
                                boxShadow: "none",
                                bgcolor: "transparent",
                                width: 36,
                                height: 36,
                                p: 0.5,
                                "&:hover": {
                                  bgcolor: "transparent",
                                  borderColor: "transparent",
                                },
                                "&:active": { bgcolor: "transparent" },
                                "& .MuiSvgIcon-root": { fontSize: 18 },
                              },
                            },
                          },
                          textField: {
                            placeholder: "",
                            InputLabelProps: { shrink: true },
                          },
                        }}
                        label='Study Date End'
                        name='studyDateEnd'
                        value={filters.studyDateEnd}
                        onChange={handleStudyDateChange}
                        sx={{
                          width: "100%",
                        }}
                      />
                    </Grid>
                  </LocalizationProvider>
                </Grid>
                {/* <Grid item size={{ xs: 12, md: 6 }}>
                      <Stack
                        direction={{ xs: "column", sm: "row" }}
                        spacing={2}
                        sx={{ width: "100%" }}>
                        <FormControl sx={{ m: 1, width: 300 }}>
                          <InputLabel id='ageMin-label'>Age Min</InputLabel>
                          <TextField
                            labelId='ageMin-label'
                            name='ageMin'
                            type='number'
                            value={filters.ageMin}
                            // onChange={handleChange}
                            slotProps={{
                              input: {
                                min: ageRange?.min ?? 0,
                                max: ageRange?.max ?? 150,
                                endAdornment: (
                                  <InputAdornment position='end'>
                                    years
                                  </InputAdornment>
                                ),
                              },
                            }}
                          />
                        </FormControl>
                        <FormControl sx={{ m: 1, width: 300 }}>
                          <InputLabel id='ageMax-label'>Age Max</InputLabel>
                          <TextField
                            labelId='ageMax-label'
                            name='ageMax'
                            type='number'
                            value={filters.ageMax}
                            // onChange={handleChange}
                            slotProps={{
                              input: {
                                min: ageRange?.min ?? 0,
                                max: ageRange?.max ?? 150,
                                endAdornment: (
                                  <InputAdornment position='end'>
                                    years
                                  </InputAdornment>
                                ),
                              },
                            }}
                          />
                        </FormControl>
                      </Stack>
                    </Grid> */}

                {/* {activeFilters.length > 0 && (
                    <Stack
                      direction='row'
                      spacing={1}
                      flexWrap='wrap'
                      alignItems='center'>
                      <Typography variant='body2' color='text.secondary'>
                        Active Filters:
                      </Typography>
                      {activeFilters.map((filter) => (
                        <Chip
                          key={filter.key}
                          label={filter.label}
                          onDelete={() => clearFilterValue(filter.key)}
                          size='small'
                        />
                      ))}
                    </Stack>
                  )} */}
              </Stack>

              {/* <Grid container spacing={2}>
                <Grid item size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    size='small'
                    label='Patient ID'
                    name='patientId'
                    value={filters.patientId}
                    onChange={handleFilterChange}
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 6 }}>
                  <DatePicker
                    enableAccessibleFieldDOMStructure={false}
                    slots={{
                      textField: TextField,
                    }}
                    slotProps={{
                      actionBar: {
                        actions: ["clear", "today", "accept"],
                      },
                      openPickerButton: {
                        sx: {
                          "&&": {
                            border: "none",
                            boxShadow: "none",
                            bgcolor: "transparent",
                            width: 36,
                            height: 36,
                            p: 0.5,
                            "&:hover": {
                              bgcolor: "transparent",
                              borderColor: "transparent",
                            },
                            "&:active": { bgcolor: "transparent" },
                            "& .MuiSvgIcon-root": { fontSize: 18 },
                          },
                        },
                      },
                    }}
                    label='Study Date'
                    name='studyDate'
                    value={filters.studyDate}
                    onChange={handleStudyDateChange}
                    sx={{
                      width: "100%",
                    }}
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    size='small'
                    label='Series Description'
                    name='seriesDescription'
                    value={filters.seriesDescription}
                    onChange={handleFilterChange}
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    size='small'
                    label='Series Instance UID'
                    name='seriesInstanceUid'
                    value={filters.seriesInstanceUid}
                    onChange={handleFilterChange}
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    size='small'
                    label='Max results'
                    name='limit'
                    type='number'
                    inputProps={{ min: 1, max: 200 }}
                    value={filters.limit}
                    onChange={handleFilterChange}
                  />
                </Grid>
              </Grid> */}
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
                    // size='small'
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
                    // size='small'
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
                  <DateTimePicker
                    enableAccessibleFieldDOMStructure={false}
                    slots={{
                      textField: TextField,
                    }}
                    slotProps={{
                      actionBar: {
                        actions: ["clear", "today", "accept"],
                      },
                      openPickerButton: {
                        sx: {
                          "&&": {
                            border: "none",
                            boxShadow: "none",
                            bgcolor: "transparent",
                            width: 36,
                            height: 36,
                            p: 0.5,
                            "&:hover": {
                              bgcolor: "transparent",
                              borderColor: "transparent",
                            },
                            "&:active": { bgcolor: "transparent" },
                            "& .MuiSvgIcon-root": { fontSize: 18 },
                          },
                        },
                      },
                      textField: {
                        placeholder: "Leave empty to start now",
                        InputLabelProps: { shrink: true },
                      },
                    }}
                    label='Start time'
                    name='start'
                    value={schedule.start}
                    onChange={handleStartDatePickerChange}
                    sx={{
                      width: "100%",
                    }}
                  />
                </Grid>
                <Grid item size={{ xs: 12, sm: 3 }}>
                  <DateTimePicker
                    enableAccessibleFieldDOMStructure={false}
                    slots={{
                      textField: TextField,
                      actionBar: CustomActionBar,
                    }}
                    slotProps={{
                      actionBar: {
                        actions: ["today", "clear", "cancel", "accept"],
                        add12Label: "+12 hours",
                        onAdd12Hours: () =>
                          handleEndDatePickerChange(
                            schedule.start.add(12, "hour"),
                          ),
                      },
                      openPickerButton: {
                        sx: {
                          "&&": {
                            border: "none",
                            boxShadow: "none",
                            bgcolor: "transparent",
                            width: 36,
                            height: 36,
                            p: 0.5,
                            "&:hover": {
                              bgcolor: "transparent",
                              borderColor: "transparent",
                            },
                            "&:active": { bgcolor: "transparent" },
                            "& .MuiSvgIcon-root": { fontSize: 18 },
                          },
                        },
                      },
                      textField: {
                        placeholder: "Leave empty to run indefinitely",
                        InputLabelProps: { shrink: true },
                      },
                    }}
                    disablePast
                    label='End time'
                    name='end'
                    value={schedule.end}
                    minDate={schedule.start}
                    onChange={handleEndDatePickerChange}
                    sx={{
                      width: "100%",
                    }}
                  />
                  {/* <TextField
                    fullWidth
                    size='small'
                    type='datetime-local'
                    label='End time'
                    name='end'
                    value={schedule.end}
                    onChange={handleScheduleChange}
                    slotProps={{ inputLabel: { shrink: true } }}
                  /> */}
                </Grid>
              </Grid>
              <Paper
                variant='outlined'
                sx={{ p: 2, bgcolor: "background.default" }}>
                <Stack spacing={1}>
                  <Typography variant='subtitle2'>Selection summary</Typography>
                  <Typography variant='body2' color='text.secondary'>
                    {selectedItems.length} series selected · Estimated transfer
                    time {formatDuration(estimatedSeconds)}
                  </Typography>
                </Stack>
              </Paper>
              <Button
                variant='contained'
                startIcon={<ScheduleRoundedIcon />}
                // disabled={disableScheduling}
                onClick={handleCreateBatch}>
                {creatingBatch
                  ? "Scheduling…"
                  : schedule.start === null
                    ? "Pull now"
                    : "Schedule pull"}
              </Button>
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
