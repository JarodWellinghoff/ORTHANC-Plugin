import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import { useNavigate } from "react-router-dom";

import { API_BASE_URL } from "./dicom/cornerstoneConfig";

const statusColorMap = {
  full: "success",
  partial: "warning",
  error: "error",
};

const statusLabelMap = {
  full: "Full",
  partial: "Global Noise",
  error: "Error",
};

const formatDateTime = (value) => {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
};

const fetchJson = async (path, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText || "Request failed");
  }
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
};

const deriveRowId = (item, index) => {
  return (
    item.series_uuid ??
    item.series_id ??
    item.series_instance_uid ??
    item.seriesId ??
    item.study_instance_uid ??
    `${item.patient_name ?? "row"}-${index}`
  );
};

const DicomViewerPage = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);
  const [loadingResults, setLoadingResults] = useState(false);
  const [availableSeries, setAvailableSeries] = useState([]);
  const [loadingAvailability, setLoadingAvailability] = useState(false);
  const [modalities, setModalities] = useState([]);
  const [selectedModality, setSelectedModality] = useState("");
  const [loadingModalities, setLoadingModalities] = useState(false);
  const [recoveringMap, setRecoveringMap] = useState({});
  const [filters, setFilters] = useState({ patient: "", protocol: "" });
  const [banner, setBanner] = useState(null);
  const [error, setError] = useState(null);

  const loadResults = useCallback(async () => {
    setLoadingResults(true);
    setError(null);
    try {
      const response = await fetchJson("/cho-results?limit=250");
      const items = Array.isArray(response)
        ? response
        : Array.isArray(response?.data)
        ? response.data
        : [];
      setResults(items);
    } catch (err) {
      console.error("Failed to load CHO results", err);
      setError(err.message ?? "Failed to load series list.");
      setResults([]);
    } finally {
      setLoadingResults(false);
    }
  }, []);

  const loadAvailableSeries = useCallback(async () => {
    setLoadingAvailability(true);
    try {
      const data = await fetchJson("/series/");
      if (Array.isArray(data)) {
        setAvailableSeries(data);
      } else {
        setAvailableSeries([]);
      }
    } catch (err) {
      console.error("Failed to load Orthanc series", err);
    } finally {
      setLoadingAvailability(false);
    }
  }, []);

  const loadModalities = useCallback(async () => {
    setLoadingModalities(true);
    try {
      const data = await fetchJson("/dicom-modalities");
      const entries = Array.isArray(data?.modalities) ? data.modalities : [];
      setModalities(entries);
      if (!selectedModality && entries.length > 0) {
        setSelectedModality(entries[0].id);
      }
    } catch (err) {
      console.error("Failed to load modalities", err);
    } finally {
      setLoadingModalities(false);
    }
  }, [selectedModality]);

  useEffect(() => {
    loadResults();
    loadAvailableSeries();
    loadModalities();
  }, [loadResults, loadAvailableSeries, loadModalities]);

  const availableSet = useMemo(
    () => new Set(availableSeries.filter(Boolean).map((value) => String(value))),
    [availableSeries]
  );

  const normalizedRows = useMemo(() => {
    return results.map((item, index) => {
      const id = String(deriveRowId(item, index));
      const seriesUuid =
        item.series_uuid ?? item.seriesUuid ?? item.series_id ?? null;
      const seriesInstanceUid =
        item.series_id ??
        item.series_instance_uid ??
        item.seriesInstanceUid ??
        null;
      const studyInstanceUid =
        item.study_id ?? item.study_instance_uid ?? item.studyInstanceUid ?? null;
      const status = String(item.test_status ?? "none").toLowerCase();
      const hasDicom =
        Boolean(seriesUuid) && availableSet.has(String(seriesUuid));

      return {
        id,
        raw: item,
        seriesUuid,
        seriesInstanceUid,
        studyInstanceUid,
        patientName: item.patient_name ?? "N/A",
        protocolName: item.protocol_name ?? "N/A",
        modality: item.modality ?? "N/A",
        latestAnalysis: item.latest_analysis_date ?? null,
        status,
        hasDicom,
      };
    });
  }, [results, availableSet]);

  const filteredRows = useMemo(() => {
    const patientFilter = filters.patient.trim().toLowerCase();
    const protocolFilter = filters.protocol.trim().toLowerCase();

    return normalizedRows.filter((row) => {
      const matchesPatient =
        !patientFilter ||
        String(row.patientName ?? "")
          .toLowerCase()
          .includes(patientFilter);
      const matchesProtocol =
        !protocolFilter ||
        String(row.protocolName ?? "")
          .toLowerCase()
          .includes(protocolFilter);
      return matchesPatient && matchesProtocol;
    });
  }, [normalizedRows, filters.patient, filters.protocol]);

  const handleRefresh = async () => {
    await Promise.all([loadResults(), loadAvailableSeries()]);
  };

  const handleRecover = useCallback(
    async (row) => {
      if (!selectedModality) {
        setBanner({
          severity: "warning",
          message: "Select a modality before requesting a DICOM pull.",
        });
        return;
      }
      if (!row.seriesInstanceUid) {
        setBanner({
          severity: "error",
          message: "Series Instance UID is required for DICOM recovery.",
        });
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
            patientId: row.raw?.patient_id ?? row.raw?.patientId ?? null,
          }),
        });
        setBanner({
          severity: "info",
          message: "Recovery requested. The series will appear once Orthanc receives it.",
        });
        await loadAvailableSeries();
      } catch (err) {
        console.error("Failed to request DICOM recovery", err);
        setBanner({
          severity: "error",
          message: err.message ?? "Failed to request DICOM recovery.",
        });
      } finally {
        setRecoveringMap((prev) => ({ ...prev, [row.id]: false }));
      }
    },
    [selectedModality, loadAvailableSeries]
  );

  const handleRowNavigation = useCallback(
    (row) => {
      if (!row) return;
      const targetId =
        row.seriesInstanceUid ?? row.seriesUuid ?? row.id ?? null;
      if (!targetId) {
        setBanner({
          severity: "warning",
          message: "Unable to locate a valid identifier for this series.",
        });
        return;
      }
      navigate(`/dicom-viewer/${encodeURIComponent(String(targetId))}`, {
        state: {
          row,
          meta: {
            seriesUuid: row.seriesUuid ?? null,
            seriesInstanceUid: row.seriesInstanceUid ?? null,
            studyInstanceUid: row.studyInstanceUid ?? null,
            patientId:
              row.raw?.patient_id ?? row.raw?.patientId ?? row.patientId ?? null,
          },
        },
      });
    },
    [navigate]
  );

  const columns = useMemo(
    () => [
      {
        field: "patientName",
        headerName: "Patient",
        flex: 1.2,
        minWidth: 180,
      },
      {
        field: "protocolName",
        headerName: "Protocol",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "status",
        headerName: "Status",
        width: 140,
        renderCell: (params) => {
          const value = params.value ?? "none";
          const color = statusColorMap[value] ?? "default";
          const label = statusLabelMap[value] ?? value;
          return (
            <Chip
              size='small'
              color={color}
              variant={color === "default" ? "outlined" : "filled"}
              label={label}
            />
          );
        },
      },
      {
        field: "latestAnalysis",
        headerName: "Updated",
        flex: 1,
        minWidth: 160,
        valueFormatter: (value) => formatDateTime(value),
      },
      {
        field: "hasDicom",
        headerName: "DICOM",
        width: 120,
        renderCell: (params) => {
          const hasDicom = params.value;
          return (
            <Chip
              size='small'
              color={hasDicom ? "success" : "default"}
              variant={hasDicom ? "filled" : "outlined"}
              label={hasDicom ? "Available" : "Missing"}
            />
          );
        },
      },
      {
        field: "actions",
        headerName: "Actions",
        width: 150,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params) => {
          const row = params.row;
          const isRecovering = recoveringMap[row.id] ?? false;
          return (
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
                handleRecover(row);
              }}>
              {isRecovering ? "Recovering" : "Pull DICOM"}
            </Button>
          );
        },
      },
    ],
    [handleRecover, recoveringMap, selectedModality]
  );

  return (
    <Stack spacing={3}>
      {banner && (
        <Alert
          severity={banner.severity}
          onClose={() => setBanner(null)}
          sx={{ mb: 1 }}>
          {banner.message}
        </Alert>
      )}
      {error && (
        <Alert severity='error' onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Stack direction='row' spacing={2} justifyContent='space-between'>
        <Typography variant='h5'>Orthanc DICOM Viewer</Typography>
        <Button
          size='small'
          variant='contained'
          startIcon={<RefreshRoundedIcon />}
          onClick={handleRefresh}
          disabled={loadingResults || loadingAvailability}>
          Refresh
        </Button>
      </Stack>

      <Paper elevation={0} variant='outlined' sx={{ p: 2 }}>
        <Stack spacing={2}>
          <Typography variant='subtitle1'>Series List</Typography>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <TextField
              label='Patient filter'
              value={filters.patient}
              onChange={(event) =>
                setFilters((prev) => ({
                  ...prev,
                  patient: event.target.value,
                }))
              }
              size='small'
              fullWidth
            />
            <TextField
              label='Protocol filter'
              value={filters.protocol}
              onChange={(event) =>
                setFilters((prev) => ({
                  ...prev,
                  protocol: event.target.value,
                }))
              }
              size='small'
              fullWidth
            />
          </Stack>
          <FormControl size='small' sx={{ maxWidth: 320 }}>
            <InputLabel id='modality-label'>Recovery modality</InputLabel>
            <Select
              labelId='modality-label'
              label='Recovery modality'
              value={selectedModality}
              onChange={(event) => setSelectedModality(event.target.value)}
              disabled={loadingModalities || modalities.length === 0}>
              {modalities.map((item) => (
                <MenuItem key={item.id} value={item.id}>
                  {item.title ?? item.id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ height: 520 }}>
            <DataGrid
              rows={filteredRows}
              columns={columns}
              loading={loadingResults || loadingAvailability}
              disableColumnSelector
              disableDensitySelector
              disableRowSelectionOnClick
              onRowClick={(params) => handleRowNavigation(params.row)}
              pageSizeOptions={[10, 25, 50]}
              initialState={{
                pagination: {
                  paginationModel: { pageSize: 10, page: 0 },
                },
                sorting: {
                  sortModel: [{ field: "latestAnalysis", sort: "desc" }],
                },
              }}
              sx={{
                "& .MuiDataGrid-row": {
                  cursor: "pointer",
                },
              }}
            />
          </Box>
        </Stack>
      </Paper>
    </Stack>
  );
};

export default DicomViewerPage;

