import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from "@mui/material";
import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import CloudDownloadRoundedIcon from "@mui/icons-material/CloudDownloadRounded";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import DicomMprViewer from "./dicom/DicomMprViewer";
import { API_BASE_URL } from "./dicom/cornerstoneConfig";

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

const formatValue = (value) => {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return String(value);
};

const DicomViewerRoute = () => {
  const { seriesId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const rowFromState = location.state?.row ?? null;
  const meta = location.state?.meta ?? {};

  const [seriesData, setSeriesData] = useState(
    rowFromState?.raw ?? rowFromState ?? null
  );
  const [seriesUuid, setSeriesUuid] = useState(
    meta.seriesUuid ?? rowFromState?.seriesUuid ?? null
  );
  const [seriesInstanceUid, setSeriesInstanceUid] = useState(
    meta.seriesInstanceUid ??
      rowFromState?.seriesInstanceUid ??
      seriesId ??
      null
  );
  const [studyInstanceUid, setStudyInstanceUid] = useState(
    meta.studyInstanceUid ?? rowFromState?.studyInstanceUid ?? null
  );
  const [patientId, setPatientId] = useState(
    meta.patientId ??
      rowFromState?.raw?.patient_id ??
      rowFromState?.patientId ??
      null
  );

  const [loadingSummary, setLoadingSummary] = useState(!rowFromState);
  const [summaryError, setSummaryError] = useState(null);
  const [modalities, setModalities] = useState([]);
  const [selectedModality, setSelectedModality] = useState("");
  const [loadingModalities, setLoadingModalities] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [availability, setAvailability] = useState({
    checking: false,
    hasDicom: false,
  });
  const [availabilityTick, setAvailabilityTick] = useState(0);
  const [banner, setBanner] = useState(null);

  const loadSummary = useCallback(async () => {
    if (!seriesId || rowFromState) {
      return;
    }

    setLoadingSummary(true);
    setSummaryError(null);

    try {
      const data = await fetchJson(
        `/cho-results/${encodeURIComponent(seriesId)}`
      );
      if (data && typeof data === "object") {
        setSeriesData(data);
        setSeriesUuid(
          data.series_uuid ?? data.seriesUuid ?? data.series_id ?? null
        );
        setSeriesInstanceUid(
          data.series_instance_uid ?? data.seriesInstanceUid ?? seriesId ?? null
        );
        setStudyInstanceUid(
          data.study_instance_uid ?? data.studyInstanceUid ?? null
        );
        setPatientId(data.patient_id ?? data.patientId ?? null);
      } else {
        setSummaryError("Series details not found.");
      }
    } catch (err) {
      console.error("Failed to load series details", err);
      setSummaryError(err.message ?? "Failed to load series details.");
    } finally {
      setLoadingSummary(false);
    }
  }, [seriesId, rowFromState]);

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

  const checkAvailability = useCallback(async () => {
    if (!seriesUuid) {
      setAvailability({ checking: false, hasDicom: false });
      return;
    }

    setAvailability({ checking: true, hasDicom: false });
    try {
      const list = await fetchJson("/series/");
      const hasDicom =
        Array.isArray(list) &&
        list.map((item) => String(item)).includes(String(seriesUuid));
      setAvailability({ checking: false, hasDicom });
    } catch (err) {
      console.error("Failed to check series availability", err);
      setAvailability({ checking: false, hasDicom: false });
    }
  }, [seriesUuid]);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  useEffect(() => {
    loadModalities();
  }, [loadModalities]);

  useEffect(() => {
    checkAvailability();
  }, [checkAvailability, availabilityTick]);

  const handleRecover = useCallback(async () => {
    if (!selectedModality) {
      setBanner({
        severity: "warning",
        message: "Select a modality before requesting a DICOM pull.",
      });
      return;
    }
    if (!seriesInstanceUid) {
      setBanner({
        severity: "error",
        message: "Series Instance UID is required for DICOM recovery.",
      });
      return;
    }

    setRecovering(true);
    try {
      await fetchJson("/dicom-pull/recover", {
        method: "POST",
        body: JSON.stringify({
          modality: selectedModality,
          seriesInstanceUID: seriesInstanceUid,
          studyInstanceUID: studyInstanceUid,
          patientId,
        }),
      });
      setBanner({
        severity: "info",
        message:
          "Recovery requested. Refresh availability once Orthanc receives the series.",
      });
    } catch (err) {
      console.error("Failed to request DICOM recovery", err);
      setBanner({
        severity: "error",
        message: err.message ?? "Failed to request DICOM recovery.",
      });
    } finally {
      setRecovering(false);
    }
  }, [patientId, selectedModality, seriesInstanceUid, studyInstanceUid]);

  const summaryItems = useMemo(() => {
    const source = seriesData ?? {};
    return [
      { label: "Patient", value: source.patient_name ?? source.PatientName },
      {
        label: "Patient ID",
        value:
          patientId ??
          source.patient_id ??
          source.PatientID ??
          source.PatientId,
      },
      {
        label: "Protocol",
        value: source.protocol_name ?? source.series_description,
      },
      {
        label: "Series Instance UID",
        value:
          seriesInstanceUid ??
          source.series_instance_uid ??
          source.SeriesInstanceUID,
      },
      {
        label: "Orthanc Series UUID",
        value: seriesUuid ?? source.series_uuid ?? "Unknown",
      },
      {
        label: "Study Instance UID",
        value:
          studyInstanceUid ??
          source.study_instance_uid ??
          source.StudyInstanceUID,
      },
      {
        label: "Last Updated",
        value: source.latest_analysis_date ?? source.updated_at,
      },
    ];
  }, [patientId, seriesData, seriesInstanceUid, seriesUuid, studyInstanceUid]);

  const hasSeriesUuid = Boolean(seriesUuid);
  const displaySeriesUuid =
    availability.hasDicom && hasSeriesUuid ? seriesUuid : null;

  return (
    <Stack spacing={3}>
      <Stack direction='row' spacing={2} alignItems='center'>
        <Button
          size='small'
          variant='text'
          startIcon={<ArrowBackRoundedIcon />}
          onClick={() => navigate("/dicom-viewer")}>
          Back to list
        </Button>
        <Typography variant='h5' sx={{ flexGrow: 1 }}>
          DICOM Viewer
        </Typography>
        <Stack direction='row' spacing={2} alignItems='center'>
          <FormControl size='small' sx={{ minWidth: 200 }}>
            <InputLabel id='viewer-modality-label'>
              Recovery modality
            </InputLabel>
            <Select
              labelId='viewer-modality-label'
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
          <Button
            size='small'
            variant='outlined'
            startIcon={
              recovering ? (
                <CircularProgress size={14} />
              ) : (
                <CloudDownloadRoundedIcon fontSize='small' />
              )
            }
            disabled={recovering}
            onClick={handleRecover}>
            {recovering ? "Recovering" : "Pull DICOM"}
          </Button>
          <Button
            size='small'
            variant='outlined'
            startIcon={<RefreshRoundedIcon />}
            disabled={availability.checking}
            onClick={() => setAvailabilityTick((count) => count + 1)}>
            {availability.checking ? "Checking..." : "Refresh availability"}
          </Button>
        </Stack>
      </Stack>

      {banner && (
        <Alert
          severity={banner.severity}
          onClose={() => setBanner(null)}
          sx={{ mb: 1 }}>
          {banner.message}
        </Alert>
      )}

      {summaryError && (
        <Alert
          severity='error'
          onClose={() => setSummaryError(null)}
          sx={{ mb: 1 }}>
          {summaryError}
        </Alert>
      )}
      <Paper elevation={0} variant='outlined' sx={{ p: 2 }}>
        <Accordion>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls='panel1-content'
            id='panel1-header'>
            <Typography variant='subtitle1' gutterBottom>
              Series Summary
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            {loadingSummary ? (
              <Stack direction='row' spacing={1} alignItems='center'>
                <CircularProgress size={18} />
                <Typography variant='body2'>
                  Loading series details...
                </Typography>
              </Stack>
            ) : (
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                  gap: 2,
                }}>
                {summaryItems.map((item) => (
                  <Stack key={item.label} spacing={0.5}>
                    <Typography variant='caption' color='text.secondary'>
                      {item.label}
                    </Typography>
                    <Typography variant='body2'>
                      {formatValue(item.value)}
                    </Typography>
                  </Stack>
                ))}
              </Box>
            )}
          </AccordionDetails>
        </Accordion>
      </Paper>
      {!hasSeriesUuid && (
        <Alert severity='warning'>
          Unable to determine the Orthanc series identifier for this entry. The
          DICOM viewer requires the series to be present in Orthanc and to have
          its UUID recorded in the results.
        </Alert>
      )}

      {!availability.hasDicom && hasSeriesUuid && (
        <Alert severity='info'>
          This series is not currently stored in Orthanc. Request a pull or
          refresh availability once the import completes.
        </Alert>
      )}

      <Paper elevation={0} variant='outlined' sx={{ p: 2, minHeight: 800 }}>
        <DicomMprViewer seriesUuid={displaySeriesUuid} />
      </Paper>
    </Stack>
  );
};

export default DicomViewerRoute;
