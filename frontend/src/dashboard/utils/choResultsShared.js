// ─────────────────────────────────────────────────────────────────────────────
// choResultsShared.js
//
// Shared helpers and constants used by the pages that render the CHO results
// catalog — `BulkTestsPage` (testing) and `ResultsPage` (viewing). Both pages
// read the same `summary` slice from DashboardContext and render mostly the
// same column lineup; pulling the duplicated bits out into one module keeps
// them from drifting after the split.
// ─────────────────────────────────────────────────────────────────────────────

const apiBase = import.meta.env.VITE_API_URL;

export const defaultChoParams = {
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

export const statusColorMap = {
  full: "success",
  partial: "warning",
  error: "error",
  none: "default",
  pending: "default",
};

export const statusLabelMap = {
  full: "Full",
  partial: "Global Noise",
  error: "Error",
  none: "Unknown",
  pending: "Pending",
};

export const fetchJson = async (path, options = {}) => {
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

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const formatDateTime = (value) => {
  if (!value) return "--";
  try {
    return new Date(value).toLocaleString();
  } catch (error) {
    console.debug("Failed to format date", error);
    return value;
  }
};

export const deriveRowId = (item, index) => {
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

export const resolveSeriesKey = (row) => {
  if (!row) return null;
  if (row.seriesUuid) return String(row.seriesUuid);
  if (row.seriesInstanceUid) return String(row.seriesInstanceUid);
  if (row.studyInstanceUid) return String(row.studyInstanceUid);
  if (row.raw?.series_id) return String(row.raw.series_id);
  if (row.raw?.series_uuid) return String(row.raw.series_uuid);
  return null;
};

/**
 * Map an API series item onto the row shape the DataGrid expects. `availableSet`
 * is the set of series UUIDs known to be present in Orthanc; we use it to flag
 * `hasDicom` so the testing page can show "Pull DICOM" for missing ones.
 */
export const normalizeChoRow = (item, index, availableSet) => {
  const id = deriveRowId(item, index);
  const seriesUuid =
    item.series_uuid ?? item.seriesUuid ?? item.series_id ?? null;
  const seriesInstanceUid =
    item.series_id ??
    item.series_instance_uid ??
    item.seriesInstanceUid ??
    null;
  const studyInstanceUid =
    item.study_id ?? item.study_instance_uid ?? item.studyInstanceUid ?? null;

  const statusRaw = (item.test_status ?? "").toLowerCase();
  const status =
    statusRaw === "full" || statusRaw === "partial" || statusRaw === "error"
      ? statusRaw
      : statusRaw || "none";

  const hasDicom =
    Boolean(seriesUuid) &&
    availableSet instanceof Set &&
    availableSet.has(String(seriesUuid));

  return {
    id,
    raw: item,
    seriesUuid,
    seriesInstanceUid,
    studyInstanceUid,
    patientId: item.patient_id ?? "N/A",
    patientName: item.patient_name ?? "N/A",
    institutionName: item.institution_name ?? "N/A",
    protocolName: item.protocol_name ?? "N/A",
    scannerModel: item.scanner_model ?? "N/A",
    stationName: item.station_name ?? "N/A",
    latestAnalysis: item.latest_analysis_date ?? null,
    testStatus: status,
    hasDicom,
    pullScheduleName: item.pull_schedule_name ?? null,
  };
};
