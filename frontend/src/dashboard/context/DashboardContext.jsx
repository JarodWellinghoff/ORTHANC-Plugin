import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { applyChoProgress, backgroundChoRun } from "../utils/choResultFlow";
const AGE_FILTER_MIN = 0;
const AGE_FILTER_MAX = 200;
const DashboardContext = createContext(null);

const defaultFilters = {
  patientIdSearch: [],
  patientNameSearch: [],
  instituteSearch: [],
  protocolNameSearch: [],
  scannerModelSearch: [],
  scannerStationSearch: [],
  pullScheduleSearch: [],
  studyDateStartSearch: null,
  studyDateEndSearch: null,
  ageStartSearch: AGE_FILTER_MIN,
  ageEndSearch: AGE_FILTER_MAX,
};

const defaultChoParams = {
  testType: "full",
  lesionSet: "standard",
  lesionHUs: {
    standard: [-10, -30, -50],
    "low-contrast": [-5, -15, -25],
    "high-contrast": [-20, -60, -100],
  },
  spatialResolution: "auto",
  mtf50: 0.434,
  mtf10: null,
};

const createInitialChoState = () => ({
  open: false,
  seriesUuid: null,
  seriesId: null,
  seriesSummary: null,
  stage: "config",
  params: { ...defaultChoParams },
  progress: { value: 0, message: "", stage: "initialization" },
  results: null,
  runMode: null,
  runId: null,
  previousRunId: null,
  reviewRequired: false,
  storedResults: { loading: false, data: null, error: null },
  pollError: null,
});

// Initial state for the delete dialog
const createInitialDeleteDialog = () => ({
  open: false,
  // Identifiers
  seriesId: null, // series_instance_uid — used for DB result delete
  seriesUuid: null, // Orthanc UUID — used for DICOM delete
  // Metadata
  calculationType: null,
  patientName: null,
  // What exists
  hasResults: false, // Whether DB results exist for this series
  hasDicom: false, // Whether DICOMs are present in Orthanc
  // User's current selections
  deleteResults: false,
  deleteDicom: false,
  // Loading state
  loading: false,
});

const fetchJson = async (url, options) => {
  const response = await fetch(`${import.meta.env.VITE_API_URL}${url}`, {
    credentials: "include",
    ...options,
  });
  if (response.status === 204) return null;
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

const toSearchParams = (params) => {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.append(key, value);
    }
  });
  return search;
};

const serializeChoPayload = (seriesUuid, params) => ({
  series_uuid: seriesUuid,
  testType: params.testType,
  resamples: Number(params.resamples) || 500,
  internalNoise: Number(params.internalNoise) || 2.25,
  resamplingMethod: params.resamplingMethod || "Bootstrap",
  roiSize: Number(params.roiSize) || 6,
  thresholdLow: Number(params.thresholdLow) || 0,
  thresholdHigh: Number(params.thresholdHigh) || 150,
  windowLength: Number(params.windowLength) || 15,
  stepSize: Number(params.stepSize) || 5,
  channelType: params.channelType || "Gabor",
  lesionSet: params.lesionSet || "standard",
  saveResults:
    typeof params.saveResults === "boolean" ? params.saveResults : true,
  spatialResolution: params.spatialResolution || "auto",
  mtf50: params.mtf50 ? Number(params.mtf50) : null,
  mtf10: params.mtf10 ? Number(params.mtf10) : null,
});

const deriveSeriesSummary = (data, fallback = null) => {
  if (!data || typeof data !== "object") {
    return fallback ?? null;
  }

  const patient = data.patient ?? {};
  const series = data.series ?? {};
  const scanner = data.scanner ?? {};
  const study = data.study ?? {};

  const summary = {
    patient_name:
      patient.patient_name ??
      data.patient_name ??
      fallback?.patient_name ??
      null,
    series_uuid:
      series.series_instance_uid ??
      series.series_uuid ??
      data.series_uuid ??
      data.series_instance_uid ??
      fallback?.series_uuid ??
      null,
    series_id:
      data.series_id ?? series.series_id ?? fallback?.series_id ?? null,
    study_id:
      study.study_instance_uid ??
      study.study_id ??
      data.study_id ??
      fallback?.study_id ??
      null,
  };

  const hasValue = Object.values(summary).some(
    (value) => value !== null && value !== undefined && value !== "",
  );

  if (!hasValue) {
    return fallback ?? null;
  }

  return summary;
};
export const DashboardProvider = ({ children }) => {
  const [status, setStatusState] = useState({ message: "", severity: "idle" });
  const [filters, setFilters] = useState(defaultFilters);
  const [filterOptions, setFilterOptions] = useState({
    institutes: [],
    scanner_stations: [],
    protocol_names: [],
    scanner_models: [],
    date_range: null,
    age_range: null,
  });
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false);
  const [stats, setStats] = useState({
    total: 0,
    detectability: 0,
    noise: 0,
    errors: 0,
  });
  const [summaryItems, setSummaryItems] = useState([]);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 25,
    total: 0,
    pages: 1,
  });
  const [availableSeries, setAvailableSeries] = useState([]);
  const [choModal, setChoModal] = useState(createInitialChoState);
  const [deleteDialog, setDeleteDialog] = useState(createInitialDeleteDialog);
  const [calculationStates, setCalculationStates] = useState({});

  const pollingRef = useRef(null);
  const storedResultsRequestRef = useRef(0);
  const choStartLockRef = useRef(false);
  const mountedRef = useRef(true);
  const sseRef = useRef(null);
  const sseReconnectRef = useRef(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
      if (sseReconnectRef.current) {
        clearTimeout(sseReconnectRef.current);
        sseReconnectRef.current = null;
      }
    };
  }, []);

  const setStatus = useCallback((message, severity = "idle") => {
    setStatusState({ message, severity });
  }, []);

  const resetFilters = useCallback(() => {
    setFilters(defaultFilters);
    setPagination((prev) => ({ ...prev, page: 1 }));
  }, []);

  const updateFilter = useCallback((name, value) => {
    setFilters((prev) => ({ ...prev, [name]: value ?? "" }));
  }, []);

  const clearFilterValue = useCallback((name) => {
    setFilters((prev) => ({ ...prev, [name]: "" }));
  }, []);

  const toggleAdvancedFilters = useCallback(() => {
    setAdvancedFiltersOpen((prev) => !prev);
  }, []);

  const buildFilterParams = useCallback(
    (overrides = {}) => {
      const p = { ...filters, ...overrides };
      const params = {};
      const ageMin = Number(p.ageStartSearch);
      const ageMax = Number(p.ageEndSearch);
      if (p.patientIdSearch) params.patient_id = p.patientIdSearch.join(",");
      if (p.patientNameSearch)
        params.patient_name = p.patientNameSearch.join(",");
      if (p.instituteSearch) params.institute = p.instituteSearch.join(",");
      if (p.scannerStationSearch)
        params.scanner_station = p.scannerStationSearch.join(",");
      if (p.protocolNameSearch)
        params.protocol_name = p.protocolNameSearch.join(",");
      if (p.scannerModelSearch)
        params.scanner_model = p.scannerModelSearch.join(",");
      if (p.studyDateStartSearch)
        params.exam_date_from = p.studyDateStartSearch;
      if (p.studyDateEndSearch) params.exam_date_to = p.studyDateEndSearch;
      if (Number.isFinite(ageMin) && ageMin > AGE_FILTER_MIN)
        params.patient_age_min = ageMin;
      if (Number.isFinite(ageMax) && ageMax < AGE_FILTER_MAX)
        params.patient_age_max = ageMax;
      if (p.sort_by) params.sort_by = p.sort_by;
      if (p.sort_order) params.sort_order = p.sort_order;
      return params;
    },
    [filters],
  );

  const loadFilterOptions = useCallback(async () => {
    try {
      const data = await fetchJson("/cho-filter-options");
      if (mountedRef.current && data) {
        setFilterOptions({
          institutes: data.institutes ?? [],
          scanner_stations: data.scanner_stations ?? [],
          protocol_names: data.protocol_names ?? [],
          scanner_models: data.scanner_models ?? [],
          pull_schedules: data.pull_schedule_names ?? [],
          date_range: data.date_range ?? null,
          age_range: data.age_range ?? null,
        });
      }
    } catch (error) {
      console.error("Failed to load filter options", error);
    }
  }, []);

  const loadSummary = useCallback(
    async (overrides = {}) => {
      if (!mountedRef.current) return;
      setSummaryLoading(true);
      // If Overrides is completely empty, make everything empty
      if (Object.keys(overrides).length === 0) {
        setSummaryItems([]);
        setPagination({ page: 1, limit: 25, total: 0, pages: 1 });
        setAvailableSeries([]);
        setStatus("No data available", "idle");
        setSummaryLoading(false);
        return;
      }

      try {
        const targetPage = overrides.page ?? pagination.page ?? 1;
        const targetLimit = overrides.limit ?? pagination.limit ?? 25;

        const filterParams = buildFilterParams(overrides);
        console.log(
          "Loading summary with filters:",
          filterParams,
          "and pagination:",
          { page: targetPage, limit: targetLimit },
        );
        const searchParams = toSearchParams({
          ...filterParams,
          page: targetPage,
          limit: targetLimit,
        });

        const [summaryResponse, orthancSeries] = await Promise.all([
          fetchJson(`/cho-results?${searchParams.toString()}`),
          fetchJson("/series/").catch(() => []),
        ]);

        if (!mountedRef.current) return;

        setPagination((prev) => ({
          ...prev,
          page: targetPage,
          limit: targetLimit,
        }));

        let items = [];
        let total = 0;
        let pages = 1;
        let page = targetPage;

        if (Array.isArray(summaryResponse)) {
          items = summaryResponse;
          total = summaryResponse.length;
          pages = Math.max(1, Math.ceil(total / targetLimit));
        } else if (summaryResponse) {
          items = summaryResponse.data ?? [];
          total = summaryResponse.total ?? items.length;
          pages =
            summaryResponse.pages ??
            Math.max(1, Math.ceil(total / targetLimit));
          page = summaryResponse.page ?? targetPage;
        }

        setSummaryItems(items);
        setPagination({ page, limit: targetLimit, total, pages });
        setAvailableSeries(Array.isArray(orthancSeries) ? orthancSeries : []);
        setStatus(
          `Loaded ${items.length} series (Page ${page} of ${pages})`,
          "success",
        );
      } catch (error) {
        console.error("Failed to load dashboard data", error);
        if (mountedRef.current) {
          setStatus(`Error loading data: ${error.message}`, "error");
        }
      } finally {
        if (mountedRef.current) {
          setSummaryLoading(false);
        }
      }
    },
    [buildFilterParams, pagination.page, pagination.limit, setStatus],
  );

  useEffect(() => {
    loadFilterOptions();
  }, [loadFilterOptions]);

  //   useEffect(() => {
  //     loadSummary();
  //   }, [loadSummary]);
  useEffect(() => {
    loadSummary();
    // Mount only. Pages drive every subsequent load with their own params
    // (filters/page/sort); re-running this on loadSummary identity changes
    // re-fires the no-arg "clear everything" path on every pagination change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const refresh = useCallback(() => {
    loadSummary();
  }, [loadSummary]);

  const changePage = useCallback(
    (page, extras = {}) => {
      setPagination((prev) => ({ ...prev, page }));
      loadSummary({ page, ...extras });
    },
    [loadSummary],
  );

  const changePageSize = useCallback(
    (limit, extras = {}) => {
      setPagination((prev) => ({ ...prev, page: 1, limit }));
      loadSummary({ page: 1, limit, ...extras });
    },
    [loadSummary],
  );

  /**
   * Open the delete dialog.
   *
   * @param {string|null} seriesId    - The series_instance_uid (for DB result delete)
   * @param {object}      opts
   * @param {string|null}   opts.seriesUuid         - Orthanc UUID (for DICOM delete)
   * @param {string|null}   opts.calculationType
   * @param {string|null}   opts.patientName
   * @param {boolean}       opts.hasResults         - Whether DB results exist
   * @param {boolean}       opts.hasDicom           - Whether DICOMs exist in Orthanc
   * @param {boolean}       opts.initialDeleteResults - Pre-check "delete results" box
   * @param {boolean}       opts.initialDeleteDicom   - Pre-check "delete DICOM" box
   */
  const openDeleteDialogAction = useCallback(
    (
      seriesId,
      {
        seriesUuid = null,
        calculationType = null,
        patientName = null,
        hasResults = false,
        hasDicom = false,
        initialDeleteResults = true,
        initialDeleteDicom = false,
      } = {},
    ) => {
      setDeleteDialog({
        open: true,
        seriesId: seriesId ?? null,
        seriesUuid: seriesUuid ?? null,
        calculationType: calculationType ?? null,
        patientName: patientName ?? null,
        hasResults,
        hasDicom,
        // Only pre-check options that are actually possible
        deleteResults: initialDeleteResults && hasResults,
        deleteDicom: initialDeleteDicom && hasDicom,
        loading: false,
      });
    },
    [],
  );

  const closeDeleteDialog = useCallback(() => {
    setDeleteDialog(createInitialDeleteDialog());
  }, []);

  /**
   * Toggle one of the delete checkboxes inside the dialog.
   * @param {"deleteResults"|"deleteDicom"} field
   */
  const toggleDeleteOption = useCallback((field) => {
    setDeleteDialog((prev) => ({ ...prev, [field]: !prev[field] }));
  }, []);

  /**
   * Execute the confirmed deletion(s).
   * Runs result-delete and/or DICOM-delete based on the user's checkbox selections.
   */
  const confirmDelete = useCallback(async () => {
    const {
      seriesId,
      seriesUuid,
      calculationType,
      deleteResults,
      deleteDicom,
    } = deleteDialog;

    if (!deleteResults && !deleteDicom) return;

    setDeleteDialog((prev) => ({ ...prev, loading: true }));
    setStatus("Deleting...", "loading");

    const errors = [];

    try {
      // ── Delete DB results ─────────────────────────────────────────────
      if (deleteResults && seriesId) {
        try {
          const query = calculationType
            ? `?calculation_type=${encodeURIComponent(calculationType)}`
            : "";
          await fetchJson(`/cho-results/${seriesId}${query}`, {
            method: "DELETE",
          });
        } catch (err) {
          console.error("Failed to delete results", err);
          errors.push(`Results: ${err.message}`);
        }
      }

      // ── Delete DICOM from Orthanc ────────────────────────────────────
      if (deleteDicom && seriesUuid) {
        try {
          await fetchJson(`/cho-dicom/${seriesUuid}`, {
            method: "DELETE",
          });
        } catch (err) {
          console.error("Failed to delete DICOM", err);
          errors.push(`DICOM: ${err.message}`);
        }
      }

      if (!mountedRef.current) return;

      if (errors.length === 0) {
        setStatus("Successfully deleted", "success");
      } else if (
        errors.length <
        (deleteResults ? 1 : 0) + (deleteDicom ? 1 : 0)
      ) {
        setStatus(`Partially deleted. Errors: ${errors.join("; ")}`, "warning");
      } else {
        setStatus(`Error deleting: ${errors.join("; ")}`, "error");
      }

      closeDeleteDialog();
      loadSummary();
    } catch (error) {
      console.error("Unexpected error during delete", error);
      if (mountedRef.current) {
        setStatus(`Error deleting: ${error.message}`, "error");
        setDeleteDialog((prev) => ({ ...prev, loading: false }));
      }
    }
  }, [deleteDialog, closeDeleteDialog, loadSummary, setStatus]);

  const exportAllResults = useCallback(async () => {
    setStatus("Exporting to CSV...", "loading");
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/cho-results-export`,
        {
          method: "GET",
        },
      );
      if (!response.ok) throw new Error("Failed to export");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "cho-results.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      setStatus("Export complete", "success");
    } catch (error) {
      setStatus(`Export failed: ${error.message}`, "error");
    }
  }, [setStatus]);

  const stopChoPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  useEffect(() => {
    const apiBase = import.meta.env.VITE_API_URL;
    if (!apiBase) {
      return undefined;
    }

    const connect = () => {
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }

      const source = new EventSource(`${apiBase}/cho-progress/stream`, {
        withCredentials: true,
      });
      sseRef.current = source;

      source.onopen = () => {
        if (sseReconnectRef.current) {
          clearTimeout(sseReconnectRef.current);
          sseReconnectRef.current = null;
        }
      };

      source.addEventListener("snapshot", (event) => {
        if (!mountedRef.current) return;
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          console.debug("Failed to parse snapshot event", error);
          return;
        }
        const next = {};
        const register = (entry) => {
          if (!entry || typeof entry !== "object") return;
          const id = entry.series_id ?? entry.series_uuid ?? entry.seriesId;
          if (id) next[String(id)] = { ...entry, _event: "snapshot" };
        };
        if (Array.isArray(payload)) payload.forEach(register);
        else if (payload?.calculations) Object.values(payload.calculations).forEach(register);
        else register(payload);
        setCalculationStates(next);
        setChoModal((previous) => applyChoProgress(previous,
          next[String(previous.seriesUuid)] ?? next[String(previous.seriesId)]));
      });

      source.addEventListener("cho-calculation", (event) => {
        if (!mountedRef.current) return;
        let entry;
        try {
          entry = JSON.parse(event.data);
        } catch {
          return;
        }
        const rawId =
          entry.series_id ?? entry.series_uuid ?? entry.seriesId ?? null;
        if (!rawId) return;
        const id = String(rawId);
        const eventType = entry.eventType ?? "progress";

        // Bulk page state — keep as before
        setCalculationStates((prev) => ({
          ...prev,
          [id]: { ...(prev[id] ?? {}), ...entry, _event: eventType },
        }));

        // Only a locally started run may update this page. Background results
        // and staged reruns never become the visible result through SSE.
        setChoModal((previous) => applyChoProgress(previous, entry));
      });

      source.onerror = () => {
        if (!mountedRef.current) return;
        if (sseRef.current) {
          sseRef.current.close();
          sseRef.current = null;
        }
        sseReconnectRef.current = setTimeout(connect, 5000);
      };
    };

    connect();

    return () => {
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
      if (sseReconnectRef.current) {
        clearTimeout(sseReconnectRef.current);
        sseReconnectRef.current = null;
      }
    };
  }, []);

  const openChoModal = useCallback((series) => {
    if (!series) return;
    const uuid = series.series_uuid ?? series.seriesUuid ?? series.uuid ?? null;
    const id =
      series.series_id ?? series.series_instance_uid ?? series.seriesId ?? null;

    setChoModal((prev) => ({
      ...createInitialChoState(),
      open: true,
      seriesUuid: uuid,
      seriesId: id,
      seriesSummary: series,
    }));
  }, []);

  const closeChoModal = useCallback(() => {
    stopChoPolling();
    setChoModal(createInitialChoState());
  }, [stopChoPolling]);

  const updateChoParam = useCallback((name, value) => {
    setChoModal((prev) => ({
      ...prev,
      params: { ...prev.params, [name]: value },
    }));
  }, []);

  const choModalRef = useRef(choModal);
  useEffect(() => {
    choModalRef.current = choModal;
  }, [choModal]);

  const reloadStoredChoResults = useCallback(async ({ clearResults = false } = {}) => {
    const { seriesId } = choModalRef.current;
    if (!seriesId) return false;
    const requestId = ++storedResultsRequestRef.current;
    const isCurrent = (state) => mountedRef.current && state.seriesId === seriesId &&
      storedResultsRequestRef.current === requestId;
    setChoModal((previous) => isCurrent(previous) ? {
      ...previous,
      ...(clearResults ? { results: null, runMode: null, reviewRequired: false, stage: "config" } : {}),
      storedResults: { ...previous.storedResults, loading: true, error: null },
    } : previous);
    try {
      const data = await fetchJson(`/cho-results/${encodeURIComponent(seriesId)}`, { cache: "no-store" });
      if (!isCurrent(choModalRef.current)) return false;
      setChoModal((previous) => {
        if (!isCurrent(previous)) return previous;
        const next = { ...previous, storedResults: { loading: false, data, error: null } };
        if (data?.series_uuid && (!previous.seriesUuid || previous.seriesUuid === previous.seriesId)) {
          next.seriesUuid = data.series_uuid;
        }
        return next;
      });
      return true;
    } catch (error) {
      setChoModal((previous) => isCurrent(previous) ? {
        ...previous,
        storedResults: { ...previous.storedResults, loading: false, error: error.message ?? "Failed to load results" },
      } : previous);
      return false;
    }
  }, []);

  // Auto-fetch stored results whenever seriesId is set
  useEffect(() => {
    if (choModal.seriesId) {
      reloadStoredChoResults();
    }
  }, [choModal.seriesId, reloadStoredChoResults]);

  const continueChoInBackground = useCallback(() => {
    setChoModal(backgroundChoRun);
  }, []);

  const startChoAnalysis = useCallback(async ({ background = false } = {}) => {
    const { seriesUuid, seriesId, params, stage } = choModalRef.current;
    if (choStartLockRef.current || stage === "running") return;
    if (!seriesUuid) {
      setStatus("No series selected", "error");
      return;
    }
    choStartLockRef.current = true;
    const previousRunId = calculationStates[String(seriesUuid)]?.metadata?.run_id ?? null;
    setChoModal((previous) => ({
      ...previous, stage: "running", runMode: background ? "background" : "foreground",
      runId: null, previousRunId, reviewRequired: false,
      progress: { value: 0, message: "Starting analysis...", stage: "initialization" },
      results: null, pollError: null,
    }));
    try {
      // First analyses save normally; the server stages reruns without overwriting.
      const payload = serializeChoPayload(seriesUuid, { ...params, saveResults: true });
      await fetchJson("/cho-analysis-modal", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
    } catch (error) {
      if (!mountedRef.current) return;
      setChoModal((previous) => previous.seriesId === seriesId && previous.seriesUuid === seriesUuid ? {
        ...previous, stage: "config", runMode: null,
        pollError: error.message ?? "Failed to start analysis",
      } : previous);
      setStatus(`Failed to start analysis: ${error.message}`, "error");
    } finally {
      choStartLockRef.current = false;
    }
  }, [calculationStates, setStatus]);

  const discardChoResults = useCallback(async () => {
    if (!choModal.seriesUuid) return;
    try {
      await fetchJson(
        `/cho-calculation-status?series_id=${encodeURIComponent(
          choModal.seriesUuid,
        )}&action=discard`,
      );
      if (mountedRef.current) {
        setStatus("Results discarded", "success");
        closeChoModal();
        loadSummary();
      }
    } catch (error) {
      if (mountedRef.current) {
        setStatus(`Failed to discard results: ${error.message}`, "error");
      }
    }
  }, [choModal.seriesUuid, closeChoModal, loadSummary, setStatus]);

  const exportSeries = useCallback(
    async (seriesIdOrIds) => {
      // Accept either a single id (legacy: ChoAnalysisPage's "Export XLS"
      // button) or an array of ids (ResultsPage's new "Export Selected"
      // bulk action). Normalize up front so the rest of the function only
      // has to think about arrays.
      const seriesIds = Array.isArray(seriesIdOrIds)
        ? seriesIdOrIds.filter(Boolean)
        : seriesIdOrIds
          ? [seriesIdOrIds]
          : [];
      if (seriesIds.length === 0) {
        setStatus("No series selected for export", "error");
        return;
      }

      const isBulk = seriesIds.length > 1;
      setStatus(
        isBulk ? `Exporting ${seriesIds.length} series…` : "Exporting…",
        "loading",
      );

      try {
        const response = await fetch(
          `${import.meta.env.VITE_API_URL}/cho-export-results`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ series_ids: seriesIds }),
            credentials: "include",
          },
        );
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || response.statusText);
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        // Filename adapts to the request shape: single id keeps the
        // historical `<uid>-results.xls` shape so individual-case exports
        // still drop in with the same name. Bulk exports get a timestamp
        // so the user can do multiple exports in a row without clobbering
        // each other's downloads.
        if (isBulk) {
          const stamp = new Date().toISOString().replace(/[:.]/g, "-");
          link.download = `cho-results-${seriesIds.length}-series-${stamp}.xls`;
        } else {
          link.download = `${seriesIds[0]}-results.xls`;
        }
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
        setStatus(
          isBulk ? `Export ready (${seriesIds.length} series)` : "Export ready",
          "success",
        );
      } catch (error) {
        console.debug(error);
        setStatus(`Download failed: ${error.message}`, "error");
      }
    },
    [setStatus],
  );

  const value = useMemo(() => {
    return {
      status,
      calculationStates,
      filters,
      filterOptions,
      advancedFiltersOpen,
      stats,
      summary: {
        items: summaryItems,
        loading: summaryLoading,
        pagination,
        availableSeries,
      },
      choModal,
      deleteDialog,
      actions: {
        setStatus,
        resetFilters,
        updateFilter,
        clearFilterValue,
        toggleAdvancedFilters,
        refresh,
        changePage,
        changePageSize,
        openDeleteDialog: openDeleteDialogAction,
        closeDeleteDialog,
        toggleDeleteOption,
        confirmDelete,
        exportAllResults,
        exportSeries,
        reloadStoredChoResults,
        openChoModal,
        closeChoModal,
        updateChoParam,
        startChoAnalysis,
        continueChoInBackground,
        discardChoResults,
        loadFilterOptions,
        loadSummary,
      },
    };
  }, [
    status,
    filters,
    filterOptions,
    advancedFiltersOpen,
    stats,
    summaryItems,
    summaryLoading,
    pagination,
    availableSeries,
    calculationStates,
    choModal,
    deleteDialog,
    setStatus,
    resetFilters,
    updateFilter,
    clearFilterValue,
    toggleAdvancedFilters,
    refresh,
    changePage,
    changePageSize,
    openDeleteDialogAction,
    closeDeleteDialog,
    toggleDeleteOption,
    confirmDelete,
    exportAllResults,
    exportSeries,
    reloadStoredChoResults,
    openChoModal,
    closeChoModal,
    updateChoParam,
    startChoAnalysis,
    continueChoInBackground,
    discardChoResults,
    loadFilterOptions,
    loadSummary,
  ]);

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error("useDashboard must be used within a DashboardProvider");
  }
  return context;
};
