import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

const DashboardContext = createContext(null);

const defaultFilters = {
  patientSearch: "",
  institute: "",
  scannerStation: "",
  protocolName: "",
  scannerModel: "",
  examDateFrom: "",
  examDateTo: "",
  ageMin: "",
  ageMax: "",
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
      if (p.patientSearch) params.patient_search = p.patientSearch;
      if (p.institute) params.institute = p.institute;
      if (p.scannerStation) params.scanner_station = p.scannerStation;
      if (p.protocolName) params.protocol_name = p.protocolName;
      if (p.scannerModel) params.scanner_model = p.scannerModel;
      if (p.examDateFrom) params.exam_date_from = p.examDateFrom;
      if (p.examDateTo) params.exam_date_to = p.examDateTo;
      if (p.ageMin) params.age_min = p.ageMin;
      if (p.ageMax) params.age_max = p.ageMax;
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

      try {
        const targetPage = overrides.page ?? pagination.page ?? 1;
        const targetLimit = overrides.limit ?? pagination.limit ?? 25;

        const filterParams = buildFilterParams(overrides);
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

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  const refresh = useCallback(() => {
    loadSummary();
  }, [loadSummary]);

  const changePage = useCallback(
    (page) => {
      setPagination((prev) => ({ ...prev, page }));
      loadSummary({ page });
    },
    [loadSummary],
  );

  const changePageSize = useCallback(
    (limit) => {
      setPagination((prev) => ({ ...prev, page: 1, limit }));
      loadSummary({ page: 1, limit });
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
        setCalculationStates(() => {
          const next = {};
          const register = (entry, defaultEvent = "snapshot") => {
            if (!entry || typeof entry !== "object") return;
            const rawId =
              entry.series_id ?? entry.series_uuid ?? entry.seriesId ?? null;
            if (!rawId) return;
            const id = String(rawId);
            next[id] = { ...entry, _event: defaultEvent };
          };

          if (Array.isArray(payload)) {
            payload.forEach((entry) => register(entry));
          } else if (payload && typeof payload === "object") {
            if (payload.calculations) {
              Object.values(payload.calculations).forEach((entry) =>
                register(entry),
              );
            } else {
              register(payload);
            }
          }
          return next;
        });
      });

      source.addEventListener("progress", (event) => {
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
        setCalculationStates((prev) => ({
          ...prev,
          [id]: { ...(prev[id] ?? {}), ...entry, _event: "progress" },
        }));
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
    const uuid =
      series.series_uuid ??
      series.seriesUuid ??
      series.uuid ??
      series.series_id ??
      series.series_instance_uid ??
      null;
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

  // Fix reloadStoredChoResults to read from the ref
  const reloadStoredChoResults = useCallback(async () => {
    const { seriesId } = choModalRef.current;
    if (!seriesId) return;

    setChoModal((prev) => ({
      ...prev,
      storedResults: { loading: true, data: null, error: null },
    }));

    try {
      const data = await fetchJson(`/cho-results/${seriesId}`);
      setChoModal((prev) => ({
        ...prev,
        storedResults: { loading: false, data, error: null },
      }));
    } catch (error) {
      setChoModal((prev) => ({
        ...prev,
        storedResults: {
          loading: false,
          data: null,
          error: error.message ?? "Failed to load results",
        },
      }));
    }
  }, []); // no stale dependency

  // Auto-fetch stored results whenever seriesId is set
  useEffect(() => {
    if (choModal.seriesId) {
      reloadStoredChoResults();
    }
  }, [choModal.seriesId, reloadStoredChoResults]);

  const startChoAnalysis = useCallback(async () => {
    const { seriesUuid, params } = choModal;
    if (!seriesUuid) {
      setStatus("No series selected", "error");
      return;
    }

    setChoModal((prev) => ({
      ...prev,
      stage: "running",
      progress: {
        value: 0,
        message: "Starting analysis...",
        stage: "initialization",
      },
      results: null,
      pollError: null,
    }));

    try {
      const payload = serializeChoPayload(seriesUuid, params);
      await fetchJson("/cho-analysis-modal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (error) {
      setChoModal((prev) => ({
        ...prev,
        stage: "config",
        pollError: error.message ?? "Failed to start analysis",
      }));
      setStatus(`Failed to start analysis: ${error.message}`, "error");
    }
  }, [choModal, setStatus]);

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
    async (seriesId) => {
      try {
        const response = await fetch(
          `${import.meta.env.VITE_API_URL}/cho-export-results`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ series_ids: [seriesId] }),
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
        link.download = `${seriesId}-results.xls`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
        setStatus("Export ready", "success");
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
