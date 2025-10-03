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
  testType: "global",
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

const createInitialChoState = () => ({
  open: false,
  seriesUuid: null,
  seriesSummary: null,
  stage: "config",
  params: { ...defaultChoParams },
  progress: { value: 0, message: "", stage: "initialization" },
  results: null,
  saving: false,
  pollError: null,
});

const fetchJson = async (url, options) => {
  const response = await fetch(
    `${import.meta.env.VITE_API_URL}${url}`,
    options
  );
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
  resamples: Number(params.resamples) || 0,
  internalNoise: Number(params.internalNoise) || 0,
  resamplingMethod: params.resamplingMethod,
  roiSize: Number(params.roiSize) || 0,
  thresholdLow: Number(params.thresholdLow) || 0,
  thresholdHigh: Number(params.thresholdHigh) || 0,
  windowLength: Number(params.windowLength) || 0,
  stepSize: Number(params.stepSize) || 0,
  channelType: params.channelType,
  lesionSet: params.lesionSet,
  saveResults: false,
});

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
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(true);
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
  const [detailsModal, setDetailsModal] = useState({
    open: false,
    loading: false,
    seriesId: null,
    data: null,
  });
  const [choModal, setChoModal] = useState(createInitialChoState);
  const [deleteDialog, setDeleteDialog] = useState({
    open: false,
    seriesId: null,
    calculationType: null,
    patientName: null,
    loading: false,
  });

  const pollingRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
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
    setPagination((prev) => ({ ...prev, page: 1 }));
  }, []);

  const clearFilterValue = useCallback(
    (name) => {
      updateFilter(name, "");
    },
    [updateFilter]
  );

  const toggleAdvancedFilters = useCallback(() => {
    setAdvancedFiltersOpen((prev) => !prev);
  }, []);

  const buildFilterParams = useCallback(() => {
    return {
      patient_search: filters.patientSearch,
      institute: filters.institute,
      scanner_station: filters.scannerStation,
      protocol_name: filters.protocolName,
      scanner_model: filters.scannerModel,
      exam_date_from: filters.examDateFrom,
      exam_date_to: filters.examDateTo,
      patient_age_min: filters.ageMin,
      patient_age_max: filters.ageMax,
    };
  }, [filters]);

  const loadFilterOptions = useCallback(async () => {
    try {
      const data = await fetchJson("/cho-filter-options");
      if (mountedRef.current && data) {
        setFilterOptions(data);
      }
    } catch (error) {
      console.error("Failed to load filter options", error);
      setStatus(`Failed to load filter options: ${error.message}`, "error");
    }
  }, [setStatus]);

  const loadSummary = useCallback(
    async (overrides = {}) => {
      const targetPage = overrides.page ?? pagination.page;
      const targetLimit = overrides.limit ?? pagination.limit;

      setSummaryLoading(true);
      setStatus("Loading data...", "loading");

      try {
        const params = buildFilterParams();
        params.page = targetPage;
        params.limit = targetLimit;
        const search = toSearchParams(params);

        const [statsResponse, summaryResponse, orthancSeries] =
          await Promise.all([
            fetchJson("/results-statistics"),
            fetchJson(`/cho-results?${search.toString()}`),
            fetchJson("/series/"),
          ]);

        console.debug("Stats Response:", statsResponse);
        console.debug("Summary Response:", summaryResponse);
        console.debug("Orthanc Series:", orthancSeries);
        console.debug("mountedRef.current:", mountedRef.current);
        if (mountedRef.current) {
          if (statsResponse) {
            setStats({
              total: statsResponse.total_results_count ?? 0,
              detectability: statsResponse.detectability_count ?? 0,
              noise: statsResponse.global_noise_count ?? 0,
              errors: statsResponse.error_count ?? 0,
            });
          }

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
            "success"
          );
        }
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
    [buildFilterParams, pagination.page, pagination.limit, setStatus]
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
    [loadSummary]
  );

  const changePageSize = useCallback(
    (limit) => {
      setPagination((prev) => ({ ...prev, page: 1, limit }));
      loadSummary({ page: 1, limit });
    },
    [loadSummary]
  );

  const openDetails = useCallback(
    async (seriesId) => {
      setDetailsModal({ open: true, loading: true, seriesId, data: null });
      setStatus("Loading series details...", "loading");
      try {
        const data = await fetchJson(`/cho-results/${seriesId}`);
        if (mountedRef.current) {
          setDetailsModal({ open: true, loading: false, seriesId, data });
          setStatus("", "idle");
        }
      } catch (error) {
        console.error("Failed to load series details", error);
        if (mountedRef.current) {
          setDetailsModal({ open: true, loading: false, seriesId, data: null });
          setStatus(`Error loading details: ${error.message}`, "error");
        }
      }
    },
    [setStatus]
  );

  const closeDetails = useCallback(() => {
    setDetailsModal({
      open: false,
      loading: false,
      seriesId: null,
      data: null,
    });
  }, []);

  const openDeleteDialogAction = useCallback(
    (seriesId, calculationType, patientName) => {
      setDeleteDialog({
        open: true,
        seriesId,
        calculationType: calculationType ?? null,
        patientName: patientName ?? null,
        loading: false,
      });
    },
    []
  );

  const closeDeleteDialog = useCallback(() => {
    setDeleteDialog({
      open: false,
      seriesId: null,
      calculationType: null,
      patientName: null,
      loading: false,
    });
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!deleteDialog.seriesId) return;

    setDeleteDialog((prev) => ({ ...prev, loading: true }));
    setStatus("Deleting...", "loading");

    try {
      const query = deleteDialog.calculationType
        ? `?calculation_type=${encodeURIComponent(
            deleteDialog.calculationType
          )}`
        : "";
      await fetchJson(`/cho-results/${deleteDialog.seriesId}${query}`, {
        method: "DELETE",
      });
      if (mountedRef.current) {
        setStatus("Successfully deleted", "success");
        closeDeleteDialog();
        loadSummary();
      }
    } catch (error) {
      console.error("Failed to delete result", error);
      if (mountedRef.current) {
        setStatus(`Error deleting: ${error.message}`, "error");
        setDeleteDialog((prev) => ({ ...prev, loading: false }));
      }
    }
  }, [deleteDialog, closeDeleteDialog, loadSummary, setStatus]);

  const exportAllResults = useCallback(() => {
    setStatus("Exporting to CSV...", "loading");
    try {
      window.open("/cho-results-export", "_blank", "noopener");
      setStatus("Export initiated", "success");
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

  const checkChoCalculationStatus = useCallback(
    async (seriesUuidParam) => {
      const targetUuid = seriesUuidParam ?? choModal.seriesUuid;
      if (!targetUuid) return;
      try {
        const data = await fetchJson(
          `/cho-calculation-status?series_id=${encodeURIComponent(
            targetUuid
          )}&action=check`
        );
        if (!mountedRef.current) return;
        if (data.status === "running") {
          setChoModal((prev) => ({
            ...prev,
            stage: "progress",
            progress: {
              value: data.progress ?? prev.progress.value,
              message: data.message ?? prev.progress.message,
              stage: data.current_stage ?? prev.progress.stage,
            },
            pollError: null,
          }));
        } else if (data.status === "completed") {
          stopChoPolling();
          setChoModal((prev) => ({
            ...prev,
            stage: "results",
            progress: { value: 100, message: "Completed", stage: "finalizing" },
            results: data.results ?? prev.results,
            pollError: null,
          }));
        } else if (data.status === "failed") {
          stopChoPolling();
          setChoModal((prev) => ({
            ...prev,
            stage: "results",
            pollError: data.error ?? "Analysis failed",
            results: null,
          }));
        }
      } catch (error) {
        if (mountedRef.current) {
          setChoModal((prev) => ({ ...prev, pollError: error.message }));
        }
      }
    },
    [choModal.seriesUuid, stopChoPolling]
  );

  const startChoPolling = useCallback(
    (seriesUuid) => {
      stopChoPolling();
      pollingRef.current = setInterval(() => {
        checkChoCalculationStatus(seriesUuid);
      }, 1000);
    },
    [checkChoCalculationStatus, stopChoPolling]
  );

  const checkExistingChoCalculation = useCallback(
    async (seriesUuid) => {
      try {
        const data = await fetchJson(
          `/cho-calculation-status?series_id=${encodeURIComponent(
            seriesUuid
          )}&action=check`
        );
        if (!mountedRef.current) return;
        if (data.status === "running") {
          setChoModal((prev) => ({
            ...prev,
            stage: "progress",
            progress: {
              value: data.progress ?? 0,
              message: data.message ?? "Processing...",
              stage: data.current_stage ?? "analysis",
            },
            pollError: null,
          }));
          startChoPolling(seriesUuid);
        } else if (data.status === "completed") {
          setChoModal((prev) => ({
            ...prev,
            stage: "results",
            progress: { value: 100, message: "Completed", stage: "finalizing" },
            results: data.results ?? null,
            pollError: null,
          }));
        }
      } catch {
        // ignore
      }
    },
    [startChoPolling]
  );

  const openChoModal = useCallback(
    (series) => {
      if (!series) return;
      stopChoPolling();
      setChoModal({
        ...createInitialChoState(),
        open: true,
        seriesUuid: series.series_uuid,
        seriesSummary: series,
        params: { ...defaultChoParams, testType: "global" },
      });
      checkExistingChoCalculation(series.series_uuid);
    },
    [checkExistingChoCalculation, stopChoPolling]
  );

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

  const startChoAnalysis = useCallback(async () => {
    if (!choModal.seriesUuid) {
      setStatus("No series selected for analysis", "error");
      return;
    }
    const payload = serializeChoPayload(choModal.seriesUuid, choModal.params);
    setChoModal((prev) => ({
      ...prev,
      stage: "progress",
      progress: {
        value: 0,
        message: "Initializing...",
        stage: "initialization",
      },
      results: null,
      pollError: null,
    }));
    try {
      await fetchJson("/cho-analysis-modal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (mountedRef.current) {
        startChoPolling(choModal.seriesUuid);
      }
    } catch (error) {
      if (mountedRef.current) {
        setChoModal((prev) => ({
          ...prev,
          stage: "config",
          pollError: error.message,
        }));
        setStatus(`Failed to start analysis: ${error.message}`, "error");
      }
    }
  }, [choModal.seriesUuid, choModal.params, setStatus, startChoPolling]);

  const saveChoResults = useCallback(async () => {
    if (!choModal.results) {
      setStatus("No results to save", "error");
      return;
    }
    setChoModal((prev) => ({ ...prev, saving: true }));
    try {
      await fetchJson("/cho-save-results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient: choModal.results.patient,
          study: choModal.results.study,
          scanner: choModal.results.scanner,
          series: choModal.results.series,
          ct: choModal.results.ct,
          converted_results: choModal.results.results,
        }),
      });
      if (mountedRef.current) {
        setStatus("Results saved to database", "success");
        closeChoModal();
        loadSummary();
      }
    } catch (error) {
      if (mountedRef.current) {
        setChoModal((prev) => ({ ...prev, saving: false }));
        setStatus(`Failed to save results: ${error.message}`, "error");
      }
    }
  }, [choModal.results, closeChoModal, loadSummary, setStatus]);

  const discardChoResults = useCallback(async () => {
    if (!choModal.seriesUuid) return;
    try {
      await fetchJson(
        `/cho-calculation-status?series_id=${encodeURIComponent(
          choModal.seriesUuid
        )}&action=discard`
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
          }
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
    [setStatus]
  );

  const value = useMemo(() => {
    return {
      status,
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
      detailsModal,
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
        openDetails,
        closeDetails,
        openDeleteDialog: openDeleteDialogAction,
        closeDeleteDialog,
        confirmDelete,
        exportAllResults,
        exportSeries,
        openChoModal,
        closeChoModal,
        updateChoParam,
        startChoAnalysis,
        saveChoResults,
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
    detailsModal,
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
    openDetails,
    closeDetails,
    openDeleteDialogAction,
    closeDeleteDialog,
    confirmDelete,
    exportAllResults,
    exportSeries,
    openChoModal,
    closeChoModal,
    updateChoParam,
    startChoAnalysis,
    saveChoResults,
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
