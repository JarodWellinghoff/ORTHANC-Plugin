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

const fetchJson = async (url, options) => {
  const response = await fetch(
    `${import.meta.env.VITE_API_URL}${url}`,
    options
  );
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
    protocol_name:
      series.protocol_name ??
      data.protocol_name ??
      fallback?.protocol_name ??
      null,
    institution_name:
      scanner.institution_name ??
      study.institution_name ??
      data.institution_name ??
      fallback?.institution_name ??
      null,
    scanner_model:
      scanner.model_name ??
      scanner.scanner_model ??
      data.scanner_model ??
      fallback?.scanner_model ??
      null,
    station_name:
      scanner.station_name ??
      data.station_name ??
      fallback?.station_name ??
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
    (value) => value !== null && value !== undefined && value !== ""
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
  const [deleteDialog, setDeleteDialog] = useState({
    open: false,
    seriesId: null,
    calculationType: null,
    patientName: null,
    loading: false,
  });
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

  const exportAllResults = useCallback(async () => {
    setStatus("Exporting to CSV...", "loading");
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/cho-results-export`,
        {
          method: "GET",
        }
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
              entry.series_id ?? entry.seriesUuid ?? entry.seriesInstanceUid;
            if (!rawId) return;
            const key = String(rawId);
            next[key] = {
              ...entry,
              eventType: entry.eventType ?? defaultEvent,
            };
          };
          if (Array.isArray(payload.history)) {
            payload.history.forEach((item) => register(item, "history"));
          }
          if (Array.isArray(payload.active)) {
            payload.active.forEach((item) => register(item, "snapshot"));
          }
          return next;
        });
      });

      source.addEventListener("cho-calculation", (event) => {
        if (!mountedRef.current) return;
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          console.debug("Failed to parse calculation event", error);
          return;
        }
        const rawId =
          payload?.series_id ??
          payload?.seriesUuid ??
          payload?.seriesInstanceUid ??
          null;
        if (!rawId) {
          return;
        }
        const key = String(rawId);
        setCalculationStates((prev) => {
          if (
            payload?.eventType === "cleanup" ||
            payload?.eventType === "history-cleanup" ||
            payload?.status === "removed"
          ) {
            if (!(key in prev)) {
              return prev;
            }
            const next = { ...prev };
            delete next[key];
            return next;
          }
          const nextState = { ...(prev[key] ?? {}), ...payload };
          return { ...prev, [key]: nextState };
        });
      });

      source.onerror = () => {
        if (!mountedRef.current) {
          return;
        }
        if (sseReconnectRef.current) {
          return;
        }
        source.close();
        sseRef.current = null;
        sseReconnectRef.current = setTimeout(() => {
          sseReconnectRef.current = null;
          connect();
        }, 5000);
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
  }, [stopChoPolling]);

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
          //   startChoPolling(seriesUuid);
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

  const activeChoSeriesId = choModal.seriesUuid
    ? String(choModal.seriesUuid)
    : null;

  useEffect(() => {
    if (!activeChoSeriesId) {
      return;
    }
    const state = calculationStates[activeChoSeriesId];
    if (!state) {
      return;
    }

    setChoModal((prev) => {
      if (prev.seriesUuid !== activeChoSeriesId) {
        return prev;
      }

      const status = state.status ?? state.eventType;

      if (status === "running") {
        return {
          ...prev,
          stage: "progress",
          progress: {
            value:
              typeof state.progress === "number"
                ? state.progress
                : prev.progress.value,
            message: state.message ?? prev.progress.message ?? "Processing...",
            stage: state.current_stage ?? prev.progress.stage ?? "analysis",
          },
          pollError: null,
        };
      }

      if (status === "completed") {
        stopChoPolling();
        return {
          ...prev,
          stage: "results",
          progress: {
            value: 100,
            message: state.message ?? "Completed",
            stage: state.current_stage ?? "completed",
          },
          results: state.results ?? prev.results,
          pollError: null,
        };
      }

      if (status === "failed") {
        stopChoPolling();
        return {
          ...prev,
          stage: "results",
          progress: {
            value:
              typeof state.progress === "number"
                ? state.progress
                : prev.progress.value ?? 0,
            message: state.message ?? "Failed",
            stage: state.current_stage ?? "failed",
          },
          pollError: state.error ?? state.message ?? "Analysis failed",
          results: null,
        };
      }

      if (status === "cancelled") {
        stopChoPolling();
        return {
          ...prev,
          stage: "results",
          progress: {
            value:
              typeof state.progress === "number"
                ? state.progress
                : prev.progress.value ?? 0,
            message: state.message ?? "Cancelled",
            stage: state.current_stage ?? "cancelled",
          },
          pollError: state.message ?? "Analysis cancelled",
        };
      }

      return prev;
    });
  }, [activeChoSeriesId, calculationStates, stopChoPolling]);

  const loadStoredChoResults = useCallback(
    async (seriesId) => {
      if (!seriesId) {
        setChoModal((prev) => ({
          ...prev,
          storedResults: { loading: false, data: null, error: null },
        }));
        return;
      }
      setChoModal((prev) => ({
        ...prev,
        storedResults: { ...prev.storedResults, loading: true, error: null },
      }));
      setStatus("Loading series details...", "loading");
      try {
        const data = await fetchJson(`/cho-results/${seriesId}`);
        if (!mountedRef.current) return;
        setChoModal((prev) => {
          const derivedSummary = deriveSeriesSummary(data, prev.seriesSummary);
          const nextSeriesUuid =
            derivedSummary?.series_uuid ?? prev.seriesUuid ?? null;
          const nextSeriesId =
            derivedSummary?.series_id ?? prev.seriesId ?? null;
          return {
            ...prev,
            seriesUuid: nextSeriesUuid,
            seriesId: nextSeriesId,
            seriesSummary: derivedSummary ?? prev.seriesSummary,
            storedResults: { loading: false, data, error: null },
          };
        });
        setStatus("", "idle");
      } catch (error) {
        if (!mountedRef.current) return;
        setChoModal((prev) => ({
          ...prev,
          storedResults: { loading: false, data: null, error: error.message },
        }));
        setStatus(`Error loading details: ${error.message}`, "error");
      }
    },
    [setStatus]
  );

  const reloadStoredChoResults = useCallback(() => {
    if (choModal.seriesId) {
      loadStoredChoResults(choModal.seriesId);
    }
  }, [choModal.seriesId, loadStoredChoResults]);

  const openChoModal = useCallback(
    (series) => {
      if (!series) return;
      stopChoPolling();
      const seriesUuid =
        series.series_uuid ??
        series.series_instance_uid ??
        series.seriesId ??
        null;
      const seriesId =
        series.series_id ??
        series.seriesId ??
        series.series_instance_uid ??
        null;
      const storedResultsState = seriesId
        ? { loading: true, data: null, error: null }
        : {
            loading: false,
            data: null,
            error: "Stored CHO results unavailable for this series.",
          };
      setChoModal({
        ...createInitialChoState(),
        open: true,
        seriesUuid,
        seriesId,
        seriesSummary: series,
        params: { ...defaultChoParams, testType: series.testType ?? "full" },
        storedResults: storedResultsState,
      });
      if (seriesId) {
        loadStoredChoResults(seriesId);
      }
      if (seriesUuid) {
        checkExistingChoCalculation(seriesUuid);
      }
    },
    [checkExistingChoCalculation, loadStoredChoResults, stopChoPolling]
  );

  const closeChoModal = useCallback(() => {
    stopChoPolling();
    setChoModal(createInitialChoState());
  }, [stopChoPolling]);

  const updateChoParam = useCallback((name, value) => {
    console.debug("Updating CHO param", name, value);
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
      //   if (mountedRef.current) {
      //     startChoPolling(choModal.seriesUuid);
      //   }
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
  }, [choModal.seriesUuid, choModal.params, setStatus]);

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
