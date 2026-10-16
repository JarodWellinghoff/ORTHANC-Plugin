import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, Stack, Typography } from "@mui/material";
import FiltersPanel, { FILTER_FIELDS } from "./FiltersPanel";
import PlottingWorkspace from "./plots/PlottingWorkspace";
import { normalizeProtocolName } from "./plots/protocolPlotState";
import { useDashboard } from "../context/DashboardContext";
import { useFilters } from "../../hooks/useFilters";
import { fetchAggregateRecords } from "../utils/aggregateMetrics";
import { buildChoQueryParams, fetchJson, hasStoredResults, normalizeChoRow } from "../utils/choResultsShared";
const PAGE_LIMIT = 1000;
const EMPTY_SET = new Set();
const VISIBLE_FIELDS = [FILTER_FIELDS.institute, FILTER_FIELDS.protocolName, FILTER_FIELDS.scannerModel,
  FILTER_FIELDS.scannerStation, FILTER_FIELDS.studyDate, FILTER_FIELDS.age];
const fetchAllSummaryItems = async (filters, signal) => {
  const params = buildChoQueryParams({ ...filters, protocolNameSearch: [] });
  const items = [];
  for (let page = 1; ; page += 1) {
    const search = new URLSearchParams({ ...params, page: String(page), limit: String(PAGE_LIMIT) });
    const response = await fetchJson(`/cho-results?${search}`, { signal });
    const batch = Array.isArray(response) ? response : response?.data ?? [];
    items.push(...batch);
    const total = Array.isArray(response) ? batch.length : response?.total ?? batch.length;
    if (!batch.length || items.length >= total) return items;
  }
};
export default function PlottingPage() {
  const { filters, updateFilter, resetFilters } = useFilters();
  const { filterOptions } = useDashboard();
  const protocolOptions = useMemo(() => [...new Set((filterOptions.protocol_names ?? []).map(normalizeProtocolName))]
    .sort((a, b) => a.localeCompare(b)), [filterOptions.protocol_names]);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [failed, setFailed] = useState(0);
  const currentRequest = useRef(null);
  const lastQuery = useRef(filters);
  const query = useCallback(async (queryFilters) => {
    currentRequest.current?.abort();
    const controller = new AbortController();
    currentRequest.current = controller;
    const { signal } = controller;
    lastQuery.current = queryFilters;
    setLoading(true); setError(null); setFailed(0);
    try {
      const allItems = await fetchAllSummaryItems(queryFilters, signal);
      if (signal.aborted) return;
      const selected = new Set((queryFilters.protocolNameSearch ?? []).map(normalizeProtocolName));
      const items = selected.size ? allItems.filter((item) => selected.has(normalizeProtocolName(item.protocol_name))) : allItems;
      const rows = items.map((item, index) => normalizeChoRow(item, index, EMPTY_SET)).filter(hasStoredResults);
      const result = await fetchAggregateRecords(rows, { signal });
      if (signal.aborted) return;
      setRecords(result.records); setFailed(result.failures.length);
    } catch (reason) {
      if (signal.aborted) return;
      setError(reason.message ?? "Failed to load results"); setRecords([]);
    } finally { if (!signal.aborted) setLoading(false); }
  }, []);
  useEffect(() => {
    query(lastQuery.current);
    const onReviewed = (event) => { if (event.detail?.decision === "overwrite") query(lastQuery.current); };
    window.addEventListener("ctpro-review-resolved", onReviewed);
    return () => { currentRequest.current?.abort(); window.removeEventListener("ctpro-review-resolved", onReviewed); };
  }, [query]);
  return <Stack spacing={3}>
    <Typography variant='h4' component='h1'>Statistics Report</Typography>
    <FiltersPanel filters={{ ...filters, protocolNameSearch: [...new Set((filters.protocolNameSearch ?? []).map(normalizeProtocolName))] }}
      filterOptions={{ ...filterOptions, protocol_names: protocolOptions }}
      onChange={(name, value) => updateFilter(name, name === "protocolNameSearch" ? [...new Set(value.map(normalizeProtocolName))] : value)}
      onQuery={() => query(filters)} onReset={resetFilters} visibleFields={VISIBLE_FIELDS} />
    {error && <Alert severity='error'>{error}</Alert>}
    {failed > 0 && <Alert severity='warning'>{failed} series could not be loaded and are not included.</Alert>}
    <PlottingWorkspace records={records} loading={loading} />
  </Stack>;
}
