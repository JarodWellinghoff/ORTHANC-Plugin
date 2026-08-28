import { useCallback, useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { useSnackbar } from "notistack";
import { alpha } from "@mui/material/styles";

import FiltersPanel, { FILTER_FIELDS } from "./FiltersPanel";
import CustomizablePlotCard from "./plots/CustomizablePlotCard";
import PlotStatsTable from "./plots/PlotStatsTable";
import { useFilters } from "../../hooks/useFilters";
import { UNGROUPED_KEY, fetchAggregateRecords } from "../utils/aggregateMetrics";
import {
  buildChoQueryParams,
  fetchJson,
  hasStoredResults,
  normalizeChoRow,
} from "../utils/choResultsShared";

// ─────────────────────────────────────────────────────────────────────────────
// PlottingPage
//
// Cohort-plotting counterpart to ResultsPage. Instead of selecting rows out of
// a grid, every series matching the filters is pulled in and plotted — there
// is no patient-level identification here, so the patient ID / name filters
// are hidden. Four independently-configurable plots are embedded directly on
// the page (no popup dialog), each backed by the same fetched cohort; their
// generated summary tables live in a separate, scrollable section below.
// ─────────────────────────────────────────────────────────────────────────────

// `/cho-results` caps `limit` at 1000 server-side; page through it so a
// filtered cohort larger than that still gets plotted in full.
const PAGE_LIMIT = 1000;
const EMPTY_SET = new Set();

const PLOTTING_VISIBLE_FIELDS = [
  FILTER_FIELDS.institute,
  FILTER_FIELDS.protocolName,
  FILTER_FIELDS.scannerModel,
  FILTER_FIELDS.scannerStation,
  FILTER_FIELDS.studyDate,
  FILTER_FIELDS.age,
  FILTER_FIELDS.pullSchedule,
];

const DEFAULT_PLOT_CONFIGS = [
  {
    mode: "scatter",
    xKey: "ssde",
    yKey: "average_index_of_detectability",
    groupKey: UNGROUPED_KEY,
    showTrend: true,
  },
  {
    mode: "histogram",
    xKey: "average_index_of_detectability",
    yKey: "average_index_of_detectability",
    groupKey: UNGROUPED_KEY,
    showTrend: true,
  },
  {
    mode: "box",
    xKey: "ssde",
    yKey: "average_index_of_detectability",
    groupKey: "protocol_name",
    showTrend: true,
  },
  {
    mode: "scatter",
    xKey: "ctdivol_avg",
    yKey: "ssde",
    groupKey: UNGROUPED_KEY,
    showTrend: true,
  },
];

const fetchAllSummaryItems = async (filters) => {
  const baseParams = buildChoQueryParams(filters);
  const items = [];
  let page = 1;

  for (;;) {
    const searchParams = new URLSearchParams({
      ...baseParams,
      page: String(page),
      limit: String(PAGE_LIMIT),
    });
    const response = await fetchJson(`/cho-results?${searchParams.toString()}`);
    const pageItems = Array.isArray(response) ? response : (response?.data ?? []);
    items.push(...pageItems);

    const total = Array.isArray(response)
      ? pageItems.length
      : (response?.total ?? pageItems.length);

    if (pageItems.length === 0 || items.length >= total) break;
    page += 1;
  }

  return items;
};

const PlottingPage = () => {
  const { enqueueSnackbar } = useSnackbar();
  const { filters, updateFilter, resetFilters } = useFilters();

  const [plotConfigs, setPlotConfigs] = useState(DEFAULT_PLOT_CONFIGS);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [seriesCount, setSeriesCount] = useState(0);
  const [skippedCount, setSkippedCount] = useState(0);
  const [failedCount, setFailedCount] = useState(0);

  const updatePlotConfig = useCallback((index, patch) => {
    setPlotConfigs((prev) =>
      prev.map((config, i) => (i === index ? { ...config, ...patch } : config)),
    );
  }, []);

  const handleQuery = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const summaryItems = await fetchAllSummaryItems(filters);
      const normalized = summaryItems.map((item, index) =>
        normalizeChoRow(item, index, EMPTY_SET),
      );
      const plottable = normalized.filter(hasStoredResults);

      setSeriesCount(normalized.length);
      setSkippedCount(normalized.length - plottable.length);

      const { records: fetched, failures } = await fetchAggregateRecords(
        plottable,
      );
      setRecords(fetched);
      setFailedCount(failures.length);

      if (normalized.length === 0) {
        enqueueSnackbar("No series match these filters.", {
          variant: "warning",
        });
      }
    } catch (error) {
      console.error("Failed to load results for plotting", error);
      setLoadError(error.message ?? "Failed to load results");
      setRecords([]);
      setSeriesCount(0);
      setSkippedCount(0);
      setFailedCount(0);
    } finally {
      setLoading(false);
    }
  }, [enqueueSnackbar, filters]);

  useEffect(() => {
    handleQuery();
    // Mount only — subsequent loads go through the "Query" button.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Stack spacing={3}>
      {/* Hero */}
      <Box
        variant='outlined'
        sx={(theme) => ({
          position: "relative",
          overflow: "hidden",
          px: { xs: 3, md: 6 },
          py: { xs: 4, md: 7 },
          borderRadius: 1,
          border: "1px solid",
          borderColor: "divider",
          backgroundImage: `linear-gradient(135deg, ${alpha(
            theme.palette.secondary.main,
            0.14,
          )} 0%, ${alpha(theme.palette.primary.dark, 0.04)} 100%)`,
        })}>
        <Stack spacing={2.5} sx={{ maxWidth: 880, position: "relative" }}>
          <Typography
            variant='h3'
            component='h1'
            sx={{
              fontWeight: 600,
              letterSpacing: "-0.02em",
              fontSize: { xs: "2rem", md: "2.75rem" },
            }}>
            Plotting
          </Typography>
          <Typography
            variant='body1'
            color='text.secondary'
            sx={{ fontSize: "1.05rem", lineHeight: 1.65 }}>
            Filter the analyzed cohort by protocol, scanner, or institute and
            plot every matching series' metrics against each other — no
            per-case selection required.
          </Typography>
        </Stack>
      </Box>

      <FiltersPanel
        filters={filters}
        onChange={updateFilter}
        onQuery={handleQuery}
        onReset={resetFilters}
        visibleFields={PLOTTING_VISIBLE_FIELDS}
      />

      {loadError && <Alert severity='error'>{loadError}</Alert>}

      {!loading && !loadError && (
        <Typography variant='body2' color='text.secondary'>
          {records.length} of {seriesCount} matching series plotted
          {skippedCount > 0
            ? ` (${skippedCount} skipped — no stored results)`
            : ""}
          {failedCount > 0 ? ` (${failedCount} failed to load)` : ""}.
        </Typography>
      )}

      {/* Plots */}
      <Stack spacing={2}>
        <Typography variant='h5' component='h2' fontWeight={600}>
          Plots
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr" },
            gap: 3,
          }}>
          {plotConfigs.map((config, index) => (
            <CustomizablePlotCard
              key={index}
              title={`Plot ${index + 1}`}
              records={records}
              config={config}
              onConfigChange={(patch) => updatePlotConfig(index, patch)}
              loading={loading}
            />
          ))}
        </Box>
      </Stack>

      {/* Tables */}
      <Stack spacing={2}>
        <Typography variant='h5' component='h2' fontWeight={600}>
          Tables
        </Typography>
        <Paper
          variant='outlined'
          sx={{ p: 2, maxHeight: 640, overflowY: "auto" }}>
          <Stack spacing={3} divider={<Divider />}>
            {plotConfigs.map((config, index) => (
              <PlotStatsTable
                key={index}
                title={`Plot ${index + 1}`}
                records={records}
                config={config}
              />
            ))}
          </Stack>
        </Paper>
      </Stack>
    </Stack>
  );
};

export default PlottingPage;
