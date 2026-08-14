// ─────────────────────────────────────────────────────────────────────────────
// AggregatePlotsDialog.jsx
//
// Cohort view for the "Plot Selected" action on ResultsPage. Takes the selected
// grid rows, pulls each series' stored record from `/cho-results/{id}`, and
// plots the scalar metrics across the whole selection.
//
// Only scalar columns are offered. The array columns — ctdivol, dw, location,
// nps, cho_detectability, spatial_frequency, ssde_inc — are per-slice or
// per-frequency and already have a home in ChoPlots on the single-series view;
// there is no meaningful way to put a cohort's worth of them on one axis.
//
// Three modes cover the questions this data gets asked:
//   Scatter   — one metric against another, e.g. SSDE against d′, which is the
//               dose-versus-quality trade-off the platform exists to measure.
//   Histogram — how a single metric is distributed across the cohort.
//   Box       — how a single metric shifts between protocols, scanners, or
//               kernels.
// ─────────────────────────────────────────────────────────────────────────────

import * as React from "react";
import Plot from "react-plotly.js";

import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import IconButton from "@mui/material/IconButton";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Typography from "@mui/material/Typography";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import FileDownloadIcon from "@mui/icons-material/FileDownload";

import {
  GROUP_FIELDS,
  GROUP_PALETTE,
  SCALAR_METRICS,
  SCALAR_METRIC_MAP,
  UNGROUPED_KEY,
  buildAggregateCsv,
  describe,
  fetchAggregateRecords,
  formatValue,
  groupRecords,
  linearFit,
  pearson,
  toFiniteNumber,
} from "../../utils/aggregateMetrics";
import {
  PLOT_CONFIG,
  PLOT_STYLE,
  useBuildLayout,
} from "../../theme/customizations/plotTheme";

const PLOT_HEIGHT = 460;

const MODES = [
  { key: "scatter", label: "Scatter" },
  { key: "histogram", label: "Histogram" },
  { key: "box", label: "Box" },
];

const AggregatePlotsDialog = ({ open, onClose, rows = [] }) => {
  const buildLayout = useBuildLayout();

  const [loading, setLoading] = React.useState(false);
  const [records, setRecords] = React.useState([]);
  const [failures, setFailures] = React.useState([]);
  const [loadError, setLoadError] = React.useState(null);

  const [mode, setMode] = React.useState("scatter");
  const [xKey, setXKey] = React.useState("ssde");
  const [yKey, setYKey] = React.useState("average_index_of_detectability");
  const [groupKey, setGroupKey] = React.useState(UNGROUPED_KEY);
  const [showTrend, setShowTrend] = React.useState(true);

  // Fetch once per open. An AbortController-style `cancelled` flag keeps a
  // closed-then-reopened dialog from writing stale records into fresh state.
  React.useEffect(() => {
    if (!open || rows.length === 0) return undefined;

    let cancelled = false;
    const signal = { aborted: false };

    const run = async () => {
      setLoading(true);
      setLoadError(null);
      try {
        const result = await fetchAggregateRecords(rows, { signal });
        if (cancelled) return;
        setRecords(result.records);
        setFailures(result.failures);
      } catch (error) {
        if (!cancelled) {
          setLoadError(error.message ?? "Could not load results");
          setRecords([]);
          setFailures([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => {
      cancelled = true;
      signal.aborted = true;
    };
  }, [open, rows]);

  const xMetric = SCALAR_METRIC_MAP.get(xKey);
  const yMetric = SCALAR_METRIC_MAP.get(yKey);
  const isScatter = mode === "scatter";

  const groups = React.useMemo(
    () => groupRecords(records, groupKey),
    [records, groupKey],
  );

  // ── Traces ────────────────────────────────────────────────────────────────

  const plot = React.useMemo(() => {
    if (records.length === 0 || !xMetric) return null;

    const colorFor = (index) => GROUP_PALETTE[index % GROUP_PALETTE.length];

    if (isScatter) {
      if (!yMetric) return null;

      const traces = groups.map((group, index) => {
        const points = group.records
          .map((record) => ({
            x: toFiniteNumber(record[xMetric.key]),
            y: toFiniteNumber(record[yMetric.key]),
            label: record.__label,
          }))
          .filter((point) => point.x !== null && point.y !== null);

        return {
          name: `${group.label} (n=${points.length})`,
          x: points.map((p) => p.x),
          y: points.map((p) => p.y),
          text: points.map((p) => p.label),
          hovertemplate:
            "%{text}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>",
          type: "scatter",
          mode: "markers",
          marker: {
            size: 9,
            color: colorFor(index),
            line: { width: 1, color: "rgba(0,0,0,0.35)" },
          },
        };
      });

      if (showTrend) {
        const allX = records.map((record) => record[xMetric.key]);
        const allY = records.map((record) => record[yMetric.key]);
        const fit = linearFit(allX, allY);
        if (fit) {
          traces.push({
            name: `Fit: y = ${fit.slope.toPrecision(3)}x ${
              fit.intercept >= 0 ? "+" : "−"
            } ${Math.abs(fit.intercept).toPrecision(3)}`,
            x: fit.x,
            y: fit.y,
            type: "scatter",
            mode: "lines",
            hoverinfo: "skip",
            line: { width: 2, dash: "dash", color: "#7f7f7f" },
          });
        }
      }

      return {
        data: traces,
        layout: buildLayout({
          title: { text: `${yMetric.label} vs ${xMetric.label}` },
          xaxis: { title: { text: xMetric.axisLabel } },
          yaxis: { title: { text: yMetric.axisLabel } },
          showlegend: true,
        }),
      };
    }

    if (mode === "histogram") {
      const traces = groups.map((group, index) => {
        const values = group.records
          .map((record) => toFiniteNumber(record[xMetric.key]))
          .filter((value) => value !== null);
        return {
          name: `${group.label} (n=${values.length})`,
          x: values,
          type: "histogram",
          opacity: groups.length > 1 ? 0.6 : 0.85,
          marker: {
            color: colorFor(index),
            line: { width: 1, color: "rgba(0,0,0,0.25)" },
          },
        };
      });

      return {
        data: traces,
        layout: buildLayout({
          title: { text: `Distribution of ${xMetric.label}` },
          xaxis: { title: { text: xMetric.axisLabel } },
          yaxis: { title: { text: "Series count" } },
          barmode: "overlay",
          showlegend: groups.length > 1,
        }),
      };
    }

    // Box
    const traces = groups.map((group, index) => {
      const values = group.records
        .map((record) => toFiniteNumber(record[xMetric.key]))
        .filter((value) => value !== null);
      return {
        name: `${group.label} (n=${values.length})`,
        y: values,
        type: "box",
        boxpoints: "outliers",
        marker: { color: colorFor(index), size: 6 },
        line: { width: 2 },
      };
    });

    return {
      data: traces,
      layout: buildLayout({
        title: { text: `${xMetric.label} by group` },
        yaxis: { title: { text: xMetric.axisLabel } },
        xaxis: { title: { text: "" } },
        showlegend: false,
      }),
    };
  }, [
    buildLayout,
    groups,
    isScatter,
    mode,
    records,
    showTrend,
    xMetric,
    yMetric,
  ]);

  // ── Summary table ─────────────────────────────────────────────────────────

  // One row per group per metric. Scatter reports both axes so the two
  // distributions behind a correlation are visible alongside it.
  const statRows = React.useMemo(() => {
    const metrics = isScatter ? [xMetric, yMetric] : [xMetric];
    const output = [];
    for (const metric of metrics) {
      if (!metric) continue;
      for (const group of groups) {
        const stats = describe(
          group.records.map((record) => record[metric.key]),
        );
        output.push({
          id: `${metric.key}-${group.label}`,
          metric: metric.label,
          unit: metric.unit,
          group: group.label,
          decimals: metric.decimals,
          ...stats,
        });
      }
    }
    return output;
  }, [groups, isScatter, xMetric, yMetric]);

  const correlation = React.useMemo(() => {
    if (!isScatter || !xMetric || !yMetric || records.length === 0) return null;
    return pearson(
      records.map((record) => record[xMetric.key]),
      records.map((record) => record[yMetric.key]),
    );
  }, [isScatter, records, xMetric, yMetric]);

  // ── CSV ───────────────────────────────────────────────────────────────────

  const handleDownloadCsv = React.useCallback(() => {
    if (records.length === 0) return;
    const csv = buildAggregateCsv(records);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.href = url;
    link.download = `cho-aggregate-${records.length}-series-${stamp}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }, [records]);

  // ── Render ────────────────────────────────────────────────────────────────

  const renderBody = () => {
    if (loading) {
      return (
        <Stack
          alignItems='center'
          justifyContent='center'
          spacing={2}
          sx={{ height: PLOT_HEIGHT }}>
          <CircularProgress />
          <Typography variant='body2' color='text.secondary'>
            Loading results for {rows.length}{" "}
            {rows.length === 1 ? "series" : "series"}…
          </Typography>
        </Stack>
      );
    }

    if (loadError) {
      return (
        <Alert severity='error' sx={{ my: 2 }}>
          {loadError}
        </Alert>
      );
    }

    if (records.length === 0) {
      return (
        <Stack
          alignItems='center'
          justifyContent='center'
          spacing={1}
          sx={{ height: PLOT_HEIGHT }}>
          <Typography variant='subtitle1'>
            No stored results for this selection
          </Typography>
          <Typography variant='body2' color='text.secondary'>
            Run an analysis on these series from the Tests page, then plot them
            here.
          </Typography>
        </Stack>
      );
    }

    return (
      <Stack spacing={2}>
        {failures.length > 0 && (
          <Alert severity='warning'>
            {failures.length} of {rows.length} selected series could not be
            loaded and are excluded from the plot.
          </Alert>
        )}

        <Box
          sx={{
            height: PLOT_HEIGHT,
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            p: 1,
          }}>
          {plot ? (
            <Plot
              data={plot.data}
              layout={plot.layout}
              config={PLOT_CONFIG}
              style={PLOT_STYLE}
              useResizeHandler
            />
          ) : (
            <Stack
              alignItems='center'
              justifyContent='center'
              sx={{ height: "100%" }}>
              <Typography variant='body2' color='text.secondary'>
                No series in this selection carry values for the chosen metrics.
              </Typography>
            </Stack>
          )}
        </Box>

        {correlation?.r !== null && correlation?.r !== undefined && (
          <Typography variant='body2' color='text.secondary'>
            Pearson r = {correlation.r.toFixed(3)} across {correlation.n}{" "}
            complete pairs. Correlation only; it does not establish a causal
            relationship between these metrics.
          </Typography>
        )}

        <Divider />

        <TableContainer>
          <Table size='small'>
            <TableHead>
              <TableRow>
                <TableCell>Metric</TableCell>
                <TableCell>Group</TableCell>
                <TableCell align='right'>n</TableCell>
                <TableCell align='right'>Mean</TableCell>
                <TableCell align='right'>SD</TableCell>
                <TableCell align='right'>Median</TableCell>
                <TableCell align='right'>Min</TableCell>
                <TableCell align='right'>Max</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {statRows.map((row) => (
                <TableRow key={row.id}>
                  <TableCell>
                    {row.metric}
                    {row.unit ? (
                      <Typography
                        component='span'
                        variant='caption'
                        color='text.secondary'
                        sx={{ ml: 0.5 }}>
                        ({row.unit})
                      </Typography>
                    ) : null}
                  </TableCell>
                  <TableCell>{row.group}</TableCell>
                  <TableCell align='right'>{row.n}</TableCell>
                  <TableCell align='right'>
                    {formatValue(row.mean, row.decimals)}
                  </TableCell>
                  <TableCell align='right'>
                    {formatValue(row.sd, row.decimals)}
                  </TableCell>
                  <TableCell align='right'>
                    {formatValue(row.median, row.decimals)}
                  </TableCell>
                  <TableCell align='right'>
                    {formatValue(row.min, row.decimals)}
                  </TableCell>
                  <TableCell align='right'>
                    {formatValue(row.max, row.decimals)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Stack>
    );
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth='xl' fullWidth>
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          pr: 1,
        }}>
        <Stack>
          <Typography variant='h6' component='span'>
            Aggregate results
          </Typography>
          <Typography variant='body2' color='text.secondary'>
            {records.length} of {rows.length} selected series loaded
          </Typography>
        </Stack>
        <IconButton aria-label='Close aggregate plots' onClick={onClose}>
          <CloseRoundedIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers>
        <Stack spacing={2}>
          <Stack
            direction='row'
            spacing={2}
            alignItems='center'
            flexWrap='wrap'
            useFlexGap>
            <ToggleButtonGroup
              size='small'
              exclusive
              value={mode}
              onChange={(_, next) => next && setMode(next)}>
              {MODES.map((item) => (
                <ToggleButton key={item.key} value={item.key}>
                  {item.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            <FormControl size='small' sx={{ minWidth: 220 }}>
              <InputLabel id='aggregate-x-label'>
                {isScatter ? "X axis" : "Metric"}
              </InputLabel>
              <Select
                labelId='aggregate-x-label'
                label={isScatter ? "X axis" : "Metric"}
                value={xKey}
                onChange={(event) => setXKey(event.target.value)}>
                {SCALAR_METRICS.map((metric) => (
                  <MenuItem key={metric.key} value={metric.key}>
                    {metric.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {isScatter && (
              <FormControl size='small' sx={{ minWidth: 220 }}>
                <InputLabel id='aggregate-y-label'>Y axis</InputLabel>
                <Select
                  labelId='aggregate-y-label'
                  label='Y axis'
                  value={yKey}
                  onChange={(event) => setYKey(event.target.value)}>
                  {SCALAR_METRICS.map((metric) => (
                    <MenuItem key={metric.key} value={metric.key}>
                      {metric.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            <FormControl size='small' sx={{ minWidth: 200 }}>
              <InputLabel id='aggregate-group-label'>Group by</InputLabel>
              <Select
                labelId='aggregate-group-label'
                label='Group by'
                value={groupKey}
                onChange={(event) => setGroupKey(event.target.value)}>
                {GROUP_FIELDS.map((field) => (
                  <MenuItem key={field.key} value={field.key}>
                    {field.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {isScatter && (
              <FormControlLabel
                control={
                  <Switch
                    size='small'
                    checked={showTrend}
                    onChange={(event) => setShowTrend(event.target.checked)}
                  />
                }
                label='Trend line'
              />
            )}
          </Stack>

          {renderBody()}
        </Stack>
      </DialogContent>

      <DialogActions>
        <Button
          startIcon={<FileDownloadIcon />}
          onClick={handleDownloadCsv}
          disabled={records.length === 0}>
          Download CSV
        </Button>
        <Button variant='contained' onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AggregatePlotsDialog;
