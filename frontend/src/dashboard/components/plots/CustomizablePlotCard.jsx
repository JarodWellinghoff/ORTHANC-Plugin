// ─────────────────────────────────────────────────────────────────────────────
// CustomizablePlotCard.jsx
//
// One embedded, independently-configurable cohort plot (scatter / histogram /
// box) — the plotting-page equivalent of the controls + chart half of the old
// AggregatePlotsDialog, minus the dialog chrome and minus the stats table
// (that lives in PlotStatsTable, in its own page section).
// ─────────────────────────────────────────────────────────────────────────────

import * as React from "react";
import Plot from "react-plotly.js";

import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Typography from "@mui/material/Typography";

import { GROUP_FIELDS, SCALAR_METRICS } from "../../utils/aggregateMetrics";
import { PLOT_CONFIG, PLOT_STYLE } from "../../theme/customizations/plotTheme";
import { useAggregatePlot } from "./useAggregatePlot";

export const PLOT_CARD_HEIGHT = 420;

const MODES = [
  { key: "scatter", label: "Scatter" },
  { key: "histogram", label: "Histogram" },
  { key: "box", label: "Box" },
];

const CustomizablePlotCard = ({
  title,
  records,
  config,
  onConfigChange,
  loading,
}) => {
  const { plot, isScatter } = useAggregatePlot(records, config);
  const idPrefix = React.useId();

  const set = (patch) => onConfigChange(patch);

  return (
    <Paper variant='outlined' sx={{ p: 2, height: "100%" }}>
      <Stack spacing={2}>
        <Typography variant='subtitle1' fontWeight={600}>
          {title}
        </Typography>

        <Stack
          direction='row'
          spacing={1.5}
          alignItems='center'
          flexWrap='wrap'
          useFlexGap>
          <ToggleButtonGroup
            size='small'
            exclusive
            value={config.mode}
            onChange={(_, next) => next && set({ mode: next })}>
            {MODES.map((item) => (
              <ToggleButton key={item.key} value={item.key}>
                {item.label}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>

          <FormControl size='small' sx={{ minWidth: 190 }}>
            <InputLabel id={`${idPrefix}-x-label`}>
              {isScatter ? "X axis" : "Metric"}
            </InputLabel>
            <Select
              labelId={`${idPrefix}-x-label`}
              label={isScatter ? "X axis" : "Metric"}
              value={config.xKey}
              onChange={(event) => set({ xKey: event.target.value })}>
              {SCALAR_METRICS.map((metric) => (
                <MenuItem key={metric.key} value={metric.key}>
                  {metric.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {isScatter && (
            <FormControl size='small' sx={{ minWidth: 190 }}>
              <InputLabel id={`${idPrefix}-y-label`}>Y axis</InputLabel>
              <Select
                labelId={`${idPrefix}-y-label`}
                label='Y axis'
                value={config.yKey}
                onChange={(event) => set({ yKey: event.target.value })}>
                {SCALAR_METRICS.map((metric) => (
                  <MenuItem key={metric.key} value={metric.key}>
                    {metric.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          <FormControl size='small' sx={{ minWidth: 170 }}>
            <InputLabel id={`${idPrefix}-group-label`}>Group by</InputLabel>
            <Select
              labelId={`${idPrefix}-group-label`}
              label='Group by'
              value={config.groupKey}
              onChange={(event) => set({ groupKey: event.target.value })}>
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
                  checked={config.showTrend}
                  onChange={(event) =>
                    set({ showTrend: event.target.checked })
                  }
                />
              }
              label='Trend'
            />
          )}
        </Stack>

        <Box
          sx={{
            height: PLOT_CARD_HEIGHT,
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            p: 1,
          }}>
          {loading ? (
            <Stack
              alignItems='center'
              justifyContent='center'
              sx={{ height: "100%" }}>
              <CircularProgress size={28} />
            </Stack>
          ) : records.length === 0 ? (
            <Stack
              alignItems='center'
              justifyContent='center'
              sx={{ height: "100%", px: 2, textAlign: "center" }}>
              <Typography variant='body2' color='text.secondary'>
                No results match the current filters yet. Query above to load
                data.
              </Typography>
            </Stack>
          ) : plot ? (
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
                No series in this selection carry values for the chosen
                metrics.
              </Typography>
            </Stack>
          )}
        </Box>
      </Stack>
    </Paper>
  );
};

export default CustomizablePlotCard;
