import { useId, useMemo } from "react";
import Plot from "react-plotly.js";
import CheckBoxIcon from "@mui/icons-material/CheckBox";
import CheckBoxOutlineBlankIcon from "@mui/icons-material/CheckBoxOutlineBlank";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import FormControl from "@mui/material/FormControl";
import FormHelperText from "@mui/material/FormHelperText";
import IconButton from "@mui/material/IconButton";
import InputLabel from "@mui/material/InputLabel";
import ListItemText from "@mui/material/ListItemText";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";

import { SCALAR_METRICS, toFiniteNumber } from "../../utils/aggregateMetrics";
import { PLOT_CONFIG, PLOT_STYLE } from "../../theme/customizations/plotTheme";
import { useAggregatePlot } from "./useAggregatePlot";
import {
  MAX_PLOT_PROTOCOLS,
  PROTOCOL_GROUP_KEY,
  resolvePlotProtocols,
  selectPlotRecords,
} from "./protocolPlotState";

const ProtocolPlotCard = ({
  config,
  protocolGroups,
  protocolOptions,
  onConfigChange,
  loading,
  canRemove,
  onRemove,
}) => {
  const idPrefix = useId();
  const selectedProtocols = useMemo(
    () => resolvePlotProtocols(config.protocols, protocolOptions),
    [config.protocols, protocolOptions],
  );
  const plotRecords = useMemo(
    () => selectPlotRecords(protocolGroups, selectedProtocols),
    [protocolGroups, selectedProtocols],
  );
  const fixedConfig = useMemo(
    () => ({
      ...config,
      mode: config.mode === "box" ? "box" : "histogram",
      groupKey: PROTOCOL_GROUP_KEY,
    }),
    [config],
  );
  const { plot, xMetric } = useAggregatePlot(plotRecords, fixedConfig);
  const hasValues = useMemo(
    () =>
      plotRecords.some(
        (record) => toFiniteNumber(record[config.xKey]) !== null,
      ),
    [plotRecords, config.xKey],
  );
  const figure = useMemo(() => {
    if (!plot) return null;
    return {
      ...plot,
      // Shared bins make overlaid protocol histograms directly comparable.
      data:
        fixedConfig.mode === "histogram"
          ? plot.data.map((trace) => ({ ...trace, bingroup: "protocols" }))
          : plot.data,
      layout:
        fixedConfig.mode === "box"
          ? {
              ...plot.layout,
              title: {
                ...plot.layout.title,
                text: `${xMetric.label} by protocol`,
              },
              xaxis: {
                ...plot.layout.xaxis,
                type: "category",
                title: { text: "Protocol" },
                automargin: true,
              },
            }
          : plot.layout,
    };
  }, [plot, fixedConfig.mode, xMetric]);

  let emptyMessage =
    "No series in the selected protocols have values for this metric.";
  if (protocolGroups.length === 0) {
    emptyMessage =
      "No results match the current filters. Query above to load data.";
  } else if (selectedProtocols.length === 0) {
    emptyMessage = "Select at least one protocol to display this plot.";
  }

  return (
    <Paper variant='outlined' sx={{ p: 2, minWidth: 0 }}>
      <Stack spacing={2}>
        <Stack
          direction='row'
          alignItems='center'
          justifyContent='space-between'>
          <Typography variant='subtitle1' component='h4' fontWeight={600}>
            {config.title}
          </Typography>
          <Tooltip
            title={
              canRemove
                ? `Remove ${config.title}`
                : "Keep at least one plot in this column"
            }>
            <span>
              <IconButton
                size='small'
                aria-label={`Remove ${config.title}`}
                disabled={!canRemove}
                onClick={onRemove}>
                <DeleteOutlineIcon fontSize='small' />
              </IconButton>
            </span>
          </Tooltip>
        </Stack>
        <Stack direction='row' spacing={2}>
          <TextField
            select
            fullWidth
            size='small'
            id={`${idPrefix}-metric`}
            label='Metric'
            value={config.xKey}
            onChange={(event) => onConfigChange({ xKey: event.target.value })}>
            {SCALAR_METRICS.map((metric) => (
              <MenuItem key={metric.key} value={metric.key}>
                {metric.label}
              </MenuItem>
            ))}
          </TextField>
          <FormControl
            fullWidth
            size='small'
            disabled={loading || protocolOptions.length === 0}>
            <InputLabel id={`${idPrefix}-protocols-label`} shrink>
              Protocols to show
            </InputLabel>
            <Select
              labelId={`${idPrefix}-protocols-label`}
              id={`${idPrefix}-protocols`}
              multiple
              displayEmpty
              notched
              label='Protocols to show'
              aria-describedby={`${idPrefix}-protocols-helper`}
              value={selectedProtocols}
              onChange={(event) => {
                const value = event.target.value;
                // Autofill may emit one string. Do not split protocol names on commas.
                const next = Array.isArray(value) ? value : [value];
                onConfigChange({
                  protocols: resolvePlotProtocols(next, protocolOptions),
                });
              }}
              renderValue={(selected) => (
                <Typography
                  component='span'
                  variant='inherit'
                  noWrap
                  color={selected.length === 0 ? "text.secondary" : "inherit"}
                  sx={{
                    display: "block",
                    width: "100%",
                    minWidth: 0,
                  }}>
                  {selected.length === 0
                    ? "Choose protocols"
                    : selected.join(", ")}
                </Typography>
              )}
              SelectDisplayProps={{ title: selectedProtocols.join(", ") }}
              MenuProps={{ slotProps: { paper: { sx: { maxHeight: 360 } } } }}>
              {protocolOptions.map((protocol) => {
                const selected = selectedProtocols.includes(protocol);
                const SelectionIcon = selected
                  ? CheckBoxIcon
                  : CheckBoxOutlineBlankIcon;
                return (
                  <MenuItem
                    key={protocol}
                    value={protocol}
                    disabled={
                      selectedProtocols.length >= MAX_PLOT_PROTOCOLS &&
                      !selected
                    }>
                    {/* MenuItem owns selection semantics; icons are visual indicators only. */}
                    <SelectionIcon
                      fontSize='small'
                      aria-hidden='true'
                      sx={{
                        mr: 1.5,
                        color: selected ? "primary.main" : "text.secondary",
                      }}
                    />
                    <ListItemText
                      primary={protocol}
                      sx={{ whiteSpace: "normal", overflowWrap: "anywhere" }}
                    />
                  </MenuItem>
                );
              })}
            </Select>
            <FormHelperText id={`${idPrefix}-protocols-helper`}>
              {selectedProtocols.length}/{MAX_PLOT_PROTOCOLS} selected.
              {selectedProtocols.length >= MAX_PLOT_PROTOCOLS}
            </FormHelperText>
          </FormControl>
        </Stack>
        <Box
          sx={{
            height: 420,
            minWidth: 0,
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
              <CircularProgress
                size={28}
                aria-label={`Loading ${config.title}`}
              />
            </Stack>
          ) : figure && hasValues ? (
            <Plot
              data={figure.data}
              layout={figure.layout}
              config={PLOT_CONFIG}
              style={PLOT_STYLE}
              useResizeHandler
            />
          ) : (
            <Stack
              alignItems='center'
              justifyContent='center'
              sx={{ height: "100%", px: 2, textAlign: "center" }}>
              <Typography variant='body2' color='text.secondary'>
                {emptyMessage}
              </Typography>
            </Stack>
          )}
        </Box>
      </Stack>
    </Paper>
  );
};

export default ProtocolPlotCard;
