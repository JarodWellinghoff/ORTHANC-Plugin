import { useId } from "react";
import Plot from "react-plotly.js";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { Box, Checkbox, CircularProgress, FormControl, FormHelperText, IconButton,
  InputLabel, ListItemText, MenuItem, Paper, Select, Stack, TextField, Tooltip, Typography } from "@mui/material";
import { SCALAR_METRICS } from "../../utils/aggregateMetrics";
import { PLOT_CONFIG, PLOT_STYLE, useBuildLayout } from "../../theme/customizations/plotTheme";
import { MAX_PLOT_PROTOCOLS, resolvePlotProtocols } from "./protocolPlotState";
import { buildProtocolTraces, countPatients, finiteNumber, getReportMetrics, PATIENT_COUNT_KEY } from "./protocolReport";

const METRICS = getReportMetrics(SCALAR_METRICS);
export default function ProtocolPlotCard({ config, protocolGroups, onConfigChange, loading, canRemove, onRemove }) {
  const id = useId();
  const buildLayout = useBuildLayout();
  const multiple = config.mode === "box";
  const metric = METRICS.find((item) => item.key === config.xKey) ?? METRICS[0];
  const patientCount = metric.key === PATIENT_COUNT_KEY;
  const selected = resolvePlotProtocols(config.protocols, protocolGroups, config.mode);
  const groups = protocolGroups.filter((group) => selected.includes(group.label));
  const missing = groups.reduce((sum, group) => sum + countPatients(group.records).missing, 0);
  const hasValues = patientCount ? groups.length > 0 : groups.some((group) =>
    group.records.some((record) => finiteNumber(record[metric.key]) !== null));
  const figure = { data: buildProtocolTraces(groups, metric, config.mode), layout: buildLayout({
    title: { text: "" }, margin: { t: 16, b: 64, l: 64, r: 16 },
    xaxis: multiple || patientCount ? { type: "category", title: { text: "Protocol" }, automargin: true }
      : { title: { text: metric.axisLabel }, automargin: true },
    yaxis: { title: { text: patientCount || multiple ? metric.axisLabel : "Series count" }, automargin: true,
      ...(!multiple || patientCount ? { rangemode: "tozero" } : {}), ...(patientCount ? { tickformat: ",d" } : {}) },
    showlegend: false,
  }) };
  const label = multiple ? "Protocols" : "Protocol";
  const chartLabel = `${metric.label} ${multiple ? "by protocol" : "for " + (selected[0] ?? "a protocol")}`;
  return <Paper variant='outlined' sx={{ p: 2, minWidth: 0 }}><Stack spacing={2}>
    <Stack direction='row' spacing={1} alignItems='flex-start'>
      <TextField select fullWidth size='small' label='Metric' id={`${id}-metric`} value={metric.key}
        onChange={(event) => onConfigChange({ xKey: event.target.value })}>
        {METRICS.map((item) => <MenuItem key={item.key} value={item.key}>{item.label}</MenuItem>)}
      </TextField>
      <FormControl fullWidth size='small' disabled={loading || protocolGroups.length === 0}>
        <InputLabel id={`${id}-protocol-label`} shrink>{label}</InputLabel>
        <Select labelId={`${id}-protocol-label`} label={label} multiple={multiple} displayEmpty notched
          value={multiple ? selected : selected[0] ?? ""}
          onChange={(event) => onConfigChange({ protocols: resolvePlotProtocols(
            Array.isArray(event.target.value) ? event.target.value : [event.target.value], protocolGroups, config.mode) })}
          renderValue={(value) => <Typography component='span' variant='inherit' noWrap sx={{ display: "block" }}>
            {(Array.isArray(value) ? value.join(", ") : value) || "Choose protocols"}
          </Typography>}
          SelectDisplayProps={{ title: selected.join(", ") }} MenuProps={{ slotProps: { paper: { sx: { maxHeight: 360 } } } }}>
          {protocolGroups.map((group) => <MenuItem key={group.label} value={group.label}
            disabled={multiple && selected.length >= MAX_PLOT_PROTOCOLS && !selected.includes(group.label)}>
            {multiple && <Checkbox checked={selected.includes(group.label)} tabIndex={-1} disableRipple sx={{ pointerEvents: "none" }} />}
            <ListItemText primary={group.label} secondary={`${group.records.length} series`} sx={{ whiteSpace: "normal", overflowWrap: "anywhere" }} />
          </MenuItem>)}
        </Select>
        {multiple && <FormHelperText>{selected.length}/{MAX_PLOT_PROTOCOLS}</FormHelperText>}
      </FormControl>
      <Tooltip title={canRemove ? "Remove plot" : "Keep at least one plot in this column"}><span>
        <IconButton size='small' aria-label={`Remove ${chartLabel}`} disabled={!canRemove} onClick={onRemove}><DeleteOutlineIcon fontSize='small' /></IconButton>
      </span></Tooltip>
    </Stack>
    {patientCount && <Typography variant='caption' color='text.secondary'>
      Unique patient IDs{missing ? `; ${missing} series without a patient ID excluded` : ""}.
    </Typography>}
    <Box role='figure' aria-label={chartLabel} sx={{ height: 420, minWidth: 0 }}>
      {loading ? <Stack alignItems='center' justifyContent='center' sx={{ height: "100%" }}><CircularProgress size={28} /></Stack>
        : hasValues ? <Plot data={figure.data} layout={figure.layout} config={PLOT_CONFIG} style={PLOT_STYLE} useResizeHandler />
        : <Stack alignItems='center' justifyContent='center' sx={{ height: "100%", textAlign: "center", px: 2 }}>
          <Typography variant='body2' color='text.secondary'>{!protocolGroups.length ? "No results match these filters." : !selected.length ? "Select a protocol." : "No values for this metric."}</Typography>
        </Stack>}
    </Box>
  </Stack></Paper>;
}
