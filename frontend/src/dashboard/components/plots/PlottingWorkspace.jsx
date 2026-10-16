import { useMemo, useReducer } from "react";
import AddIcon from "@mui/icons-material/Add";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import { Box, Button, Stack, Tooltip, Typography } from "@mui/material";
import { SCALAR_METRICS } from "../../utils/aggregateMetrics";
import ProtocolPlotCard from "./ProtocolPlotCard";
import { createInitialPlotState, groupProtocolRecords, PLOT_COLUMNS, protocolPlotReducer } from "./protocolPlotState";
import { buildReportRows, getReportMetrics, reportRowsToCsv } from "./protocolReport";
const METRICS = getReportMetrics(SCALAR_METRICS);
export default function PlottingWorkspace({ records, loading }) {
  const [state, dispatch] = useReducer(protocolPlotReducer, undefined, createInitialPlotState);
  const groups = useMemo(() => groupProtocolRecords(records), [records]);
  const exportCsv = () => {
    const rows = buildReportRows(PLOT_COLUMNS.flatMap((column) => state.columns[column.mode]), groups, METRICS);
    const url = URL.createObjectURL(new Blob([reportRowsToCsv(rows)], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `ctpro-statistics-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <Stack spacing={2}>
    <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
      <Tooltip title='Export every protocol in the current query'><span><Button variant='outlined' startIcon={<FileDownloadIcon />}
        onClick={exportCsv} disabled={loading || groups.length === 0}>Export CSV</Button></span></Tooltip>
    </Box>
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "minmax(0, 1fr)", lg: "repeat(2, minmax(0, 1fr))" }, gap: 3, alignItems: "start" }}>
      {PLOT_COLUMNS.map((column) => <Stack key={column.mode} component='section' aria-label={column.title} spacing={2} sx={{ minWidth: 0 }}>
        <Typography variant='h6' component='h2'>{column.title}</Typography>
        {state.columns[column.mode].map((plot) => <ProtocolPlotCard key={plot.id} config={plot} protocolGroups={groups} loading={loading}
          canRemove={state.columns[column.mode].length > 1}
          onRemove={() => dispatch({ type: "remove", mode: column.mode, id: plot.id })}
          onConfigChange={(patch) => dispatch({ type: "configure", mode: column.mode, id: plot.id, patch })} />)}
        <Box sx={{ display: "flex", justifyContent: "center" }}><Button size='small' variant='outlined' startIcon={<AddIcon />}
          onClick={() => dispatch({ type: "add", mode: column.mode })}>{column.addLabel}</Button></Box>
      </Stack>)}
    </Box>
  </Stack>;
}
