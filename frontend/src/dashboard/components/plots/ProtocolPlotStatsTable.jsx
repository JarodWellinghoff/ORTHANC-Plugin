import { useMemo } from "react";
import CircularProgress from "@mui/material/CircularProgress";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";

import { describe, formatValue, SCALAR_METRIC_MAP } from "../../utils/aggregateMetrics";
import { getAllPlotGroups } from "./protocolPlotState";

const STAT_COLUMNS = [
  { key: "mean", label: "Mean" },
  { key: "sd", label: "SD" },
  { key: "median", label: "Median" },
  { key: "min", label: "Min" },
  { key: "max", label: "Max" },
];

const ProtocolPlotStatsTable = ({ plots, protocolGroups, loading }) => {
  const rows = useMemo(() => getAllPlotGroups(plots, protocolGroups).flatMap((group) => {
    const metric = SCALAR_METRIC_MAP.get(group.metricKey);
    if (!metric) return [];
    return [{
      ...group,
      metric,
      stats: describe(group.records.map((record) => record[metric.key])),
    }];
  }), [plots, protocolGroups]);

  if (loading) {
    return (
      <Paper variant='outlined' sx={{ p: 3 }}>
        <Stack direction='row' spacing={2} alignItems='center' role='status'>
          <CircularProgress size={24} />
          <Typography variant='body2'>Loading all plot results...</Typography>
        </Stack>
      </Paper>
    );
  }

  return (
    <TableContainer component={Paper} variant='outlined' sx={{ maxHeight: 640 }}>
      <Table size='small' stickyHeader aria-label='All plot results for every protocol'>
        <TableHead>
          <TableRow>
            <TableCell>Plot</TableCell>
            <TableCell>Metric</TableCell>
            <TableCell>Protocol</TableCell>
            <TableCell align='right'>n</TableCell>
            {STAT_COLUMNS.map((column) => <TableCell key={column.key} align='right'>{column.label}</TableCell>)}
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.length === 0 ? (
            <TableRow>
              <TableCell colSpan={9} sx={{ py: 3 }}>
                <Typography variant='body2' color='text.secondary'>No data to summarize yet.</Typography>
              </TableCell>
            </TableRow>
          ) : rows.map((row) => (
            <TableRow key={row.id}>
              <TableCell>{row.plotTitle}</TableCell>
              <TableCell>
                {row.metric.label}
                {row.metric.unit ? <Typography component='span' variant='caption' color='text.secondary' sx={{ ml: 0.5 }}>({row.metric.unit})</Typography> : null}
              </TableCell>
              <TableCell>{row.protocol}</TableCell>
              <TableCell align='right'>{row.stats.n}</TableCell>
              {STAT_COLUMNS.map((column) => (
                <TableCell key={column.key} align='right'>{formatValue(row.stats[column.key], row.metric.decimals)}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ProtocolPlotStatsTable;
