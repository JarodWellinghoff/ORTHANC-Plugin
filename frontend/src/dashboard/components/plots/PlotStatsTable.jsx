// ─────────────────────────────────────────────────────────────────────────────
// PlotStatsTable.jsx
//
// The summary-statistics table a CustomizablePlotCard produces, rendered on
// its own so the plotting page can group all four tables into a dedicated,
// scrollable "Tables" section separate from the plots themselves. Reads the
// same {records, config} pair as the card it corresponds to.
// ─────────────────────────────────────────────────────────────────────────────

import * as React from "react";

import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";

import { formatValue } from "../../utils/aggregateMetrics";
import { useAggregatePlot } from "./useAggregatePlot";

const PlotStatsTable = ({ title, records, config }) => {
  const { statRows, correlation, isScatter } = useAggregatePlot(
    records,
    config,
  );

  return (
    <Box>
      <Typography variant='subtitle1' fontWeight={600} sx={{ mb: 0.5 }}>
        {title}
      </Typography>

      {statRows.length === 0 ? (
        <Typography variant='body2' color='text.secondary'>
          No data to summarize yet.
        </Typography>
      ) : (
        <>
          {isScatter && correlation?.r !== null && correlation?.r !== undefined && (
            <Typography variant='body2' color='text.secondary' sx={{ mb: 1 }}>
              Pearson r = {correlation.r.toFixed(3)} across {correlation.n}{" "}
              complete pairs. Correlation only; it does not establish a
              causal relationship between these metrics.
            </Typography>
          )}

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
        </>
      )}
    </Box>
  );
};

export default PlotStatsTable;
