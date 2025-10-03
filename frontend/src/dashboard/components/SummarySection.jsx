import * as React from "react";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TablePagination from "@mui/material/TablePagination";
import TableRow from "@mui/material/TableRow";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import ScienceIcon from "@mui/icons-material/Science";
import VisibilityIcon from "@mui/icons-material/Visibility";
import DeleteIcon from "@mui/icons-material/Delete";
import InfoIcon from "@mui/icons-material/Info";

import { useDashboard } from "../context/DashboardContext";

const statusLabelMap = {
  full: "Full Analysis",
  partial: "Global Noise",
  headers_only: "Headers Only",
  error: "Error",
  none: "No Analysis",
};

const statusColorMap = {
  full: "success",
  partial: "warning",
  headers_only: "info",
  error: "error",
  none: "default",
};

const formatDateTime = (value) => {
  if (!value) return "N/A";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
};

const SummarySection = () => {
  const { summary, actions } = useDashboard();
  const availableLookup = React.useMemo(
    () => new Set(summary.availableSeries ?? []),
    [summary.availableSeries]
  );
  const { items, loading, pagination } = summary;

  const handleChangePage = (_event, newPage) => {
    actions.changePage(newPage + 1);
  };

  const handleChangeRowsPerPage = (event) => {
    actions.changePageSize(parseInt(event.target.value, 10));
  };

  const rows = loading ? [] : items;

  return (
    <Paper variant='outlined'>
      <TableContainer>
        <Table size='small'>
          <TableHead>
            <TableRow>
              <TableCell>Patient</TableCell>
              <TableCell>Institute</TableCell>
              <TableCell>Scanner Model</TableCell>
              <TableCell>Station</TableCell>
              <TableCell>Protocol</TableCell>
              {/* <TableCell>Status</TableCell> */}
              <TableCell>Last Analysis</TableCell>
              <TableCell align='center'>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={8} align='center'>
                  <Typography variant='body2' color='text.secondary'>
                    Loading series summary...
                  </Typography>
                </TableCell>
              </TableRow>
            ) : rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} align='center'>
                  <Typography variant='body2' color='text.secondary'>
                    No series found matching the filters.
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              rows.map((row) => {
                const statusKey = row.test_status ?? "none";
                const chipColor = statusColorMap[statusKey] ?? "default";
                const chipLabel = statusLabelMap[statusKey] ?? "No Analysis";
                const hasSeries =
                  row.series_uuid && availableLookup.has(row.series_uuid);

                return (
                  <TableRow
                    hover
                    key={`${row.series_id}-${row.test_status ?? "na"}`}
                    onClick={() => actions.openDetails(row.series_id)}
                    sx={{ cursor: "pointer" }}>
                    <TableCell>{row.patient_name ?? "N/A"}</TableCell>
                    <TableCell>{row.institution_name ?? "N/A"}</TableCell>
                    <TableCell>{row.scanner_model ?? "N/A"}</TableCell>
                    <TableCell>{row.station_name ?? "N/A"}</TableCell>
                    <TableCell>{row.protocol_name ?? "N/A"}</TableCell>
                    {/* <TableCell>
                      <Chip size="small" label={chipLabel} color={chipColor} variant={chipColor === "default" ? "outlined" : "filled"} />
                    </TableCell> */}
                    <TableCell>
                      {formatDateTime(row.latest_analysis_date)}
                    </TableCell>
                    <TableCell
                      align='center'
                      onClick={(event) => event.stopPropagation()}>
                      {/* <Tooltip title='View Details'>
                        <IconButton
                          size='small'
                          onClick={() => actions.openDetails(row.series_id)}>
                          <InfoIcon fontSize='small' />
                        </IconButton>
                      </Tooltip> */}
                      <Tooltip
                        title={
                          hasSeries
                            ? "Run CHO Analysis"
                            : "Series not found in DICOM database"
                        }>
                        <span>
                          <IconButton
                            size='small'
                            onClick={() => actions.openChoModal(row)}
                            disabled={!hasSeries}>
                            <ScienceIcon fontSize='small' />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip
                        title={
                          hasSeries
                            ? "Open Viewer"
                            : "Series not found in DICOM database"
                        }>
                        <span>
                          <IconButton
                            size='small'
                            onClick={() =>
                              window.open(
                                `/ohif/viewer?StudyInstanceUIDs=${encodeURIComponent(
                                  row.study_id
                                )}`,
                                "_blank",
                                "noopener"
                              )
                            }
                            disabled={!hasSeries}>
                            <VisibilityIcon fontSize='small' />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title='Delete'>
                        <IconButton
                          size='small'
                          onClick={() =>
                            actions.openDeleteDialog(
                              row.series_id,
                              null,
                              row.patient_name
                            )
                          }>
                          <DeleteIcon fontSize='small' />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component='div'
        count={pagination.total}
        page={Math.max(0, pagination.page - 1)}
        onPageChange={handleChangePage}
        rowsPerPage={pagination.limit}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[10, 25, 50, 100]}
      />
    </Paper>
  );
};

export default SummarySection;
