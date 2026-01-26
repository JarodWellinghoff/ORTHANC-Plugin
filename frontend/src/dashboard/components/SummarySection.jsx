import * as React from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteIcon from "@mui/icons-material/Delete";
import { DataGrid } from "@mui/x-data-grid";
import { useNavigate } from "react-router-dom";

import { useDashboard } from "../context/DashboardContext";

const formatDateTime = (value) => {
  if (!value) return "N/A";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
};

const NoRowsOverlay = () => (
  <Box sx={{ p: 2, textAlign: "center" }}>
    <Typography variant='body2' color='text.secondary'>
      No series found matching the filters.
    </Typography>
  </Box>
);

const LoadingOverlay = () => (
  <Box sx={{ p: 2, textAlign: "center" }}>
    <Typography variant='body2' color='text.secondary'>
      Loading series summary...
    </Typography>
  </Box>
);

const SummarySection = () => {
  const { summary, actions } = useDashboard();
  const navigate = useNavigate();
  const { items, loading, pagination } = summary;

  const openSeriesDetails = React.useCallback(
    (series) => {
      if (!series) return;
      actions.openChoModal(series);
      const targetId =
        series.series_id ??
        series.series_uuid ??
        series.series_instance_uid ??
        series.seriesId ??
        series.seriesUuid ??
        null;
      if (targetId !== null && targetId !== undefined && targetId !== "") {
        navigate(`/results/${encodeURIComponent(String(targetId))}`);
      }
    },
    [actions, navigate]
  );

  const paginationModel = React.useMemo(
    () => ({
      page: Math.max(0, (pagination.page ?? 1) - 1),
      pageSize: pagination.limit ?? 25,
    }),
    [pagination.page, pagination.limit]
  );

  const handlePaginationModelChange = React.useCallback(
    (model) => {
      if (model.page !== paginationModel.page) {
        actions.changePage(model.page + 1);
      }
      if (model.pageSize !== paginationModel.pageSize) {
        actions.changePageSize(model.pageSize);
      }
    },
    [actions, paginationModel.page, paginationModel.pageSize]
  );

  const columns = React.useMemo(
    () => [
      {
        field: "patient_name",
        headerName: "Patient",
        flex: 1,
        minWidth: 180,
        valueFormatter: (value) => value ?? "N/A",
      },
      {
        field: "institution_name",
        headerName: "Institute",
        flex: 1,
        minWidth: 180,
        valueFormatter: (value) => value ?? "N/A",
      },
      {
        field: "scanner_model",
        headerName: "Scanner Model",
        flex: 1,
        minWidth: 180,
        valueFormatter: (value) => value ?? "N/A",
      },
      {
        field: "station_name",
        headerName: "Station",
        flex: 1,
        minWidth: 160,
        valueFormatter: (value) => value ?? "N/A",
      },
      {
        field: "protocol_name",
        headerName: "Protocol",
        flex: 1,
        minWidth: 180,
        valueFormatter: (value) => value ?? "N/A",
      },
      {
        field: "latest_analysis_date",
        headerName: "Last Analysis",
        flex: 1,
        minWidth: 200,
        valueFormatter: (value) => formatDateTime(value),
      },
      {
        field: "actions",
        headerName: "Actions",
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        width: 100,
        align: "center",
        renderCell: (params) => (
          <Tooltip title='Delete'>
            <span>
              <IconButton
                size='small'
                onClick={(event) => {
                  event.stopPropagation();
                  actions.openDeleteDialog(
                    params.row.series_id,
                    null,
                    params.row.patient_name
                  );
                }}>
                <DeleteIcon fontSize='small' />
              </IconButton>
            </span>
          </Tooltip>
        ),
      },
    ],
    [actions]
  );

  const getRowId = React.useCallback((row) => {
    const baseId =
      row.series_id ??
      row.series_uuid ??
      row.series_instance_uid ??
      row.seriesId ??
      row.seriesUuid ??
      row.study_id ??
      null;
    if (baseId !== null && baseId !== undefined && baseId !== "") {
      return String(baseId);
    }
    return `${row.patient_name ?? "patient"}-${
      row.latest_analysis_date ?? "na"
    }`;
  }, []);

  return (
    <DataGrid
      autoHeight
      rows={loading ? [] : items ?? []}
      columns={columns}
      getRowId={getRowId}
      rowCount={pagination.total ?? items?.length ?? 0}
      paginationMode='server'
      paginationModel={paginationModel}
      onPaginationModelChange={handlePaginationModelChange}
      loading={loading}
      disableRowSelectionOnClick
      disableColumnSelector
      disableDensitySelector
      pageSizeOptions={[10, 25, 50, 100]}
      initialState={{
        pagination: { paginationModel: { pageSize: 25 } },
        sorting: {
          sortModel: [{ field: "latest_analysis_date", sort: "desc" }],
        },
      }}
      onRowClick={(params) => openSeriesDetails(params.row)}
      showToolbar
      slots={{
        noRowsOverlay: NoRowsOverlay,
        loadingOverlay: LoadingOverlay,
      }}
    />
  );
};

export default SummarySection;
