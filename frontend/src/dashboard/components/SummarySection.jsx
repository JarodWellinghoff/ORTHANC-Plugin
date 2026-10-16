import { useMemo } from "react";
import { Box, IconButton, Tooltip, Typography } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import { DataGrid } from "@mui/x-data-grid";
import { useNavigate } from "react-router-dom";
import { useDashboard } from "../context/DashboardContext";
const formatDateTime = (value) => {
  if (!value) return "N/A";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};
const NoRowsOverlay = () => <Box sx={{ p: 2, textAlign: "center" }}><Typography variant='body2'>No series found matching the filters.</Typography></Box>;
const LoadingOverlay = () => <Box sx={{ p: 2, textAlign: "center" }}><Typography variant='body2'>Loading series summary...</Typography></Box>;
const RESULT_STATUSES = new Set(["full", "partial"]);
export default function SummarySection() {
  const { summary, actions } = useDashboard();
  const { items, loading, pagination, availableSeries } = summary;
  const navigate = useNavigate();
  const available = useMemo(() => new Set((availableSeries ?? []).filter(Boolean).map(String)), [availableSeries]);
  const paginationModel = { page: Math.max(0, (pagination.page ?? 1) - 1), pageSize: pagination.limit ?? 25 };
  const columns = useMemo(() => [
    ...[["patient_name", "Patient"], ["institution_name", "Institute"], ["scanner_model", "Scanner Model"],
      ["station_name", "Station"], ["protocol_name", "Protocol"]].map(([field, headerName]) => ({
        field, headerName, flex: 1, minWidth: 180, valueFormatter: (value) => value ?? "N/A" })),
    { field: "latest_analysis_date", headerName: "Last Analysis", flex: 1, minWidth: 200, valueFormatter: formatDateTime },
    { field: "actions", headerName: "Actions", sortable: false, filterable: false, disableColumnMenu: true, width: 100, align: "center",
      renderCell: ({ row }) => {
        const seriesUuid = row.series_uuid ?? row.seriesUuid ?? null;
        const hasResults = RESULT_STATUSES.has(String(row.test_status ?? "none").toLowerCase());
        const hasDicom = Boolean(seriesUuid) && available.has(String(seriesUuid));
        return <Tooltip title='Delete'><IconButton size='small' aria-label='Delete series' onClick={(event) => {
          event.stopPropagation();
          actions.openDeleteDialog(row.series_id, { seriesUuid, patientName: row.patient_name, hasResults, hasDicom,
            initialDeleteResults: hasResults, initialDeleteDicom: !hasResults && hasDicom });
        }}><DeleteIcon fontSize='small' /></IconButton></Tooltip>;
      } },
  ], [actions, available]);
  const getRowId = (row) => String(row.series_id ?? row.series_uuid ?? row.series_instance_uid ?? row.seriesId ?? row.seriesUuid ?? row.study_id ?? `${row.patient_name ?? "patient"}-${row.latest_analysis_date ?? "na"}`);
  return <DataGrid autoHeight={false} rows={loading ? [] : items ?? []} columns={columns} getRowId={getRowId}
    rowCount={pagination.total ?? items?.length ?? 0} paginationMode='server' paginationModel={paginationModel}
    onPaginationModelChange={(model) => { if (model.page !== paginationModel.page) actions.changePage(model.page + 1); }}
    loading={loading} disableRowSelectionOnClick disableColumnSelector disableDensitySelector pageSizeOptions={[]}
    initialState={{ sorting: { sortModel: [{ field: "latest_analysis_date", sort: "desc" }] } }}
    onRowClick={({ row }) => {
      actions.openChoModal(row);
      const target = row.series_id ?? row.series_uuid ?? row.series_instance_uid ?? row.seriesId ?? row.seriesUuid;
      if (target != null && target !== "") navigate(`/results/${encodeURIComponent(String(target))}`);
    }} showToolbar slots={{ noRowsOverlay: NoRowsOverlay, loadingOverlay: LoadingOverlay }} />;
}
