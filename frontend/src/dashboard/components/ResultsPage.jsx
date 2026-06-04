import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Stack from "@mui/material/Stack";
import Divider from "@mui/material/Divider";
import Tooltip from "@mui/material/Tooltip";
import ContentPasteSearchIcon from "@mui/icons-material/ContentPasteSearch";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import { useNavigate } from "react-router-dom";
import { useDashboard } from "../context/DashboardContext";
import { useSnackbar } from "notistack";
import {
  DataGrid,
  Toolbar,
  ToolbarButton,
  ColumnsPanelTrigger,
  ExportCsv,
  ExportPrint,
} from "@mui/x-data-grid";
import FiltersPanel from "./FiltersPanel";
import { useFilters } from "../../hooks/useFilters";
import {
  fetchJson,
  formatDateTime,
  normalizeChoRow,
  statusColorMap,
  statusLabelMap,
} from "../utils/choResultsShared";

// ─────────────────────────────────────────────────────────────────────────────
// GridToolbar — same shape as the one on BulkTestsPage. Kept locally so each
// page can evolve its toolbar independently without coupling.
// ─────────────────────────────────────────────────────────────────────────────
const GridToolbar = () => {
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const exportMenuTriggerRef = useRef(null);

  return (
    <Toolbar>
      <Tooltip title='Columns'>
        <ColumnsPanelTrigger render={<ToolbarButton />}>
          <ViewColumnIcon fontSize='small' />
        </ColumnsPanelTrigger>
      </Tooltip>
      <Tooltip title='Filters'></Tooltip>
      <Divider
        orientation='vertical'
        variant='middle'
        flexItem
        sx={{ mx: 0.5 }}
      />
      <Tooltip title='Export'>
        <ToolbarButton
          ref={exportMenuTriggerRef}
          id='export-menu-trigger'
          aria-controls='export-menu'
          aria-haspopup='true'
          aria-expanded={exportMenuOpen ? "true" : undefined}
          onClick={() => setExportMenuOpen(true)}>
          <FileDownloadIcon fontSize='small' />
        </ToolbarButton>
      </Tooltip>
      <Menu
        id='export-menu'
        anchorEl={exportMenuTriggerRef.current}
        open={exportMenuOpen}
        onClose={() => setExportMenuOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          list: {
            "aria-labelledby": "export-menu-trigger",
          },
        }}>
        <ExportPrint
          render={<MenuItem />}
          onClick={() => setExportMenuOpen(false)}>
          Print
        </ExportPrint>
        <ExportCsv
          render={<MenuItem />}
          onClick={() => setExportMenuOpen(false)}>
          Download as CSV
        </ExportCsv>
      </Menu>
    </Toolbar>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// ResultsPage
//
// Read-only counterpart to BulkTestsPage. Same filter shape, same data source
// (DashboardContext `summary`), but stripped of every testing-related concern:
// no modality picker, no testType, no per-row run/recover buttons, no
// beforeunload guard, no calculationStates wiring. The one action per row is
// "View Results", which deep-links into the existing ChoAnalysisRoute view.
// ─────────────────────────────────────────────────────────────────────────────
const ResultsPage = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const { summary, actions } = useDashboard();
  const { filters, updateFilter, resetFilters } = useFilters();
  const { items, pagination } = summary;

  const [filterModel, setFilterModel] = useState({ items: [] });
  const [sortModel, setSortModel] = useState([
    { field: "latestAnalysis", sort: "desc" },
  ]);
  const sortRef = useRef(sortModel);

  const [availableSeries, setAvailableSeries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectionModel, setSelectionModel] = useState({
    type: "include",
    ids: new Set(),
  });

  const handleQuery = () => actions.loadSummary(filters);

  const handleSortModelChange = useCallback(
    (model) => {
      setSortModel(model);
      sortRef.current = model;
      const sort = model[0];
      actions.loadSummary({
        ...filters,
        page: 1,
        sort_by: sort?.field,
        sort_order: sort?.sort ?? "asc",
      });
    },
    [actions, filters],
  );

  const loadAvailableSeries = useCallback(async () => {
    try {
      const data = await fetchJson("/series/");
      if (Array.isArray(data)) {
        setAvailableSeries(data);
      } else {
        console.warn("Unexpected response for /series/", data);
        setAvailableSeries([]);
      }
    } catch (err) {
      console.error("Failed to load available series", err);
    }
  }, []);

  // Kick off the initial summary + Orthanc availability fetch. The summary call
  // is what populates the grid; the availability call is only used here to
  // light up the "DICOM available" badge in the status column.
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      try {
        await actions.loadSummary(filters);
      } catch (err) {
        if (!cancelled) {
          console.error("Failed to load results summary", err);
          enqueueSnackbar("Failed to load results", { variant: "error" });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    loadAvailableSeries();
    return () => {
      cancelled = true;
    };
    // Intentionally only on mount — `actions` is stable and `filters` is the
    // initial value; subsequent queries go through `handleQuery`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const availableSet = useMemo(
    () => new Set(availableSeries.filter(Boolean).map((v) => String(v))),
    [availableSeries],
  );

  const normalizedResults = useMemo(
    () =>
      items.map((item, index) => normalizeChoRow(item, index, availableSet)),
    [items, availableSet],
  );

  const columns = useMemo(() => {
    return [
      {
        field: "patientId",
        headerName: "Patient ID",
        flex: 0.8,
        minWidth: 120,
      },
      {
        field: "patientName",
        headerName: "Patient",
        flex: 1.1,
        minWidth: 160,
      },
      {
        field: "institutionName",
        headerName: "Institute",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "protocolName",
        headerName: "Protocol",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "pullScheduleName",
        headerName: "Pull Schedule",
        flex: 1,
        minWidth: 160,
        valueFormatter: (value) => value ?? "—",
      },
      {
        field: "testStatus",
        headerName: "Status",
        width: 140,
        renderCell: (params) => {
          const value = params.row.testStatus ?? "none";
          const chipColor = statusColorMap[value] ?? "default";
          const label = statusLabelMap[value] ?? value;
          return (
            <Chip
              size='small'
              label={label}
              color={chipColor}
              variant={chipColor === "default" ? "outlined" : "filled"}
            />
          );
        },
      },
      {
        field: "latestAnalysis",
        headerName: "Latest Analysis",
        flex: 1,
        minWidth: 180,
        valueFormatter: (value) => formatDateTime(value),
      },
      {
        field: "actions",
        headerName: "Actions",
        width: 120,
        sortable: false,
        filterable: false,
        disableColumnMenu: true,
        renderCell: (params) => {
          const row = params.row;
          const seriesInstanceUid = row.seriesInstanceUid ?? row.seriesUuid;
          const hasResults =
            row.testStatus &&
            row.testStatus !== "none" &&
            row.testStatus !== "pending";
          const disabled = !hasResults || !seriesInstanceUid;
          const tooltip = disabled
            ? "No results recorded for this series yet"
            : "View results for this series";
          return (
            <Tooltip title={tooltip}>
              <span>
                <IconButton
                  size='small'
                  color='primary'
                  disabled={disabled}
                  onClick={(event) => {
                    event.stopPropagation();
                    navigate(
                      `/results/${encodeURIComponent(seriesInstanceUid)}`,
                    );
                  }}>
                  <ContentPasteSearchIcon fontSize='small' />
                </IconButton>
              </span>
            </Tooltip>
          );
        },
      },
    ];
  }, [navigate]);

  const paginationModel = useMemo(
    () => ({
      page: Math.max(0, (pagination.page ?? 1) - 1),
      pageSize: pagination.limit ?? 25,
    }),
    [pagination.page, pagination.limit],
  );

  const getRowId = useCallback((row) => {
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

  const handlePaginationModelChange = useCallback(
    (model) => {
      const sort = sortRef.current[0];
      const sortExtras = sort
        ? { sort_by: sort.field, sort_order: sort.sort ?? "asc" }
        : {};

      if (model.page !== paginationModel.page) {
        actions.changePage(model.page + 1, sortExtras);
      }
      if (model.pageSize !== paginationModel.pageSize) {
        actions.changePageSize(model.pageSize, sortExtras);
      }
    },
    [actions, paginationModel.page, paginationModel.pageSize],
  );

  return (
    <Stack spacing={3}>
      <FiltersPanel
        filters={filters}
        onChange={updateFilter}
        onQuery={handleQuery}
        onReset={resetFilters}
      />

      <DataGrid
        rows={loading ? [] : (normalizedResults ?? [])}
        columns={columns}
        getRowId={getRowId}
        rowCount={pagination.total ?? normalizedResults?.length ?? 0}
        paginationMode='server'
        filterMode='client'
        filterModel={filterModel}
        onFilterModelChange={setFilterModel}
        paginationModel={paginationModel}
        onPaginationModelChange={handlePaginationModelChange}
        sortingMode='server'
        sortModel={sortModel}
        onSortModelChange={handleSortModelChange}
        checkboxSelection
        disableRowSelectionOnClick
        loading={loading}
        pageSizeOptions={[25, 50, 100]}
        slots={{ toolbar: GridToolbar }}
        showToolbar
        initialState={{
          pagination: { paginationModel: paginationModel },
        }}
        onRowSelectionModelChange={(model) => setSelectionModel(model)}
        rowSelectionModel={selectionModel}
        slotProps={{
          toolbar: {
            showQuickFilter: true,
            quickFilterProps: { debounceMs: 500 },
          },
        }}
      />
    </Stack>
  );
};

export default ResultsPage;
