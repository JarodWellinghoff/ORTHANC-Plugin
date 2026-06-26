import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
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
import { alpha } from "@mui/material/styles";
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
      {/* <Divider
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
      </Menu> */}
    </Toolbar>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// ResultsPage
//
// Read-only counterpart to BulkTestsPage. Same filter shape, same data source
// (DashboardContext `summary`), but stripped of every testing-related concern:
// no modality picker, no testType, no per-row run/recover buttons, no
// beforeunload guard, no calculationStates wiring. The per-row action is
// "View Results", which deep-links into the existing ChoAnalysisRoute view.
//
// The page-level action is "Export Selected" — a bulk counterpart to the
// per-series "Export XLS" button in ChoAnalysisPage. It posts the selected
// series ids to /cho-export-results and the backend returns a single .xls
// workbook with one sheet per selected series.
// ─────────────────────────────────────────────────────────────────────────────
const ResultsPage = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const { summary, actions } = useDashboard();
  const { filters, updateFilter, resetFilters } = useFilters();
  const { items, pagination } = summary;

  const [filterModel, setFilterModel] = useState({
    items: [{ field: "testStatus", operator: "equals", value: "full" }],
  });
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
  // In-flight flag for the Export Selected button. Drives the spinner icon and
  // the disabled state so a user can't fire a second export while the first
  // request is still streaming the workbook back.
  const [exporting, setExporting] = useState(false);

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

  // Resolve the current selection (handling both the "include" and "exclude"
  // selection-model shapes) down to an ordered queue of rows on this page.
  // Mirrors `resolveSelectedQueue` from BulkTestsPage so behavior across the
  // two pages stays consistent.
  const resolveSelectedQueue = useCallback(() => {
    const type = selectionModel.type;
    const rowsById = new Map(normalizedResults.map((row) => [row.id, row]));
    let selectedIds = null;

    if (type === "include") {
      if (selectionModel.ids.size === 0) {
        return [];
      }
      selectedIds = selectionModel.ids.intersection
        ? selectionModel.ids.intersection(rowsById)
        : new Set([...selectionModel.ids].filter((id) => rowsById.has(id)));
    } else if (type === "exclude") {
      if (selectionModel.ids.size === rowsById.size) {
        return [];
      }
      selectedIds = new Set(rowsById.keys()).difference
        ? new Set(rowsById.keys()).difference(selectionModel.ids)
        : new Set(
            [...rowsById.keys()].filter((id) => !selectionModel.ids.has(id)),
          );
    }

    const queue = [];
    if (selectedIds) {
      for (const value of selectedIds) {
        if (rowsById.has(value)) queue.push(rowsById.get(value));
      }
    }
    return queue;
  }, [normalizedResults, selectionModel]);

  // Bulk export entry point. Filters the selection down to rows that actually
  // have results in the database — anything else would just produce empty
  // sheets and waste a backend round-trip. Surfaces any skipped rows so the
  // user knows why their selection count might not match the export count.
  const handleExportSelected = useCallback(async () => {
    if (exporting) return;

    const queue = resolveSelectedQueue();
    if (queue.length === 0) {
      enqueueSnackbar("Select at least one row to export.", {
        variant: "warning",
      });
      return;
    }

    const exportable = queue.filter((row) => {
      const status = row.testStatus;
      const hasUid = Boolean(row.seriesInstanceUid ?? row.seriesUuid);
      return (
        hasUid &&
        status &&
        status !== "none" &&
        status !== "pending" &&
        status !== "untested"
      );
    });
    const skipped = queue.length - exportable.length;

    if (exportable.length === 0) {
      enqueueSnackbar("None of the selected series have results to export.", {
        variant: "error",
      });
      return;
    }
    if (skipped > 0) {
      enqueueSnackbar(
        `${skipped} selected series ${
          skipped === 1 ? "has" : "have"
        } no results yet and will be skipped.`,
        { variant: "warning" },
      );
    }

    // Backend keys on series_instance_uid for /cho-results/{id}; the
    // normalized row stashes that on `seriesInstanceUid` (with `seriesUuid`
    // as a last-resort fallback for older items that lack it).
    const seriesIds = exportable.map(
      (row) => row.seriesInstanceUid ?? row.seriesUuid,
    );

    setExporting(true);
    try {
      await actions.exportSeries(seriesIds);
      // exportSeries surfaces success/failure via the context's status
      // channel; we add a snackbar here so the feedback is also visible in
      // the page where the user took the action.
      enqueueSnackbar(`Export started for ${seriesIds.length} series.`, {
        variant: "info",
      });
    } finally {
      setExporting(false);
    }
  }, [actions, enqueueSnackbar, exporting, resolveSelectedQueue]);

  // Count of currently selected rows on this page, for the action bar button.
  // Handles both selection-model shapes ("include" of explicit ids, or
  // "exclude" of unchecked ids after a "select all").
  const selectedCount = useMemo(() => {
    const ids = selectionModel?.ids;
    if (!ids) return 0;
    if (selectionModel.type === "exclude") {
      return Math.max(0, normalizedResults.length - ids.size);
    }
    return ids.size;
  }, [selectionModel, normalizedResults.length]);

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
        field: "scannerModel",
        headerName: "Scanner Model",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "stationName",
        headerName: "Scanner Station",
        flex: 1,
        minWidth: 160,
      },
      {
        field: "studyDate",
        headerName: "Study Date",
        flex: 1,
        minWidth: 50,
        valueFormatter: (value) =>
          value.replace(/^(\d{4})-(\d{2})-(\d{2})$/, "$2/$3/$1"),
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
                  }}
                  sx={{
                    height: "2.5rem",
                    width: "2.5rem",
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
      {/* Hero */}
      <Box
        variant='outlined'
        sx={(theme) => ({
          position: "relative",
          overflow: "hidden",
          px: { xs: 3, md: 6 },
          py: { xs: 4, md: 7 },
          borderRadius: 1,
          border: "1px solid",
          borderColor: "divider",
          backgroundImage: `linear-gradient(135deg, ${alpha(
            theme.palette.secondary.main,
            0.14,
          )} 0%, ${alpha(theme.palette.primary.dark, 0.04)} 100%)`,
        })}>
        <Stack spacing={2.5} sx={{ maxWidth: 880, position: "relative" }}>
          <Typography
            variant='h3'
            component='h1'
            sx={{
              fontWeight: 600,
              letterSpacing: "-0.02em",
              fontSize: { xs: "2rem", md: "2.75rem" },
            }}>
            Results
          </Typography>
          <Typography
            variant='body1'
            color='text.secondary'
            sx={{ fontSize: "1.05rem", lineHeight: 1.65 }}>
            Read-only view of analyzed series. Filter by patient, protocol, or
            institute and drill into any series' CHO, NPS, MTF, and dose
            metrics.
          </Typography>
        </Stack>
      </Box>
      <FiltersPanel
        filters={filters}
        onChange={updateFilter}
        onQuery={handleQuery}
        onReset={resetFilters}>
        {/* Bulk action bar lives inside the FiltersPanel children slot, the
            same way BulkTestsPage hosts its "Run Selected" button. Keeping it
            here means the action sits visually adjacent to the filters that
            shape the selection. */}
        <Stack direction='row' spacing={1.5} alignItems='center'>
          <Tooltip
            title={
              selectedCount === 0
                ? "Select one or more series to export"
                : "Export the selected series as a single XLS with one sheet per case"
            }>
            <span>
              <Button
                variant='contained'
                startIcon={
                  exporting ? (
                    <CircularProgress size={16} color='inherit' />
                  ) : (
                    <FileDownloadIcon />
                  )
                }
                disabled={exporting || selectedCount === 0}
                onClick={handleExportSelected}>
                {exporting
                  ? "Exporting…"
                  : `Export Selected${
                      selectedCount ? ` (${selectedCount})` : ""
                    }`}
              </Button>
            </span>
          </Tooltip>
        </Stack>
      </FiltersPanel>

      <DataGrid
        rows={loading ? [] : (normalizedResults ?? [])}
        columns={columns}
        getRowId={getRowId}
        rowCount={pagination.total ?? normalizedResults?.length ?? 0}
        paginationMode='server'
        filterMode='client'
        disableColumnFilter
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
          filter: {
            filterModel: {
              items: [
                { field: "testStatus", operator: "is", value: "Complete" },
              ],
            },
          },
          pagination: { paginationModel: paginationModel },
          columns: {
            columnVisibilityModel: {
              latestAnalysis: false,
              pullScheduleName: false,
              studyDate: false,
              stationName: false,
              patientId: false,
              testStatus: false,
            },
          },
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
