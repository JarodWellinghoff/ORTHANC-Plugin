import * as React from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Collapse from "@mui/material/Collapse";
import Grid from "@mui/material/Grid";
import InputAdornment from "@mui/material/InputAdornment";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import FilterAltIcon from "@mui/icons-material/FilterAlt";
import FilterAltOffIcon from "@mui/icons-material/FilterAltOff";
import SearchIcon from "@mui/icons-material/Search";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DownloadIcon from "@mui/icons-material/Download";
import RefreshIcon from "@mui/icons-material/Refresh";
import Select from "@mui/material/Select";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import dayjs from "dayjs";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";

import { useDashboard } from "../context/DashboardContext";

const filterLabels = {
  patientSearch: "Patient",
  institute: "Institute",
  scannerStation: "Station",
  protocolName: "Protocol",
  scannerModel: "Scanner",
  examDateFrom: "Exam From",
  examDateTo: "Exam To",
  ageMin: "Age Min",
  ageMax: "Age Max",
};

const defaultActionItems = [
  "advancedFilters",
  "clearFilters",
  "refresh",
  "exportCsv",
];

const FiltersPanel = ({ actionItems }) => {
  const { filters, filterOptions, advancedFiltersOpen, actions } =
    useDashboard();
  const {
    resetFilters,
    toggleAdvancedFilters,
    refresh,
    exportAllResults,
    clearFilterValue,
    updateFilter,
  } = actions;

  const activeFilters = React.useMemo(() => {
    return Object.entries(filters)
      .filter(
        ([, value]) => value !== undefined && value !== null && value !== "",
      )
      .map(([key, value]) => ({
        key,
        label: `${filterLabels[key] ?? key}: ${value}`,
      }));
  }, [filters]);

  const handleChange = (event) => {
    console.debug(event);
    const { name, value } = event.target;
    updateFilter(name, value);
  };

  const { date_range: dateRange, age_range: ageRange } = filterOptions;

  const examDateFromValue = React.useMemo(() => {
    if (!filters.examDateFrom) {
      return null;
    }
    const parsed = dayjs(filters.examDateFrom);
    return parsed.isValid() ? parsed : null;
  }, [filters.examDateFrom]);

  const examDateToValue = React.useMemo(() => {
    if (!filters.examDateTo) {
      return null;
    }
    const parsed = dayjs(filters.examDateTo);
    return parsed.isValid() ? parsed : null;
  }, [filters.examDateTo]);

  const minExamDate = React.useMemo(() => {
    if (!dateRange?.min) {
      return undefined;
    }
    const parsed = dayjs(dateRange.min);
    return parsed.isValid() ? parsed : undefined;
  }, [dateRange]);

  const maxExamDate = React.useMemo(() => {
    if (!dateRange?.max) {
      return undefined;
    }
    const parsed = dayjs(dateRange.max);
    return parsed.isValid() ? parsed : undefined;
  }, [dateRange]);

  const handleDateChange = React.useCallback(
    (name) => (newValue) => {
      if (!newValue || !newValue.isValid()) {
        updateFilter(name, "");
        return;
      }
      updateFilter(name, newValue.format("YYYY-MM-DD"));
    },
    [updateFilter],
  );

  const resolvedActionItems = React.useMemo(() => {
    const items =
      Array.isArray(actionItems) && actionItems.length > 0
        ? actionItems
        : defaultActionItems;

    const normalizeKey = (value) => {
      if (!value) {
        return null;
      }
      if (typeof value === "string" && defaultActionItems.includes(value)) {
        return value;
      }
      const normalized = String(value).trim().toLowerCase();
      switch (normalized) {
        case "advanced":
        case "advancedfilters":
        case "advanced_filters":
          return "advancedFilters";
        case "clear":
        case "clearfilters":
        case "clear_filters":
          return "clearFilters";
        case "refresh":
          return "refresh";
        case "export":
        case "exportcsv":
        case "export_csv":
          return "exportCsv";
        default:
          return null;
      }
    };

    const actionContext = {
      filters,
      advancedFiltersOpen,
      resetFilters,
      toggleAdvancedFilters,
      refresh,
      exportAllResults,
    };

    return items
      .map((item, index) => {
        if (!item) {
          return null;
        }
        if (React.isValidElement(item)) {
          return React.cloneElement(item, {
            key: item.key ?? `filters-panel-custom-${index}`,
          });
        }
        if (typeof item === "function") {
          const result = item(actionContext);
          if (!React.isValidElement(result)) {
            return null;
          }
          return React.cloneElement(result, {
            key: result.key ?? `filters-panel-fn-${index}`,
          });
        }
        const key = normalizeKey(item);
        const elementKey = `filters-panel-action-${key ?? "custom"}-${index}`;
        if (!key) {
          return null;
        }
        switch (key) {
          case "advancedFilters":
            return (
              <Button
                key={elementKey}
                variant={advancedFiltersOpen ? "contained" : "outlined"}
                startIcon={<FilterAltIcon />}
                endIcon={
                  advancedFiltersOpen ? (
                    <ExpandLessIcon fontSize='small' />
                  ) : (
                    <ExpandMoreIcon fontSize='small' />
                  )
                }
                sx={{ whiteSpace: "nowrap" }}
                onClick={toggleAdvancedFilters}>
                Advanced Filters
              </Button>
            );
          case "clearFilters":
            return (
              <Button
                key={elementKey}
                variant='outlined'
                startIcon={<FilterAltOffIcon />}
                sx={{ whiteSpace: "nowrap" }}
                onClick={resetFilters}>
                Clear Filters
              </Button>
            );
          case "refresh":
            return (
              <Button
                key={elementKey}
                variant='outlined'
                startIcon={<RefreshIcon />}
                sx={{ whiteSpace: "nowrap" }}
                onClick={refresh}>
                Refresh
              </Button>
            );
          case "exportCsv":
            return (
              <Button
                key={elementKey}
                variant='contained'
                sx={{ whiteSpace: "nowrap" }}
                startIcon={<DownloadIcon />}
                onClick={exportAllResults}>
                Export CSV
              </Button>
            );
          default:
            return null;
        }
      })
      .filter(Boolean);
  }, [
    actionItems,
    filters,
    advancedFiltersOpen,
    resetFilters,
    toggleAdvancedFilters,
    refresh,
    exportAllResults,
  ]);

  return (
    <Paper variant='outlined' sx={{ p: 3 }}>
      <Stack spacing={3}>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={2}
          alignItems={{ xs: "stretch", md: "center" }}>
          <TextField
            name='patientSearch'
            value={filters.patientSearch}
            onChange={handleChange}
            placeholder='Search by patient name or ID'
            // sx={{
            //   width: "50%",
            // }}
            fullWidth
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position='start'>
                    <SearchIcon fontSize='small' />
                  </InputAdornment>
                ),
              },
            }}
          />
          <Stack direction='row' spacing={1} justifyContent='flex-end'>
            {resolvedActionItems}
          </Stack>
        </Stack>

        <Collapse in={advancedFiltersOpen} unmountOnExit>
          <Box
            sx={{
              borderRadius: 2,
              border: "1px solid",
              borderColor: "divider",
              p: 2,
              mt: 1,
            }}>
            <Grid container spacing={2}>
              <FormControl sx={{ m: 1, width: 300 }}>
                <InputLabel id='institute-label'>Institute</InputLabel>
                <Select
                  labelId='institute-label'
                  name='institute'
                  value={filters.institute}
                  onChange={handleChange}
                  fullWidth>
                  {filterOptions.institutes?.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl sx={{ m: 1, width: 300 }}>
                <InputLabel id='scannerStation-label'>Scanner Name</InputLabel>
                <Select
                  labelId='scannerStation-label'
                  name='scannerStation'
                  value={filters.scannerStation}
                  onChange={handleChange}
                  fullWidth>
                  {filterOptions.scanner_stations?.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl sx={{ m: 1, width: 300 }}>
                <InputLabel id='protocolName-label'>Protocol Name</InputLabel>
                <Select
                  labelId='protocolName-label'
                  name='protocolName'
                  value={filters.protocolName}
                  onChange={handleChange}
                  fullWidth>
                  {filterOptions.protocol_names?.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl sx={{ m: 1, width: 300 }}>
                <InputLabel id='scannerModel-label'>Scanner Model</InputLabel>
                <Select
                  labelId='scannerModel-label'
                  name='scannerModel'
                  value={filters.scannerModel}
                  onChange={handleChange}
                  fullWidth>
                  {filterOptions.scanner_models?.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={2}
                  sx={{ width: "100%" }}>
                  <DatePicker
                    label='Exam Date From'
                    name='examDateFrom'
                    value={examDateFromValue}
                    onChange={handleDateChange("examDateFrom")}
                    format='YYYY-MM-DD'
                    slotProps={{
                      field: {
                        clearable: true,
                      },
                    }}
                    minDate={minExamDate}
                    maxDate={maxExamDate}
                    sx={{ m: 1, width: 300 }}
                  />
                  <DatePicker
                    label='Exam Date To'
                    name='examDateTo'
                    value={examDateToValue}
                    onChange={handleDateChange("examDateTo")}
                    format='YYYY-MM-DD'
                    slotProps={{
                      field: {
                        clearable: true,
                      },
                    }}
                    minDate={examDateFromValue ?? minExamDate}
                    maxDate={maxExamDate}
                    sx={{ m: 1, width: 300 }}
                  />
                </Stack>
              </LocalizationProvider>
              <Grid item size={{ xs: 12, md: 6 }}>
                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={2}
                  sx={{ width: "100%" }}>
                  <FormControl sx={{ m: 1, width: 300 }}>
                    <InputLabel id='ageMin-label'>Age Min</InputLabel>
                    <TextField
                      labelId='ageMin-label'
                      name='ageMin'
                      type='number'
                      value={filters.ageMin}
                      onChange={handleChange}
                      slotProps={{
                        input: {
                          min: ageRange?.min ?? 0,
                          max: ageRange?.max ?? 150,
                          endAdornment: (
                            <InputAdornment position='end'>
                              years
                            </InputAdornment>
                          ),
                        },
                      }}
                    />
                  </FormControl>
                  <FormControl sx={{ m: 1, width: 300 }}>
                    <InputLabel id='ageMax-label'>Age Max</InputLabel>
                    <TextField
                      labelId='ageMax-label'
                      name='ageMax'
                      type='number'
                      value={filters.ageMax}
                      onChange={handleChange}
                      slotProps={{
                        input: {
                          min: ageRange?.min ?? 0,
                          max: ageRange?.max ?? 150,
                          endAdornment: (
                            <InputAdornment position='end'>
                              years
                            </InputAdornment>
                          ),
                        },
                      }}
                    />
                  </FormControl>
                </Stack>
              </Grid>
            </Grid>
          </Box>
        </Collapse>

        {activeFilters.length > 0 && (
          <Stack
            direction='row'
            spacing={1}
            flexWrap='wrap'
            alignItems='center'>
            <Typography variant='body2' color='text.secondary'>
              Active Filters:
            </Typography>
            {activeFilters.map((filter) => (
              <Chip
                key={filter.key}
                label={filter.label}
                onDelete={() => clearFilterValue(filter.key)}
                size='small'
              />
            ))}
          </Stack>
        )}
      </Stack>
    </Paper>
  );
};

export default FiltersPanel;
