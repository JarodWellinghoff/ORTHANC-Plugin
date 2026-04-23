import { useState } from "react";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Paper from "@mui/material/Paper";
import Autocomplete from "@mui/material/Autocomplete";
import Slider from "@mui/material/Slider";
import SvgIcon from "@mui/material/SvgIcon";
import PersonIcon from "@mui/icons-material/Person";
import LocationCityIcon from "@mui/icons-material/LocationCity";
import Stack from "@mui/material/Stack";
import InputAdornment from "@mui/material/InputAdornment";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Grid";
import SearchIcon from "@mui/icons-material/Search";
import LabelIcon from "@mui/icons-material/Label";
import PermIdentityIcon from "@mui/icons-material/PermIdentity";
import { useDashboard } from "../context/DashboardContext";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import dayjs from "dayjs";
import ProtocolNameIcon from "../../assets/protocol_name.svg?react";
import ScannerModelIcon from "../../assets/scanner_model.svg?react";
import ScannerStationIcon from "../../assets/scanner_station.svg?react";
import ClearIcon from "@mui/icons-material/Clear";

const MIN_AGE = 0;
const MAX_AGE = 200;
const ITEM_GRID_SIZE = { xs: 12, md: 6, lg: 3 };

const FiltersPanel = ({ filters, onChange, onQuery, onReset }) => {
  const { filterOptions } = useDashboard();
  console.log("filters:", filters);

  return (
    <Paper variant='outlined' sx={{ p: 3 }}>
      <Grid container spacing={3}>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            options={[]}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.patientIdSearch}
            onChange={(_, newValue) => onChange("patientIdSearch", newValue)}
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                name='patientIdSearch'
                placeholder='Patient IDs'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.patientIdSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <PermIdentityIcon />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            options={[]}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.patientNameSearch}
            onChange={(_, newValue) => onChange("patientNameSearch", newValue)}
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                name='patientNameSearch'
                placeholder='Patient Names'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.patientNameSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <PersonIcon />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            id='institutions'
            options={filterOptions.institutes}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.instituteSearch}
            onChange={(_, newValue) => onChange("instituteSearch", newValue)}
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                name='institutesSearch'
                placeholder='Institutes'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.instituteSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <LocationCityIcon />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            id='protocol-names'
            options={filterOptions.protocol_names ?? []}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.protocolNameSearch}
            onChange={(_, newValue) => onChange("protocolNameSearch", newValue)}
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                name='protocolNameSearch'
                placeholder='Protocol Name'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.protocolNameSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <SvgIcon
                            component={ProtocolNameIcon}
                            inheritViewBox
                          />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            id='scanner-models'
            options={filterOptions.scanner_models ?? []}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.scannerModelSearch}
            onChange={(_, newValue) => onChange("scannerModelSearch", newValue)}
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder='Scanner Models'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.scannerModelSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <SvgIcon
                            component={ScannerModelIcon}
                            inheritViewBox
                          />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            id='scanner-stations'
            options={filterOptions.scanner_stations ?? []}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.scannerStationSearch}
            onChange={(_, newValue) =>
              onChange("scannerStationSearch", newValue)
            }
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder='Scanner Stations'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.scannerStationSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <SvgIcon
                            component={ScannerStationIcon}
                            inheritViewBox
                          />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <Stack direction='row' spacing={2} alignItems='center'>
              <DatePicker
                enableAccessibleFieldDOMStructure={false}
                slots={{ textField: TextField }}
                slotProps={{
                  actionBar: { actions: ["clear", "today", "accept"] },
                  openPickerButton: {
                    sx: {
                      "&&": {
                        border: "none",
                        boxShadow: "none",
                        bgcolor: "transparent",
                        width: 36,
                        height: 36,
                        p: 0.5,
                        "&:hover": {
                          bgcolor: "transparent",
                          borderColor: "transparent",
                        },
                        "&:active": { bgcolor: "transparent" },
                        "& .MuiSvgIcon-root": { fontSize: 18 },
                      },
                    },
                  },
                  textField: {
                    placeholder: "",
                    InputLabelProps: { shrink: true },
                  },
                }}
                label='Study Date Start'
                name='studyDateStart'
                value={
                  filters?.studyDateStartSearch
                    ? dayjs(filters?.studyDateStartSearch)
                    : null
                }
                onChange={(newValue) =>
                  onChange("studyDateStartSearch", newValue)
                }
                sx={{ width: "100%" }}
              />
              <DatePicker
                enableAccessibleFieldDOMStructure={false}
                slots={{ textField: TextField }}
                slotProps={{
                  actionBar: { actions: ["clear", "today", "accept"] },
                  openPickerButton: {
                    sx: {
                      "&&": {
                        border: "none",
                        boxShadow: "none",
                        bgcolor: "transparent",
                        width: 36,
                        height: 36,
                        p: 0.5,
                        "&:hover": {
                          bgcolor: "transparent",
                          borderColor: "transparent",
                        },
                        "&:active": { bgcolor: "transparent" },
                        "& .MuiSvgIcon-root": { fontSize: 18 },
                      },
                    },
                  },
                  textField: {
                    placeholder: "",
                    InputLabelProps: { shrink: true },
                  },
                }}
                label='Study Date End'
                name='studyDateEnd'
                value={
                  filters?.studyDateEndSearch
                    ? dayjs(filters?.studyDateEndSearch)
                    : null
                }
                onChange={(newValue) =>
                  onChange("studyDateEndSearch", newValue)
                }
                sx={{ width: "100%" }}
              />
            </Stack>
          </LocalizationProvider>
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Stack direction='row' spacing={2} alignItems='center'>
            <TextField
              value={filters?.ageStartSearch}
              label='Min Age'
              type='number'
              onChange={(e) => {
                const value = Math.min(
                  Math.max(Number(e.target.value), MIN_AGE),
                  filters?.ageEndSearch,
                );
                onChange("ageStartSearch", value);
              }}
            />
            <Slider
              getAriaLabel={() => "Age range"}
              value={[filters?.ageStartSearch, filters?.ageEndSearch]}
              min={MIN_AGE}
              max={MAX_AGE}
              onChange={(event, newValue) => {
                onChange("ageStartSearch", newValue[0]);
                onChange("ageEndSearch", newValue[1]);
              }}
              valueLabelDisplay='auto'
              getAriaValueText={(value) => `${value} years`}
            />
            <TextField
              value={filters?.ageEndSearch}
              label='Max Age'
              type='number'
              onChange={(e) => {
                const value = Math.min(
                  Math.max(Number(e.target.value), filters?.ageStartSearch),
                  MAX_AGE,
                );
                onChange("ageEndSearch", value);
              }}
            />
          </Stack>
        </Grid>
        <Grid item size={ITEM_GRID_SIZE}>
          <Autocomplete
            multiple
            fullWidth
            options={[]}
            freeSolo
            sx={{
              ".MuiOutlinedInput-root": {
                paddingY: 0,
                height: "auto",
                alignItems: "center",
              },
              ".MuiInputAdornment-root": {
                alignSelf: "center",
              },
            }}
            value={filters?.pullScheduleSearch}
            onChange={(event, newValue) =>
              onChange("pullScheduleSearch", newValue)
            }
            renderValue={(value, getItemProps) =>
              value.map((option, index) => {
                const { key, ...itemProps } = getItemProps({ index });
                return (
                  <Chip
                    variant='outlined'
                    label={option}
                    key={key}
                    {...itemProps}
                  />
                );
              })
            }
            renderInput={(params) => (
              <TextField
                {...params}
                name='pullScheduleSearch'
                placeholder='Pull Schedules'
                slotProps={{
                  input: {
                    ...params.InputProps,
                    startAdornment:
                      filters === undefined ||
                      filters?.pullScheduleSearch.length === 0 ? (
                        <InputAdornment position='start'>
                          <LabelIcon />
                        </InputAdornment>
                      ) : (
                        params.InputProps.startAdornment
                      ),
                  },
                }}
              />
            )}
          />
        </Grid>
        <Grid item size={12}>
          <Stack
            direction='row'
            spacing={3}
            alignItems='center'
            justifyContent='flex-end'>
            <Button
              startIcon={<SearchIcon />}
              variant='contained'
              color='primary'
              onClick={onQuery}>
              Query
            </Button>
            <Button
              startIcon={<ClearIcon />}
              variant='contained'
              color='secondary'
              onClick={onReset}>
              Clear Filters
            </Button>
          </Stack>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default FiltersPanel;
