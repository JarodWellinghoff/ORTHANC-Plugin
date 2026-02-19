import * as React from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Divider from "@mui/material/Divider";
import Button from "@mui/material/Button";
import FormControl from "@mui/material/FormControl";
import InputAdornment from "@mui/material/InputAdornment";
import FormControlLabel from "@mui/material/FormControlLabel";
import { NumberField } from "@base-ui-components/react/number-field";
import { Image } from "mui-image";
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
import Grid from "@mui/material/Grid";
import InputLabel from "@mui/material/InputLabel";
import TextField from "@mui/material/TextField";
import LinearProgress from "@mui/material/LinearProgress";
import CircularProgress from "@mui/material/CircularProgress";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Typography from "@mui/material/Typography";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import CloseIcon from "@mui/icons-material/Close";
import DeleteForeverIcon from "@mui/icons-material/DeleteForever";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import KeyboardArrowDownRoundedIcon from "@mui/icons-material/KeyboardArrowDownRounded";
import KeyboardArrowUpRoundedIcon from "@mui/icons-material/KeyboardArrowUpRounded";
import { useNavigate } from "react-router-dom";
import { drawerWidth, userSectionHeight } from "./SideMenu";

import { API_BASE_URL } from "./dicom/cornerstoneConfig";
import { useDashboard } from "../context/DashboardContext";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import ChoPlots from "./plots/ChoPlots";
import MTFInteractivePlot from "./plots/MTFInteractivePlot";
import MetadataSections from "./metadata/MetadataSections";
import buildMetadataSections from "./metadata/metadataSections.utils";
import CornerstoneViewport from "./dicom/CornerstoneViewport";
import ResultsComparison from "./ResultsComparison";
import DicomTable from "./DicomTable";
import { Icon, Tooltip } from "@mui/material";
import { alpha } from "@mui/material/styles";
// import ChoNumberField from "./ChoNumberField";

const stageOrder = [
  "initialization",
  "loading",
  "preprocessing",
  "analysis",
  "finalizing",
];

const formatStageLabel = (value) => {
  if (!value || typeof value !== "string") {
    return null;
  }
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
};

const LesionGridCell = ({ children }) => {
  return (
    <Box
      sx={{
        // let flex handle height so borders don't break % math
        flex: 1,
        p: { xs: 1, md: 2 },
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        // horizontal lines (between rows)
        "&:not(:first-of-type)": {
          borderTop: "1px solid",
          borderColor: "divider",
        },
      }}>
      {children}
    </Box>
  );
};

const LesionGridColumn = ({ children }) => {
  return (
    <Stack
      sx={{
        width: "25%",
        // vertical lines (between columns)
        "&:not(:last-of-type)": {
          borderRight: "1px solid",
          borderColor: "divider",
        },
        // stack children evenly
        "& > *": { minHeight: 0 }, // avoid overflow
      }}>
      {children}
    </Stack>
  );
};

const LesionGrid = ({ params, recon_diameter_mm, rows }) => {
  console.log("params", params);
  //   console.log("recon_diameter_mm", recon_diameter_mm);
  //   console.log("rows", rows);
  return (
    <Box
      sx={{
        borderRadius: 2,
        border: "1px solid",
        borderColor: "divider",
        p: { xs: 1, md: 2 },
      }}>
      <Stack direction='row' sx={{ height: "100%" }}>
        <LesionGridColumn>
          <LesionGridCell></LesionGridCell>
          <LesionGridCell>
            <Typography
              variant='body1'
              color='text.secondary'
              sx={{
                fontSize: "1.875rem",
              }}>
              3 mm
            </Typography>
          </LesionGridCell>
          <LesionGridCell>
            <Typography
              variant='body2'
              color='text.secondary'
              sx={{
                fontSize: "1.875rem",
              }}>
              6 mm
            </Typography>
          </LesionGridCell>
          <LesionGridCell>
            <Typography
              variant='body2'
              color='text.secondary'
              sx={{
                fontSize: "1.875rem",
              }}>
              9 mm
            </Typography>
          </LesionGridCell>
        </LesionGridColumn>
        <LesionGridColumn>
          <LesionGridCell>
            <Typography
              variant='body2'
              color='text.secondary'
              sx={{
                fontSize: "1.875rem",
              }}>
              {params.lesionHUs[params.lesionSet][0]}
              &nbsp;HU
            </Typography>
          </LesionGridCell>
          <LesionGridCell></LesionGridCell>
          <LesionGridCell>
            <LesionImage
              lesionSet={params.lesionSet}
              index={2}
              recon_diameter_mm={recon_diameter_mm}
              rows={rows}
            />
          </LesionGridCell>
          <LesionGridCell></LesionGridCell>
        </LesionGridColumn>
        <LesionGridColumn>
          <LesionGridCell>
            <Typography
              variant='body2'
              color='text.secondary'
              sx={{
                fontSize: "1.875rem",
              }}>
              {params.lesionHUs[params.lesionSet][1]}
              &nbsp;HU
            </Typography>
          </LesionGridCell>
          <LesionGridCell>
            <LesionImage
              lesionSet={params.lesionSet}
              index={0}
              recon_diameter_mm={recon_diameter_mm}
              rows={rows}
            />
          </LesionGridCell>
          <LesionGridCell>
            <LesionImage
              lesionSet={params.lesionSet}
              index={3}
              recon_diameter_mm={recon_diameter_mm}
              rows={rows}
            />
          </LesionGridCell>
          <LesionGridCell>
            <LesionImage
              lesionSet={params.lesionSet}
              index={1}
              recon_diameter_mm={recon_diameter_mm}
              rows={rows}
            />
          </LesionGridCell>
        </LesionGridColumn>
        <LesionGridColumn>
          <LesionGridCell>
            <Typography
              variant='body2'
              color='text.secondary'
              sx={{
                fontSize: "1.875rem",
              }}>
              {params.lesionHUs[params.lesionSet][2]}
              &nbsp;HU
            </Typography>
          </LesionGridCell>
          <LesionGridCell></LesionGridCell>
          <LesionGridCell>
            <LesionImage
              lesionSet={params.lesionSet}
              index={4}
              recon_diameter_mm={recon_diameter_mm}
              rows={rows}
            />
          </LesionGridCell>
          <LesionGridCell></LesionGridCell>
        </LesionGridColumn>
      </Stack>
    </Box>
  );
};

const LesionImage = ({ lesionSet, index, recon_diameter_mm, rows }) => {
  const params = new URLSearchParams({
    set: String(lesionSet),
    index: String(index),
    recon_diameter_mm: String(recon_diameter_mm || 480),
    rows: String(rows || 1024),
  });
  //   params.set("_", String(Date.now()));
  if (!params) return null;
  console.log("LesionImage params:", params.toString());
  console.log("lesionSet:", lesionSet);
  console.log("index:", index);
  console.log("recon_diameter_mm:", recon_diameter_mm);
  console.log("rows:", rows);

  const url = `${import.meta.env.VITE_API_URL}/create-lesion?${params}`;
  return (
    <Image
      key={`${lesionSet}-${index}`}
      src={url}
      alt={`Lesion Model ${index}`}
      showLoading={<CircularProgress size='100%' />}
      duration={1500}
      wrapperStyle={{ width: "100%", height: "100%", position: "relative" }}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "contain",
        display: "block",
      }}
    />
  );
};

const InfoRow = ({ label, value }) => (
  <Stack
    direction='row'
    spacing={1}
    justifyContent='space-between'
    sx={{
      maxWidth: "400px",
      minWidth: "250px",
    }}>
    <Typography variant='body2' color='text.secondary'>
      {label}
    </Typography>
    <Typography variant='body2' sx={{ fontWeight: 500 }}>
      {value ?? "N/A"}
    </Typography>
  </Stack>
);

const ChoAnalysisPage = () => {
  const { choModal, actions, summary } = useDashboard();
  const {
    seriesUuid,
    seriesId,
    stage,
    params,
    progress,
    results,
    pollError,
    storedResults,
  } = choModal;
  console.log("storedResults:", storedResults);
  console.log("results:", results);

  const { availableSeries } = summary;

  const navigate = useNavigate();

  const [csvData, setCsvData] = React.useState(null);
  const [csvLoading, setCsvLoading] = React.useState(true);
  const [csvError, setCsvError] = React.useState(null);
  const defaultMtfc50Data = {
    "50HU": 3.62,
    "30HU": 3.482,
    "10HU": 3.093,
  };
  const [mtfc50Data, setMtfc50Data] = React.useState(defaultMtfc50Data);
  const defaultMtfc10Data = {
    "50HU": 7.204,
    "30HU": 6.922,
    "10HU": 6.117,
  };
  const [mtfc10Data, setMtfc10Data] = React.useState(defaultMtfc10Data);
  const [currentContrast, setCurrentContrast] = React.useState("10HU");
  const displayedResults = results ?? storedResults?.data ?? null;
  const hasComparison = Boolean(results && storedResults?.data);
  const baselineResults = hasComparison ? storedResults.data : null;
  const comparisonCurrent = hasComparison ? results : null;
  const isStoredLoading = Boolean(storedResults?.loading && !results);
  const storedError = storedResults?.error;
  const isAvailableSeries = availableSeries.includes(seriesUuid);
  const [currentDICOMIndex, setCurrentDICOMIndex] = React.useState(0);
  const [seriesInstances, setSeriesInstances] = React.useState([]);
  const [dicomTags, setDicomTags] = React.useState([]);
  const currentStageIndex = stageOrder.indexOf(
    progress?.stage ?? "initialization",
  );
  React.useEffect(() => {
    fetch("/mtfc_data.csv")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load CSV file");
        }
        return response.text();
      })
      .then((data) => {
        setCsvData(data);
        setCsvLoading(false);
      })
      .catch((err) => {
        setCsvError(err.message);
        setCsvLoading(false);
      });
  }, []);

  React.useEffect(() => {
    if (!seriesUuid) return;
    fetch(`${API_BASE_URL}/series/${seriesUuid}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load DICOM instances");
        }
        return response.json();
      })
      .then((data) => {
        setSeriesInstances(data.Instances);
      })
      .catch((err) => {
        console.error("Error fetching DICOM instances:", err);
      });
  }, [seriesUuid]);

  React.useEffect(() => {
    if (!seriesInstances.length) return;

    function flattenDicomData(data) {
      const out = [];
      for (const [tag, elem] of Object.entries(data)) {
        pushElem(tag, elem, out);
      }
      // add id = index
      return out.map((item, index) => ({ ...item, id: index }));
    }

    function pushElem(tag, elem, out) {
      const row = { Name: elem.Name, Type: elem.Type, Tag: `(${tag})` };

      if (elem.Type === "Sequence" && Array.isArray(elem.Value)) {
        // Show only child tag references
        const childTags = new Set();
        elem.Value.forEach((seqItem) => {
          Object.keys(seqItem || {}).forEach((childTag) =>
            childTags.add(childTag),
          );
        });

        row.Value = `[${[...childTags].map((t) => `(${t})`).join(", ")}]`;
        out.push(row);

        // Add children as separate rows
        elem.Value.forEach((seqItem) => {
          Object.entries(seqItem || {}).forEach(([childTag, child]) => {
            pushElem(childTag, child, out);
          });
        });
      } else {
        // Normal element
        let v = elem.Value;
        if (Array.isArray(v)) v = v.join("\\"); // handle multi-valued primitives
        row.Value = v;
        out.push(row);
      }
    }

    const currentInstance = seriesInstances[currentDICOMIndex];
    fetch(`${API_BASE_URL}/instances/${currentInstance}/tags`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load DICOM tags");
        }
        return response.json();
      })
      .then((data) => {
        // const tagsArray = Object.entries(data).map(([key, value]) => ({
        //   ...value,
        //   Tag: key,
        // }));
        // const tagsArray = flattenDicomData(data);
        // console.log("DICOM tags:", tagsArray);
        setDicomTags(data);
      })
      .catch((err) => {
        console.error("Error fetching DICOM tags:", err);
      });
  }, [seriesInstances, currentDICOMIndex]);

  const resolvedSummary = React.useMemo(() => {
    if (!displayedResults) return null;

    const scanner = displayedResults?.scanner ?? displayedResults;
    const ct = displayedResults?.ct ?? displayedResults;
    const patient = displayedResults?.patient ?? displayedResults;
    const study = displayedResults?.study ?? displayedResults;
    const series = displayedResults?.series ?? displayedResults;

    console.log("displayedResults", displayedResults);

    return {
      make: scanner.manufacturer ?? null,
      location: scanner.institution_name ?? null,
      scanner_model: scanner.scanner_model ?? null,
      station_name: scanner.station_name ?? null,
      collimation: `${
        (ct.total_collimation_width_mm ?? null) /
        (ct.single_collimation_width_mm ?? null)
      } x ${ct.single_collimation_width_mm ?? "N/A"} mm`,
      qrm: ct.qrm ?? null,
      helical_pitch: ct.spiral_pitch_factor ?? null,
      tube_potential: ct.kvp ?? null,
      effective_mAs: ct.effective_mAs ?? null,
      recon_fov: ct.recon_diameter_mm ?? null,
      slice_thickness: ct.slice_thickness_mm ?? null,
      rotation_time: ct.exposure_time_ms ?? null,
      ctdi_vol: ct.ctdi_vol ?? null,
      ctdi_phantom_type: ct.ctdi_phantom_type ?? "BODY",
      patient_name: patient.patient_name ?? null,
      patient_id: patient.patient_id ?? null,
      patient_sex: patient.patient_sex ?? null,
      birth_date: patient.birth_date ?? null,
      study_date: study.study_date ?? null,
      study_id: study.study_id ?? null,
      study_description: study.study_description ?? null,
      study_time: study.study_time ?? null,
      kernel: series.convolution_kernel ?? null,
      pixel_size: series.pixel_spacing_mm?.[0] ?? null,
      slice_interval: series.slice_interval_mm ?? null,
      series_description: series.series_description ?? null,
      pixel_spacing_mm: series.pixel_spacing_mm ?? null,
      protocol_name: series.protocol_name ?? null,
    };
  }, [displayedResults]);

  const memoizedSeriesDetails = React.useMemo(() => {
    if (!resolvedSummary) return null;
    console.log("resolvedSummary", resolvedSummary);
    return (
      <Stack
        spacing={3}
        direction={"row"}
        sx={{
          width: "100%",
          height: { sm: "100%", md: "55vh" },
          overflow: "auto",
          justifyContent: "space-evenly",
        }}>
        <Stack
          spacing={1.5}
          sx={{
            height: { sm: "100%", md: "55vh", alignItems: "left" },
          }}>
          <InfoRow label='Make' value={resolvedSummary.make} />
          <InfoRow label='Location' value={resolvedSummary.location} />
          <InfoRow label='Model' value={resolvedSummary.scanner_model} />
          <InfoRow label='Station Name' value={resolvedSummary.station_name} />
          <InfoRow label='Collimation' value={resolvedSummary.collimation} />
          <InfoRow label='QRM' value={resolvedSummary.qrm} />
          <InfoRow
            label='Helical Pitch'
            value={resolvedSummary.helical_pitch}
          />
          <InfoRow
            label='Tube Potential'
            value={resolvedSummary.tube_potential}
          />
          <InfoRow
            label='Effective mAs'
            value={resolvedSummary.effective_mAs}
          />
          <InfoRow label='Recon FOV' value={resolvedSummary.recon_fov} />
          <InfoRow
            label='Slice Thickness'
            value={resolvedSummary.slice_thickness}
          />
          <InfoRow
            label='Rotation Time'
            value={resolvedSummary.rotation_time}
          />
          <InfoRow label='CTDI vol' value={resolvedSummary.ctdi_vol} />
          <InfoRow
            label='CTDI Phantom Type'
            value={resolvedSummary.ctdi_phantom_type}
          />
        </Stack>
        <Divider orientation='vertical' flexItem />
        <Stack
          spacing={1.5}
          sx={{
            height: { sm: "100%", md: "55vh" },
            overflowY: "auto",
            paddingRight: 1.5,
          }}>
          <InfoRow label='Patient Name' value={resolvedSummary.patient_name} />
          <InfoRow label='Patient ID' value={resolvedSummary.patient_id} />
          <InfoRow label='Patient Sex' value={resolvedSummary.patient_sex} />
          <InfoRow label='Birth Date' value={resolvedSummary.birth_date} />
          <InfoRow label='Study ID' value={resolvedSummary.study_id} />
          <InfoRow
            label='Study Description'
            value={resolvedSummary.study_description}
          />
          <InfoRow label='Study Date' value={resolvedSummary.study_date} />
          <InfoRow label='Study Time' value={resolvedSummary.study_time} />
          <InfoRow
            label='Protocol Name'
            value={resolvedSummary.protocol_name}
          />
          <InfoRow
            label='Series Description'
            value={resolvedSummary.series_description}
          />
          <InfoRow label='Kernel' value={resolvedSummary.kernel} />
          {/* <InfoRow label='Pixel Size' value={resolvedSummary.pixel_size} /> */}
          <InfoRow
            label='Slice Interval'
            value={resolvedSummary.slice_interval}
          />
          {/* <InfoRow
            label='Pixel Spacing'
            value={resolvedSummary.pixel_spacing}
          /> */}
        </Stack>
      </Stack>
    );
  }, [resolvedSummary]);

  const [detailsTab, setDetailsTab] = React.useState("series-details");
  const [isConfigOpen, setConfigOpen] = React.useState(false);
  const [dialogDismissedDuringRun, setDialogDismissedDuringRun] =
    React.useState(false);
  const [showButtonProgress, setShowButtonProgress] = React.useState(false);
  const isAnalysisRunning = stage === "progress";
  const rawProgressValue =
    typeof progress?.value === "number" ? progress.value : 0;
  const buttonProgressValue = isAnalysisRunning
    ? Math.min(100, Math.max(0, Math.round(rawProgressValue)))
    : 0;
  const progressStageLabel = formatStageLabel(progress?.stage);
  const normalizedProgressMessage =
    typeof progress?.message === "string" && progress.message.trim().length > 0
      ? progress.message
      : null;
  const analysisTooltip = isAnalysisRunning
    ? normalizedProgressMessage || progressStageLabel || "Processing..."
    : "Start CHO analysis";
  const startButtonDisabled =
    (!seriesId && !isAnalysisRunning) || !isAvailableSeries;

  const metadataSections = React.useMemo(() => {
    if (!displayedResults) return [];
    return buildMetadataSections(displayedResults);
  }, [displayedResults]);

  const filteredMetadataSections = React.useMemo(
    () =>
      metadataSections
        .map((section) => ({
          ...section,
          items: section.items.filter((item) => item.value !== "N/A"),
        }))
        .filter((section) => section.items.length > 0),
    [metadataSections],
  );

  const allMetadataContent = React.useMemo(() => {
    if (dicomTags.length === 0) return null;

    return <DicomTable data={dicomTags} />;
  }, [dicomTags]);

  //   const allMetadataContent = React.useMemo(() => {
  //     if (filteredMetadataSections.length === 0) return null;

  //     return (
  //       <Stack
  //         spacing={2}
  //         sx={{
  //           height: { sm: "100%", md: "55vh" },
  //           overflowY: "auto",
  //           paddingRight: 1.5,
  //         }}>
  //         {filteredMetadataSections.map((section) => {
  //           if (section.title === "Results") return null;
  //           return (
  //             <Box key={section.title}>
  //               <Typography variant='subtitle2' sx={{ fontWeight: 600, mb: 1 }}>
  //                 {section.title}
  //               </Typography>
  //               <Stack spacing={1.5}>
  //                 {section.items.map((item) => (
  //                   <Stack
  //                     key={item.label}
  //                     direction='row'
  //                     spacing={1}
  //                     justifyContent='space-between'>
  //                     <Typography variant='body2' color='text.secondary'>
  //                       {item.label}
  //                     </Typography>
  //                     <Typography variant='body2' sx={{ fontWeight: 500 }}>
  //                       {item.value}
  //                       {item.unit ? ` ${item.unit}` : ""}
  //                     </Typography>
  //                   </Stack>
  //                 ))}
  //               </Stack>
  //             </Box>
  //           );
  //         })}
  //       </Stack>
  //     );
  //   }, [filteredMetadataSections]);

  const detailsTabs = React.useMemo(() => {
    const tabs = [
      {
        label: "Series Summary",
        value: "series-details",
        content: memoizedSeriesDetails,
        available: Boolean(memoizedSeriesDetails),
      },
      {
        label: "All DICOM Tags",
        value: "all-dicom-tags",
        content: allMetadataContent,
        available: Boolean(allMetadataContent),
      },
    ];

    return tabs;
  }, [memoizedSeriesDetails, allMetadataContent]);

  React.useEffect(() => {
    const activeTab = detailsTabs.find(
      (tab) => tab.value === detailsTab && tab.available,
    );
    if (!activeTab) {
      const fallback = detailsTabs.find((tab) => tab.available);
      if (fallback && fallback.value !== detailsTab) {
        setDetailsTab(fallback.value);
      }
    }
  }, [detailsTab, detailsTabs]);

  React.useEffect(() => {
    if (isAnalysisRunning && !dialogDismissedDuringRun) {
      setConfigOpen(true);
    }
  }, [isAnalysisRunning, dialogDismissedDuringRun]);

  React.useEffect(() => {
    if (!isAnalysisRunning) {
      setDialogDismissedDuringRun(false);
      setShowButtonProgress(false);
    } else if (isConfigOpen) {
      setShowButtonProgress(false);
    }
  }, [isAnalysisRunning, isConfigOpen]);
  const handleDetailsTabChange = React.useCallback((event, value) => {
    setDetailsTab(value);
  }, []);

  const handleStartAnalysis = React.useCallback(() => {
    actions.startChoAnalysis();
  }, [actions]);

  const handleFieldChange = (name) => (event) => {
    actions.updateChoParam(name, event.target.value);
  };

  const handleBackToDashboard = React.useCallback(() => {
    actions.closeChoModal();
    navigate("/main-dashboard");
  }, [actions, navigate]);

  const activeDetailsTab = detailsTabs.find(
    (tab) => tab.value === detailsTab && tab.available,
  );
  const hasAvailableDetails = detailsTabs.some((tab) => tab.available);

  return (
    <Box
      sx={{
        py: 4,
        pb: { xs: 16, md: 14 },
      }}>
      <Stack spacing={3}>
        {/* <Box>
          <Typography variant='h4' sx={{ fontWeight: 600 }}>
            CHO Analysis Workspace
          </Typography>
          <Typography variant='body2' color='text.secondary'>
            Review existing results, adjust configuration, and rerun the CHO
            analysis with updated parameters.
          </Typography>
        </Box> */}

        <Grid container spacing={3}>
          <Grid item size={12} sx={{ width: "100%" }}>
            <Stack
              spacing={2}
              direction={{ xs: "column", md: "row" }}
              sx={{ width: "100%" }}>
              <Paper
                variant='outlined'
                sx={{ p: 2, borderRadius: 2, width: "100%", minHeight: 500 }}>
                <Stack spacing={1.5} sx={{ height: "100%" }}>
                  <CornerstoneViewport
                    seriesUuid={seriesUuid}
                    currentIndex={currentDICOMIndex}
                    setCurrentIndex={setCurrentDICOMIndex}
                  />
                </Stack>
              </Paper>
              <Paper
                variant='outlined'
                sx={{
                  p: 2,
                  borderRadius: 2,
                  minHeight: 180,
                  width: "100%",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                }}>
                <Tabs
                  value={detailsTab}
                  onChange={handleDetailsTabChange}
                  variant='scrollable'
                  allowScrollButtonsMobile
                  sx={{ mb: 1, justifyContent: "center" }}>
                  {detailsTabs.map((tab) => (
                    <Tab
                      key={tab.value}
                      label={tab.label}
                      value={tab.value}
                      disabled={!tab.available}
                    />
                  ))}
                </Tabs>
                <Divider />
                <Box sx={{ flex: 1, mt: 1 }}>
                  {hasAvailableDetails ? (
                    (activeDetailsTab?.content ?? (
                      <Typography variant='body2' color='text.secondary'>
                        Select a section to view details.
                      </Typography>
                    ))
                  ) : (
                    <Typography variant='body2' color='text.secondary'>
                      Select a series from the dashboard to begin.
                    </Typography>
                  )}
                </Box>
              </Paper>
            </Stack>
          </Grid>
          <Grid item size={12} sx={{ width: "100%" }}>
            <Paper
              variant='outlined'
              sx={{
                p: 3,
                borderRadius: 2,
                width: "100%",
              }}>
              <Stack spacing={2}>
                {displayedResults ? (
                  <Grid container spacing={3}>
                    <Grid
                      item
                      size={{
                        sm: 12,
                        // md: hasComparison && comparisonCurrent ? 12 : 9,
                        md: 12,
                      }}>
                      <Stack spacing={3}>
                        <Box>
                          <Typography
                            variant='subtitle1'
                            sx={{ fontWeight: 600 }}>
                            Analysis Configuration
                          </Typography>
                        </Box>
                        {!seriesId && (
                          <Alert severity='info'>
                            Select a series from the dashboard to enable CHO
                            analysis configuration.
                          </Alert>
                        )}
                        <Grid container spacing={2}>
                          <Grid size={{ sm: 12, md: 6 }}>
                            <Stack
                              spacing={2}
                              sx={{
                                border: "1px solid",
                                borderColor: "divider",
                                borderRadius: 2,
                                p: 2,
                              }}>
                              <FormControl fullWidth>
                                <InputLabel id='cho-lesion-set-label'>
                                  Lesion Set
                                </InputLabel>
                                <Select
                                  labelId='cho-lesion-set-label'
                                  label='Lesion Set'
                                  value={params.lesionSet}
                                  onChange={handleFieldChange("lesionSet")}
                                  fullWidth>
                                  <MenuItem value='standard'>Standard</MenuItem>
                                  <MenuItem value='low-contrast' disabled>
                                    Low Contrast
                                  </MenuItem>
                                  <MenuItem value='high-contrast' disabled>
                                    High Contrast
                                  </MenuItem>
                                </Select>
                              </FormControl>
                              <LesionGrid
                                params={params}
                                recon_diameter_mm={
                                  displayedResults.recon_diameter_mm
                                }
                                rows={displayedResults.rows}
                              />
                            </Stack>
                          </Grid>
                          <Grid container size={{ sm: 12, md: 6 }}>
                            <Grid container spacing={2}>
                              <Grid
                                container
                                size={12}
                                sx={{
                                  border: "1px solid",
                                  borderColor: "divider",
                                  borderRadius: 2,
                                  p: 2,
                                }}>
                                <Grid item size={{ sm: 12, md: 6 }}>
                                  <FormControl fullWidth>
                                    <InputLabel id='cho-spatial-resolution-label'>
                                      Contrast Dependent Spatial Resolution
                                    </InputLabel>
                                    <Select
                                      labelId='cho-spatial-resolution-label'
                                      label='Contrast Dependent Spatial Resolution'
                                      value={params.spatialResolution}
                                      onChange={handleFieldChange(
                                        "spatialResolution",
                                      )}
                                      fullWidth>
                                      <MenuItem value='auto'>Auto</MenuItem>
                                      <MenuItem value='custom-full'>
                                        Custom
                                      </MenuItem>
                                    </Select>
                                  </FormControl>
                                </Grid>
                                <Grid item size={{ sm: 12, md: 6 }}></Grid>
                                <Grid item size={{ sm: 12, md: 3 }}>
                                  <FormControl fullWidth>
                                    <InputLabel id='contrast-label'>
                                      Contrast
                                    </InputLabel>
                                    <Select
                                      label='Contrast'
                                      labelId='contrast-label'
                                      //   disabled={
                                      //     params.spatialResolution === "auto"
                                      //   }
                                      value={currentContrast}
                                      onChange={(e) =>
                                        setCurrentContrast(e.target.value)
                                      }
                                      fullWidth>
                                      <MenuItem value='10HU'>-10HU</MenuItem>
                                      <MenuItem value='30HU'>-30HU</MenuItem>
                                      <MenuItem value='50HU'>-50HU</MenuItem>
                                    </Select>
                                  </FormControl>
                                </Grid>
                                <Grid item size={{ sm: 12, md: 4 }}>
                                  <TextField
                                    id='mtf-50-field'
                                    label='MTF 50%'
                                    type='number'
                                    value={
                                      params.spatialResolution === "auto"
                                        ? defaultMtfc50Data[currentContrast]
                                        : mtfc50Data[currentContrast]
                                    }
                                    // value={mtfc50Data[currentContrast] ?? null}
                                    slotProps={{
                                      input: {
                                        endAdornment: (
                                          <InputAdornment position='end'>
                                            cycles/cm
                                          </InputAdornment>
                                        ),
                                        slotProps: {
                                          input: {
                                            step: "0.01",
                                          },
                                        },
                                      },
                                    }}
                                    onChange={(e) =>
                                      setMtfc50Data((prev) => ({
                                        ...prev,
                                        [currentContrast]:
                                          e.target.value ?? null,
                                      }))
                                    }
                                    disabled={
                                      params.spatialResolution === "auto"
                                    }
                                  />
                                </Grid>
                                <Grid item size={{ sm: 12, md: 4 }}>
                                  <TextField
                                    id='mtf-10-field'
                                    label='MTF 10%'
                                    // type='number'
                                    type={
                                      params.spatialResolution === "custom-50"
                                        ? "text"
                                        : "number"
                                    }
                                    placeholder='Gaussian'
                                    value={
                                      params.spatialResolution === "auto"
                                        ? defaultMtfc10Data[currentContrast]
                                        : params.spatialResolution ===
                                            "custom-50"
                                          ? "Gaussian"
                                          : params.spatialResolution ===
                                              "custom-full"
                                            ? mtfc10Data[currentContrast]
                                            : null
                                    }
                                    slotProps={{
                                      input: {
                                        endAdornment: (
                                          <InputAdornment position='end'>
                                            cycles/cm
                                          </InputAdornment>
                                        ),
                                        slotProps: {
                                          input: {
                                            step: "0.01",
                                          },
                                        },
                                      },
                                    }}
                                    onChange={(e) =>
                                      setMtfc10Data((prev) => ({
                                        ...prev,
                                        [currentContrast]:
                                          e.target.value ?? null,
                                      }))
                                    }
                                    disabled={
                                      params.spatialResolution !== "custom-full"
                                    }
                                  />
                                </Grid>
                                <Grid item size={{ sm: 12, md: 1 }}>
                                  <Tooltip title='Reset MTF values to default'>
                                    <IconButton
                                      aria-label='reset-mtf-values'
                                      onClick={() => {
                                        setMtfc50Data(defaultMtfc50Data);
                                        setMtfc10Data(defaultMtfc10Data);
                                      }}
                                      sx={{
                                        "&&": {
                                          border: "none",
                                          padding: 0,
                                          height: "100%",
                                          width: "100%",
                                        },
                                      }}>
                                      <RestartAltRoundedIcon />
                                    </IconButton>
                                  </Tooltip>
                                </Grid>
                                <Grid size={12}>
                                  <MTFInteractivePlot
                                    originalPlotData={csvData}
                                    loading={csvLoading}
                                    error={csvError}
                                    editable50={
                                      //   params.spatialResolution !== "auto"
                                      false
                                    }
                                    editable10={
                                      //   params.spatialResolution === "custom-full"
                                      false
                                    }
                                    mtfc50Data={mtfc50Data}
                                    mtfc10Data={mtfc10Data}
                                    setMtfc50Data={setMtfc50Data}
                                    setMtfc10Data={setMtfc10Data}
                                    currentContrast={currentContrast}
                                  />
                                </Grid>
                              </Grid>
                            </Grid>
                          </Grid>
                        </Grid>
                      </Stack>
                    </Grid>
                  </Grid>
                ) : !isStoredLoading ? (
                  <Typography variant='body2' color='text.secondary'>
                    No results available yet. Run the analysis to generate new
                    results or reload stored data.
                  </Typography>
                ) : null}
              </Stack>
            </Paper>
          </Grid>
          <Grid item size={12} sx={{ width: "100%" }}>
            <Paper
              variant='outlined'
              sx={{
                p: 3,
                borderRadius: 2,
                width: "100%",
              }}>
              <Stack spacing={2}>
                <Box>
                  <Typography variant='subtitle1' sx={{ fontWeight: 600 }}>
                    Analysis Results
                  </Typography>
                </Box>

                {pollError && <Alert severity='error'>{pollError}</Alert>}
                {storedError && !results && !isStoredLoading && (
                  <Alert severity='warning'>{storedError}</Alert>
                )}

                {isAnalysisRunning && !displayedResults && !isStoredLoading && (
                  <Typography variant='body2' color='text.secondary'>
                    An analysis is currently running. Open the Start Analysis
                    dialog to monitor progress.
                  </Typography>
                )}

                {isStoredLoading && (
                  <Stack alignItems='center' spacing={1} sx={{ py: 4 }}>
                    <LinearProgress sx={{ width: "100%" }} />
                    <Typography variant='body2' color='text.secondary'>
                      Loading stored analysis results...
                    </Typography>
                  </Stack>
                )}

                {displayedResults ? (
                  <Grid container spacing={3}>
                    <Grid item size={12}>
                      <ChoPlots
                        data={displayedResults}
                        comparison={baselineResults}
                        direction={"row"}>
                        {/* <Box
                          sx={{
                            border: "1px solid",
                            borderColor: "divider",
                            borderRadius: 2,
                            p: 2,
                            height: "100%",
                          }}> */}
                        <MetadataSections
                          data={displayedResults}
                          visibleSections={["Results"]}
                        />
                        {/* </Box> */}
                      </ChoPlots>
                    </Grid>
                    {/* <Grid
                      item
                      size={{
                        sm: 12,
                        md: hasComparison && comparisonCurrent ? 12 : 3,
                      }}>
                      <Box
                        sx={{
                          border: "1px solid",
                          borderColor: "divider",
                          borderRadius: 2,
                          p: 2,
                          height: "100%",
                        }}>
                        {hasComparison && comparisonCurrent ? (
                          <ResultsComparison
                            baseline={baselineResults}
                            current={comparisonCurrent}
                          />
                        ) : (
                          <MetadataSections
                            data={displayedResults}
                            visibleSections={["Results"]}
                          />
                        )}
                      </Box>
                    </Grid> */}
                  </Grid>
                ) : !isStoredLoading ? (
                  <Typography variant='body2' color='text.secondary'>
                    No results available yet. Run the analysis to generate new
                    results or reload stored data.
                  </Typography>
                ) : null}
              </Stack>
            </Paper>
          </Grid>
        </Grid>
      </Stack>

      <Box
        sx={{
          display: { xs: "none", md: "block" },
          position: "fixed",
          bottom: 0,
          left: drawerWidth,
          right: 0,
          borderTop: "1px solid",
          borderColor: "divider",
          bgcolor: "background.paper",
          py: 1.5,
          mt: 3,
          height: userSectionHeight,
          zIndex: (theme) => theme.zIndex.drawer - 1,
        }}>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={1.5}
          justifyContent={{ xs: "stretch", md: "space-between" }}
          alignItems={{ xs: "stretch", md: "center" }}
          sx={{ px: { xs: 1, sm: 2 } }}>
          <Typography
            variant='body2'
            color='text.secondary'
            sx={{ mb: { xs: 1, md: 0 } }}>
            {resolvedSummary
              ? `${resolvedSummary.patient_name ?? "Anonymous Patient"} - ${
                  resolvedSummary.protocol_name ?? "Anonymous Protocol"
                }`
              : "Select a series from the dashboard to enable all analysis actions."}
          </Typography>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={1}
            justifyContent='flex-end'
            flexWrap='wrap'>
            <Button
              variant='outlined'
              startIcon={<ArrowBackIcon />}
              onClick={handleBackToDashboard}>
              Back to Dashboard
            </Button>
            <Button
              variant='outlined'
              startIcon={<RestartAltIcon />}
              onClick={actions.reloadStoredChoResults}
              disabled={!displayedResults || storedResults?.loading}>
              Reload Stored
            </Button>
            <Button
              variant='outlined'
              startIcon={<FileDownloadIcon />}
              onClick={() => seriesId && actions.exportSeries(seriesId)}
              disabled={!displayedResults}>
              Export XLS
            </Button>
            <Tooltip title={analysisTooltip}>
              <span style={{ display: "inline-flex" }}>
                <Button
                  variant='contained'
                  startIcon={<PlayArrowIcon />}
                  onClick={handleStartAnalysis}
                  disabled={isAnalysisRunning || startButtonDisabled}
                  sx={{
                    position: "relative",
                    overflow: "hidden",
                    ...(isAnalysisRunning && {
                      "&::after": {
                        content: '""',
                        position: "absolute",
                        inset: 0,
                        width: `${buttonProgressValue}%`,
                        bgcolor: (theme) =>
                          alpha(theme.palette.success.main, 0.45),
                        transition: "width 250ms ease",
                        pointerEvents: "none",
                        zIndex: 0,
                      },
                      "& .analysis-button-content": {
                        position: "relative",
                        zIndex: 1,
                      },
                      "& .MuiButton-startIcon": {
                        position: "relative",
                        zIndex: 1,
                      },
                    }),
                  }}>
                  <Box component='span' className='analysis-button-content'>
                    {isAnalysisRunning ? "Analysis Running" : "Start Analysis"}
                  </Box>
                </Button>
              </span>
            </Tooltip>
            <Box sx={{ position: "relative", display: "inline-flex" }}></Box>
            <Button
              variant='outlined'
              color='error'
              startIcon={<DeleteForeverIcon />}
              onClick={actions.discardChoResults}
              disabled={!results || stage !== "results"}>
              Discard Results
            </Button>
          </Stack>
        </Stack>
      </Box>
    </Box>
  );
};

export default ChoAnalysisPage;
