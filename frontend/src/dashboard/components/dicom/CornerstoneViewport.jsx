import { useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import OpenWithRoundedIcon from "@mui/icons-material/OpenWithRounded";

import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Button,
  Grid,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import LayersRoundedIcon from "@mui/icons-material/LayersRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import SettingsBrightnessRoundedIcon from "@mui/icons-material/SettingsBrightnessRounded";
import ZoomInMapRoundedIcon from "@mui/icons-material/ZoomInMapRounded";
import PanToolAltRoundedIcon from "@mui/icons-material/PanToolAltRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";

import {
  API_BASE_URL,
  ensureCornerstoneReady,
  ensureCornerstoneToolsReady,
  cornerstoneApi,
} from "./cornerstoneConfig";

const { cornerstone, cornerstoneTools } = cornerstoneApi;

const toolDefinitions = [
  {
    value: "Wwwc",
    label: "W/L",
    icon: <SettingsBrightnessRoundedIcon fontSize='small' />,
    tooltip: "Adjust window level",
  },
  {
    value: "Zoom",
    label: "Zoom",
    icon: <SearchRoundedIcon fontSize='small' />,
    tooltip: "Zoom in/out",
  },
  {
    value: "Pan",
    label: "Pan",
    icon: <OpenWithRoundedIcon fontSize='small' />,
    tooltip: "Pan the image",
  },
];

const sortInstances = (instances) => {
  return instances.slice().sort((a, b) => {
    const aNumber = Number(
      a?.MainDicomTags?.InstanceNumber ?? a?.IndexInSeries ?? 0
    );
    const bNumber = Number(
      b?.MainDicomTags?.InstanceNumber ?? b?.IndexInSeries ?? 0
    );
    return aNumber - bNumber;
  });
};

const ResetButton = ({ onClick, disabled }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <Tooltip title='Reset all viewports'>
      <IconButton
        size='small'
        disabled={disabled}
        onClick={onClick}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        sx={{
          position: "absolute",
          top: (theme) => theme.spacing(1),
          right: (theme) => theme.spacing(1),
        }}>
        <RestartAltRoundedIcon
          sx={{
            transform: hovered ? "rotate(360deg)" : "none",
            transition: "all 0.5s ease",
          }}
        />
      </IconButton>
    </Tooltip>
  );
};

const ToolButtons = ({ activeTool, onChange }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <Box
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      sx={{
        transition: "all 0.3s ease",
        position: "absolute",
        bgcolor: hovered ? "black" : "transparent",
      }}>
      <ToggleButtonGroup
        size='small'
        value={activeTool}
        exclusive
        orientation='vertical'
        onChange={(event, value) => {
          if (value) onChange(value);
        }}
        sx={{}}>
        {toolDefinitions.map((tool) => (
          <Tooltip title={tool.tooltip} placement='left'>
            <ToggleButton
              key={tool.value}
              value={tool.value}
              sx={{
                display: "flex",
                justifyContent: "flex-start",
              }}>
              <Stack
                direction='row'
                spacing={hovered ? 1 : 0}
                sx={{ transition: "all 0.3s ease" }}>
                {tool.icon}
                <Typography
                  variant='caption'
                  noWrap
                  sx={{
                    opacity: hovered ? 1 : 0,
                    maxWidth: hovered ? "200px" : "0px",
                    overflow: "hidden",
                    transition: "all 0.3s ease",
                  }}>
                  {tool.label}
                </Typography>
              </Stack>
            </ToggleButton>
          </Tooltip>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
};

const buildImageIds = (instances) => {
  return instances
    .map((item) => item?.ID)
    .filter(Boolean)
    .map((id) => `wadouri:${API_BASE_URL}/instances/${id}/file`);
};

const clampIndex = (index, length) => {
  if (length === 0) return 0;
  return Math.min(Math.max(index, 0), length - 1);
};

const CornerstoneViewport = ({ seriesUuid, currentIndex, setCurrentIndex }) => {
  const elementRef = useRef(null);
  const [imageIds, setImageIds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTool, setActiveTool] = useState("Wwwc");
  const [windowLevel, setWindowLevel] = useState({ width: 0, center: 0 });
  const [zoomPercent, setZoomPercent] = useState(100);

  useEffect(() => {
    ensureCornerstoneReady();
    ensureCornerstoneToolsReady();
  }, []);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    cornerstone.enable(element);

    return () => {
      try {
        cornerstone.disable(element);
      } catch (err) {
        console.debug("Failed to disable cornerstone element", err);
      }
    };
  }, []);
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const updateInfo = () => {
      const viewport = cornerstone.getViewport(element);
      if (!viewport) return;
      setWindowLevel({
        width: Math.round(viewport.voi.windowWidth),
        center: Math.round(viewport.voi.windowCenter),
      });
      setZoomPercent(Math.round(viewport.scale * 100));
    };

    element.addEventListener("cornerstoneimagerendered", updateInfo);
    return () =>
      element.removeEventListener("cornerstoneimagerendered", updateInfo);
  }, []);
  useEffect(() => {
    if (!seriesUuid) {
      setImageIds([]);
      setCurrentIndex(0);
      setError("No DICOM series identifier available for this selection.");
      return;
    }

    let isActive = true;
    const controller = new AbortController();

    const loadInstances = async () => {
      setLoading(true);
      setError(null);
      setCurrentIndex(0);

      try {
        console.log("seriesUuid", seriesUuid);
        const response = await fetch(
          `${API_BASE_URL}/series/${seriesUuid}/instances?expanded`,
          { signal: controller.signal }
        );
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error("Series not found");
          }
          throw new Error(
            response.statusText || "Failed to load series instances"
          );
        }
        const data = await response.json();
        if (!isActive) return;
        const instances = Array.isArray(data) ? data : [];
        const sorted = sortInstances(instances);
        const ids = buildImageIds(sorted);
        setImageIds(ids);
        setLoading(false);
        if (ids.length === 0) {
          setError("This series does not contain any DICOM slices.");
        }
      } catch (err) {
        if (!isActive || controller.signal.aborted) return;
        setLoading(false);
        setImageIds([]);
        setError(
          err instanceof Error
            ? err.message
            : "Unable to retrieve series images"
        );
      }
    };

    loadInstances();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, [seriesUuid]);

  useEffect(() => {
    setCurrentIndex((prev) => clampIndex(prev, imageIds.length));
  }, [imageIds]);

  useEffect(() => {
    const element = elementRef.current;
    if (!element || imageIds.length === 0) return;

    let cancelled = false;
    const imageId = imageIds[currentIndex];

    cornerstone
      .loadAndCacheImage(imageId)
      .then((image) => {
        if (cancelled) return;
        cornerstone.displayImage(element, image);
        cornerstone.fitToWindow(element);

        // Read initial W/L and zoom
        const viewport = cornerstone.getViewport(element);
        setWindowLevel({
          width: Math.round(viewport.voi.windowWidth),
          center: Math.round(viewport.voi.windowCenter),
        });
        setZoomPercent(Math.round(viewport.scale * 100));
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Image display failed");
      });

    return () => {
      cancelled = true;
    };
  }, [imageIds, currentIndex]);
  useEffect(() => {
    const element = elementRef.current;
    if (!element || imageIds.length === 0) return;

    ["Wwwc", "Zoom", "Pan"].forEach((tool) => {
      try {
        cornerstoneTools.addToolForElement(element, tool);
      } catch (error) {
        // console.debug(`Unable to add ${tool} tool`, error);
      }
    });

    ["Wwwc", "Zoom", "Pan"].forEach((tool) => {
      try {
        cornerstoneTools.setToolPassiveForElement(element, tool);
      } catch (error) {
        console.debug(`Unable to set ${tool} passive`, error);
      }
    });

    try {
      cornerstoneTools.setToolActiveForElement(element, activeTool, {
        mouseButtonMask: 1,
      });
      console.debug("Tool activated", activeTool);
    } catch (error) {
      console.debug("Failed to activate tool", activeTool, error);
    }
  }, [activeTool, imageIds.length]);

  useEffect(() => {
    const element = elementRef.current;
    if (!element || imageIds.length === 0) return;

    const handleWheel = (event) => {
      event.preventDefault();
      const direction = event.deltaY > 0 ? 1 : -1;
      setCurrentIndex((prev) => clampIndex(prev + direction, imageIds.length));
    };

    element.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      element.removeEventListener("wheel", handleWheel);
    };
  }, [imageIds.length]);

  const navigationDisabled = imageIds.length <= 1;

  const resetViewports = () => {
    try {
      cornerstone.reset(elementRef.current);
    } catch (error) {
      console.debug("Failed to reset plane", error);
    }
  };
  return (
    <>
      <Stack
        direction='row'
        sx={{
          position: "relative",
          borderRadius: 2,
          border: "1px solid",
          borderColor: "divider",
          overflow: "hidden",
          bgcolor: "black",
          height: "100%",
          p: 1,
          // minHeight: 360,
        }}>
        {/* <Stack
        direction='row'
        spacing={2}
        alignItems='flex-start'
        justifyContent='space-between'
        sx={{ p: 1, position: "absolute", width: "100%" }}> */}
        <ToolButtons activeTool={activeTool} onChange={setActiveTool} />
        {/* </Stack> */}
        <Box
          ref={elementRef}
          sx={{
            width: "100%",
            height: "100%",
            //   height: 360,
          }}
        />

        {loading && (
          <Stack
            alignItems='center'
            justifyContent='center'
            sx={{
              position: "absolute",
              inset: 0,
              bgcolor: "rgba(0,0,0,0.6)",
              color: "common.white",
            }}
            spacing={1}>
            <CircularProgress size={32} sx={{ color: "common.white" }} />
            <Typography variant='body2'>Loading DICOM images�</Typography>
          </Stack>
        )}

        {!loading && error && (
          <Stack
            alignItems='center'
            justifyContent='center'
            sx={{
              position: "absolute",
              inset: 0,
              bgcolor: "rgba(0,0,0,0.8)",
              color: "common.white",
              textAlign: "center",
              p: 2,
            }}
            spacing={1}>
            <Typography variant='subtitle2'>Viewer unavailable</Typography>
            <Typography variant='body2'>{error}</Typography>
          </Stack>
        )}

        {/* {!loading && !error && imageIds.length > 0 && (
          <Stack
            direction='row'
            spacing={1}
            alignItems='center'
            sx={{
              position: "absolute",
              bottom: 8,
              left: "50%",
              transform: "translateX(-50%)",
              bgcolor: "rgba(0,0,0,0.6)",
              borderRadius: 20,
              px: 1,
              py: 0.5,
            }}>
            <Tooltip title='Previous slice'>
              <span>
                <IconButton
                  size='small'
                  onClick={() =>
                    setCurrentIndex((prev) =>
                      clampIndex(prev - 1, imageIds.length)
                    )
                  }
                  disabled={navigationDisabled}>
                  <ChevronLeftIcon
                    sx={{ color: "common.white" }}
                    fontSize='small'
                  />
                </IconButton>
              </span>
            </Tooltip>
            <Typography
              variant='caption'
              sx={{ color: "common.white", minWidth: 80, textAlign: "center" }}>
              {currentIndex + 1} / {imageIds.length}
            </Typography>
            <Tooltip title='Next slice'>
              <span>
                <IconButton
                  size='small'
                  onClick={() =>
                    setCurrentIndex((prev) =>
                      clampIndex(prev + 1, imageIds.length)
                    )
                  }
                  disabled={navigationDisabled}>
                  <ChevronRightIcon
                    sx={{ color: "common.white" }}
                    fontSize='small'
                  />
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        )} */}
        <ResetButton onClick={resetViewports} disabled={navigationDisabled} />
        {/* <Tooltip title='Reset all viewports'>
          <IconButton
            size='small'
            onClick={resetViewports}
            disabled={navigationDisabled}
            sx={{
              position: "absolute",
              top: (theme) => theme.spacing(1),
              right: (theme) => theme.spacing(1),
            }}>
            <RestartAltRoundedIcon
              sx={{
                ":hover": { transform: "rotate(360deg)" },
                transition: "all 0.5s ease",
              }}
            />
          </IconButton>
        </Tooltip> */}
      </Stack>
      {!loading && !error && imageIds.length > 0 && (
        <Grid
          container
          spacing={1}
          sx={{
            "--Grid-borderWidth": "1px",
            borderColor: "divider",
            "& > div": {
              borderRight: "var(--Grid-borderWidth) solid",
              borderColor: "divider",
            },
            "div:last-child": {
              borderRight: "none",
            },
            mt: 1,
            px: 1,
          }}>
          <Grid
            item
            size={{ xs: 12, sm: 2 }}
            display='flex'
            justifyContent='center'
            alignItems='center'>
            <Typography variant='caption' color='text.secondary'>
              W/L:&nbsp;
            </Typography>
            <Typography variant='caption'>
              {windowLevel.width} / {windowLevel.center}
            </Typography>
          </Grid>
          <Grid
            item
            size={{ xs: 12, sm: 2 }}
            display='flex'
            justifyContent='center'
            alignItems='center'>
            <Typography variant='caption' color='text.secondary'>
              Zoom:&nbsp;
            </Typography>
            <Typography variant='caption'>{zoomPercent}%</Typography>
          </Grid>
          <Grid
            item
            size={{ xs: 12, sm: 2 }}
            display='flex'
            justifyContent='center'
            alignItems='center'>
            <Typography variant='caption' color='text.secondary'>
              Slice:&nbsp;
            </Typography>
            <Typography variant='caption'>
              {currentIndex + 1} of {imageIds.length}
            </Typography>
          </Grid>
          <Grid
            item
            size={{ xs: 12, sm: 6 }}
            display='flex'
            justifyContent='center'
            alignItems='center'>
            <Typography variant='caption' color='text.secondary'>
              Use the mouse wheel to navigate slices.
            </Typography>
          </Grid>
        </Grid>
      )}
    </>
  );
};

export default CornerstoneViewport;
