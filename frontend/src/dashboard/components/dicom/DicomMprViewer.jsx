import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  CircularProgress,
  Grid,
  IconButton,
  Slider,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from "@mui/material";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import LayersRoundedIcon from "@mui/icons-material/LayersRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import SettingsBrightnessRoundedIcon from "@mui/icons-material/SettingsBrightnessRounded";
import ZoomInMapRoundedIcon from "@mui/icons-material/ZoomInMapRounded";
import PanToolAltRoundedIcon from "@mui/icons-material/PanToolAltRounded";

import {
  API_BASE_URL,
  ensureCornerstoneReady,
  ensureCornerstoneToolsReady,
  cornerstoneApi,
} from "./cornerstoneConfig";

const { cornerstone, cornerstoneTools } = cornerstoneApi;

const planeDefinitions = [
  { value: "axial", label: "Axial" },
  //   { value: "coronal", label: "Coronal" },
  //   { value: "sagittal", label: "Sagittal" },
];

const toolDefinitions = [
  {
    value: "Wwwc",
    label: "W/L",
    icon: <SettingsBrightnessRoundedIcon fontSize='small' />,
  },
  {
    value: "Zoom",
    label: "Zoom",
    icon: <ZoomInMapRoundedIcon fontSize='small' />,
  },
  {
    value: "Pan",
    label: "Pan",
    icon: <PanToolAltRoundedIcon fontSize='small' />,
  },
];

const volumeStore = new Map();
let mprLoaderRegistered = false;

const clamp = (value, min, max) => {
  if (Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
};

const parseSpacing = (value) => {
  if (!value) return null;
  if (Array.isArray(value)) return value.map((item) => Number(item) || 0);
  if (typeof value === "string") {
    return value
      .split(/\\|,/)
      .map((item) => Number(item.trim()))
      .filter((item) => !Number.isNaN(item));
  }
  return null;
};

const parsePosition = (value) => {
  const parts = parseSpacing(value);
  if (!parts || parts.length < 3) return null;
  return { x: parts[0], y: parts[1], z: parts[2] };
};

const computeSliceSpacing = (instances) => {
  if (!Array.isArray(instances) || instances.length < 2) {
    const fallback = parseSpacing(
      instances?.[0]?.MainDicomTags?.SpacingBetweenSlices
    )?.[0];
    return (
      fallback || Number(instances?.[0]?.MainDicomTags?.SliceThickness) || 1
    );
  }

  const positions = instances
    .map((item) => parsePosition(item?.MainDicomTags?.ImagePositionPatient))
    .filter(Boolean);

  if (positions.length >= 2) {
    const first = positions[0];
    const last = positions[positions.length - 1];
    const distance = Math.sqrt(
      (last.x - first.x) ** 2 +
        (last.y - first.y) ** 2 +
        (last.z - first.z) ** 2
    );
    const slices = positions.length - 1;
    if (slices > 0) {
      const spacing = Math.abs(distance / slices);
      if (spacing > 0) {
        return spacing;
      }
    }
  }

  const fallback =
    parseSpacing(instances[0]?.MainDicomTags?.SpacingBetweenSlices)?.[0] ??
    Number(instances[0]?.MainDicomTags?.SliceThickness) ??
    1;
  return fallback > 0 ? fallback : 1;
};

const sortInstances = (instances) => {
  if (!Array.isArray(instances)) return [];
  const clone = instances.slice();
  clone.sort((a, b) => {
    const posA = parsePosition(a?.MainDicomTags?.ImagePositionPatient);
    const posB = parsePosition(b?.MainDicomTags?.ImagePositionPatient);
    if (posA && posB) {
      if (posA.z !== posB.z) return posA.z - posB.z;
      if (posA.y !== posB.y) return posA.y - posB.y;
      if (posA.x !== posB.x) return posA.x - posB.x;
    }
    const aNumber = Number(
      a?.MainDicomTags?.InstanceNumber ?? a?.IndexInSeries ?? a?.ID ?? 0
    );
    const bNumber = Number(
      b?.MainDicomTags?.InstanceNumber ?? b?.IndexInSeries ?? b?.ID ?? 0
    );
    return aNumber - bNumber;
  });
  return clone;
};

const buildImageObject = ({
  pixels,
  width,
  height,
  spacing,
  stats,
  meta,
  imageId,
}) => ({
  imageId,
  minPixelValue: stats.min,
  maxPixelValue: stats.max,
  slope: meta.slope,
  intercept: meta.intercept,
  windowCenter: meta.windowCenter,
  windowWidth: meta.windowWidth,
  rows: height,
  columns: width,
  height,
  width,
  color: false,
  columnPixelSpacing: spacing.column,
  rowPixelSpacing: spacing.row,
  sizeInBytes: pixels.byteLength,
  bitsAllocated: meta.bitsAllocated,
  bitsStored: meta.bitsStored,
  highBit: meta.highBit,
  pixelRepresentation: meta.pixelRepresentation,
  invert: false,
  getPixelData: () => pixels,
  render: cornerstone.renderGrayscaleImage,
});

const registerMprImageLoader = () => {
  if (mprLoaderRegistered || typeof window === "undefined") {
    return;
  }

  cornerstone.registerImageLoader("mpr", (imageId) => {
    const [, seriesUuid, plane, indexPart] = imageId.split(":");
    const targetIndex = Number(indexPart);
    const cached = volumeStore.get(seriesUuid);
    if (!cached) {
      return Promise.reject(new Error("Series volume not cached."));
    }

    const { volume } = cached;
    const {
      rows,
      columns,
      slices,
      data,
      typedArrayConstructor,
      pixelSpacing,
      meta,
    } = volume;

    const indices = {
      axial: slices - 1,
      //   coronal: rows - 1,
      //   sagittal: columns - 1,
    };

    const clampedIndex = clamp(targetIndex, 0, indices[plane] ?? 0);
    const cacheMap = volume.cache[plane];
    if (cacheMap?.has(clampedIndex)) {
      return Promise.resolve(cacheMap.get(clampedIndex));
    }

    const TypedArray = typedArrayConstructor;
    let width = columns;
    let height = rows;
    let rowSpacing = pixelSpacing.row;
    let columnSpacing = pixelSpacing.column;
    let pixels = null;

    // if (plane === "coronal") {
    //   width = columns;
    //   height = slices;
    //   rowSpacing = pixelSpacing.slice;
    //   columnSpacing = pixelSpacing.column;
    //   pixels = new TypedArray(width * height);
    //   let offset = 0;
    //   for (let slice = 0; slice < slices; slice += 1) {
    //     const base = slice * rows * columns + clampedIndex * columns;
    //     for (let col = 0; col < columns; col += 1) {
    //       const value = data[base + col];
    //       pixels[offset] = value;
    //       offset += 1;
    //     }
    //   }
    // } else if (plane === "sagittal") {
    //   width = rows;
    //   height = slices;
    //   rowSpacing = pixelSpacing.slice;
    //   columnSpacing = pixelSpacing.row;
    //   pixels = new TypedArray(width * height);
    //   let offset = 0;
    //   for (let slice = 0; slice < slices; slice += 1) {
    //     const base = slice * rows * columns;
    //     for (let row = 0; row < rows; row += 1) {
    //       const value = data[base + row * columns + clampedIndex];
    //       pixels[offset] = value;
    //       offset += 1;
    //     }
    //   }
    // } else {
    //   return Promise.reject(new Error(`Unsupported plane: ${plane}`));
    // }

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < pixels.length; i += 1) {
      const value = pixels[i];
      if (value < min) min = value;
      if (value > max) max = value;
    }

    const image = buildImageObject({
      pixels,
      width,
      height,
      spacing: { row: rowSpacing, column: columnSpacing },
      stats: { min, max },
      meta,
      imageId,
    });

    cacheMap.set(clampedIndex, image);
    return Promise.resolve(image);
  });

  mprLoaderRegistered = true;
};

const fetchJson = async (url, { signal } = {}) => {
  const response = await fetch(url, {
    credentials: "include",
    signal,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText || "Request failed");
  }
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
};

const fetchInstanceTags = async (instanceId, signal) => {
  if (!instanceId) return [];
  try {
    const tags = await fetchJson(
      `${API_BASE_URL}/instances/${instanceId}/tags?simplify`,
      { signal }
    );
    if (!tags || typeof tags !== "object") return [];
    return Object.entries(tags)
      .map(([name, value]) => ({
        name,
        value: Array.isArray(value) ? value.join(", ") : String(value ?? ""),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  } catch (error) {
    console.debug("Failed to fetch instance tags", error);
    return [];
  }
};

const loadSeriesVolume = async (seriesUuid, signal) => {
  const instancesRaw = await fetchJson(
    `${API_BASE_URL}/series/${seriesUuid}/instances?expanded`,
    { signal }
  );

  if (!Array.isArray(instancesRaw) || instancesRaw.length === 0) {
    throw new Error("Series has no instances available.");
  }

  const instances = sortInstances(instancesRaw);
  const imageIds = instances
    .map((item) => item?.ID)
    .filter(Boolean)
    .map((id) => `wadouri:${API_BASE_URL}/instances/${id}/file`);

  ensureCornerstoneReady();

  const images = [];
  for (const imageId of imageIds) {
    const image = await cornerstone.loadAndCacheImage(imageId);
    images.push(image);
  }

  if (images.length === 0) {
    throw new Error("Failed to load DICOM images for this series.");
  }

  const firstImage = images[0];
  const rows = firstImage.rows;
  const columns = firstImage.columns;
  const slices = images.length;
  const samplePixels = firstImage.getPixelData();
  const TypedArray = samplePixels.constructor;
  const sliceSize = rows * columns;
  const voxels = new TypedArray(sliceSize * slices);

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  images.forEach((image, sliceIndex) => {
    const pixels = image.getPixelData();
    const offset = sliceIndex * sliceSize;
    voxels.set(pixels, offset);
    for (let i = 0; i < pixels.length; i += 1) {
      const value = pixels[i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
  });

  const pixelSpacingRaw = parseSpacing(
    instances[0]?.MainDicomTags?.PixelSpacing
  ) ?? [firstImage.rowPixelSpacing ?? 1, firstImage.columnPixelSpacing ?? 1];
  const sliceSpacing = computeSliceSpacing(instances);

  const slope = firstImage.slope ?? 1;
  const intercept = firstImage.intercept ?? 0;
  const windowCenterRaw = firstImage.windowCenter;
  const windowWidthRaw = firstImage.windowWidth;
  const windowCenter = Array.isArray(windowCenterRaw)
    ? windowCenterRaw[0]
    : windowCenterRaw ?? (min + max) / 2;
  const windowWidth = Array.isArray(windowWidthRaw)
    ? windowWidthRaw[0]
    : windowWidthRaw ?? Math.max(1, max - min);

  const meta = {
    slope,
    intercept,
    windowCenter,
    windowWidth,
    bitsAllocated: firstImage.bitsAllocated,
    bitsStored: firstImage.bitsStored,
    highBit: firstImage.highBit,
    pixelRepresentation: firstImage.pixelRepresentation,
  };

  const firstInstance = instances[0] ?? {};

  const tags = await fetchInstanceTags(firstInstance.ID, signal);

  return {
    volume: {
      seriesUuid,
      imageIds,
      rows,
      columns,
      slices,
      data: voxels,
      typedArrayConstructor: TypedArray,
      pixelSpacing: {
        row: pixelSpacingRaw?.[0] ?? 1,
        column: pixelSpacingRaw?.[1] ?? 1,
        slice: sliceSpacing ?? 1,
      },
      //   cache: {
      //     coronal: new Map(),
      //     sagittal: new Map(),
      //   },
      meta,
      min,
      max,
    },
    metadata: {
      patient: firstInstance.PatientMainDicomTags ?? {},
      study: firstInstance.StudyMainDicomTags ?? {},
      series: firstInstance.MainDicomTags ?? {},
      seriesUuid,
    },
    tags,
  };
};

const ToolButtons = ({ activeTool, onChange }) => (
  <ToggleButtonGroup
    size='small'
    value={activeTool}
    exclusive
    onChange={(event, value) => {
      if (value) onChange(value);
    }}>
    {toolDefinitions.map((tool) => (
      <ToggleButton key={tool.value} value={tool.value}>
        <Stack direction='row' spacing={1} alignItems='center'>
          {tool.icon}
          <Typography variant='caption'>{tool.label}</Typography>
        </Stack>
      </ToggleButton>
    ))}
  </ToggleButtonGroup>
);

const PlaneToggleButtons = ({ visiblePlanes, onChange }) => (
  <ToggleButtonGroup
    size='small'
    value={visiblePlanes}
    onChange={(event, next) => {
      if (Array.isArray(next) && next.length > 0) {
        onChange(next);
      }
    }}>
    {planeDefinitions.map((plane) => (
      <ToggleButton key={plane.value} value={plane.value}>
        <Stack direction='row' spacing={1} alignItems='center'>
          <LayersRoundedIcon fontSize='small' />
          <Typography variant='caption'>{plane.label}</Typography>
        </Stack>
      </ToggleButton>
    ))}
  </ToggleButtonGroup>
);

const PlaneViewport = forwardRef(
  ({ plane, volume, index, onIndexChange, activeTool }, ref) => {
    const elementRef = useRef(null);
    const latestImageRef = useRef(null);
    const initializedRef = useRef(false);

    const limits = useMemo(() => {
      if (!volume) return { min: 0, max: 0 };
      if (plane === "axial") {
        return { min: 0, max: Math.max(0, volume.slices - 1) };
      }
      //   if (plane === "coronal") {
      //     return { min: 0, max: Math.max(0, volume.rows - 1) };
      //   }
      //   return { min: 0, max: Math.max(0, volume.columns - 1) };
    }, [volume, plane]);

    useImperativeHandle(
      ref,
      () => ({
        reset: () => {
          const element = elementRef.current;
          if (!element) return;
          try {
            cornerstone.reset(element);
          } catch (error) {
            console.debug("Failed to reset viewport", error);
          }
        },
      }),
      []
    );

    useEffect(() => {
      ensureCornerstoneReady();
      ensureCornerstoneToolsReady();
      registerMprImageLoader();
    }, []);

    useEffect(() => {
      const element = elementRef.current;
      if (!element || !volume) return;

      if (!initializedRef.current) {
        initializedRef.current = true;
        cornerstone.enable(element);
        ["Wwwc", "Zoom", "Pan"].forEach((tool) => {
          try {
            cornerstoneTools.addToolForElement(element, tool);
          } catch (error) {
            // ignore duplicate registrations
          }
        });
      }

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
      } catch (error) {
        console.debug("Failed to activate tool", activeTool, error);
      }
    }, [activeTool, volume]);

    useEffect(() => {
      const element = elementRef.current;
      if (!element || !volume) return;

      const targetIndex = clamp(index, limits.min, limits.max);
      let requestAborted = false;

      const loadImage = async () => {
        try {
          let imagePromise = null;
          if (plane === "axial") {
            const imageId = volume.imageIds[targetIndex];
            if (!imageId) {
              throw new Error("Missing image identifier.");
            }
            imagePromise = cornerstone.loadAndCacheImage(imageId);
          } else {
            const imageId = `mpr:${volume.seriesUuid}:${plane}:${targetIndex}`;
            imagePromise = cornerstone.loadImage(imageId);
          }
          const image = await imagePromise;
          if (requestAborted) return;
          latestImageRef.current = image;
          const viewport = cornerstone.getViewport(element);
          cornerstone.displayImage(element, image);
          if (viewport) {
            cornerstone.setViewport(element, { ...viewport });
          }
        } catch (error) {
          if (!requestAborted) {
            console.debug("Failed to display image", error);
          }
        }
      };

      loadImage();

      return () => {
        requestAborted = true;
      };
    }, [volume, plane, index, limits.min, limits.max]);

    useEffect(() => {
      const element = elementRef.current;
      if (!element || !volume) return;

      const handleWheel = (event) => {
        event.preventDefault();
        const delta = event.deltaY > 0 ? 1 : -1;
        const nextIndex = clamp(index + delta, limits.min, limits.max);
        if (nextIndex !== index) {
          onIndexChange(nextIndex);
        }
      };

      element.addEventListener("wheel", handleWheel, { passive: false });
      return () => {
        element.removeEventListener("wheel", handleWheel);
      };
    }, [index, limits.min, limits.max, onIndexChange, volume]);

    useEffect(() => {
      return () => {
        const element = elementRef.current;
        if (!element) return;
        try {
          cornerstone.disable(element);
        } catch (error) {
          console.debug("Failed to disable viewport", error);
        }
      };
    }, []);

    return (
      <Box
        sx={{
          position: "relative",
          borderRadius: 2,
          border: "1px solid",
          borderColor: "divider",
          overflow: "hidden",
          bgcolor: "black",
          height: "70vh",
        }}>
        <Box
          ref={elementRef}
          sx={{
            width: "100%",
            height: "100%",
            cursor:
              activeTool === "Pan"
                ? "grab"
                : activeTool === "Zoom"
                ? "zoom-in"
                : "crosshair",
          }}
        />
        <Box
          sx={{
            position: "absolute",
            bottom: 8,
            left: "50%",
            transform: "translateX(-50%)",
            bgcolor: "rgba(0,0,0,0.6)",
            borderRadius: 12,
            px: 1.5,
            py: 0.5,
          }}>
          <Typography variant='caption' sx={{ color: "common.white" }}>
            {plane.toUpperCase()} {index + 1} / {limits.max + 1}
          </Typography>
        </Box>
      </Box>
    );
  }
);

const MetadataSection = ({ metadata }) => {
  if (!metadata) return null;

  const patientItems = Object.entries(metadata.patient ?? {}).map(
    ([key, value]) => ({
      label: key.replace(/([A-Z])/g, " $1"),
      value: Array.isArray(value) ? value.join(", ") : String(value ?? ""),
    })
  );
  const studyItems = Object.entries(metadata.study ?? {}).map(
    ([key, value]) => ({
      label: key.replace(/([A-Z])/g, " $1"),
      value: Array.isArray(value) ? value.join(", ") : String(value ?? ""),
    })
  );
  const seriesItems = Object.entries(metadata.series ?? {}).map(
    ([key, value]) => ({
      label: key.replace(/([A-Z])/g, " $1"),
      value: Array.isArray(value) ? value.join(", ") : String(value ?? ""),
    })
  );

  const renderItems = (items) => (
    <Stack spacing={0.5}>
      {items.map((item) => (
        <Stack
          key={item.label}
          direction='row'
          justifyContent='space-between'
          spacing={2}>
          <Typography variant='caption' color='text.secondary'>
            {item.label}
          </Typography>
          <Typography variant='caption'>{item.value || "N/A"}</Typography>
        </Stack>
      ))}
    </Stack>
  );

  return (
    <Stack spacing={2}>
      <Box>
        <Typography variant='subtitle2' gutterBottom>
          Patient
        </Typography>
        {renderItems(patientItems)}
      </Box>
      <Box>
        <Typography variant='subtitle2' gutterBottom>
          Study
        </Typography>
        {renderItems(studyItems)}
      </Box>
      <Box>
        <Typography variant='subtitle2' gutterBottom>
          Series
        </Typography>
        {renderItems(seriesItems)}
      </Box>
    </Stack>
  );
};

const TagsAccordion = ({ tags }) => {
  if (!Array.isArray(tags) || tags.length === 0) {
    return (
      <Typography variant='body2' color='text.secondary'>
        No DICOM tags available for this instance.
      </Typography>
    );
  }

  return (
    <Accordion disableGutters elevation={0}>
      <AccordionSummary expandIcon={<ExpandMoreRoundedIcon />}>
        <Typography variant='subtitle2'>DICOM Tags</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={0.5} sx={{ maxHeight: 240, overflow: "auto" }}>
          {tags.map((item) => (
            <Box
              key={item.name}
              sx={{
                display: "grid",
                gridTemplateColumns: "180px 1fr",
                gap: 1,
                pb: 0.5,
                borderBottom: "1px solid",
                borderColor: "divider",
              }}>
              <Typography variant='caption' color='text.secondary'>
                {item.name}
              </Typography>
              <Typography variant='caption'>{item.value || "N/A"}</Typography>
            </Box>
          ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
};

const DicomMprViewer = ({ seriesUuid }) => {
  const [volume, setVolume] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [tags, setTags] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [visiblePlanes, setVisiblePlanes] = useState([
    "axial",
    // "coronal",
    // "sagittal",
  ]);
  const [indices, setIndices] = useState({
    axial: 0,
    // coronal: 0,
    // sagittal: 0,
  });
  const [activeTool, setActiveTool] = useState("Wwwc");
  const [reloadId, setReloadId] = useState(0);

  const axialRef = useRef(null);
  //   const coronalRef = useRef(null);
  //   const sagittalRef = useRef(null);

  useEffect(() => {
    ensureCornerstoneReady();
    ensureCornerstoneToolsReady();
    registerMprImageLoader();
  }, []);

  useEffect(() => {
    if (!seriesUuid) {
      setVolume(null);
      setMetadata(null);
      setTags([]);
      setError(null);
      return;
    }

    let ignore = false;
    const controller = new AbortController();

    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        let cached = volumeStore.get(seriesUuid);
        if (!cached) {
          const loaded = await loadSeriesVolume(seriesUuid, controller.signal);
          cached = loaded;
          volumeStore.set(seriesUuid, loaded);
        }
        if (ignore) return;
        setVolume(cached.volume);
        setMetadata(cached.metadata);
        setTags(cached.tags);
        setIndices((prev) => ({
          axial: clamp(prev.axial, 0, Math.max(0, cached.volume.slices - 1)),
          //   coronal: clamp(prev.coronal, 0, Math.max(0, cached.volume.rows - 1)),
          //   sagittal: clamp(
          //     prev.sagittal,
          //     0,
          //     Math.max(0, cached.volume.columns - 1)
          //   ),
        }));
      } catch (err) {
        if (ignore || controller.signal.aborted) return;
        console.error("Failed to load series volume", err);
        setError(err.message ?? "Failed to load selected series.");
        setVolume(null);
        setMetadata(null);
        setTags([]);
      } finally {
        if (!ignore) setLoading(false);
      }
    };

    run();

    return () => {
      ignore = true;
      controller.abort();
    };
  }, [seriesUuid, reloadId]);

  const resetViewports = () => {
    [
      axialRef,
      // coronalRef, sagittalRef
    ].forEach((ref) => {
      console.debug("elementRef.current", ref.current);
      try {
        ref.current?.reset?.();
      } catch (error) {
        console.debug("Failed to reset plane", error);
      }
    });
  };

  const renderViewport = (plane, ref) => {
    const limit =
      plane === "axial"
        ? volume?.slices ?? 0
        : plane === "coronal"
        ? volume?.rows ?? 0
        : volume?.columns ?? 0;

    return (
      <Stack key={plane} spacing={1}>
        <PlaneViewport
          ref={ref}
          plane={plane}
          volume={volume}
          index={indices[plane]}
          activeTool={activeTool}
          onIndexChange={(next) =>
            setIndices((prev) => ({ ...prev, [plane]: next }))
          }
        />
        <Slider
          size='small'
          value={indices[plane]}
          min={0}
          max={Math.max(0, limit - 1)}
          step={1}
          valueLabelDisplay='auto'
          onChange={(event, value) => {
            const next = Array.isArray(value) ? value[0] : value;
            setIndices((prev) => ({
              ...prev,
              [plane]: clamp(next, 0, Math.max(0, limit - 1)),
            }));
          }}
        />
      </Stack>
    );
  };

  const visibleViewports = useMemo(
    () =>
      planeDefinitions
        .filter((plane) => visiblePlanes.includes(plane.value))
        .map((plane) =>
          renderViewport(
            plane.value,
            plane.value === "axial"
              ? axialRef
              : plane.value === "coronal"
              ? coronalRef
              : sagittalRef
          )
        ),
    [visiblePlanes, volume, indices, activeTool]
  );

  if (!seriesUuid) {
    return (
      <Stack
        spacing={2}
        sx={{
          alignItems: "center",
          justifyContent: "center",
          height: "70vh",
          border: "1px dashed",
          borderColor: "divider",
          borderRadius: 2,
          p: 4,
        }}>
        <Typography variant='subtitle1'>
          Select a series from the table to load the DICOM viewer.
        </Typography>
        <Typography variant='body2' color='text.secondary'>
          Scroll slices with the mouse wheel and adjust window/level, zoom, or
          pan using the controls once a series is loaded.
        </Typography>
      </Stack>
    );
  }

  return (
    <Stack spacing={3}>
      <Stack
        direction='row'
        spacing={2}
        alignItems='center'
        justifyContent='space-between'>
        <Stack direction='row' spacing={2} alignItems='center'>
          <PlaneToggleButtons
            visiblePlanes={visiblePlanes}
            onChange={setVisiblePlanes}
          />
          <ToolButtons activeTool={activeTool} onChange={setActiveTool} />
        </Stack>
        <Stack direction='row' spacing={1}>
          <Tooltip title='Reset all viewports'>
            <IconButton size='small' onClick={resetViewports}>
              <RestartAltRoundedIcon />
            </IconButton>
          </Tooltip>
          <Button
            size='small'
            variant='outlined'
            onClick={() => {
              if (seriesUuid) {
                volumeStore.delete(seriesUuid);
              }
              setReloadId((value) => value + 1);
            }}>
            Reload
          </Button>
        </Stack>
      </Stack>

      {error && <Alert severity='error'>{error}</Alert>}

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 8 }}>
          <Stack spacing={2}>
            {loading && (
              <Stack
                alignItems='center'
                justifyContent='center'
                sx={{
                  height: "70vh",
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: 2,
                }}
                spacing={2}>
                <CircularProgress />
                <Typography variant='body2'>
                  Loading DICOM volume for viewing...
                </Typography>
              </Stack>
            )}
            {!loading && volume && visibleViewports}
          </Stack>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Stack spacing={2}>
            <Box
              sx={{
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 2,
                p: 2,
              }}>
              <Typography variant='subtitle1' gutterBottom>
                Series Metadata
              </Typography>
              <MetadataSection metadata={metadata} />
            </Box>
            <Box
              sx={{
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 2,
                p: 2,
              }}>
              <TagsAccordion tags={tags} />
            </Box>
          </Stack>
        </Grid>
      </Grid>
    </Stack>
  );
};

export default DicomMprViewer;
