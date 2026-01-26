import * as React from "react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Plot from "react-plotly.js";
import { useTheme } from "@mui/material/styles";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";

const isTruthyArray = (arr) => !arr?.every((x) => x === null) || false;

const useImageAvailable = (url) => {
  const [available, setAvailable] = React.useState(false);

  React.useEffect(() => {
    if (!url) {
      setAvailable(false);
      return;
    }
    let isMounted = true;
    const img = new Image();
    img.onload = () => {
      if (isMounted) setAvailable(true);
    };
    img.onerror = () => {
      if (isMounted) setAvailable(false);
    };
    img.src = url;
    return () => {
      isMounted = false;
    };
  }, [url]);

  return available;
};

const chartContainerSx = {
  borderRadius: 2,
  border: "1px solid",
  borderColor: "divider",
  position: "relative",
  p: 1,
  height: 359,
  //   flex: 1,
};

const API_BASE_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "";
const buildApiUrl = (pathname) =>
  API_BASE_URL ? `${API_BASE_URL}${pathname}` : pathname;

const mergeDeep = (target, source) => {
  for (const key in source) {
    if (
      source[key] instanceof Object &&
      key in target &&
      target[key] instanceof Object
    ) {
      mergeDeep(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
};

const ChoPlots = ({ children, data, comparison, direction = "column" }) => {
  const seriesData = data?.series ?? data;
  const comparisonSeries = comparison?.series ?? comparison;
  const theme = useTheme();
  const resultsData = data?.results ?? data;
  const comparisonResults = comparison?.results ?? comparison;
  const hasComparison = Boolean(comparisonResults);
  const currentSuffix = hasComparison ? " (new)" : "";
  const storedSuffix = hasComparison ? " (stored)" : "";
  const [expandedPlot, setExpandedPlot] = React.useState(null);
  const seriesInstanceUid = seriesData?.series_instance_uid;
  const coronalImageUrl = seriesInstanceUid
    ? buildApiUrl(`/minio-images/${seriesInstanceUid}`)
    : null;
  const imageAvailable = useImageAvailable(coronalImageUrl);
  console.log("seriesData", seriesData);

  const handleOpenPlotModal = React.useCallback((plot, title) => {
    if (!plot) return;

    const clone = (value) =>
      value !== undefined ? JSON.parse(JSON.stringify(value)) : value;

    setExpandedPlot({
      title,
      plot: {
        data: Array.isArray(plot.data) ? clone(plot.data) : [],
        layout: plot.layout ? clone(plot.layout) : undefined,
      },
    });
  }, []);

  const handleClosePlotModal = React.useCallback(() => {
    setExpandedPlot(null);
  }, []);

  const lastLocation = React.useMemo(() => {
    const candidates = [
      seriesData?.location,
      resultsData?.location,
      resultsData?.location_sparse,
      comparisonSeries?.location,
      comparisonResults?.location,
      comparisonResults?.location_sparse,
    ];
    let max = 0;
    candidates.forEach((arr) => {
      if (Array.isArray(arr) && arr.length) {
        const candidate = Number(arr[arr.length - 1]);
        if (!Number.isNaN(candidate)) {
          max = Math.max(max, candidate);
        }
      }
    });
    return max;
  }, [comparisonResults, comparisonSeries, resultsData, seriesData]);
  const baseLayout = React.useMemo(
    () => ({
      title: {
        font: {
          size: 16,
          color: theme.palette.primary.contrastText,
          weight: 1000,
        },
      },
      xaxis: {
        title: {
          font: {
            color: theme.palette.primary.contrastText,
          },
        },
        tickfont: {
          color: theme.palette.primary.contrastText,
        },
        color: theme.palette.divider,
        gridcolor: theme.palette.divider,
      },
      yaxis: {
        title: {
          font: {
            color: theme.palette.primary.contrastText,
          },
        },
        tickfont: {
          color: theme.palette.primary.contrastText,
        },
        color: theme.palette.divider,
        gridcolor: theme.palette.divider,
        showgrid: false,
      },
      yaxis2: {
        title: {
          font: {
            color: theme.palette.primary.contrastText,
          },
        },
        tickfont: {
          color: theme.palette.primary.contrastText,
        },
        color: theme.palette.divider,
        gridcolor: theme.palette.divider,
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      plot_bgcolor: "rgba(0, 0, 0, 0)",
      paper_bgcolor: "rgba(0, 0, 0, 0)",
      margin: { l: 60, r: 40, t: 60, b: 60 },
      legend: {
        x: 0.0,
        y: 1.02,
        bgcolor: "rgba(255,255,255,0.8)",
        bordercolor: "#e2e8f0",
        borderwidth: 1,
        yanchor: "bottom",
        autosize: true,
      },
      showlegend: hasComparison,
      font: { family: "Arial, sans-serif" },
    }),
    [hasComparison, theme]
  );
  const buildLayout = React.useCallback(
    (options) =>
      mergeDeep(JSON.parse(JSON.stringify(baseLayout)), options || {}),
    [baseLayout]
  );

  const npsPlot = React.useMemo(() => {
    const spatial = resultsData?.spatial_frequency;
    const nps = resultsData?.nps;
    const comparisonSpatial = comparisonResults?.spatial_frequency;
    const comparisonNps = comparisonResults?.nps;

    if (
      (!isTruthyArray(spatial) || !isTruthyArray(nps)) &&
      (!isTruthyArray(comparisonSpatial) || !isTruthyArray(comparisonNps))
    ) {
      return null;
    }

    const traces = [];
    if (spatial && nps) {
      traces.push({
        name: `NPS${currentSuffix}`,
        x: spatial,
        y: nps,
        type: "scatter",
        mode: "lines",
        line: { width: 3, shape: "spline", color: "#00FFFF" },
      });
    }
    if (comparisonSpatial && comparisonNps) {
      traces.push({
        name: `NPS${storedSuffix}`,
        x: comparisonSpatial,
        y: comparisonNps,
        type: "scatter",
        mode: "lines",
        line: {
          width: 2,
          shape: "spline",
          color: "#94a3b8",
          dash: "dash",
        },
      });
    }

    if (traces.length === 0) return null;

    return {
      data: traces,
      layout: buildLayout({
        title: {
          text: "Noise Power Spectrum (NPS)",
        },
        xaxis: {
          title: {
            text: "Spatial Frequency (cm<sup>-1</sup>)",
          },
        },
        yaxis: {
          title: {
            text: "NPS",
          },
        },
        showlegend: traces.length > 1,
      }),
    };
  }, [
    buildLayout,
    comparisonResults,
    currentSuffix,
    storedSuffix,
    resultsData,
  ]);

  const ctdiPlot = React.useMemo(() => {
    const location = resultsData?.location || [];
    const ctdi = resultsData?.ctdivol || [];
    const ssde_inc = resultsData?.ssde_inc || [];
    // const comparisonLocation = comparisonResults?.location || [];
    // const comparisonCtdi = comparisonResults?.ctdivol || [];
    if (
      (!isTruthyArray(location) || !isTruthyArray(ctdi)) &&
      //   (!isTruthyArray(comparisonLocation) || !isTruthyArray(comparisonCtdi)) &&
      !isTruthyArray(ssde_inc)
    ) {
      return null;
    }

    const traces = [];
    if (location && ctdi) {
      traces.push({
        name: `CTDIvol${currentSuffix}`,
        x: location,
        y: ctdi,
        type: "scatter",
        mode: "lines",
        line: { width: 3, shape: "spline", color: "#FFA500" },
        marker: { size: 4 },
        legendgroup: "ctdi",
      });
      traces.push({
        name: `SSDE`,
        x: location,
        y: ssde_inc,
        type: "scatter",
        mode: "lines",
        line: { width: 3, shape: "spline", color: "#FFFF00" },
        marker: { size: 4 },
        legendgroup: "ssde",
      });
    }
    // if (comparisonLocation && comparisonCtdi) {
    //   traces.push({
    //     name: `CTDIvol${storedSuffix}`,
    //     x: comparisonLocation,
    //     y: comparisonCtdi,
    //     type: "scatter",
    //     mode: "lines",
    //     line: {
    //       width: 2,
    //       shape: "spline",
    //       color: "#94a3b8",
    //       dash: "dash",
    //     },
    //     marker: { size: 4 },
    //     legendgroup: "ctdi",
    //   });
    // }
    if (traces.length === 0) return null;

    const baseLayout = buildLayout({
      title: {
        text: "CTDIvol & SSDE",
      },
      xaxis: {
        title: { text: "Z Location (cm)" },
        range: [0, lastLocation || 0],
      },
      yaxis: {
        title: { text: "CTDIvol & SSDE (mGy)" },
      },
      yaxis2: {
        title: { text: "SSDE" },
      },
    });

    const layout =
      imageAvailable && coronalImageUrl
        ? {
            ...baseLayout,
            images: [
              {
                x: 0,
                y: 0,
                sizex: lastLocation || 1,
                sizey: 1,
                source: coronalImageUrl,
                xref: "x",
                yref: "paper",
                xanchor: "left",
                yanchor: "bottom",
                opacity: 1,
                layer: "below",
                sizing: "stretch",
              },
            ],
          }
        : baseLayout;

    return {
      data: traces,
      layout: {
        ...layout,
        showlegend: true,
      },
    };
  }, [
    buildLayout,
    // comparisonResults?.ctdivol,
    // comparisonResults?.location,
    coronalImageUrl,
    currentSuffix,
    imageAvailable,
    lastLocation,
    resultsData?.ctdivol,
    resultsData?.location,
    resultsData?.ssde_inc,
    // storedSuffix,
  ]);

  const dwPlot = React.useMemo(() => {
    const location = resultsData?.location;
    const dw = resultsData?.dw;
    const ssde_inc = resultsData?.ssde_inc;
    const comparisonLocation = comparisonResults?.location;
    const comparisonDw = comparisonResults?.dw;

    if (
      (!isTruthyArray(location) || !isTruthyArray(dw)) &&
      (!isTruthyArray(comparisonLocation) || !isTruthyArray(comparisonDw))
    ) {
      return null;
    }

    const traces = [];
    if (location && dw && ssde_inc) {
      traces.push({
        name: `Water Equivalent Diameter${currentSuffix}`,
        x: location,
        y: dw,
        type: "scatter",
        mode: "lines",
        line: { width: 3, shape: "spline", color: "#00FF00" },
        marker: { size: 4 },
        legendgroup: "dw",
      });
    }
    if (comparisonLocation && comparisonDw) {
      traces.push({
        name: `Water Equivalent Diameter${storedSuffix}`,
        x: comparisonLocation,
        y: comparisonDw,
        type: "scatter",
        mode: "lines",

        line: {
          width: 2,
          shape: "spline",
          color: "#f97373",
          dash: "dash",
        },
        marker: { size: 4 },
        legendgroup: "dw",
      });
    }

    if (traces.length === 0) return null;

    const baseLayout = buildLayout({
      title: {
        text: "Water Equivalent Diameter (Dw)",
      },
      xaxis: {
        title: { text: "Z Location (cm)" },
        range: [0, lastLocation || 0],
      },
      yaxis: {
        title: {
          text: "Dw (mm)",
        },
      },
    });

    const layout =
      imageAvailable && coronalImageUrl
        ? {
            ...baseLayout,
            images: [
              {
                x: 0,
                y: 0,
                sizex: lastLocation || 1,
                sizey: 1,
                source: coronalImageUrl,
                xref: "x",
                yref: "paper",
                xanchor: "left",
                yanchor: "bottom",
                opacity: 1,
                layer: "below",
                sizing: "stretch",
              },
            ],
          }
        : baseLayout;

    return {
      data: traces,
      layout: {
        ...layout,
        showlegend: traces.length > 2,
      },
    };
  }, [
    buildLayout,
    comparisonResults,
    coronalImageUrl,
    currentSuffix,
    imageAvailable,
    lastLocation,
    resultsData,
    storedSuffix,
  ]);

  const noisePlot = React.useMemo(() => {
    const location = resultsData?.location_sparse;
    const noise = resultsData?.noise_level;
    const comparisonLocation = comparisonResults?.location_sparse;
    const comparisonNoise = comparisonResults?.noise_level;

    if (
      (!isTruthyArray(location) || !isTruthyArray(noise)) &&
      (!isTruthyArray(comparisonLocation) || !isTruthyArray(comparisonNoise))
    ) {
      return null;
    }

    const traces = [];
    if (location && noise) {
      traces.push({
        name: `Local Noise Level${currentSuffix}`,
        x: location,
        y: noise,
        type: "scatter",
        mode: "lines+markers",
        line: {
          width: 3,
          shape: "spline",
          color: "#FF00FF",
          dash: "dot",
        },
        marker: { size: 8 },
        legendgroup: "noise",
      });
    }
    if (comparisonLocation && comparisonNoise) {
      traces.push({
        name: `Local Noise Level${storedSuffix}`,
        x: comparisonLocation,
        y: comparisonNoise,
        type: "scatter",
        mode: "lines+markers",
        line: {
          width: 2,
          shape: "spline",
          color: "#94a3b8",
          dash: "dot",
        },
        marker: { size: 8, symbol: "circle-open" },
        legendgroup: "noise",
      });
    }

    if (traces.length === 0) return null;

    const baseLayout = buildLayout({
      title: {
        text: "Noise Level ",
      },
      xaxis: {
        title: { text: "Z Location (cm)" },
        range: [0, lastLocation || 0],
      },
      yaxis: {
        title: { text: "Noise Level (HU)" },
      },
    });

    const layout =
      imageAvailable && coronalImageUrl
        ? {
            ...baseLayout,
            images: [
              {
                x: 0,
                y: 0,
                sizex: lastLocation || 1,
                sizey: 1,
                source: coronalImageUrl,
                xref: "x",
                yref: "paper",
                xanchor: "left",
                yanchor: "bottom",
                opacity: 1,
                layer: "below",
                sizing: "stretch",
              },
            ],
          }
        : baseLayout;

    return {
      data: traces,
      layout: {
        ...layout,
        showlegend: traces.length > 2,
      },
    };
  }, [
    buildLayout,
    comparisonResults,
    coronalImageUrl,
    currentSuffix,
    imageAvailable,
    lastLocation,
    resultsData,
    storedSuffix,
  ]);

  const choPlot = React.useMemo(() => {
    const location = resultsData?.location_sparse;
    const detectability = resultsData?.cho_detectability;
    const comparisonLocation = comparisonResults?.location_sparse;
    const comparisonDetectability = comparisonResults?.cho_detectability;

    if (
      (!isTruthyArray(location) || !isTruthyArray(detectability)) &&
      (!isTruthyArray(comparisonLocation) ||
        !isTruthyArray(comparisonDetectability))
    ) {
      return null;
    }

    const traces = [];
    if (location && detectability) {
      traces.push({
        name: `Detectability Index${currentSuffix}`,
        x: location,
        // y: [
        //   1.16858955, 1.19313878, 1.52170035, 1.62096009, 1.47527275,
        //   1.54435674,
        // ],
        y: detectability,
        type: "scatter",
        mode: "lines+markers",
        line: {
          width: 3,
          shape: "spline",
          color: "#ff0000ff",
          dash: "dot",
        },
        marker: { size: 8 },
        legendgroup: "detectability",
      });
    }
    if (comparisonLocation && comparisonDetectability) {
      traces.push({
        name: `Detectability Index${storedSuffix}`,
        x: comparisonLocation,
        y: comparisonDetectability,
        type: "scatter",
        mode: "lines+markers",
        line: {
          width: 2,
          shape: "spline",
          color: "#f97373",
          dash: "dot",
        },
        marker: { size: 8, symbol: "square-open" },
        legendgroup: "detectability",
      });
    }

    if (traces.length === 0) return null;

    const baseLayout = buildLayout({
      title: {
        text: "Detectability Index ",
      },
      xaxis: {
        title: { text: "Z Location (cm)" },
        range: [0, lastLocation || 0],
      },
      yaxis: {
        title: { text: "Detectability Index" },
      },
    });

    const layout =
      imageAvailable && coronalImageUrl
        ? {
            ...baseLayout,
            images: [
              {
                x: 0,
                y: 0,
                sizex: lastLocation || 1,
                sizey: 1,
                source: coronalImageUrl,
                xref: "x",
                yref: "paper",
                xanchor: "left",
                yanchor: "bottom",
                opacity: 1,
                layer: "below",
                sizing: "stretch",
              },
            ],
          }
        : baseLayout;

    return {
      data: traces,
      layout: {
        ...layout,
        showlegend: traces.length > 2,
      },
    };
  }, [
    buildLayout,
    comparisonResults,
    coronalImageUrl,
    currentSuffix,
    imageAvailable,
    lastLocation,
    resultsData,
    storedSuffix,
  ]);

  const plotConfig = { displayModeBar: false, responsive: true };
  const plotStyle = { width: "100%", height: "100%" };

  return (
    <>
      <Grid container spacing={2}>
        <Grid size={{ sm: 12, md: 4 }}>
          {ctdiPlot ? (
            <Box sx={chartContainerSx}>
              <IconButton
                aria-label='Expand CTDI/Dw plot'
                onClick={() =>
                  handleOpenPlotModal(
                    ctdiPlot,
                    ctdiPlot?.layout?.title?.text ?? "CTDI/Dw"
                  )
                }
                sx={{
                  position: "absolute",
                  bottom: (theme) => theme.spacing(1),
                  right: (theme) => theme.spacing(1),
                  zIndex: 1,
                }}>
                <SearchRoundedIcon fontSize='small' />
              </IconButton>
              <Plot
                data={ctdiPlot.data}
                layout={ctdiPlot.layout}
                config={plotConfig}
                style={plotStyle}
                useResizeHandler={true}
              />
            </Box>
          ) : (
            <EmptyPlot message='CTDI and SSDE data unavailable' />
          )}
        </Grid>
        <Grid size={{ sm: 12, md: 4 }}>
          {dwPlot ? (
            <Box sx={chartContainerSx}>
              <IconButton
                aria-label='Expand CTDI/Dw plot'
                onClick={() =>
                  handleOpenPlotModal(
                    dwPlot,
                    dwPlot?.layout?.title?.text ?? "CTDI/Dw"
                  )
                }
                sx={{
                  position: "absolute",
                  bottom: (theme) => theme.spacing(1),
                  right: (theme) => theme.spacing(1),
                  zIndex: 1,
                }}>
                <SearchRoundedIcon fontSize='small' />
              </IconButton>
              <Plot
                data={dwPlot.data}
                layout={dwPlot.layout}
                config={plotConfig}
                style={plotStyle}
                useResizeHandler={true}
              />
            </Box>
          ) : (
            <EmptyPlot message='Dw data unavailable' />
          )}
        </Grid>
        <Grid size={{ sm: 12, md: 4 }}>
          {noisePlot ? (
            <Box sx={chartContainerSx}>
              <IconButton
                aria-label='Expand noise and detectability plot'
                onClick={() =>
                  handleOpenPlotModal(
                    noisePlot,
                    noisePlot?.layout?.title?.text ?? "Noise and detectability"
                  )
                }
                sx={{
                  position: "absolute",
                  bottom: (theme) => theme.spacing(1),
                  right: (theme) => theme.spacing(1),
                  zIndex: 1,
                }}>
                <SearchRoundedIcon fontSize='small' />
              </IconButton>
              <Plot
                data={noisePlot.data}
                layout={noisePlot.layout}
                config={plotConfig}
                style={plotStyle}
                useResizeHandler={true}
              />
            </Box>
          ) : (
            <EmptyPlot message='Noise and detectability data unavailable' />
          )}
        </Grid>
        <Grid size={{ sm: 12, md: 4 }}>
          {choPlot ? (
            <Box sx={chartContainerSx}>
              <IconButton
                aria-label='Expand noise and detectability plot'
                onClick={() =>
                  handleOpenPlotModal(
                    choPlot,
                    choPlot?.layout?.title?.text ?? "Noise and detectability"
                  )
                }
                sx={{
                  position: "absolute",
                  bottom: (theme) => theme.spacing(1),
                  right: (theme) => theme.spacing(1),
                  zIndex: 1,
                }}>
                <SearchRoundedIcon fontSize='small' />
              </IconButton>
              <Plot
                data={choPlot.data}
                layout={choPlot.layout}
                config={plotConfig}
                style={plotStyle}
                useResizeHandler={true}
              />
            </Box>
          ) : (
            <EmptyPlot message='Noise and detectability data unavailable' />
          )}
        </Grid>
        <Grid size={{ sm: 12, md: 4 }}>
          {npsPlot ? (
            <Box sx={chartContainerSx}>
              <IconButton
                aria-label='Expand noise power spectrum plot'
                onClick={() =>
                  handleOpenPlotModal(
                    npsPlot,
                    npsPlot?.layout?.title?.text ?? "Noise power spectrum"
                  )
                }
                sx={{
                  position: "absolute",
                  bottom: (theme) => theme.spacing(1),
                  right: (theme) => theme.spacing(1),
                  zIndex: 1,
                }}>
                <SearchRoundedIcon fontSize='small' />
              </IconButton>
              <Plot
                data={npsPlot.data}
                layout={npsPlot.layout}
                config={plotConfig}
                style={plotStyle}
                useResizeHandler={true}
              />
            </Box>
          ) : (
            <EmptyPlot message='Noise power spectrum unavailable' />
          )}
        </Grid>
        <Grid size={{ sm: 12, md: 4 }}>
          <Box sx={chartContainerSx}>{children}</Box>
        </Grid>
      </Grid>
      <Dialog
        open={Boolean(expandedPlot)}
        onClose={handleClosePlotModal}
        maxWidth='xl'
        fullWidth
        slotProps={{
          paper: { sx: { bgcolor: "background.paper" } },
        }}>
        {expandedPlot ? (
          <>
            <DialogTitle
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                pr: 1,
              }}>
              {expandedPlot.title}
              <IconButton
                aria-label='Close enlarged plot'
                onClick={handleClosePlotModal}
                edge='end'
                size='small'>
                <CloseRoundedIcon />
              </IconButton>
            </DialogTitle>
            <DialogContent
              dividers
              sx={{
                height: { xs: 400, sm: 500, md: 600 },
                p: (theme) => theme.spacing(2),
              }}>
              <Box sx={{ height: "100%", width: "100%" }}>
                <Plot
                  key={expandedPlot.title}
                  data={expandedPlot.plot.data}
                  layout={expandedPlot.plot.layout}
                  config={plotConfig}
                  style={plotStyle}
                  useResizeHandler={true}
                />
              </Box>
            </DialogContent>
          </>
        ) : null}
      </Dialog>
    </>
  );
};

const EmptyPlot = ({ message }) => (
  <Box
    sx={{
      ...chartContainerSx,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}>
    <Typography variant='body2' color='text.secondary'>
      {message}
    </Typography>
  </Box>
);

export default ChoPlots;
