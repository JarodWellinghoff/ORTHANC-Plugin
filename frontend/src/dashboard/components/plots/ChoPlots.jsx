import * as React from "react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Plot from "react-plotly.js";

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
  p: 1,
  minHeight: 320,
  flex: 1,
};

const ChoPlots = ({ data }) => {
  const seriesData = data?.series ?? data;
  const resultsData = data?.results ?? data;
  const scanLength = seriesData?.scan_length_cm ?? 0;
  const seriesInstanceUid = seriesData?.series_instance_uid;
  const coronalImageUrl = seriesInstanceUid ? `/minio-images/${seriesInstanceUid}` : null;
  const imageAvailable = useImageAvailable(coronalImageUrl);

  const npsPlot = React.useMemo(() => {
    const spatial = resultsData?.spatial_frequency;
    const nps = resultsData?.nps;
    if (!spatial || !nps) return null;
    return {
      data: [
        {
          name: "NPS",
          x: spatial,
          y: nps,
          type: "scatter",
          mode: "lines",
          line: { width: 3, shape: "spline", color: "#667eea" },
        },
      ],
      layout: {
        title: { text: "Noise Power Spectrum vs Spatial Frequency", font: { size: 16, color: "#2d3748" } },
        xaxis: {
          title: "Spatial Frequency (cm<sup>-1</sup>)",
          titlefont: { size: 14 },
          tickfont: { size: 12 },
        },
        yaxis: {
          title: "NPS",
          titlefont: { size: 14 },
          tickfont: { size: 12 },
        },
        plot_bgcolor: "white",
        paper_bgcolor: "white",
        margin: { l: 60, r: 40, t: 60, b: 60 },
        showlegend: false,
        font: { family: "Arial, sans-serif" },
      },
    };
  }, [resultsData]);

  const ctdiDwPlot = React.useMemo(() => {
    const location = resultsData?.location;
    const ctdi = resultsData?.ctdivol;
    const dw = resultsData?.dw;
    if (!location || !ctdi || !dw) return null;
    const baseLayout = {
      xaxis: {
        title: "Location (cm)",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
        range: [0, scanLength || location[location.length - 1] || 0],
        showgrid: false,
      },
      yaxis: {
        title: "CTDIvol (mGy)",
        titlefont: { size: 14, color: "#667eea" },
        tickfont: { size: 12, color: "#667eea" },
        showgrid: false,
      },
      yaxis2: {
        title: "Water Equivalent Diameter (mm)",
        titlefont: { size: 14, color: "#e53e3e" },
        tickfont: { size: 12, color: "#e53e3e" },
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      margin: { l: 60, r: 60, t: 60, b: 60 },
      legend: {
        x: 0.02,
        y: 1.02,
        bgcolor: "rgba(255,255,255,0.8)",
        bordercolor: "#e2e8f0",
        borderwidth: 1,
        yanchor: "bottom",
      },
      font: { family: "Arial, sans-serif" },
    };

    const layout = imageAvailable && coronalImageUrl
      ? {
          ...baseLayout,
          images: [
            {
              x: 0,
              y: 0,
              sizex: scanLength || 1,
              sizey: 1,
              source: coronalImageUrl,
              xref: "x",
              yref: "paper",
              xanchor: "left",
              yanchor: "bottom",
              opacity: 0.8,
              layer: "below",
              sizing: "stretch",
            },
          ],
        }
      : baseLayout;

    return {
      data: [
        {
          name: "CTDIvol",
          x: location,
          y: ctdi,
          type: "scatter",
          mode: "lines+markers",
          line: { width: 3, shape: "spline", color: "#667eea" },
          marker: { size: 4 },
        },
        {
          name: "Water Equivalent Diameter",
          x: location,
          y: dw,
          type: "scatter",
          mode: "lines+markers",
          yaxis: "y2",
          line: { width: 3, shape: "spline", color: "#e53e3e" },
          marker: { size: 4 },
        },
      ],
      layout,
    };
  }, [resultsData, scanLength, coronalImageUrl, imageAvailable]);

  const noiseChoPlot = React.useMemo(() => {
    const location = resultsData?.location_sparse;
    const noise = resultsData?.noise_level;
    const detectability = resultsData?.cho_detectability;
    if (!location || !noise || !detectability) return null;

    const baseLayout = {
      xaxis: {
        title: "Anatomical Location (cm)",
        titlefont: { size: 14 },
        tickfont: { size: 12 },
        range: [0, scanLength || location[location.length - 1] || 0],
        showgrid: false,
      },
      yaxis: {
        title: "Noise Level (HU)",
        titlefont: { size: 14, color: "#667eea" },
        tickfont: { size: 12, color: "#667eea" },
        showgrid: false,
      },
      yaxis2: {
        title: "CHO Detectability Index",
        titlefont: { size: 14, color: "#e53e3e" },
        tickfont: { size: 12, color: "#e53e3e" },
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      margin: { l: 60, r: 60, t: 60, b: 60 },
      legend: {
        x: 0.02,
        y: 1.02,
        bgcolor: "rgba(255,255,255,0.8)",
        bordercolor: "#e2e8f0",
        borderwidth: 1,
        yanchor: "bottom",
      },
      font: { family: "Arial, sans-serif" },
    };

    const layout = imageAvailable && coronalImageUrl
      ? {
          ...baseLayout,
          images: [
            {
              x: 0,
              y: 0,
              sizex: scanLength || 1,
              sizey: 1,
              source: coronalImageUrl,
              xref: "x",
              yref: "paper",
              xanchor: "left",
              yanchor: "bottom",
              opacity: 0.8,
              layer: "below",
              sizing: "stretch",
            },
          ],
        }
      : baseLayout;

    return {
      data: [
        {
          name: "Local Noise Level",
          x: location,
          y: noise,
          type: "scatter",
          mode: "lines+markers",
          line: { width: 3, shape: "spline", color: "#667eea", dash: "dot" },
          marker: { size: 5 },
        },
        {
          name: "CHO Detectability Index",
          x: location,
          y: detectability,
          type: "scatter",
          mode: "lines+markers",
          yaxis: "y2",
          line: { width: 3, shape: "spline", color: "#e53e3e", dash: "dot" },
          marker: { size: 5 },
        },
      ],
      layout,
    };
  }, [resultsData, scanLength, coronalImageUrl, imageAvailable]);

  const plotConfig = { displayModeBar: false, responsive: true }; 
  const plotStyle = { width: "100%", height: "100%" };

  return (
    <Stack direction={{ xs: "column", lg: "row" }} spacing={2} sx={{ width: "100%" }}>
      {npsPlot ? (
        <Box sx={chartContainerSx}>
          <Plot data={npsPlot.data} layout={npsPlot.layout} config={plotConfig} style={plotStyle} useResizeHandler />
        </Box>
      ) : (
        <EmptyPlot message="Noise power spectrum unavailable" />
      )}
      {ctdiDwPlot ? (
        <Box sx={chartContainerSx}>
          <Plot data={ctdiDwPlot.data} layout={ctdiDwPlot.layout} config={plotConfig} style={plotStyle} useResizeHandler />
        </Box>
      ) : (
        <EmptyPlot message="CTDI/Dw data unavailable" />
      )}
      {noiseChoPlot ? (
        <Box sx={chartContainerSx}>
          <Plot data={noiseChoPlot.data} layout={noiseChoPlot.layout} config={plotConfig} style={plotStyle} useResizeHandler />
        </Box>
      ) : (
        <EmptyPlot message="Noise and detectability data unavailable" />
      )}
    </Stack>
  );
};

const EmptyPlot = ({ message }) => (
  <Box sx={{ ...chartContainerSx, display: "flex", alignItems: "center", justifyContent: "center" }}>
    <Typography variant="body2" color="text.secondary">
      {message}
    </Typography>
  </Box>
);

export default ChoPlots;
