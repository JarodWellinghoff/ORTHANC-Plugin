import React, { useState, useEffect, useMemo } from "react";
import Plot from "react-plotly.js";
import { useTheme } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";

/**
 * MTF Interactive Plot - React Component
 *
 * Features:
 * - 6 interactive sliders for controlling MTF thresholds
 * - Smooth cubic spline transformation algorithm
 * - Real-time curve updates
 * - Professional visualization
 */
const LN10_OVER_LN2 = Math.log(10) / Math.log(2);
const gaussianF10FromF50 = (f50) =>
  f50 != null ? Number(f50) * Math.sqrt(LN10_OVER_LN2) : null;
function handleShapesForContrast(
  contrastKey,
  mtfc50Data,
  mtfc10Data,
  visible,
  editable50,
  editable10
) {
  // Only show handles for the active contrast to avoid confusion
  if (!visible) return [];
  const x50 = mtfc50Data[contrastKey];
  const x10 = editable10
    ? mtfc10Data[contrastKey]
    : gaussianF10FromF50(mtfc50Data[contrastKey]);

  const mkVLine = (x, color, editable) => ({
    type: "line",
    xref: "x",
    yref: "paper", // span full plot height in paper coords
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    line: { color, width: 4 }, // thicker = easier to grab
    opacity: editable ? 1 : 0,
    editable: editable,
  });

  //   return [
  //     mkCircle(x50, "#fff"), // 50% handle
  //     mkCircle(x10, "#fff"), // 10% handle
  //   ];
  return [
    mkVLine(x50, "#fff", editable50), // 50% handle
    mkVLine(x10, "#fff", editable10), // 10% handle
  ];
}
function nextEditRevision(prev) {
  return (prev ?? 0) + 1;
}
// Cubic spline interpolation class (JavaScript implementation)
class CubicSpline {
  constructor(x, y) {
    this.x = [...x];
    this.y = [...y];
    this.n = x.length;
    this.computeCoefficients();
  }

  computeCoefficients() {
    const n = this.n - 1;
    const h = new Array(n);
    const alpha = new Array(n);
    const l = new Array(this.n);
    const mu = new Array(this.n);
    const z = new Array(this.n);
    const c = new Array(this.n);
    const b = new Array(n);
    const d = new Array(n);

    // Calculate h values
    for (let i = 0; i < n; i++) {
      h[i] = this.x[i + 1] - this.x[i];
    }

    // Calculate alpha values
    for (let i = 1; i < n; i++) {
      alpha[i] =
        (3 / h[i]) * (this.y[i + 1] - this.y[i]) -
        (3 / h[i - 1]) * (this.y[i] - this.y[i - 1]);
    }

    // Natural spline boundary conditions
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    // Solve tridiagonal system
    for (let i = 1; i < n; i++) {
      l[i] = 2 * (this.x[i + 1] - this.x[i - 1]) - h[i - 1] * mu[i - 1];
      mu[i] = h[i] / l[i];
      z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n] = 1;
    z[n] = 0;
    c[n] = 0;

    // Back substitution
    for (let j = n - 1; j >= 0; j--) {
      c[j] = z[j] - mu[j] * c[j + 1];
      b[j] =
        (this.y[j + 1] - this.y[j]) / h[j] - (h[j] * (c[j + 1] + 2 * c[j])) / 3;
      d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    this.b = b;
    this.c = c;
    this.d = d;
  }

  interpolate(xVal) {
    // Find the interval
    let i = 0;
    while (i < this.n - 1 && xVal > this.x[i + 1]) {
      i++;
    }

    // Clamp to valid range
    if (i >= this.n - 1) i = this.n - 2;
    if (i < 0) i = 0;

    const dx = xVal - this.x[i];
    return (
      this.y[i] +
      this.b[i] * dx +
      this.c[i] * dx * dx +
      this.d[i] * dx * dx * dx
    );
  }
}

// Linear interpolation helper
function linearInterp(x, xArray, yArray) {
  if (x <= xArray[0]) return yArray[0];
  if (x >= xArray[xArray.length - 1]) return yArray[yArray.length - 1];

  for (let i = 0; i < xArray.length - 1; i++) {
    if (x >= xArray[i] && x <= xArray[i + 1]) {
      const t = (x - xArray[i]) / (xArray[i + 1] - xArray[i]);
      return yArray[i] + t * (yArray[i + 1] - yArray[i]);
    }
  }
  return yArray[yArray.length - 1];
}

// Find frequency where MTF equals target value
function findFrequencyAtMTF(freqArray, mtfArray, targetMTF) {
  for (let i = 0; i < mtfArray.length - 1; i++) {
    if (
      (mtfArray[i] >= targetMTF && mtfArray[i + 1] <= targetMTF) ||
      (mtfArray[i] <= targetMTF && mtfArray[i + 1] >= targetMTF)
    ) {
      // Linear interpolation
      const t = (targetMTF - mtfArray[i]) / (mtfArray[i + 1] - mtfArray[i]);
      return freqArray[i] + t * (freqArray[i + 1] - freqArray[i]);
    }
  }
  return null;
}

// Transform MTF curve using smooth cubic spline interpolation
function transformMTFCurve(freqOriginal, mtfOriginal, freqAt05, freqAt01) {
  // Find original frequencies at thresholds
  const origFreq05 = findFrequencyAtMTF(freqOriginal, mtfOriginal, 0.5);
  const origFreq01 = findFrequencyAtMTF(freqOriginal, mtfOriginal, 0.1);

  if (!origFreq05 || !origFreq01) {
    return { freq: freqOriginal, mtf: mtfOriginal };
  }

  // Calculate scaling factors
  const scale05 = freqAt05 / origFreq05;
  const scale01 = freqAt01 / origFreq01;

  // Define anchor points for smooth scaling
  const scaleAtZero = scale01 + ((scale01 - scale05) * 0.1) / 0.4;
  const mtfAnchors = [0.0, 0.1, 0.5, 1.0];
  const scaleAnchors = [scaleAtZero, scale01, scale05, 1.0];

  // Create cubic spline interpolator
  const scaleInterpolator = new CubicSpline(mtfAnchors, scaleAnchors);

  // Apply transformation
  const freqNew = [];
  const mtfNew = [];

  for (let i = 0; i < freqOriginal.length; i++) {
    const f = freqOriginal[i];
    let mtfVal = linearInterp(f, freqOriginal, mtfOriginal);

    // Clamp MTF to valid range
    mtfVal = Math.max(0.0, Math.min(1.0, mtfVal));

    // Get smooth scale factor
    let scale = scaleInterpolator.interpolate(mtfVal);

    // Clamp scale to reasonable range
    scale = Math.max(0.1, Math.min(10.0, scale));

    freqNew.push(f * scale);
    mtfNew.push(mtfVal);
  }

  // Sort by frequency
  const sorted = freqNew
    .map((f, i) => ({ freq: f, mtf: mtfNew[i] }))
    .sort((a, b) => a.freq - b.freq);

  // Remove duplicates
  const unique = [];
  for (let i = 0; i < sorted.length; i++) {
    if (i === 0 || sorted[i].freq - sorted[i - 1].freq > 1e-10) {
      unique.push(sorted[i]);
    }
  }

  return {
    freq: unique.map((d) => d.freq),
    mtf: unique.map((d) => d.mtf),
  };
}

const chartContainerSx = {
  borderRadius: 2,
  border: "1px solid",
  borderColor: "divider",
  position: "relative",
  p: 1,
  maxHeight: 512,
  //   flex: 1,
};

const plotConfig = { displayModeBar: false, responsive: true };
const plotStyle = { width: "100%", height: "100%" };

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

const MTFInteractivePlot = ({
  originalPlotData,
  loading,
  error,
  editable50,
  editable10,
  mtfc10Data,
  mtfc50Data,
  setMtfc50Data,
  setMtfc10Data,
  currentContrast,
}) => {
  // Parse CSV data
  const [editRev, setEditRev] = React.useState(0);
  const [expandedPlot, setExpandedPlot] = React.useState(null);
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

  const onRelayout = React.useCallback(
    (relayoutData) => {
      // We only care about shape drags; relayoutData will contain keys like:
      // "shapes[0].x0", "shapes[0].x1", "shapes[0].y0", "shapes[0].y1", etc.
      // We’ll reconstruct centers for the first two shapes (50% and 10% handles).
      // Handles only exist for the currentContrast.
      const idxFromKey = (k) => {
        const m = k.match(/^shapes\[(\d+)\]\./);
        return m ? Number(m[1]) : null;
      };
      //   console.log("Relayout data:", relayoutData);

      // Gather mutated shapes centers
      const moved = {};
      for (const k in relayoutData) {
        if (!k.startsWith("shapes[")) continue;
        const idx = idxFromKey(k);
        if (idx == null) continue;
        moved[idx] = moved[idx] || { x0: null, x1: null, y0: null, y1: null };
        const part = k.slice(k.indexOf("].") + 2);
        moved[idx][part] = relayoutData[k];
      }

      // We created 2 shapes for the active contrast: [0] -> y=0.5 (50%), [1] -> y=0.1 (10%)
      // Compute the x centers and snap y back to fixed lines.

      const x50 = moved[2]?.x0 ?? moved[2]?.x1 ?? null;
      const x10 = moved[3]?.x0 ?? moved[3]?.x1 ?? null;

      if (x50 != null) {
        const fixedX50 = x50.toFixed(4);
        setMtfc50Data((prev) => ({ ...prev, [currentContrast]: fixedX50 }));
      }
      if (x10 != null && editable10) {
        const fixedX10 = x10.toFixed(4);
        setMtfc10Data((prev) => ({ ...prev, [currentContrast]: fixedX10 }));
      }
      //   console.log("Updated MTF data:", moved);
      // Force Plotly to keep/edit shapes coherently after we snap y back
      setEditRev(nextEditRevision);
    },
    [currentContrast, setMtfc50Data, setMtfc10Data, editable10]
  );
  const handleShapes = React.useMemo(() => {
    return [
      ...handleShapesForContrast(
        "10HU",
        mtfc50Data,
        mtfc10Data,
        currentContrast === "10HU",
        editable50,
        editable10
      ),
      ...handleShapesForContrast(
        "30HU",
        mtfc50Data,
        mtfc10Data,
        currentContrast === "30HU",
        editable50,
        editable10
      ),
      ...handleShapesForContrast(
        "50HU",
        mtfc50Data,
        mtfc10Data,
        currentContrast === "50HU",
        editable50,
        editable10
      ),
    ];
  }, [mtfc50Data, mtfc10Data, currentContrast, editable50, editable10]);

  const theme = useTheme();
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
            size: 16,
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
            size: 16,
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
        x: 0.9,
        y: 0.9,
        bgcolor: "rgba(0, 0, 0, 0.8)",
        bordercolor: "#e2e8f0",
        font: {
          color: theme.palette.primary.contrastText,
          size: 16,
        },
        borderwidth: 1,
        yanchor: "top",
        xanchor: "right",
        autosize: true,
      },
      showlegend: true,
      font: { family: "Arial, sans-serif" },
    }),
    [theme]
  );
  const buildLayout = React.useCallback(
    (options) =>
      mergeDeep(JSON.parse(JSON.stringify(baseLayout)), options || {}),
    [baseLayout]
  );

  const originalData = useMemo(() => {
    if (!originalPlotData) return null;

    const lines = originalPlotData.trim().split("\n");
    const freq = [];
    const mtf10hu = [];
    const mtf30hu = [];
    const mtf50hu = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(",");
      if (values.length >= 4) {
        freq.push(parseFloat(values[0] * 10));
        mtf10hu.push(parseFloat(values[1]));
        mtf30hu.push(parseFloat(values[2]));
        mtf50hu.push(parseFloat(values[3]));
      }
    }

    return { freq, mtf10hu, mtf30hu, mtf50hu };
  }, [originalPlotData]);

  // Find original threshold frequencies
  const originalThresholds = useMemo(() => {
    if (!originalData) return null;

    return {
      freq10hu_05: findFrequencyAtMTF(
        originalData.freq,
        originalData.mtf10hu,
        0.5
      ),
      freq10hu_01: findFrequencyAtMTF(
        originalData.freq,
        originalData.mtf10hu,
        0.1
      ),
      freq30hu_05: findFrequencyAtMTF(
        originalData.freq,
        originalData.mtf30hu,
        0.5
      ),
      freq30hu_01: findFrequencyAtMTF(
        originalData.freq,
        originalData.mtf30hu,
        0.1
      ),
      freq50hu_05: findFrequencyAtMTF(
        originalData.freq,
        originalData.mtf50hu,
        0.5
      ),
      freq50hu_01: findFrequencyAtMTF(
        originalData.freq,
        originalData.mtf50hu,
        0.1
      ),
    };
  }, [originalData]);

  const finalPlot = React.useMemo(() => {
    if (!originalData || !originalThresholds) return null;

    const f10_10 = editable10
      ? mtfc10Data["10HU"]
      : gaussianF10FromF50(mtfc50Data["10HU"]);
    const f10_30 = editable10
      ? mtfc10Data["30HU"]
      : gaussianF10FromF50(mtfc50Data["30HU"]);
    const f10_50 = editable10
      ? mtfc10Data["50HU"]
      : gaussianF10FromF50(mtfc50Data["50HU"]);

    const transformed10hu = transformMTFCurve(
      originalData.freq,
      originalData.mtf10hu,
      mtfc50Data["10HU"],
      f10_10
    );

    const transformed30hu = transformMTFCurve(
      originalData.freq,
      originalData.mtf30hu,
      mtfc50Data["30HU"],
      f10_30
    );

    const transformed50hu = transformMTFCurve(
      originalData.freq,
      originalData.mtf50hu,
      mtfc50Data["50HU"],
      f10_50
    );

    const plotData = [
      // Original curves (dotted)
      //   {
      //     x: originalData.freq,
      //     y: originalData.mtf10hu,
      //     type: "scatter",
      //     mode: "lines",
      //     name: "10HU (Original)",
      //     line: { color: "blue", width: 1.5, dash: "dot" },
      //     opacity: currentContrast === "10HU" ? 1.0 : 0,
      //   },
      //   {
      //     x: originalData.freq,
      //     y: originalData.mtf30hu,
      //     type: "scatter",
      //     mode: "lines",
      //     name: "30HU (Original)",
      //     line: { color: "green", width: 1.5, dash: "dot" },
      //     opacity: currentContrast === "30HU" ? 1.0 : 0,
      //   },
      //   {
      //     x: originalData.freq,
      //     y: originalData.mtf50hu,
      //     type: "scatter",
      //     mode: "lines",
      //     name: "50HU (Original)",
      //     line: { color: "red", width: 1.5, dash: "dot" },
      //     opacity: currentContrast === "50HU" ? 1.0 : 0,
      //   },
      // Transformed curves (solid)
      {
        x: editable50 ? transformed10hu.freq : originalData.freq,
        y: editable50 ? transformed10hu.mtf : originalData.mtf10hu,
        type: "scatter",
        mode: "lines",
        // name: "10HU (Simulated)",
        name: "-10HU",
        line: { color: "#00FFFF", width: 2 },
        // opacity: currentContrast === "10HU" ? 1.0 : 0,
        opacity: 1,
      },
      {
        x: editable50 ? transformed30hu.freq : originalData.freq,
        y: editable50 ? transformed30hu.mtf : originalData.mtf30hu,
        type: "scatter",
        mode: "lines",
        // name: "30HU (Simulated)",
        name: "-30HU",
        line: { color: "#FF00FF", width: 2 },
        // opacity: currentContrast === "30HU" ? 1.0 : 0,
        opacity: 1,
      },
      {
        x: editable50 ? transformed50hu.freq : originalData.freq,
        y: editable50 ? transformed50hu.mtf : originalData.mtf50hu,
        type: "scatter",
        mode: "lines",
        // name: "50HU (Simulated)",
        name: "-50HU",
        line: { color: "#FFFF00", width: 2 },
        // opacity: currentContrast === "50HU" ? 1.0 : 0,
        opacity: 1,
      },
    ];

    if (plotData.length === 0) return null;

    return {
      data: plotData,
      layout: buildLayout({
        hovermode: false,
        dragmode: false,
        edits: { shapePosition: true, shapeSize: false },
        editrevision: editRev,
        xaxis: {
          title: {
            text: "Spatial Frequency (cm<sup>-1</sup>)",
          },
        },
        yaxis: {
          title: {
            text: "MTFc",
          },
        },
        shapes: [
          // Reference line at MTF=0.5
          {
            type: "line",
            x0: 0,
            x1: 1,
            xref: "paper",
            y0: 0.5,
            y1: 0.5,
            line: {
              color: "gray",
              width: 1,
              dash: "dash",
            },
          },
          // Reference line at MTF=0.1
          {
            type: "line",
            x0: 0,
            x1: 1,
            xref: "paper",
            y0: 0.1,
            y1: 0.1,
            line: {
              color: "gray",
              width: 1,
              dash: "dash",
            },
          },
          ...handleShapes,
        ],
        annotations: [
          {
            x: 0.98,
            y: 0.55,
            xref: "paper",
            yref: "y",
            text: "MTF = 0.5",
            showarrow: false,
            font: { size: 16, color: "gray" },
          },
          {
            x: 0.98,
            y: 0.15,
            xref: "paper",
            yref: "y",
            text: "MTF = 0.1",
            showarrow: false,
            font: { size: 16, color: "gray" },
          },
        ],
      }),
    };
  }, [
    originalData,
    originalThresholds,
    mtfc50Data,
    mtfc10Data,
    currentContrast,
    buildLayout,
    editRev,
    handleShapes,
    editable50,
  ]);

  if (loading) {
    return <div className='loading'>Loading MTF data...</div>;
  }

  if (error) {
    return <div className='error'>Error loading MTF data: {error}</div>;
  }
  if (finalPlot) {
    return (
      <>
        <Box sx={chartContainerSx}>
          <IconButton
            aria-label='MTF plot'
            onClick={() =>
              handleOpenPlotModal(
                finalPlot,
                finalPlot?.layout?.title?.text ?? "Noise power spectrum"
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
            data={finalPlot.data}
            layout={finalPlot.layout}
            config={plotConfig}
            style={plotStyle}
            useResizeHandler
            onRelayout={onRelayout}
          />
        </Box>
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
  }
};

export default MTFInteractivePlot;
