// ─────────────────────────────────────────────────────────────────────────────
// plotTheme.js
//
// Theme-aware Plotly layout helpers, extracted so more than one plot surface
// can share them.
//
// The colour resolution and base layout here are lifted from the inline
// implementation in ChoPlots.jsx — same palette lookup order, same transparent
// backgrounds, same axis treatment — so aggregate plots and per-series plots
// stay visually consistent in both light and dark mode. ChoPlots is deliberately
// left untouched; it can adopt these hooks later without any behaviour change.
// ─────────────────────────────────────────────────────────────────────────────

import * as React from "react";
import { useTheme, useColorScheme } from "@mui/material/styles";

export const mergeDeep = (target, source) => {
  for (const key in source) {
    if (
      source[key] instanceof Object &&
      !Array.isArray(source[key]) &&
      key in target &&
      target[key] instanceof Object &&
      !Array.isArray(target[key])
    ) {
      mergeDeep(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
};

/**
 * Resolve the effective colour mode, honouring the "system" setting, then pull
 * text/divider colours out of whichever palette shape is available.
 */
export const usePlotColors = () => {
  const theme = useTheme();
  const colorScheme = useColorScheme();
  const schemeMode = colorScheme?.mode;
  const schemeSystemMode = colorScheme?.systemMode;
  const themeMode = theme.palette?.mode ?? "light";

  const resolvedColorMode = React.useMemo(() => {
    if (!schemeMode) return themeMode;
    if (schemeMode === "system") return schemeSystemMode ?? themeMode;
    return schemeMode;
  }, [schemeMode, schemeSystemMode, themeMode]);

  return React.useMemo(() => {
    const paletteSource =
      theme.colorSchemes?.[resolvedColorMode]?.palette ??
      (theme.vars || theme).palette ??
      theme.palette ??
      {};
    const fallbackPalette = theme.palette ?? {};

    const textPrimary =
      paletteSource.text?.primary ?? fallbackPalette.text?.primary ?? "#0f172a";
    const textSecondary =
      paletteSource.text?.secondary ??
      fallbackPalette.text?.secondary ??
      textPrimary;
    const divider =
      paletteSource.divider ??
      fallbackPalette.divider ??
      (resolvedColorMode === "dark"
        ? "rgba(255,255,255,0.16)"
        : "rgba(0,0,0,0.12)");
    const legendBg =
      resolvedColorMode === "dark"
        ? "rgba(15,23,42,0.72)"
        : "rgba(255,255,255,0.72)";

    return { textPrimary, textSecondary, divider, legendBg, resolvedColorMode };
  }, [theme, resolvedColorMode]);
};

/**
 * Returns `buildLayout(overrides)` — a deep merge over the shared base layout.
 * Backgrounds stay transparent so the plot inherits the surface it sits on
 * (a Paper in ChoPlots, a Dialog here).
 */
export const useBuildLayout = () => {
  const plotColors = usePlotColors();

  const baseLayout = React.useMemo(() => {
    const createAxisLayout = (overrides = {}) => ({
      showgrid: true,
      ...overrides,
      color: overrides.color ?? plotColors.textSecondary,
      linecolor: overrides.linecolor ?? plotColors.textSecondary,
      zerolinecolor: overrides.zerolinecolor ?? plotColors.divider,
      gridcolor: overrides.gridcolor ?? plotColors.divider,
    });

    return {
      title: {
        font: { size: 16, color: plotColors.textPrimary, weight: 1000 },
      },
      xaxis: createAxisLayout(),
      yaxis: createAxisLayout({ showgrid: true }),
      plot_bgcolor: "rgba(0, 0, 0, 0)",
      paper_bgcolor: "rgba(0, 0, 0, 0)",
      margin: { l: 70, r: 40, t: 60, b: 70 },
      hovermode: "closest",
      legend: {
        x: 1.02,
        y: 1,
        xanchor: "left",
        yanchor: "top",
        bgcolor: plotColors.legendBg,
        bordercolor: plotColors.divider,
        borderwidth: 1,
        font: { color: plotColors.textPrimary },
      },
      font: { family: "Arial, sans-serif", color: plotColors.textPrimary },
    };
  }, [plotColors]);

  return React.useCallback(
    (options) =>
      mergeDeep(JSON.parse(JSON.stringify(baseLayout)), options || {}),
    [baseLayout],
  );
};

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };
export const PLOT_STYLE = { width: "100%", height: "100%" };
