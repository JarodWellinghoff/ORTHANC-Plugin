// ─────────────────────────────────────────────────────────────────────────────
// useAggregatePlot.js
//
// Computes a Plotly figure + summary-table rows for one "customizable plot"
// slot, given a shared pool of cohort `records` and that slot's own
// {mode, xKey, yKey, groupKey, showTrend} config. Extracted from the old
// AggregatePlotsDialog so a page can render several independently-configured
// plots (and their tables) against the same fetched records without
// duplicating the scatter/histogram/box logic per instance.
// ─────────────────────────────────────────────────────────────────────────────

import * as React from "react";

import {
  GROUP_PALETTE,
  SCALAR_METRIC_MAP,
  UNGROUPED_KEY,
  describe,
  groupRecords,
  linearFit,
  pearson,
  toFiniteNumber,
} from "../../utils/aggregateMetrics";
import { useBuildLayout } from "../../theme/customizations/plotTheme";

export const useAggregatePlot = (records, config) => {
  const buildLayout = useBuildLayout();
  const { mode, xKey, yKey, groupKey, showTrend } = config;

  const xMetric = SCALAR_METRIC_MAP.get(xKey);
  const yMetric = SCALAR_METRIC_MAP.get(yKey);
  const isScatter = mode === "scatter";

  const groups = React.useMemo(
    () => groupRecords(records, groupKey ?? UNGROUPED_KEY),
    [records, groupKey],
  );

  const plot = React.useMemo(() => {
    if (records.length === 0 || !xMetric) return null;

    const colorFor = (index) => GROUP_PALETTE[index % GROUP_PALETTE.length];

    if (isScatter) {
      if (!yMetric) return null;

      const traces = groups.map((group, index) => {
        const points = group.records
          .map((record) => ({
            x: toFiniteNumber(record[xMetric.key]),
            y: toFiniteNumber(record[yMetric.key]),
            label: record.__label,
          }))
          .filter((point) => point.x !== null && point.y !== null);

        return {
          name: `${group.label} (n=${points.length})`,
          x: points.map((p) => p.x),
          y: points.map((p) => p.y),
          text: points.map((p) => p.label),
          hovertemplate:
            "%{text}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra></extra>",
          type: "scatter",
          mode: "markers",
          marker: {
            size: 9,
            color: colorFor(index),
            line: { width: 1, color: "rgba(0,0,0,0.35)" },
          },
        };
      });

      if (showTrend) {
        const allX = records.map((record) => record[xMetric.key]);
        const allY = records.map((record) => record[yMetric.key]);
        const fit = linearFit(allX, allY);
        if (fit) {
          traces.push({
            name: `Fit: y = ${fit.slope.toPrecision(3)}x ${
              fit.intercept >= 0 ? "+" : "−"
            } ${Math.abs(fit.intercept).toPrecision(3)}`,
            x: fit.x,
            y: fit.y,
            type: "scatter",
            mode: "lines",
            hoverinfo: "skip",
            line: { width: 2, dash: "dash", color: "#7f7f7f" },
          });
        }
      }

      return {
        data: traces,
        layout: buildLayout({
          title: { text: `${yMetric.label} vs ${xMetric.label}` },
          xaxis: { title: { text: xMetric.axisLabel } },
          yaxis: { title: { text: yMetric.axisLabel } },
          showlegend: true,
        }),
      };
    }

    if (mode === "histogram") {
      const traces = groups.map((group, index) => {
        const values = group.records
          .map((record) => toFiniteNumber(record[xMetric.key]))
          .filter((value) => value !== null);
        return {
          name: `${group.label} (n=${values.length})`,
          x: values,
          type: "histogram",
          opacity: groups.length > 1 ? 0.6 : 0.85,
          marker: {
            color: colorFor(index),
            line: { width: 1, color: "rgba(0,0,0,0.25)" },
          },
        };
      });

      return {
        data: traces,
        layout: buildLayout({
          title: { text: `Distribution of ${xMetric.label}` },
          xaxis: { title: { text: xMetric.axisLabel } },
          yaxis: { title: { text: "Series count" } },
          barmode: "overlay",
          showlegend: groups.length > 1,
        }),
      };
    }

    // Box
    const traces = groups.map((group, index) => {
      const values = group.records
        .map((record) => toFiniteNumber(record[xMetric.key]))
        .filter((value) => value !== null);
      return {
        name: `${group.label} (n=${values.length})`,
        y: values,
        type: "box",
        boxpoints: "outliers",
        marker: { color: colorFor(index), size: 6 },
        line: { width: 2 },
      };
    });

    return {
      data: traces,
      layout: buildLayout({
        title: { text: `${xMetric.label} by group` },
        yaxis: { title: { text: xMetric.axisLabel } },
        xaxis: { title: { text: "" } },
        showlegend: false,
      }),
    };
  }, [buildLayout, groups, isScatter, mode, records, showTrend, xMetric, yMetric]);

  // One row per group per metric. Scatter reports both axes so the two
  // distributions behind a correlation are visible alongside it.
  const statRows = React.useMemo(() => {
    const metrics = isScatter ? [xMetric, yMetric] : [xMetric];
    const output = [];
    for (const metric of metrics) {
      if (!metric) continue;
      for (const group of groups) {
        const stats = describe(
          group.records.map((record) => record[metric.key]),
        );
        output.push({
          id: `${metric.key}-${group.label}`,
          metric: metric.label,
          unit: metric.unit,
          group: group.label,
          decimals: metric.decimals,
          ...stats,
        });
      }
    }
    return output;
  }, [groups, isScatter, xMetric, yMetric]);

  const correlation = React.useMemo(() => {
    if (!isScatter || !xMetric || !yMetric || records.length === 0) return null;
    return pearson(
      records.map((record) => record[xMetric.key]),
      records.map((record) => record[yMetric.key]),
    );
  }, [isScatter, records, xMetric, yMetric]);

  return { plot, statRows, correlation, xMetric, yMetric, isScatter, groups };
};
