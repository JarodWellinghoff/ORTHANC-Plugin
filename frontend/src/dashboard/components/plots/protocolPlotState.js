// Plotting-page state is independent of the cohort and of chart rendering.
export const MAX_PLOT_PROTOCOLS = 10;
export const PROTOCOL_GROUP_KEY = "protocol_name";
export const DEFAULT_PLOT_METRIC = "average_index_of_detectability";

export const PLOT_COLUMNS = [
  { mode: "histogram", title: "Histograms", plotLabel: "Histogram", addLabel: "Add histogram" },
  { mode: "box", title: "Box plots", plotLabel: "Box plot", addLabel: "Add box plot" },
];

const createPlot = (mode, id) => ({
  id: `plot-${id}`,
  mode,
  xKey: DEFAULT_PLOT_METRIC,
  groupKey: PROTOCOL_GROUP_KEY,
  // null selects the first ten available protocols; [] means none selected.
  protocols: null,
});

export const createInitialPlotState = () => ({
  columns: {
    histogram: [createPlot("histogram", 1)],
    box: [createPlot("box", 2)],
  },
  nextId: 3,
});

export const protocolPlotReducer = (state, action) => {
  if (!PLOT_COLUMNS.some((column) => column.mode === action.mode)) return state;
  const column = state.columns[action.mode];
  let nextColumn;

  switch (action.type) {
    case "add":
      return {
        columns: {
          ...state.columns,
          [action.mode]: [...column, createPlot(action.mode, state.nextId)],
        },
        nextId: state.nextId + 1,
      };
    case "remove":
      // Both columns must retain at least one plot, even for direct dispatches.
      if (column.length <= 1 || !column.some((plot) => plot.id === action.id)) {
        return state;
      }
      nextColumn = column.filter((plot) => plot.id !== action.id);
      break;
    case "configure":
      nextColumn = column.map((plot) => {
        if (plot.id !== action.id) return plot;
        const patch = action.patch ?? {};
        return {
          ...plot,
          ...(typeof patch.xKey === "string" ? { xKey: patch.xKey } : {}),
          ...(Array.isArray(patch.protocols)
            ? { protocols: [...new Set(patch.protocols)].slice(0, MAX_PLOT_PROTOCOLS) }
            : {}),
        };
      });
      break;
    default:
      return state;
  }

  return { ...state, columns: { ...state.columns, [action.mode]: nextColumn } };
};

/** Keep selections valid after a new query without interpreting [] as "all". */
export const resolvePlotProtocols = (protocols, options) => {
  const available = [...new Set(options)];
  if (protocols == null) return available.slice(0, MAX_PLOT_PROTOCOLS);
  const selected = new Set(protocols);
  return available.filter((protocol) => selected.has(protocol)).slice(0, MAX_PLOT_PROTOCOLS);
};

/** Only chart records are restricted; never pass this result to the table. */
export const selectPlotRecords = (protocolGroups, protocols) => {
  const selected = new Set(resolvePlotProtocols(
    protocols,
    protocolGroups.map((group) => group.label),
  ));
  return protocolGroups
    .filter((group) => selected.has(group.label))
    .flatMap((group) => group.records);
};

/** All plot/protocol combinations, deliberately ignoring chart selections. */
export const getAllPlotGroups = (plots, protocolGroups) =>
  plots.flatMap((plot) => protocolGroups.map((group) => ({
    id: JSON.stringify([plot.id, group.label]),
    plotTitle: plot.title,
    metricKey: plot.xKey,
    protocol: group.label,
    records: group.records,
  })));
