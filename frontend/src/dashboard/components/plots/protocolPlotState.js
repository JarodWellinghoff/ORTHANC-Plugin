export const MAX_PLOT_PROTOCOLS = 10;
export const PROTOCOL_GROUP_KEY = "protocol_name";
export const DEFAULT_PLOT_METRIC = "average_index_of_detectability";
export const PLOT_COLUMNS = [
  { mode: "box", title: "Statistics for Multiple Protocols", addLabel: "Add box plot" },
  { mode: "histogram", title: "Histogram for a Single Protocol", addLabel: "Add histogram" },
];

export const normalizeProtocolName = (value) => {
  const full = value == null ? "" : String(value).trim();
  const first = full.split("_", 1)[0].trim();
  return first === "WIP" ? full : first || "Unspecified";
};
export const groupProtocolRecords = (records) => {
  const groups = new Map();
  for (const record of records) {
    const label = normalizeProtocolName(record.protocol_name);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(record);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b))
    .map(([label, grouped]) => ({ label, records: grouped }));
};
const createPlot = (mode, id) => ({
  id: `plot-${id}`, mode, xKey: DEFAULT_PLOT_METRIC,
  groupKey: PROTOCOL_GROUP_KEY, protocols: null,
});
export const createInitialPlotState = () => ({
  columns: { box: [createPlot("box", 1)], histogram: [createPlot("histogram", 2)] }, nextId: 3,
});
export const protocolPlotReducer = (state, action) => {
  if (!PLOT_COLUMNS.some((column) => column.mode === action.mode)) return state;
  const column = state.columns[action.mode];
  let next;
  switch (action.type) {
    case "add":
      return { columns: { ...state.columns, [action.mode]: [...column, createPlot(action.mode, state.nextId)] }, nextId: state.nextId + 1 };
    case "remove":
      if (column.length <= 1 || !column.some((plot) => plot.id === action.id)) return state;
      next = column.filter((plot) => plot.id !== action.id);
      break;
    case "configure":
      next = column.map((plot) => {
        if (plot.id !== action.id) return plot;
        const patch = action.patch ?? {};
        return { ...plot,
          ...(typeof patch.xKey === "string" ? { xKey: patch.xKey } : {}),
          ...(Array.isArray(patch.protocols) ? { protocols: [...new Set(patch.protocols)].slice(0, action.mode === "histogram" ? 1 : MAX_PLOT_PROTOCOLS) } : {}),
        };
      });
      break;
    default: return state;
  }
  return { ...state, columns: { ...state.columns, [action.mode]: next } };
};

// Rank defaults by series count, then alphabetize both options and selections.
export const resolvePlotProtocols = (protocols, options, mode = "box") => {
  const groups = options.map((option) => typeof option === "string"
    ? { label: option, count: 0 }
    : { label: option.label, count: option.records?.length ?? option.count ?? 0 });
  const available = [...new Set(groups.map((group) => group.label))].sort((a, b) => a.localeCompare(b));
  const limit = mode === "histogram" ? 1 : MAX_PLOT_PROTOCOLS;
  const defaults = () => [...new Set([...groups]
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label)).map((group) => group.label))]
    .slice(0, limit).sort((a, b) => a.localeCompare(b));
  if (protocols == null) return defaults();
  const selected = new Set(protocols);
  const valid = available.filter((label) => selected.has(label)).slice(0, limit);
  return mode === "histogram" && valid.length === 0 ? defaults() : valid;
};
export const selectPlotRecords = (groups, protocols, mode = "box") => {
  const selected = new Set(resolvePlotProtocols(protocols, groups, mode));
  return groups.filter((group) => selected.has(group.label)).flatMap((group) => group.records);
};
