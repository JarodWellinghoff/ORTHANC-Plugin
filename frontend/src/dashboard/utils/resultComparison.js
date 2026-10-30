const asText = (value) => {
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "";
};

// Metric definitions contain plain text OR { pre, main, post, decimals }.
// Return only renderable text; formatting metadata is never a React child.
export function metricTextParts(value) {
  const definition = value && typeof value === "object" && !Array.isArray(value)
    ? value : { main: value };
  return [
    { key: "pre-sup", tag: "sup", text: asText(definition.pre?.sup) },
    { key: "pre-sub", tag: "sub", text: asText(definition.pre?.sub) },
    { key: "main", tag: null, text: asText(definition.main) },
    { key: "post-sup", tag: "sup", text: asText(definition.post?.sup) },
    { key: "post-sub", tag: "sub", text: asText(definition.post?.sub) },
  ].filter((part) => part.text !== "");
}

const numeric = (value) => {
  if (typeof value !== "number" && typeof value !== "string") return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
};

const deltaText = (value, decimals) => {
  // Round before deciding the sign so a near-zero difference is not -0.00.
  const rounded = Number(value.toFixed(decimals)) || 0;
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(decimals)}`;
};

export function buildComparisonRows(baseline, current, metrics) {
  const before = baseline?.results ?? baseline;
  const after = current?.results ?? current;
  return metrics.flatMap(({ key, label, unit, decimals = 2 }) => {
    const precision = unit?.decimals ?? decimals;
    const oldValue = numeric(before?.[key]);
    const newValue = numeric(after?.[key]);
    if (oldValue === null && newValue === null) return [];
    return [{
      key,
      label,
      unit,
      baseline: oldValue === null ? "N/A" : oldValue.toFixed(precision),
      current: newValue === null ? "N/A" : newValue.toFixed(precision),
      delta: oldValue === null || newValue === null
        ? "N/A" : deltaText(newValue - oldValue, precision),
    }];
  });
}
