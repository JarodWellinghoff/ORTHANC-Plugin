// Patient count is a cohort aggregate, never a fabricated per-series value.
export const PATIENT_COUNT_KEY = "number_of_patients";
export const getReportMetrics = (metrics) => [
  ...metrics.filter((metric) => !["image_count", "processing_time"].includes(metric.key)).map((metric) => {
    if (metric.key === "dlp") return { ...metric, label: "DLP", axisLabel: "DLP (mGy\u00b7cm)" };
    if (metric.key === "dlp_ssde") return { ...metric, label: "DLP_SSDE", axisLabel: "DLP_SSDE (mGy\u00b7cm)" };
    return metric;
  }),
  { key: PATIENT_COUNT_KEY, label: "Number of patients", axisLabel: "Number of patients", unit: "", decimals: 0 },
];
export const finiteNumber = (value) => {
  if (typeof value !== "number" && typeof value !== "string") return null;
  if (typeof value === "string" && !value.trim()) return null;
  return Number.isFinite(Number(value)) ? Number(value) : null;
};
export const countPatients = (records) => {
  const ids = new Set();
  let missing = 0;
  for (const record of records) {
    const id = record.patient_id == null ? "" : String(record.patient_id).trim();
    if (!id) { missing += 1; continue; }
    ids.add(JSON.stringify([String(record.issuer_of_patient_id ?? "").trim(), id]));
  }
  return { count: ids.size, missing };
};
export const describeReportValues = (values) => {
  const clean = values.map(finiteNumber).filter((value) => value !== null).sort((a, b) => a - b);
  const n = clean.length;
  if (!n) return { n: 0, mean: null, sd: null, median: null, min: null, max: null };
  const mean = clean.reduce((sum, value) => sum + value, 0) / n;
  const sd = n > 1 ? Math.sqrt(clean.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1)) : null;
  const mid = Math.floor(n / 2);
  return { n, mean, sd, median: n % 2 ? clean[mid] : (clean[mid - 1] + clean[mid]) / 2, min: clean[0], max: clean[n - 1] };
};
// Preserve the removed table's all-protocol scope, independent of plot selection.
export const buildReportRows = (plots, groups, metrics) => {
  const byKey = new Map(metrics.map((metric) => [metric.key, metric]));
  return plots.flatMap((plot) => {
    const metric = byKey.get(plot.xKey);
    if (!metric) return [];
    return groups.map((group) => {
      const patients = countPatients(group.records);
      const aggregate = metric.key === PATIENT_COUNT_KEY;
      const stats = describeReportValues(aggregate ? [] : group.records.map((record) => record[metric.key]));
      return { report: plot.mode === "box" ? "Multiple protocols" : "Single protocol",
        metric: metric.label, unit: metric.unit, protocol: group.label,
        seriesCount: group.records.length, patientCount: patients.count, missingPatientIds: patients.missing,
        value: aggregate ? patients.count : null, ...stats, n: aggregate ? null : stats.n };
    });
  });
};
const CSV_COLUMNS = [
  ["report", "Report"], ["metric", "Metric"], ["unit", "Unit"], ["protocol", "Protocol"],
  ["seriesCount", "Series count"], ["patientCount", "Number of patients"],
  ["missingPatientIds", "Series without patient ID"], ["value", "Aggregate value"],
  ["n", "n"], ["mean", "Mean"], ["sd", "SD"], ["median", "Median"], ["min", "Min"], ["max", "Max"],
];
export const csvCell = (value) => {
  if (value == null) return "";
  const text = typeof value === "string" && /^[\s]*[=+\-@\t\r\n]/.test(value) ? `'${value}` : String(value);
  return `"${text.replace(/"/g, '""')}"`;
};
export const reportRowsToCsv = (rows) => '\uFEFF' + [
  CSV_COLUMNS.map(([, label]) => csvCell(label)).join(","),
  ...rows.map((row) => CSV_COLUMNS.map(([key]) => csvCell(row[key])).join(",")),
].join("\r\n") + "\r\n";
export const buildProtocolTraces = (groups, metric, mode) => {
  if (metric.key === PATIENT_COUNT_KEY) return [{ type: "bar",
    x: groups.map((group) => group.label), y: groups.map((group) => countPatients(group.records).count),
    hovertemplate: "%{x}<br>Number of patients: %{y}<extra></extra>" }];
  return groups.map((group) => {
    const values = group.records.map((record) => finiteNumber(record[metric.key])).filter((value) => value !== null);
    return mode === "histogram" ? { type: "histogram", name: group.label, x: values }
      : { type: "box", name: group.label, y: values, boxpoints: "outliers", marker: { size: 5 } };
  });
};
