// ─────────────────────────────────────────────────────────────────────────────
// aggregateMetrics.js
//
// Support module for the "Plot Selected" action on ResultsPage. Splits the
// stored analysis record into the scalar (per-series) columns that can be
// aggregated across a cohort, and the per-slice / per-frequency array columns
// that can't.
//
// The grid rows on ResultsPage carry metadata only (see `normalizeChoRow` in
// choResultsShared.js) — no metrics — so anything that plots values has to go
// back to `/cho-results/{series_instance_uid}` for each selected series. That
// is the same key `handleExportSelected` posts to `/cho-export-results`; note
// it is the SeriesInstanceUID, not the Orthanc UUID.
//
// The metric catalog below mirrors RESULT_METRICS in
// metadata/metadataSections.utils.js. It is kept separate because that catalog
// carries a nested label shape ({ main, post: { sub } }) built for JSX
// rendering, while Plotly axis titles need flat strings with HTML markup.
// ─────────────────────────────────────────────────────────────────────────────

import { fetchJson } from "./choResultsShared";

// Columns on analysis.results (plus a few from dicom.series) that hold one
// value per series. Everything not listed here — ctdivol, dw, location,
// location_sparse, noise_level, nps, cho_detectability, spatial_frequency,
// ssde_inc — is an array and belongs on the per-series plots in ChoPlots.
export const SCALAR_METRICS = [
  {
    key: "average_index_of_detectability",
    label: "Detectability index (d′)",
    axisLabel: "Detectability index (d′)",
    unit: "",
    decimals: 3,
  },
  {
    key: "average_noise_level",
    label: "Average noise level",
    axisLabel: "Average noise level (HU)",
    unit: "HU",
    decimals: 2,
  },
  {
    key: "ctdivol_avg",
    label: "Average CTDIvol",
    axisLabel: "Average CTDI<sub>vol</sub> (mGy)",
    unit: "mGy",
    decimals: 2,
  },
  {
    key: "ssde",
    label: "Average SSDE",
    axisLabel: "Average SSDE (mGy)",
    unit: "mGy",
    decimals: 2,
  },
  {
    key: "dw_avg",
    label: "Average Dw",
    axisLabel: "Average D<sub>w</sub> (cm)",
    unit: "cm",
    decimals: 2,
  },
  {
    key: "dlp",
    label: "DLP (CTDIvol)",
    axisLabel: "DLP<sub>CTDIvol</sub> (mGy·cm)",
    unit: "mGy·cm",
    decimals: 1,
  },
  {
    key: "dlp_ssde",
    label: "DLP (SSDE)",
    axisLabel: "DLP<sub>SSDE</sub> (mGy·cm)",
    unit: "mGy·cm",
    decimals: 1,
  },
  {
    key: "peak_frequency",
    label: "NPS peak frequency",
    axisLabel: "NPS peak frequency (cm<sup>-1</sup>)",
    unit: "cm⁻¹",
    decimals: 3,
  },
  {
    key: "average_frequency",
    label: "NPS average frequency",
    axisLabel: "NPS average frequency (cm<sup>-1</sup>)",
    unit: "cm⁻¹",
    decimals: 3,
  },
  {
    key: "percent_10_frequency",
    label: "NPS 10% peak frequency",
    axisLabel: "NPS 10% peak frequency (cm<sup>-1</sup>)",
    unit: "cm⁻¹",
    decimals: 3,
  },
  {
    key: "spatial_resolution",
    label: "Spatial resolution",
    axisLabel: "Spatial resolution (cm<sup>-1</sup>)",
    unit: "cm⁻¹",
    decimals: 3,
  },
  {
    key: "scan_length_cm",
    label: "Scan length",
    axisLabel: "Scan length (cm)",
    unit: "cm",
    decimals: 1,
  },
  {
    key: "image_count",
    label: "Image count",
    axisLabel: "Image count",
    unit: "",
    decimals: 0,
  },
  {
    key: "processing_time",
    label: "Processing time",
    axisLabel: "Processing time (s)",
    unit: "s",
    decimals: 1,
  },
];

export const SCALAR_METRIC_MAP = new Map(
  SCALAR_METRICS.map((metric) => [metric.key, metric]),
);

// Categorical columns from the metadata joins in `cho_storage.get_result`,
// offered as the "Group by" axis so a cohort can be split by acquisition
// conditions rather than treated as one undifferentiated cloud.
export const GROUP_FIELDS = [
  { key: "__none", label: "No grouping" },
  { key: "protocol_name", label: "Protocol" },
  { key: "scanner_model", label: "Scanner model" },
  { key: "station_name", label: "Scanner station" },
  { key: "institution_name", label: "Institute" },
  { key: "manufacturer", label: "Manufacturer" },
  { key: "body_part_examined", label: "Body part" },
  { key: "convolution_kernel", label: "Reconstruction kernel" },
];

export const UNGROUPED_KEY = "__none";
export const UNGROUPED_LABEL = "All series";
export const MISSING_GROUP_LABEL = "Unspecified";

// Colorblind-safe qualitative palette (Okabe–Ito). Fixed rather than derived
// from the MUI palette because group counts routinely exceed the number of
// distinct semantic colors the theme defines, and because these plots end up
// in figures where category color has to survive grayscale printing.
export const GROUP_PALETTE = [
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#D55E00",
  "#CC79A7",
  "#56B4E9",
  "#B07AA1",
  "#8C8C00",
];

// ── Value coercion ───────────────────────────────────────────────────────────

/**
 * Numeric coercion that rejects everything a plot can't use: nulls, blanks,
 * NaN, Infinity, and arrays (which is how an array column would present itself
 * if one ever got added to SCALAR_METRICS by mistake).
 */
export const toFiniteNumber = (value) => {
  if (value === null || value === undefined || value === "") return null;
  if (Array.isArray(value)) return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const formatValue = (value, decimals = 2) => {
  const parsed = toFiniteNumber(value);
  return parsed === null ? "—" : parsed.toFixed(decimals);
};

// ── Descriptive statistics ───────────────────────────────────────────────────

/**
 * Summary stats over an array of raw values. Non-numeric entries are dropped
 * rather than coerced to zero, so `n` always reports how many series actually
 * contributed — which is the number that matters when a cohort mixes global
 * noise and full analyses.
 *
 * `sd` is the sample standard deviation (n−1 denominator) and is null for
 * n < 2.
 */
export const describe = (values) => {
  const clean = values.map(toFiniteNumber).filter((v) => v !== null);
  const n = clean.length;
  if (n === 0) {
    return { n: 0, mean: null, sd: null, median: null, min: null, max: null };
  }

  const mean = clean.reduce((sum, v) => sum + v, 0) / n;
  const sd =
    n > 1
      ? Math.sqrt(clean.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1))
      : null;

  const sorted = [...clean].sort((a, b) => a - b);
  const mid = Math.floor(n / 2);
  const median =
    n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  return { n, mean, sd, median, min: sorted[0], max: sorted[n - 1] };
};

/**
 * Pearson correlation over paired values. Pairs where either side is
 * non-numeric are dropped as a unit so x and y stay aligned.
 */
export const pearson = (xs, ys) => {
  const pairs = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    const x = toFiniteNumber(xs[i]);
    const y = toFiniteNumber(ys[i]);
    if (x !== null && y !== null) pairs.push([x, y]);
  }
  const n = pairs.length;
  if (n < 3) return { r: null, n };

  const meanX = pairs.reduce((s, p) => s + p[0], 0) / n;
  const meanY = pairs.reduce((s, p) => s + p[1], 0) / n;

  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (const [x, y] of pairs) {
    cov += (x - meanX) * (y - meanY);
    varX += (x - meanX) ** 2;
    varY += (y - meanY) ** 2;
  }
  if (varX === 0 || varY === 0) return { r: null, n };
  return { r: cov / Math.sqrt(varX * varY), n };
};

/**
 * Ordinary least squares fit, returned as the two endpoints of the line over
 * the observed x range so it can be dropped straight into a Plotly trace.
 */
export const linearFit = (xs, ys) => {
  const pairs = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    const x = toFiniteNumber(xs[i]);
    const y = toFiniteNumber(ys[i]);
    if (x !== null && y !== null) pairs.push([x, y]);
  }
  const n = pairs.length;
  if (n < 2) return null;

  const meanX = pairs.reduce((s, p) => s + p[0], 0) / n;
  const meanY = pairs.reduce((s, p) => s + p[1], 0) / n;

  let cov = 0;
  let varX = 0;
  for (const [x, y] of pairs) {
    cov += (x - meanX) * (y - meanY);
    varX += (x - meanX) ** 2;
  }
  if (varX === 0) return null;

  const slope = cov / varX;
  const intercept = meanY - slope * meanX;
  const sortedX = pairs.map((p) => p[0]).sort((a, b) => a - b);
  const x0 = sortedX[0];
  const x1 = sortedX[n - 1];

  return {
    slope,
    intercept,
    n,
    x: [x0, x1],
    y: [intercept + slope * x0, intercept + slope * x1],
  };
};

// ── Fetching ─────────────────────────────────────────────────────────────────

/**
 * Run `worker` over `items` with at most `limit` in flight. Keeps a 200-series
 * selection from opening 200 simultaneous connections to the plugin backend.
 */
export const mapWithConcurrency = async (items, limit, worker) => {
  const width = Math.max(1, Math.min(limit, items.length));
  let cursor = 0;

  const runners = Array.from({ length: width }, async () => {
    for (;;) {
      const index = cursor;
      cursor += 1;
      if (index >= items.length) return;
      await worker(items[index], index);
    }
  });

  await Promise.all(runners);
};

/**
 * `/cho-results/{id}` has returned a bare record historically; tolerate a
 * `{ data: ... }` wrapper in case that ever changes, and reject anything that
 * isn't a plain object.
 */
export const unwrapResultRecord = (payload) => {
  const record = payload?.data ?? payload;
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    return null;
  }
  return record;
};

const buildPointLabel = (record, row) => {
  const patient = record.patient_name ?? row?.patientName ?? "Anonymous";
  const protocol = record.protocol_name ?? row?.protocolName ?? "No protocol";
  return `${patient} — ${protocol}`;
};

/**
 * Fetch stored results for each selected row, one request per series.
 *
 * Partial-success semantics, matching the bulk export: a single failed fetch is
 * collected into `failures` and the remaining series are still returned, so one
 * bad record can't blank out the whole plot.
 *
 * Returns `{ records, failures }` where each record is the stored row plus two
 * internal fields — `__rowId` (grid row id, for de-duplication) and `__label`
 * (hover text).
 */
export const fetchAggregateRecords = async (rows, options = {}) => {
  const { concurrency = 6, signal } = options;
  const records = [];
  const failures = [];

  await mapWithConcurrency(rows, concurrency, async (row) => {
    if (signal?.aborted) return;

    // SeriesInstanceUID, not the Orthanc UUID — `/cho-results/{id}` keys on
    // series_instance_uid. `seriesUuid` is a last-resort fallback for older
    // summary items that predate the field.
    const seriesId = row.seriesInstanceUid ?? row.seriesUuid;
    if (!seriesId) {
      failures.push({ row, reason: "No SeriesInstanceUID on this row" });
      return;
    }

    try {
      const payload = await fetchJson(
        `/cho-results/${encodeURIComponent(seriesId)}`,
      );
      const record = unwrapResultRecord(payload);
      if (!record) {
        failures.push({ row, reason: "Empty result record" });
        return;
      }
      records.push({
        ...record,
        __rowId: row.id,
        __label: buildPointLabel(record, row),
      });
    } catch (error) {
      failures.push({ row, reason: error.message ?? "Request failed" });
    }
  });

  return { records, failures };
};

// ── Grouping ─────────────────────────────────────────────────────────────────

/**
 * Bucket records by a categorical field. `UNGROUPED_KEY` collapses everything
 * into a single series so the same downstream code renders both cases.
 */
export const groupRecords = (records, groupKey) => {
  if (!groupKey || groupKey === UNGROUPED_KEY) {
    return [{ label: UNGROUPED_LABEL, records }];
  }

  const buckets = new Map();
  for (const record of records) {
    const raw = record[groupKey];
    const label =
      raw === null || raw === undefined || raw === ""
        ? MISSING_GROUP_LABEL
        : String(raw);
    if (!buckets.has(label)) buckets.set(label, []);
    buckets.get(label).push(record);
  }

  return [...buckets.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([label, groupedRecords]) => ({ label, records: groupedRecords }));
};

// ── CSV ──────────────────────────────────────────────────────────────────────

const csvCell = (value) => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

/**
 * Flat CSV of the aggregated cohort: identifying metadata, every grouping
 * field, and every scalar metric — one row per series.
 *
 * This complements the existing XLS export rather than replacing it. That
 * export writes one sheet per series including the array columns; this is the
 * cross-series table the plots are drawn from.
 */
export const buildAggregateCsv = (records) => {
  const idColumns = ["patient_id", "patient_name", "series_instance_uid"];
  const groupColumns = GROUP_FIELDS.filter(
    (field) => field.key !== UNGROUPED_KEY,
  ).map((field) => field.key);
  const metricColumns = SCALAR_METRICS.map((metric) => metric.key);
  const columns = [...idColumns, ...groupColumns, ...metricColumns];

  const header = columns.join(",");
  const lines = records.map((record) =>
    columns.map((column) => csvCell(record[column])).join(","),
  );

  return [header, ...lines].join("\n");
};
