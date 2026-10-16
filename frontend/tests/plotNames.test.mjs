// Per-plot names were removed. Test the replacement report/CSV contract.
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { buildProtocolTraces, buildReportRows, countPatients, csvCell, describeReportValues,
  finiteNumber, getReportMetrics, PATIENT_COUNT_KEY, reportRowsToCsv } from "../src/dashboard/components/plots/protocolReport.js";
import { withScrollableTables } from "../src/dashboard/theme/customizations/scrollableTables.js";
const metrics = getReportMetrics([{ key: "ssde", label: "SSDE", unit: "mGy", decimals: 2 },
  { key: "dlp", label: "DLP (CTDIvol)" }, { key: "dlp_ssde", label: "DLP (SSDE)" }, { key: "image_count" }, { key: "processing_time" }]);
test("metrics add patients, remove timing/images, and rename both DLP labels", () => {
  assert.deepEqual(metrics.map((item) => item.key), ["ssde", "dlp", "dlp_ssde", PATIENT_COUNT_KEY]);
  assert.equal(metrics[1].label, "DLP"); assert.equal(metrics[2].label, "DLP_SSDE");
});
test("patients are distinct identifiers, not series or patient names", () => {
  assert.deepEqual(countPatients([{ patient_id: "01" }, { patient_id: "01" }, { patient_id: "02" }, { patient_name: "same" }]), { count: 2, missing: 1 });
});
test("patient identifiers preserve zero and distinguish issuers", () => {
  assert.deepEqual(countPatients([{ patient_id: 0 }, { patient_id: "0" }, { patient_id: " x " },
    { patient_id: "x", issuer_of_patient_id: "other" }, { patient_id: " " }]), { count: 3, missing: 1 });
});
test("finite numbers exclude missing and invalid values, without losing real zero", () => {
  for (const value of [null, undefined, true, false, [], "", " ", NaN, Infinity]) assert.equal(finiteNumber(value), null);
  assert.equal(finiteNumber(0), 0); assert.equal(finiteNumber("2.5"), 2.5);
});
test("descriptive statistics use sample SD and handle empty/singleton distributions", () => {
  assert.deepEqual(describeReportValues([1, 2, 3, null]), { n: 3, mean: 2, sd: 1, median: 2, min: 1, max: 3 });
  assert.equal(describeReportValues([2]).sd, null);
  assert.equal(describeReportValues([]).n, 0);
});
test("patient counts use honest aggregate bars rather than invented distributions", () => {
  const traces = buildProtocolTraces([{ label: "A", records: [{ patient_id: "P" }, { patient_id: "P" }] }], { key: PATIENT_COUNT_KEY }, "box");
  assert.equal(traces[0].type, "bar"); assert.deepEqual(traces[0].y, [1]);
});
test("numeric box plots and histograms preserve every finite measurement", () => {
  const groups = [{ label: "A", records: [{ ssde: 0 }, { ssde: 2 }, { ssde: null }] }];
  assert.deepEqual(buildProtocolTraces(groups, metrics[0], "histogram")[0].x, [0, 2]);
  assert.deepEqual(buildProtocolTraces(groups, metrics[0], "box")[0].y, [0, 2]);
});
test("CSV retains all protocols from the removed table independent of selections", () => {
  const groups = [{ label: "A", records: [{ ssde: 2, patient_id: "P" }] }, { label: "B", records: [{ ssde: 4, patient_id: "Q" }] }];
  const rows = buildReportRows([{ mode: "box", xKey: "ssde", protocols: ["A"] }], groups, metrics);
  assert.equal(rows.length, 2); assert.equal(rows[1].mean, 4); assert.equal(rows[0].patientCount, 1);
});
test("patient-count export does not fabricate per-series summary statistics", () => {
  const rows = buildReportRows([{ mode: "box", xKey: PATIENT_COUNT_KEY }], [{ label: "A", records: [{ patient_id: "P" }] }], metrics);
  assert.equal(rows[0].value, 1); assert.equal(rows[0].n, null); assert.equal(rows[0].mean, null);
});
test("CSV handles quotes, newlines, formula-leading text and numeric negatives", () => {
  assert.equal(csvCell('A,"B"\nC'), '"A,""B""\nC"');
  assert.equal(csvCell("=1+1"), '"\'=1+1"');
  assert.equal(csvCell(-2), '"-2"');
  assert.ok(reportRowsToCsv([]).startsWith('\uFEFF"Report"'));
  assert.ok(reportRowsToCsv([]).endsWith("\r\n"));
});
test("workspace exports CSV without named plots or a bottom table", () => {
  const base = new URL("../src/dashboard/components/plots/", import.meta.url);
  const workspace = readFileSync(new URL("PlottingWorkspace.jsx", base), "utf8");
  const card = readFileSync(new URL("ProtocolPlotCard.jsx", base), "utf8");
  assert.match(workspace, /Export CSV/); assert.doesNotMatch(workspace, /ProtocolPlotStatsTable/);
  assert.doesNotMatch(card, /label=['"]Plot name/); assert.match(card, /multiple=\{multiple\}/);
});
test("scrollable table theme retains existing customizations and bounds all viewports", () => {
  const result = withScrollableTables({ MuiDataGrid: { styleOverrides: { root: () => ({ borderWidth: 3 }) } } });
  assert.equal(result.MuiDataGrid.defaultProps.autoHeight, false);
  const root = result.MuiDataGrid.styleOverrides.root({});
  assert.equal(root.borderWidth, 3); assert.equal(root.height, 560); assert.equal(root.maxHeight, 560);
  assert.equal(result.MuiTableContainer.styleOverrides.root.overflow, "auto");
  assert.equal(result.MuiTable.defaultProps.stickyHeader, true);
});
