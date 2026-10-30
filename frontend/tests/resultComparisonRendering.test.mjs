import test from "node:test";
import assert from "node:assert/strict";
import { buildComparisonRows, metricTextParts } from "../src/dashboard/utils/resultComparison.js";

const metrics = [
  { key: "peak_frequency", label: { main: "NPS Peak Frequency" }, unit: { main: "cm", post: { sup: "-1" }, decimals: 3 } },
  { key: "dw_avg", label: { main: "Average D", post: { sub: "w" } }, unit: { main: "cm" } },
  { key: "detectability", label: { main: "Detectability" }, unit: { decimals: 3 } },
];

test("structured labels and units produce text with subscript/superscript descriptors", () => {
  assert.deepEqual(metricTextParts(metrics[0].label), [{ key: "main", tag: null, text: "NPS Peak Frequency" }]);
  assert.deepEqual(metricTextParts(metrics[0].unit), [
    { key: "main", tag: null, text: "cm" },
    { key: "post-sup", tag: "sup", text: "-1" },
  ]);
  assert.deepEqual(metricTextParts(metrics[1].label), [
    { key: "main", tag: null, text: "Average D" },
    { key: "post-sub", tag: "sub", text: "w" },
  ]);
});

test("legacy strings, zero, and pre/post formatting remain renderable", () => {
  assert.equal(metricTextParts("mGy")[0].text, "mGy");
  assert.equal(metricTextParts(0)[0].text, "0");
  const parts = metricTextParts({ pre: { sup: 0, sub: "a" }, main: "X", post: { sup: 2, sub: "b" } });
  assert.deepEqual(parts.map(({ text }) => text), ["0", "a", "X", "2", "b"]);
  assert.equal(new Set(parts.map(({ key }) => key)).size, parts.length);
});

test("formatting-only or malformed metadata never becomes an object React child", () => {
  for (const value of [null, undefined, false, [], { decimals: 3 }, { main: { unexpected: true } }]) {
    assert.deepEqual(metricTextParts(value), []);
  }
  for (const metric of metrics) {
    for (const part of [...metricTextParts(metric.label), ...metricTextParts(metric.unit)]) {
      assert.equal(typeof part.text, "string");
    }
  }
});

test("comparison uses metric IDs instead of object labels as row keys", () => {
  const values = { peak_frequency: 1, dw_avg: 2, detectability: 3 };
  const rows = buildComparisonRows(values, { results: values }, metrics);
  assert.deepEqual(rows.map(({ key }) => key), ["peak_frequency", "dw_avg", "detectability"]);
  assert.equal(new Set(rows.map(({ key }) => key)).size, rows.length);
});

test("precision comes from unit.decimals, including unitless measurements", () => {
  const rows = buildComparisonRows(
    { peak_frequency: 1.1234, dw_avg: 20, detectability: 0 },
    { results: { peak_frequency: 1.2234, dw_avg: 22.123, detectability: 1.23456 } },
    metrics,
  );
  assert.deepEqual(rows.map(({ baseline, current, delta }) => [baseline, current, delta]), [
    ["1.123", "1.223", "+0.100"], ["20.00", "22.12", "+2.12"], ["0.000", "1.235", "+1.235"],
  ]);
});

test("missing/invalid metrics are not converted into zero measurements", () => {
  for (const value of [null, undefined, "", "  ", NaN, Infinity, -Infinity, true, [], {}]) {
    assert.deepEqual(buildComparisonRows({ dw_avg: value }, { dw_avg: value }, metrics), []);
    const [row] = buildComparisonRows({ dw_avg: value }, { dw_avg: "0" }, metrics);
    assert.equal(row.baseline, "N/A");
    assert.equal(row.current, "0.00");
    assert.equal(row.delta, "N/A");
  }
});

test("near-zero deltas do not display negative zero", () => {
  const [row] = buildComparisonRows({ dw_avg: 1.0001 }, { dw_avg: 1 }, metrics);
  assert.equal(row.delta, "0.00");
});
