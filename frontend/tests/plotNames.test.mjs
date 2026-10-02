import assert from "node:assert/strict";
import test from "node:test";
import {
  createInitialPlotState,
  getAllPlotGroups,
  getPlotTitle,
  MAX_PLOT_NAME_LENGTH,
  normalizePlotName,
  PLOT_COLUMNS,
  protocolPlotReducer,
} from "../src/dashboard/components/plots/protocolPlotState.js";

const configure = (state, mode, id, patch) =>
  protocolPlotReducer(state, { type: "configure", mode, id, patch });
const namedPlots = (state) => PLOT_COLUMNS.flatMap((column) =>
  state.columns[column.mode].map((plot, index) => ({
    ...plot,
    title: getPlotTitle(plot, index),
  })),
);

test("new and legacy plots retain their default titles until renamed", () => {
  const state = createInitialPlotState();
  assert.deepEqual(namedPlots(state).map((plot) => plot.title), ["Histogram 1", "Box plot 1"]);
  assert.ok(namedPlots(state).every((plot) => plot.name === ""));
  assert.equal(getPlotTitle({ mode: "histogram" }, 2), "Histogram 3");
});

test("renaming one plot preserves other plots, identity, mode, and metric", () => {
  const original = createInitialPlotState();
  const plot = original.columns.histogram[0];
  const state = configure(original, "histogram", plot.id, { name: "  Dose distribution  " });
  const renamed = state.columns.histogram[0];
  assert.equal(renamed.name, "Dose distribution");
  assert.equal(getPlotTitle(renamed, 0), "Dose distribution");
  assert.deepEqual({ ...renamed, name: "" }, plot);
  assert.strictEqual(state.columns.box, original.columns.box);
  assert.equal(original.columns.histogram[0].name, "");
});

test("names survive metric and protocol changes and independent additions", () => {
  let state = createInitialPlotState();
  const id = state.columns.box[0].id;
  state = configure(state, "box", id, { name: "Noise comparison" });
  state = configure(state, "box", id, { xKey: "average_noise_level", protocols: ["Chest"] });
  state = protocolPlotReducer(state, { type: "add", mode: "histogram" });
  state = protocolPlotReducer(state, { type: "add", mode: "box" });
  assert.equal(getPlotTitle(state.columns.box[0], 0), "Noise comparison");
  assert.deepEqual(state.columns.box[0].protocols, ["Chest"]);
  assert.equal(state.columns.box[0].xKey, "average_noise_level");
  assert.equal(getPlotTitle(state.columns.box[1], 1), "Box plot 2");
  assert.equal(state.columns.histogram.length, 2);
});

test("custom names remain attached to stable IDs after a neighboring removal", () => {
  let state = protocolPlotReducer(createInitialPlotState(), { type: "add", mode: "histogram" });
  const [first, second] = state.columns.histogram;
  state = configure(state, "histogram", second.id, { name: "Follow-up scans" });
  state = protocolPlotReducer(state, { type: "remove", mode: "histogram", id: first.id });
  assert.equal(state.columns.histogram[0].id, second.id);
  assert.equal(getPlotTitle(state.columns.histogram[0], 0), "Follow-up scans");
});

test("empty or whitespace-only names restore the appropriate default title", () => {
  for (const mode of ["histogram", "box"]) {
    let state = createInitialPlotState();
    const id = state.columns[mode][0].id;
    state = configure(state, mode, id, { name: "Custom" });
    state = configure(state, mode, id, { name: " \t\n " });
    assert.equal(state.columns[mode][0].name, "");
    assert.equal(getPlotTitle(state.columns[mode][0], 0), mode === "box" ? "Box plot 1" : "Histogram 1");
  }
});

test("committed names are single-line, bounded, and reject non-string patches", () => {
  assert.equal(normalizePlotName("  Chest\r\nCT\tcomparison  "), "Chest CT comparison");
  assert.equal(normalizePlotName("N".repeat(500)).length, MAX_PLOT_NAME_LENGTH);
  assert.equal(normalizePlotName(null), "");
  assert.equal(normalizePlotName({}), "");
  let state = createInitialPlotState();
  const id = state.columns.histogram[0].id;
  state = configure(state, "histogram", id, { name: "N".repeat(500) });
  assert.equal(state.columns.histogram[0].name.length, MAX_PLOT_NAME_LENGTH);
  for (const name of [null, 4, {}, [], undefined]) {
    state = configure(state, "histogram", id, { name });
    assert.equal(state.columns.histogram[0].name.length, MAX_PLOT_NAME_LENGTH);
  }
});

test("renamed table rows retain every protocol and their stable row IDs", () => {
  const groups = Array.from({ length: 15 }, (_, index) => ({
    label: `Protocol ${index + 1}`,
    records: [{ ssde: index + 1 }],
  }));
  const original = createInitialPlotState();
  const before = getAllPlotGroups(namedPlots(original), groups);
  const state = configure(original, "histogram", original.columns.histogram[0].id, {
    name: "SSDE distribution", protocols: [],
  });
  const after = getAllPlotGroups(namedPlots(state), groups);
  assert.equal(after.length, 30);
  assert.deepEqual(after.map((row) => row.id), before.map((row) => row.id));
  assert.equal(after.filter((row) => row.plotTitle === "SSDE distribution").length, 15);
  assert.deepEqual(after.map((row) => row.records), before.map((row) => row.records));
  assert.deepEqual(after.map((row) => row.metricKey), before.map((row) => row.metricKey));
});

test("duplicate display names do not collide with plot or table row identities", () => {
  let state = createInitialPlotState();
  for (const mode of ["histogram", "box"]) {
    state = configure(state, mode, state.columns[mode][0].id, { name: "Comparison" });
  }
  const plots = namedPlots(state);
  assert.deepEqual(plots.map((plot) => plot.title), ["Comparison", "Comparison"]);
  assert.equal(new Set(plots.map((plot) => plot.id)).size, 2);
  const rows = getAllPlotGroups(plots, [{ label: "Chest", records: [] }]);
  assert.equal(new Set(rows.map((row) => row.id)).size, 2);
});

test("display names remain plain text, separate from protocol and metric data", () => {
  const original = createInitialPlotState();
  const name = "CT <low-dose> & follow-up";
  const state = configure(original, "box", original.columns.box[0].id, { name });
  assert.equal(getPlotTitle(state.columns.box[0], 0), name);
  assert.equal(state.columns.box[0].groupKey, "protocol_name");
  assert.equal(state.columns.box[0].xKey, original.columns.box[0].xKey);
});
