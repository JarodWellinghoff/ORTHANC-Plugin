import assert from "node:assert/strict";
import test from "node:test";
import {
  createInitialPlotState,
  getAllPlotGroups,
  MAX_PLOT_PROTOCOLS,
  PLOT_COLUMNS,
  PROTOCOL_GROUP_KEY,
  protocolPlotReducer,
  resolvePlotProtocols,
  selectPlotRecords,
} from "../src/dashboard/components/plots/protocolPlotState.js";

const options = Array.from({ length: 15 }, (_, index) => `Protocol ${String(index + 1).padStart(2, "0")}`);
const groups = options.map((label, index) => ({ label, records: [{ protocol_name: label, ssde: index }] }));
const configure = (state, mode, id, patch) => protocolPlotReducer(state, { type: "configure", mode, id, patch });
const namedPlots = (state) => PLOT_COLUMNS.flatMap((column) => state.columns[column.mode].map((plot, index) => ({
  ...plot,
  title: `${column.plotLabel} ${index + 1}`,
})));

test("initial layout has one histogram on the left and one box plot on the right", () => {
  const state = createInitialPlotState();
  assert.deepEqual(PLOT_COLUMNS.map((column) => column.mode), ["histogram", "box"]);
  assert.equal(state.columns.histogram.length, 1);
  assert.equal(state.columns.box.length, 1);
  assert.equal(state.columns.histogram[0].mode, "histogram");
  assert.equal(state.columns.box[0].mode, "box");
  for (const plot of namedPlots(state)) {
    assert.equal(plot.groupKey, PROTOCOL_GROUP_KEY);
    assert.equal(plot.protocols, null);
    assert.ok(plot.xKey);
  }
});

test("initial state is fresh for each workspace", () => {
  const first = createInitialPlotState();
  first.columns.histogram.push({ id: "extra" });
  assert.equal(createInitialPlotState().columns.histogram.length, 1);
});

test("columns grow independently and may contain different numbers of plots", () => {
  const original = createInitialPlotState();
  let state = protocolPlotReducer(original, { type: "add", mode: "histogram" });
  state = protocolPlotReducer(state, { type: "add", mode: "histogram" });
  assert.equal(state.columns.histogram.length, 3);
  assert.equal(state.columns.box.length, 1);
  assert.strictEqual(state.columns.box, original.columns.box);
  state = protocolPlotReducer(state, { type: "add", mode: "box" });
  assert.equal(state.columns.histogram.length, 3);
  assert.equal(state.columns.box.length, 2);
  assert.equal(original.columns.histogram.length, 1);
  assert.equal(new Set(namedPlots(state).map((plot) => plot.id)).size, 5);
});

test("the last plot in either column cannot be removed", () => {
  const state = createInitialPlotState();
  for (const mode of ["histogram", "box"]) {
    assert.strictEqual(protocolPlotReducer(state, { type: "remove", mode, id: state.columns[mode][0].id }), state);
  }
});

test("removal preserves other plot settings and never reuses an ID", () => {
  let state = protocolPlotReducer(createInitialPlotState(), { type: "add", mode: "box" });
  const id = state.columns.box[1].id;
  state = configure(state, "box", id, { xKey: "ssde", protocols: [options[12]] });
  state = protocolPlotReducer(state, { type: "remove", mode: "box", id: state.columns.box[0].id });
  assert.equal(state.columns.box[0].id, id);
  assert.equal(state.columns.box[0].xKey, "ssde");
  assert.deepEqual(state.columns.box[0].protocols, [options[12]]);
  state = protocolPlotReducer(state, { type: "add", mode: "box" });
  assert.notEqual(state.columns.box[1].id, id);
  assert.notEqual(state.columns.box[1].id, "plot-2");
});

test("each metric and protocol selection is independent; mode, group, and ID cannot be changed", () => {
  const initial = createInitialPlotState();
  const id = initial.columns.histogram[0].id;
  const state = configure(initial, "histogram", id, {
    xKey: "average_noise_level",
    protocols: [options[11]],
    mode: "scatter",
    groupKey: "scanner_model",
    id: "replacement",
  });
  const plot = state.columns.histogram[0];
  assert.equal(plot.xKey, "average_noise_level");
  assert.deepEqual(plot.protocols, [options[11]]);
  assert.equal(plot.mode, "histogram");
  assert.equal(plot.groupKey, "protocol_name");
  assert.equal(plot.id, id);
  assert.strictEqual(state.columns.box, initial.columns.box);
  assert.equal(initial.columns.histogram[0].protocols, null);
});

test("protocol cap is enforced in reducer as well as selector", () => {
  const initial = createInitialPlotState();
  const state = configure(initial, "histogram", initial.columns.histogram[0].id, { protocols: [...options, options[0]] });
  assert.equal(state.columns.histogram[0].protocols.length, MAX_PLOT_PROTOCOLS);
  assert.equal(resolvePlotProtocols(options, options).length, MAX_PLOT_PROTOCOLS);
  assert.deepEqual(resolvePlotProtocols([options[0], options[0]], options), [options[0]]);
});

test("default selection displays the first ten available protocols", () => {
  assert.deepEqual(resolvePlotProtocols(null, options), options.slice(0, 10));
  assert.deepEqual(resolvePlotProtocols(undefined, options.slice(0, 3)), options.slice(0, 3));
  assert.deepEqual(resolvePlotProtocols(null, []), []);
  assert.deepEqual(resolvePlotProtocols(null, ["A", "A", "B"]), ["A", "B"]);
});

test("any protocol can be selected, including those beyond the default ten", () => {
  const chosen = [options[14], options[12]];
  assert.deepEqual(resolvePlotProtocols(chosen, options), [options[12], options[14]]);
  assert.deepEqual(selectPlotRecords(groups, chosen), [groups[12].records[0], groups[14].records[0]]);
});

test("clearing the selector produces an empty plot, not an unlimited plot", () => {
  assert.deepEqual(resolvePlotProtocols([], options), []);
  assert.deepEqual(selectPlotRecords(groups, []), []);
});

test("new queries prune unavailable selections without replacing the remaining selections", () => {
  assert.deepEqual(resolvePlotProtocols(["Old protocol", options[0]], options), [options[0]]);
  assert.deepEqual(resolvePlotProtocols(["Old protocol"], options), []);
  assert.deepEqual(resolvePlotProtocols([options[0]], []), []);
});

test("missing-protocol labels and names containing commas are selectable", () => {
  const unusual = [
    { label: "Abdomen, pelvis", records: [{ protocol_name: "Abdomen, pelvis", ssde: 2 }] },
    { label: "Unspecified", records: [{ protocol_name: null, ssde: 3 }] },
  ];
  assert.deepEqual(selectPlotRecords(unusual, ["Abdomen, pelvis"]), unusual[0].records);
  assert.deepEqual(selectPlotRecords(unusual, ["Unspecified"]), unusual[1].records);
});

test("chart records never exceed ten protocol groups even with an oversized direct selection", () => {
  assert.equal(selectPlotRecords(groups, null).length, 10);
  assert.equal(selectPlotRecords(groups, options).length, 10);
  assert.equal(groups.length, 15);
  assert.equal(groups[14].records[0].ssde, 14);
});

test("table includes every plot and every protocol regardless of chart selection or cap", () => {
  let state = createInitialPlotState();
  state = configure(state, "histogram", state.columns.histogram[0].id, { xKey: "ssde", protocols: [] });
  state = configure(state, "box", state.columns.box[0].id, { xKey: "average_noise_level", protocols: [options[0]] });
  state = protocolPlotReducer(state, { type: "add", mode: "histogram" });
  const plots = namedPlots(state);
  const rows = getAllPlotGroups(plots, groups);
  assert.equal(rows.length, 3 * 15);
  assert.equal(new Set(rows.map((row) => row.id)).size, rows.length);
  for (const plot of plots) {
    const plotRows = rows.filter((row) => row.plotTitle === plot.title);
    assert.equal(plotRows.length, 15);
    assert.ok(plotRows.some((row) => row.protocol === options[14]));
    assert.ok(plotRows.every((row) => row.metricKey === plot.xKey));
  }
  assert.strictEqual(rows[14].records, groups[14].records);
});

test("table data follows metric changes and plot removal but not protocol selections", () => {
  let state = protocolPlotReducer(createInitialPlotState(), { type: "add", mode: "box" });
  const id = state.columns.box[1].id;
  const before = getAllPlotGroups(namedPlots(state), groups);
  state = configure(state, "box", id, { protocols: [options[14]] });
  assert.deepEqual(getAllPlotGroups(namedPlots(state), groups), before);
  state = configure(state, "box", id, { xKey: "ssde" });
  assert.ok(getAllPlotGroups(namedPlots(state), groups).filter((row) => row.plotTitle === "Box plot 2").every((row) => row.metricKey === "ssde"));
  state = protocolPlotReducer(state, { type: "remove", mode: "box", id });
  assert.equal(getAllPlotGroups(namedPlots(state), groups).length, 30);
});

test("empty cohorts have no summary rows; unrelated actions leave state unchanged", () => {
  const state = createInitialPlotState();
  assert.deepEqual(getAllPlotGroups(namedPlots(state), []), []);
  assert.strictEqual(protocolPlotReducer(state, { type: "add", mode: "scatter" }), state);
  assert.strictEqual(protocolPlotReducer(state, { type: "unknown", mode: "box" }), state);
});

test("the display limit counts protocols, not series within a protocol", () => {
  const largerGroups = groups.map((group) => ({
    ...group,
    records: [...group.records, { ...group.records[0], ssde: 99 }],
  }));
  const records = selectPlotRecords(largerGroups, null);
  assert.equal(records.length, 20);
  assert.equal(new Set(records.map((record) => record.protocol_name)).size, 10);
  assert.equal(getAllPlotGroups(namedPlots(createInitialPlotState()), largerGroups).length, 30);
});
