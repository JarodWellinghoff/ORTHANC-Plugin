import assert from "node:assert/strict";
import test from "node:test";
import { createInitialPlotState, groupProtocolRecords, normalizeProtocolName, PLOT_COLUMNS,
  protocolPlotReducer, resolvePlotProtocols, selectPlotRecords } from "../src/dashboard/components/plots/protocolPlotState.js";
const groups = Array.from({ length: 15 }, (_, index) => ({
  label: `P${String(index).padStart(2, "0")}`, records: Array.from({ length: index + 1 }, () => ({ ssde: index })),
}));
const configure = (state, mode, patch) => protocolPlotReducer(state, { type: "configure", mode, id: state.columns[mode][0].id, patch });
test("multiple-protocol reports are left; single-protocol histograms are right", () => {
  assert.deepEqual(PLOT_COLUMNS.map((item) => item.mode), ["box", "histogram"]);
});
test("standard protocols use the first token, exact WIP first tokens retain the full name", () => {
  assert.equal(normalizeProtocolName(" Chest_Adult_70 "), "Chest");
  assert.equal(normalizeProtocolName("Chest"), "Chest");
  assert.equal(normalizeProtocolName("WIP_Chest_70"), "WIP_Chest_70");
  assert.equal(normalizeProtocolName("WIP"), "WIP");
  assert.equal(normalizeProtocolName("WIPtest_Chest"), "WIPtest");
  assert.equal(normalizeProtocolName("Chest_WIP_70"), "Chest");
});
test("missing protocol values have a selectable group", () => {
  for (const value of [null, undefined, "", "  ", "_suffix"]) assert.equal(normalizeProtocolName(value), "Unspecified");
});
test("normalization merges suffix variants without modifying stored metadata", () => {
  const records = [{ protocol_name: "Chest_A" }, { protocol_name: "Chest_B" }, { protocol_name: "WIP_Chest_A" }];
  const result = groupProtocolRecords(records);
  assert.deepEqual(result.map((item) => item.label), ["Chest", "WIP_Chest_A"]);
  assert.equal(result[0].records.length, 2);
  assert.equal(records[0].protocol_name, "Chest_A");
});
test("defaults use the ten most frequent groups, then alphabetize", () => {
  assert.deepEqual(resolvePlotProtocols(null, groups), groups.slice(5).map((item) => item.label));
});
test("ties are deterministic and histograms select exactly the most frequent group", () => {
  const tied = [{ label: "Z", count: 3 }, { label: "A", count: 3 }, { label: "M", count: 20 }];
  assert.deepEqual(resolvePlotProtocols(null, tied), ["A", "M", "Z"]);
  assert.deepEqual(resolvePlotProtocols(null, tied, "histogram"), ["M"]);
  assert.deepEqual(resolvePlotProtocols(null, groups, "histogram"), ["P14"]);
});
test("histograms recover a valid selection after query changes; box plots may be cleared", () => {
  assert.deepEqual(resolvePlotProtocols(["gone"], groups, "histogram"), ["P14"]);
  assert.deepEqual(resolvePlotProtocols([], groups, "histogram"), ["P14"]);
  assert.deepEqual(resolvePlotProtocols(null, [], "histogram"), []);
  assert.deepEqual(selectPlotRecords(groups, []), []);
  assert.deepEqual(resolvePlotProtocols(["P03", "gone", "P01"], groups), ["P01", "P03"]);
});
test("selection caps apply in the reducer and do not truncate series within a protocol", () => {
  const state = createInitialPlotState();
  const labels = groups.map((item) => item.label);
  assert.equal(configure(state, "histogram", { protocols: labels }).columns.histogram[0].protocols.length, 1);
  assert.equal(configure(state, "box", { protocols: labels }).columns.box[0].protocols.length, 10);
  assert.equal(selectPlotRecords(groups, ["P14"]).length, 15);
  assert.equal(selectPlotRecords(groups, null, "histogram").length, 15);
});
test("columns add and remove independently, never reusing IDs or removing their last plot", () => {
  const original = createInitialPlotState();
  assert.strictEqual(protocolPlotReducer(original, { type: "remove", mode: "box", id: original.columns.box[0].id }), original);
  let state = protocolPlotReducer(original, { type: "add", mode: "box" });
  assert.strictEqual(state.columns.histogram, original.columns.histogram);
  const removed = state.columns.box[1].id;
  state = protocolPlotReducer(state, { type: "remove", mode: "box", id: removed });
  state = protocolPlotReducer(state, { type: "add", mode: "box" });
  assert.notEqual(state.columns.box[1].id, removed);
  assert.equal(original.columns.box.length, 1);
});
test("configuration cannot alter identity/mode/group or reintroduce plot names", () => {
  const state = createInitialPlotState();
  const plot = configure(state, "box", { id: "bad", mode: "histogram", groupKey: "scanner", name: "hidden", xKey: "ssde" }).columns.box[0];
  assert.equal(plot.id, state.columns.box[0].id);
  assert.equal(plot.mode, "box"); assert.equal(plot.groupKey, "protocol_name");
  assert.equal(plot.name, undefined); assert.equal(plot.xKey, "ssde");
  assert.strictEqual(protocolPlotReducer(state, { type: "add", mode: "scatter" }), state);
});
