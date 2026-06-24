/*
 * Headless fetch-order assertions for the progressive pack loader (apps/echoes-viewer/loader.js).
 * GPU-free: stubs `fetchArray` to record order and inject a failure, then asserts the core-before-
 * refinement contract and per-layer error isolation.
 *
 *   node tests/test_loader_fetch_order.js
 */
const assert = require("assert");
const { isPackManifest, planLayerTasks, PackLoader } = require("../apps/echoes-viewer/loader.js");

function desc(file) { return { file, dtype: "<f4", count: 1, bytes: 4, sha256: "x" }; }

const manifest = {
  schema_version: "echoes.pack.v1",
  layers: [
    // intentionally out of tier order to prove the loader reorders core-first
    { id: "completed-faint", tier: "refinement", count: 1, color: "#3a78ff",
      columns: { xyz: desc("refinement/faint_xyz"), value: desc("refinement/faint_val") } },
    { id: "observed", tier: "core", count: 1, color: "#9aa0a6",
      columns: { xyz: desc("core/obs_xyz"), value: desc("core/obs_val") } },
    { id: "completed-zoa", tier: "refinement", count: 1, color: "#ff3b30",
      columns: { xyz: desc("refinement/zoa_xyz"), value: desc("refinement/zoa_val") } },
  ],
};

(async function () {
  assert.ok(isPackManifest(manifest), "recognised as a pack manifest");
  const tasks = planLayerTasks(manifest);
  assert.deepStrictEqual(tasks.map((t) => t.layerId),
    ["observed", "completed-faint", "completed-zoa"], "core layer planned first");

  // ---- order contract: every core column fetched before any refinement column ----
  const order = [];
  const fetchArray = async (d) => { order.push(d.file); return new Float32Array([0]); };
  const appended = [];
  const loader = new PackLoader({ manifest, fetchArray, onLayer: (l) => appended.push(l.id) });
  await loader.loadCore();
  const coreCount = order.length;
  assert.ok(order.every((f) => f.startsWith("core/")), "only core chunks fetched during loadCore");
  await loader.streamRest();
  const firstRefinementIdx = order.findIndex((f) => f.startsWith("refinement/"));
  assert.ok(firstRefinementIdx >= coreCount, "all core chunks fetched before any refinement chunk");
  assert.deepStrictEqual(appended.slice().sort(),
    ["completed-faint", "completed-zoa", "observed"], "every layer appended via onLayer");

  // ---- error isolation: a failing layer must not abort the others ----
  const errs = [];
  const okLayers = [];
  const flaky = async (d) => {
    if (d.file.includes("zoa")) throw new Error("sha256 mismatch (injected)");
    return new Float32Array([0]);
  };
  const l2 = new PackLoader({ manifest, fetchArray: flaky,
    onLayer: (l) => okLayers.push(l.id), onError: (e, t) => errs.push(t.layerId) });
  await l2.loadCore();
  await l2.streamRest();
  assert.ok(okLayers.includes("observed") && okLayers.includes("completed-faint"),
    "healthy layers still load when one fails");
  assert.deepStrictEqual(errs, ["completed-zoa"], "the failing layer is isolated and reported");

  console.log("loader fetch-order: ALL ASSERTIONS PASSED");
})().catch((e) => { console.error("FAIL:", e.message); process.exit(1); });
