/*
 * pickNearest + menuModel unit tests (GPU-free; `node tests/test_pick_nearest.js`).
 * Injects a `project` stub (no DOM) to exercise the screen-space nearest-point pick.
 */
const assert = require("assert");
const { pickNearest, menuModel } = require("../apps/echoes-viewer/contextmenu.js");

// 4 points; the stub projects each to a known screen px (or null = behind camera).
const positions = new Float32Array([
  0, 0, 0, 1,   // index 0 -> (100,100)
  1, 0, 0, 1,   // index 1 -> (200,200)
  2, 0, 0, 1,   // index 2 -> null (behind camera)
  3, 0, 0, 1,   // index 3 -> (105,104)  (closest to a click at 103,102)
]);
const screen = { 0: [100, 100], 1: [200, 200], 2: null, 3: [105, 104] };
const project = (p) => screen[Math.round(p[0])];

// click near index 3
let hit = pickNearest({ positions, count: 4, project, cssX: 103, cssY: 102, maxPx: 18 });
assert.ok(hit && hit.index === 3, "nearest is index 3");

// behind-camera (null) points are skipped, never returned
hit = pickNearest({ positions, count: 4, project, cssX: 999, cssY: 999, maxPx: 18 });
assert.strictEqual(hit, null, "nothing within maxPx -> null");

// exact-hit on index 0
hit = pickNearest({ positions, count: 4, project, cssX: 100, cssY: 100, maxPx: 5 });
assert.ok(hit && hit.index === 0, "exact hit index 0");

// maxPx rejection: index 1 at (200,200), click (210,210) dist ~14.1 -> within 18, outside 10
assert.ok(pickNearest({ positions, count: 4, project, cssX: 210, cssY: 210, maxPx: 18 }).index === 1, "within 18");
assert.strictEqual(pickNearest({ positions, count: 4, project, cssX: 210, cssY: 210, maxPx: 10 }), null, "outside 10");

// ---- menuModel: coordinate-only vs named ----
const m1 = menuModel({ ra: 187.70593, dec: 12.39112, z: 0.004 });
assert.ok(m1.header === "Galaxy" && m1.links.length >= 4, "anon model has core links");
assert.ok(!m1.links.find((l) => l.label.includes("Wikipedia")), "anon model: no Wikipedia");
assert.ok(m1.info.some((r) => r.value.includes("J2000")) && m1.actions.length === 2, "info + actions");

const m2 = menuModel({ ra: 187.70593, dec: 12.39112, distMpc: 16.5, ksMag: 5.8, pgc: 41361,
                       datasetLabel: "Local 2M++" });
assert.ok(m2.header.includes("Local 2M++"), "dataset label in header");
assert.ok(m2.links.find((l) => l.label.includes("Wikipedia")), "named model: Wikipedia present");
assert.ok(m2.info.find((r) => r.value === "PGC 41361"), "PGC in info");
assert.ok(m2.thumbnail.includes("hips2fits"), "thumbnail present");

console.log("pickNearest + menuModel: ALL ASSERTIONS PASSED");
