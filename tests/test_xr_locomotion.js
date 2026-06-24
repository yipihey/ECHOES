/*
 * xrLocomotion.js unit tests (GPU/headset-free; `node tests/test_xr_locomotion.js`).
 * Verifies the pure VR locomotion math: one-hand grab pins the grabbed point, two-hand grab scales
 * about the midpoint, snap-turn rotates by the exact angle about the head, fly is zero at rest.
 */
const assert = require("assert");
const XL = require("../../echoes-k3d/js/src/providers/threejs/initializers/xrLocomotion.js");

const close = (a, b, t = 1e-9) => assert.ok(Math.abs(a - b) <= t, `${a} != ${b}`);
const vclose = (a, b, t = 1e-9) => a.forEach((x, i) => close(x, b[i], t));

// helper: world point of a room-frame hand under a rig {position, scale}
const world = (h, pos, s) => [pos[0] + s * h[0], pos[1] + s * h[1], pos[2] + s * h[2]];

// ---- one-hand grab: the grabbed world point stays under the hand ----
{
  const posStart = [10, 0, -5], scaleStart = 2.0;
  const handStart = [1, 1, 1];
  const W = world(handStart, posStart, scaleStart);                  // grabbed world point
  const handNow = [1.5, 0.5, 2];                                     // hand moved
  const posNow = XL.oneHandGrabPosition({ handStart, handNow, posStart, scaleStart });
  // the grabbed point W must now sit under handNow (scale unchanged)
  vclose(world(handNow, posNow, scaleStart), W);
}

// ---- two-hand grab: double the hand separation -> scale x2, midpoint world point fixed ----
{
  const posStart = [0, 0, 0], scaleStart = 1.0;
  const aStart = [-1, 0, 0], bStart = [1, 0, 0];                     // separation 2
  const aNow = [-2, 0, 0], bNow = [2, 0, 0];                         // separation 4 -> ratio 2
  const Wmid = world([0, 0, 0], posStart, scaleStart);              // grabbed world midpoint
  const r = XL.twoHandGrabTransform({ aStart, bStart, aNow, bNow, posStart, scaleStart,
                                      minScale: 1e-6, maxScale: 1e6 });
  close(r.scale, 2.0);
  // the grabbed world midpoint stays fixed under the new transform
  vclose(world([0, 0, 0], r.position, r.scale), Wmid);
  // clamp respected
  const cl = XL.twoHandGrabTransform({ aStart, bStart, aNow, bNow, posStart, scaleStart,
                                       minScale: 1e-6, maxScale: 1.5 });
  close(cl.scale, 1.5);
}

// ---- snap-turn: rotate 90 deg about +Z through the head; a probe rotates exactly 90 deg ----
{
  const headWorld = [0, 0, 0], rigPos = [1, 0, 0], rigQuat = [0, 0, 0, 1];
  const r = XL.snapTurn({ rigPos, rigQuat, headWorld, degrees: 90 });
  // rigPos (1,0,0) about +Z by +90deg -> (0,1,0)
  vclose(r.position, [0, 1, 0], 1e-9);
  // quaternion is a +90deg rotation about Z
  vclose(r.quaternion, [0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4)], 1e-9);
}

// ---- fly: zero stick -> zero delta; forward stick moves along head forward ----
{
  const z = XL.flyStep({ headForward: [0, 1, 0], headRight: [1, 0, 0], stickX: 0, stickY: 0,
                         speed: 5, dt: 0.016, rigScale: 1 });
  vclose(z, [0, 0, 0]);
  // stickY = -1 (up) -> move +forward; magnitude speed*dt*scale
  const f = XL.flyStep({ headForward: [0, 1, 0], headRight: [1, 0, 0], stickX: 0, stickY: -1,
                         speed: 10, dt: 0.1, rigScale: 2 });
  vclose(f, [0, 2.0, 0]);                                            // 10*0.1*2 along +Y
}

console.log("xrLocomotion: ALL ASSERTIONS PASSED");
