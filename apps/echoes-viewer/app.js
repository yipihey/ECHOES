const MANIFEST_URL = "data/viewer_manifest.json";
const TWO_PI = Math.PI * 2;

const els = {};
const state = {
  manifest: null,
  manifestUrl: null,
  base: null,
  method: null,
  realization: null,
  realizationData: null,
  catalog: null,
  gpu: null,
  buffers: null,
  lineBuffers: null,
  needsRebuild: true,
  needsRender: true,
  blinking: false,
  blinkPhase: 1,
  blinkTimer: null,
  settings: {
    coordinateMode: "comoving",
    projection: "3d",
    colorBy: "provenance",
    sizeBy: "source",
    showObserved: true,
    showEchoes: true,
    zSlab: 100,
    pointScale: 1.5,
    opacity: 0.78,
    showGrid: true,
    showLabels: true,
    gridOpacity: 0.35,
    labelSize: 12,
  },
  bounds: null,
  camera: {
    yaw: -0.82,
    pitch: 0.42,
    distance: 1,
    target: [0, 0, 0],
    orthoScale: 1,
    hasFit: false,
  },
  interaction: {
    pointers: new Map(),
    lastPointer: null,
    lastCentroid: null,
    lastDistance: null,
  },
};

function $(id) {
  return document.getElementById(id);
}

function setupElements() {
  for (const id of [
    "app", "scene", "labels", "unsupported", "methodSelect", "realizationSelect",
    "coordinateSelect", "projectionSelect", "colorBy", "sizeBy", "showObserved",
    "showEchoes", "zSlab", "pointScale", "opacity", "showGrid", "showLabels",
    "gridOpacity", "labelSize", "blinkRate", "blinkBtn", "shotBtn", "fullBtn",
    "hideBtn", "status", "cosmo", "legend",
  ]) {
    els[id] = $(id);
  }
}

async function main() {
  setupElements();
  try {
    state.manifestUrl = new URL(MANIFEST_URL, window.location.href);
    state.manifest = await fetchJson(state.manifestUrl);
    state.method = state.manifest.methods[0];
    state.realization = state.method.realizations[0];
    hydrateHashState();
    populateControls();
    renderLegend();
    updateCosmologyText();
    setStatus("Loading catalog bundle...");
    state.base = await loadBaseColumns();
    state.realizationData = await loadRealization(state.realization);

    if (!navigator.gpu) {
      els.unsupported.hidden = false;
      setStatus("WebGPU is not available in this browser.");
      return;
    }

    state.gpu = await initGpu();
    attachEvents();
    resizeCanvas();
    rebuildScene(true);
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return res.json();
}

async function fetchArray(desc) {
  const url = new URL(desc.file, state.manifestUrl);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const dtype = desc.dtype;
  if (dtype === "<f4" || dtype === "|f4") return new Float32Array(buf);
  if (dtype === "|u1" || dtype === "u1") return new Uint8Array(buf);
  throw new Error(`unsupported viewer dtype ${dtype}`);
}

async function loadBaseColumns() {
  const cols = state.manifest.base.columns;
  const base = {
    ra: await fetchArray(cols.ra),
    dec: await fetchArray(cols.dec),
    weight_systot: await fetchArray(cols.weight_systot),
    provenance: await fetchArray(cols.provenance),
    observed_z: await fetchArray(cols.observed_z),
    extraColumns: {},
  };
  for (const [id, desc] of Object.entries(cols)) {
    if (id in base) continue;
    base.extraColumns[id] = await fetchArray(desc);
  }
  return base;
}

async function loadRealization(realization) {
  const chunks = realization.chunks;
  return {
    missing_z: await fetchArray(chunks.missing_z),
    extra_ra: await fetchArray(chunks.extra_ra),
    extra_dec: await fetchArray(chunks.extra_dec),
    extra_z: await fetchArray(chunks.extra_z),
    extra_provenance: await fetchArray(chunks.extra_provenance),
  };
}

function populateControls() {
  const controls = els;
  controls.methodSelect.innerHTML = "";
  for (const method of state.manifest.methods) {
    const opt = new Option(method.label, method.id);
    controls.methodSelect.append(opt);
  }
  controls.methodSelect.value = state.method.id;
  populateRealizations();

  const appearanceCols = state.manifest.columns.filter(c => ["coordinate", "weight", "categorical"].includes(c.role));
  for (const select of [controls.colorBy, controls.sizeBy]) {
    select.innerHTML = "";
    if (select === controls.sizeBy) select.append(new Option("Fixed", "fixed"));
    for (const col of appearanceCols) {
      select.append(new Option(col.label, col.id));
    }
  }
  controls.colorBy.value = state.settings.colorBy;
  controls.sizeBy.value = state.settings.sizeBy;

  controls.coordinateSelect.value = state.settings.coordinateMode;
  controls.projectionSelect.value = state.settings.projection;
  controls.showObserved.checked = state.settings.showObserved;
  controls.showEchoes.checked = state.settings.showEchoes;
  controls.zSlab.value = String(state.settings.zSlab);
  controls.pointScale.value = String(state.settings.pointScale);
  controls.opacity.value = String(state.settings.opacity);
  controls.showGrid.checked = state.settings.showGrid;
  controls.showLabels.checked = state.settings.showLabels;
  controls.gridOpacity.value = String(state.settings.gridOpacity);
  controls.labelSize.value = String(state.settings.labelSize);
}

function populateRealizations() {
  els.realizationSelect.innerHTML = "";
  for (const realization of state.method.realizations) {
    const opt = new Option(realization.label, realization.id);
    els.realizationSelect.append(opt);
  }
  els.realizationSelect.value = state.realization.id;
}

function attachEvents() {
  window.addEventListener("resize", () => {
    resizeCanvas();
    state.needsRender = true;
  });

  els.methodSelect.addEventListener("change", async () => {
    state.method = state.manifest.methods.find(m => m.id === els.methodSelect.value);
    state.realization = state.method.realizations[0];
    populateRealizations();
    setStatus("Loading method realization...");
    state.realizationData = await loadRealization(state.realization);
    rebuildScene(true);
  });

  els.realizationSelect.addEventListener("change", async () => {
    state.realization = state.method.realizations.find(r => r.id === els.realizationSelect.value);
    setStatus("Loading realization...");
    state.realizationData = await loadRealization(state.realization);
    rebuildScene(false);
  });

  const settingInputs = [
    ["coordinateSelect", "coordinateMode", true],
    ["projectionSelect", "projection", true],
    ["colorBy", "colorBy", false],
    ["sizeBy", "sizeBy", false],
    ["zSlab", "zSlab", false],
    ["pointScale", "pointScale", false],
    ["opacity", "opacity", false],
    ["gridOpacity", "gridOpacity", false],
    ["labelSize", "labelSize", false],
  ];
  for (const [id, key, refit] of settingInputs) {
    els[id].addEventListener("input", () => {
      state.settings[key] = parseSetting(els[id].value, state.settings[key]);
      rebuildScene(refit);
    });
  }
  for (const [id, key] of [["showObserved", "showObserved"], ["showEchoes", "showEchoes"], ["showGrid", "showGrid"], ["showLabels", "showLabels"]]) {
    els[id].addEventListener("change", () => {
      state.settings[key] = els[id].checked;
      rebuildScene(false);
    });
  }

  els.blinkBtn.addEventListener("click", toggleBlink);
  els.shotBtn.addEventListener("click", downloadScreenshot);
  els.fullBtn.addEventListener("click", toggleFullscreen);
  els.hideBtn.addEventListener("click", toggleUi);

  window.addEventListener("keydown", (ev) => {
    if (ev.key === "h" || ev.key === "H") toggleUi();
    if (ev.key === " ") {
      ev.preventDefault();
      toggleBlink();
    }
    if (ev.key === "r" || ev.key === "R") {
      fitCamera();
      state.needsRender = true;
    }
  });

  const canvas = els.scene;
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
}

function parseSetting(value, current) {
  if (typeof current === "number") return Number(value);
  return value;
}

function hydrateHashState() {
  const raw = window.location.hash.replace(/^#/, "");
  if (!raw) return;
  const p = new URLSearchParams(raw);
  const methodId = p.get("method");
  const realizationId = p.get("realization");
  if (methodId) {
    const method = state.manifest.methods.find(m => m.id === methodId);
    if (method) state.method = method;
  }
  if (realizationId) {
    const realization = state.method.realizations.find(r => r.id === realizationId);
    if (realization) state.realization = realization;
  }
  for (const key of ["coordinateMode", "projection", "colorBy", "sizeBy"]) {
    if (p.has(key)) state.settings[key] = p.get(key);
  }
}

function writeHashState() {
  const p = new URLSearchParams();
  p.set("method", state.method.id);
  p.set("realization", state.realization.id);
  p.set("coordinateMode", state.settings.coordinateMode);
  p.set("projection", state.settings.projection);
  p.set("colorBy", state.settings.colorBy);
  p.set("sizeBy", state.settings.sizeBy);
  history.replaceState(null, "", `#${p.toString()}`);
}

async function initGpu() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter available");
  const device = await adapter.requestDevice();
  const context = els.scene.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  const shader = device.createShaderModule({ code: pointShader() });
  const lineShader = device.createShaderModule({ code: lineShaderCode() });
  const cameraBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shader, entryPoint: "vs" },
    fragment: {
      module: shader,
      entryPoint: "fs",
      targets: [{ format, blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      } }],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });

  const linePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: lineShader,
      entryPoint: "vs",
      buffers: [
        { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
        { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x4" }] },
      ],
    },
    fragment: {
      module: lineShader,
      entryPoint: "fs",
      targets: [{ format, blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      } }],
    },
    primitive: { topology: "line-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less-equal" },
  });

  return {
    adapter,
    device,
    context,
    format,
    cameraBuffer,
    bindGroupLayout,
    pipeline,
    linePipeline,
    depthTexture: null,
  };
}

function resizeCanvas() {
  if (!state.gpu) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(els.scene.clientWidth * ratio));
  const height = Math.max(1, Math.floor(els.scene.clientHeight * ratio));
  if (els.scene.width === width && els.scene.height === height) return;
  els.scene.width = width;
  els.scene.height = height;
  state.gpu.context.configure({
    device: state.gpu.device,
    format: state.gpu.format,
    alphaMode: "opaque",
  });
  state.gpu.depthTexture = state.gpu.device.createTexture({
    size: [width, height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  state.needsRender = true;
}

function rebuildScene(refit) {
  if (!state.base || !state.realizationData || !state.gpu) return;
  writeHashState();
  const catalog = assembleCatalog();
  state.catalog = catalog;
  state.bounds = computeBounds(catalog.positions);
  if (refit || !state.camera.hasFit) fitCamera();
  uploadPointBuffers(catalog);
  buildGridBuffers();
  updateAxisLabels();
  renderLegend();
  setStatus(statusText(catalog));
  state.needsRender = true;
}

function assembleCatalog() {
  const counts = state.manifest.counts;
  const nObs = counts.observed;
  const nBase = counts.base;
  const base = state.base;
  const r = state.realizationData;
  const total = nBase + r.extra_ra.length;
  const zFull = new Float32Array(total);
  zFull.set(base.observed_z, 0);
  zFull.set(r.missing_z, nObs);
  zFull.set(r.extra_z, nBase);

  const zRange = state.manifest.columns.find(c => c.id === "z");
  const zMid = 0.5 * (zRange.min + zRange.max);
  const slab = state.settings.zSlab / 100;
  const zHalf = 0.5 * (zRange.max - zRange.min) * slab;
  const zLo = slab >= 0.999 ? -Infinity : zMid - zHalf;
  const zHi = slab >= 0.999 ? Infinity : zMid + zHalf;
  const blinkObservedOnly = state.blinking && state.blinkPhase === 0;

  const rows = [];
  for (let i = 0; i < total; i++) {
    const isExtra = i >= nBase;
    const baseIndex = isExtra ? -1 : i;
    const prov = isExtra ? r.extra_provenance[i - nBase] : base.provenance[i];
    const isObserved = !isExtra && i < nObs && prov === 0;
    const isEchoes = !isObserved;
    if (blinkObservedOnly && isEchoes) continue;
    if (!state.settings.showObserved && isObserved) continue;
    if (!state.settings.showEchoes && isEchoes) continue;
    const z = zFull[i];
    if (z < zLo || z > zHi) continue;
    rows.push({ i, isExtra, baseIndex, prov, source: isObserved ? 0 : 1, z });
  }

  const n = rows.length || 1;
  const positions = new Float32Array(n * 4);
  const colors = new Float32Array(n * 4);
  const raw = {
    ra: new Float32Array(n),
    dec: new Float32Array(n),
    z: new Float32Array(n),
    weight_systot: new Float32Array(n),
    provenance: new Float32Array(n),
    source: new Float32Array(n),
  };

  if (!rows.length) {
    positions[3] = 1;
    colors[3] = 0;
    return { positions, colors, raw, count: 0 };
  }

  const numericRanges = getNumericRanges(rows, zFull);
  for (let j = 0; j < rows.length; j++) {
    const row = rows[j];
    const idx = row.i;
    const extraOffset = idx - nBase;
    const ra = row.isExtra ? r.extra_ra[extraOffset] : base.ra[idx];
    const dec = row.isExtra ? r.extra_dec[extraOffset] : base.dec[idx];
    const wsys = row.isExtra ? 1 : base.weight_systot[idx];
    const values = { ra, dec, z: row.z, weight_systot: wsys, provenance: row.prov, source: row.source };
    for (const [key, arr] of Object.entries(base.extraColumns)) {
      values[key] = row.isExtra || idx >= arr.length ? Number.NaN : arr[idx];
    }
    const pos = computeProjectedPosition(ra, dec, row.z);
    const size = pointSizeFor(row, values, numericRanges);
    const color = colorFor(row, values, numericRanges);
    positions.set([pos[0], pos[1], pos[2], size], j * 4);
    colors.set(color, j * 4);
    raw.ra[j] = ra;
    raw.dec[j] = dec;
    raw.z[j] = row.z;
    raw.weight_systot[j] = wsys;
    raw.provenance[j] = row.prov;
    raw.source[j] = row.source;
  }
  return { positions, colors, raw, count: rows.length };
}

function getNumericRanges(rows, zFull) {
  const ranges = {};
  for (const col of state.manifest.columns) {
    if (typeof col.min === "number" && typeof col.max === "number") {
      ranges[col.id] = [col.min, col.max];
    }
  }
  ranges.z = [state.manifest.columns.find(c => c.id === "z").min, state.manifest.columns.find(c => c.id === "z").max];
  return ranges;
}

function computeProjectedPosition(raDeg, decDeg, z) {
  const mode = state.settings.coordinateMode;
  const proj = state.settings.projection;
  const wrappedRa = wrapRa(raDeg);
  const zScaled = (z - 0.525) * 280;

  if (proj === "radec") return [wrappedRa, decDeg, 0];
  if (proj === "raz") return [wrappedRa, zScaled, 0];
  if (proj === "decz") return [decDeg, zScaled, 0];

  const xyz = mode === "observed"
    ? [wrappedRa, decDeg, zScaled]
    : radecToCartesian(raDeg, decDeg, z, mode === "proper");

  if (proj === "xy") return [xyz[0], xyz[1], 0];
  if (proj === "xz") return [xyz[0], xyz[2], 0];
  if (proj === "yz") return [xyz[1], xyz[2], 0];
  return xyz;
}

function wrapRa(ra) {
  return ra > 180 ? ra - 360 : ra;
}

let distanceTable = null;
function radecToCartesian(raDeg, decDeg, z, proper) {
  if (!distanceTable) distanceTable = makeDistanceTable();
  const dComoving = interpDistance(z);
  const d = proper ? dComoving / (1 + z) : dComoving;
  const ra = raDeg * Math.PI / 180;
  const dec = decDeg * Math.PI / 180;
  const cd = Math.cos(dec);
  return [
    d * cd * Math.cos(ra),
    d * cd * Math.sin(ra),
    d * Math.sin(dec),
  ];
}

function makeDistanceTable() {
  const cosmo = state.manifest.cosmology;
  const zMax = 0.75;
  const n = 4096;
  const z = new Float64Array(n);
  const d = new Float64Array(n);
  let integral = 0;
  for (let i = 0; i < n; i++) z[i] = zMax * i / (n - 1);
  for (let i = 1; i < n; i++) {
    const z0 = z[i - 1];
    const z1 = z[i];
    integral += 0.5 * (invEz(z0, cosmo) + invEz(z1, cosmo)) * (z1 - z0);
    d[i] = cosmo.c_over_H100_Mpch * integral;
  }
  return { z, d };
}

function invEz(z, cosmo) {
  const Om = cosmo.Om;
  const w0 = cosmo.w0;
  const wa = cosmo.wa;
  const matter = Om * Math.pow(1 + z, 3);
  const de = (1 - Om) * Math.pow(1 + z, 3 * (1 + w0 + wa)) * Math.exp(-3 * wa * z / (1 + z));
  return 1 / Math.sqrt(matter + de);
}

function interpDistance(z) {
  const table = distanceTable;
  const zs = table.z;
  const ds = table.d;
  if (z <= zs[0]) return ds[0];
  if (z >= zs[zs.length - 1]) return ds[ds.length - 1];
  const u = z / zs[zs.length - 1] * (zs.length - 1);
  const i = Math.floor(u);
  const f = u - i;
  return ds[i] * (1 - f) + ds[i + 1] * f;
}

function pointSizeFor(row, values, ranges) {
  const key = state.settings.sizeBy;
  if (key === "fixed") return row.source ? 3.2 : 2.0;
  if (key === "source") return row.source ? 3.8 : 1.9;
  if (key === "provenance") return row.prov === 0 ? 1.9 : 3.4;
  const value = values[key];
  if (!Number.isFinite(value)) return 2.2;
  const t = normalize(value, ranges[key] || [0, 1]);
  return 1.6 + 4.2 * Math.sqrt(t);
}

function colorFor(row, values, ranges) {
  const key = state.settings.colorBy;
  const alpha = state.settings.opacity;
  if (key === "provenance") return [...hexToRgb(provColor(row.prov)), alpha];
  if (key === "source") return row.source ? [...hexToRgb("#41d6b0"), alpha] : [...hexToRgb("#d8dde5"), alpha * 0.62];
  const value = values[key];
  const t = normalize(value, ranges[key] || [0, 1]);
  const rgb = gradient(t);
  return [rgb[0], rgb[1], rgb[2], row.source ? alpha : alpha * 0.62];
}

function provColor(prov) {
  return state.manifest.provenance_codes[String(prov)]?.color || "#ffffff";
}

function normalize(value, range) {
  const [lo, hi] = range;
  if (!Number.isFinite(value) || hi <= lo) return 0.5;
  return Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
}

function hexToRgb(hex) {
  const s = hex.replace("#", "");
  const n = parseInt(s, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function gradient(t) {
  const a = hexToRgb("#39b5ff");
  const b = hexToRgb("#41d6b0");
  const c = hexToRgb("#ffb84d");
  if (t < 0.5) return mixColor(a, b, t * 2);
  return mixColor(b, c, (t - 0.5) * 2);
}

function mixColor(a, b, t) {
  return [a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t, a[2] * (1 - t) + b[2] * t];
}

function computeBounds(positions) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 4) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
  }
  if (!Number.isFinite(minX)) {
    minX = minY = minZ = -1;
    maxX = maxY = maxZ = 1;
  }
  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const extent = [maxX - minX, maxY - minY, maxZ - minZ];
  const radius = Math.max(extent[0], extent[1], extent[2], 1) * 0.72;
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ], center, extent, radius };
}

function fitCamera() {
  if (!state.bounds) return;
  state.camera.target = [...state.bounds.center];
  state.camera.distance = Math.max(10, state.bounds.radius * 3.0);
  state.camera.orthoScale = Math.max(1, state.bounds.radius * 1.35);
  state.camera.hasFit = true;
}

function uploadPointBuffers(catalog) {
  const { device, bindGroupLayout, cameraBuffer } = state.gpu;
  const positions = catalog.positions.byteLength ? catalog.positions : new Float32Array([0, 0, 0, 1]);
  const colors = catalog.colors.byteLength ? catalog.colors : new Float32Array([1, 1, 1, 0]);
  const pointBuffer = createBufferWithData(device, positions, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const colorBuffer = createBufferWithData(device, colors, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: pointBuffer } },
      { binding: 2, resource: { buffer: colorBuffer } },
    ],
  });
  state.buffers = { pointBuffer, colorBuffer, bindGroup, count: Math.max(1, catalog.count) };
}

function createBufferWithData(device, typedArray, usage) {
  const buffer = device.createBuffer({
    size: Math.max(4, typedArray.byteLength),
    usage,
    mappedAtCreation: true,
  });
  const dst = new typedArray.constructor(buffer.getMappedRange());
  dst.set(typedArray);
  buffer.unmap();
  return buffer;
}

function buildGridBuffers() {
  const { device } = state.gpu;
  if (!state.settings.showGrid || !state.bounds) {
    state.lineBuffers = null;
    return;
  }
  const grid = makeGridLines();
  if (!grid.positions.length) {
    state.lineBuffers = null;
    return;
  }
  const pos = new Float32Array(grid.positions);
  const col = new Float32Array(grid.colors);
  state.lineBuffers = {
    count: pos.length / 3,
    pos: createBufferWithData(device, pos, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
    col: createBufferWithData(device, col, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
  };
}

function makeGridLines() {
  const b = state.bounds;
  const positions = [];
  const colors = [];
  const opacity = state.settings.gridOpacity;
  const gridColor = [0.78, 0.83, 0.88, 0.22 * opacity];
  const axisColors = {
    x: [0.25, 0.84, 0.69, 0.85 * opacity],
    y: [1.0, 0.72, 0.30, 0.85 * opacity],
    z: [0.22, 0.71, 1.0, 0.85 * opacity],
  };
  const ticksX = ticks(b.min[0], b.max[0], 7);
  const ticksY = ticks(b.min[1], b.max[1], 7);
  const z0 = b.min[2];
  for (const x of ticksX) pushLine(positions, colors, [x, b.min[1], z0], [x, b.max[1], z0], gridColor);
  for (const y of ticksY) pushLine(positions, colors, [b.min[0], y, z0], [b.max[0], y, z0], gridColor);
  pushLine(positions, colors, [b.min[0], b.min[1], z0], [b.max[0], b.min[1], z0], axisColors.x);
  pushLine(positions, colors, [b.min[0], b.min[1], z0], [b.min[0], b.max[1], z0], axisColors.y);
  if (state.settings.projection === "3d") {
    pushLine(positions, colors, [b.min[0], b.min[1], b.min[2]], [b.min[0], b.min[1], b.max[2]], axisColors.z);
    addBoxEdges(positions, colors, b, gridColor);
  }
  return { positions, colors };
}

function pushLine(positions, colors, a, b, color) {
  positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  colors.push(...color, ...color);
}

function addBoxEdges(positions, colors, b, color) {
  const [x0, y0, z0] = b.min;
  const [x1, y1, z1] = b.max;
  const corners = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (const [i, j] of edges) pushLine(positions, colors, corners[i], corners[j], color);
}

function ticks(lo, hi, n) {
  if (hi <= lo) return [lo];
  const out = [];
  for (let i = 0; i < n; i++) out.push(lo + (hi - lo) * i / (n - 1));
  return out;
}

function frame() {
  if (state.gpu && state.buffers && state.needsRender) {
    render();
    state.needsRender = false;
  }
  requestAnimationFrame(frame);
}

function render() {
  resizeCanvas();
  const gpu = state.gpu;
  const device = gpu.device;
  const viewProj = computeViewProjection();
  state.lastViewProj = viewProj;

  const uniform = new Float32Array(20);
  uniform.set(viewProj, 0);
  uniform[16] = els.scene.width;
  uniform[17] = els.scene.height;
  uniform[18] = state.settings.pointScale;
  uniform[19] = 0;
  device.queue.writeBuffer(gpu.cameraBuffer, 0, uniform);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: gpu.context.getCurrentTexture().createView(),
      clearValue: { r: 0.035, g: 0.043, b: 0.051, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
    depthStencilAttachment: {
      view: gpu.depthTexture.createView(),
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  if (state.lineBuffers) {
    pass.setPipeline(gpu.linePipeline);
    pass.setVertexBuffer(0, state.lineBuffers.pos);
    pass.setVertexBuffer(1, state.lineBuffers.col);
    pass.setBindGroup(0, device.createBindGroup({
      layout: gpu.linePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpu.cameraBuffer } }],
    }));
    pass.draw(state.lineBuffers.count);
  }
  pass.setPipeline(gpu.pipeline);
  pass.setBindGroup(0, state.buffers.bindGroup);
  pass.draw(6, state.buffers.count);
  pass.end();
  device.queue.submit([encoder.finish()]);
  updateAxisLabels();
}

function computeViewProjection() {
  const width = Math.max(1, els.scene.width);
  const height = Math.max(1, els.scene.height);
  const aspect = width / height;
  const target = state.camera.target;
  let view;
  let proj;
  if (state.settings.projection === "3d") {
    const pitch = Math.max(-1.35, Math.min(1.35, state.camera.pitch));
    state.camera.pitch = pitch;
    const cp = Math.cos(pitch);
    const eye = [
      target[0] + state.camera.distance * cp * Math.sin(state.camera.yaw),
      target[1] + state.camera.distance * Math.sin(pitch),
      target[2] + state.camera.distance * cp * Math.cos(state.camera.yaw),
    ];
    view = mat4LookAt(eye, target, [0, 1, 0]);
    proj = mat4Perspective(Math.PI / 4, aspect, Math.max(0.1, state.camera.distance * 0.001), state.camera.distance * 8 + state.bounds.radius * 6);
  } else {
    const scale = state.camera.orthoScale;
    view = mat4LookAt([target[0], target[1], target[2] + 1000], target, [0, 1, 0]);
    proj = mat4Ortho(-scale * aspect, scale * aspect, -scale, scale, -5000, 5000);
  }
  return mat4Multiply(proj, view);
}

function updateAxisLabels() {
  els.labels.innerHTML = "";
  if (!state.settings.showLabels || !state.bounds || !state.lastViewProj) return;
  els.labels.style.fontSize = `${state.settings.labelSize}px`;
  const b = state.bounds;
  const labels = axisLabelDefinitions();
  addAxisLabel(labels[0], [b.max[0], b.min[1], b.min[2]]);
  addAxisLabel(labels[1], [b.min[0], b.max[1], b.min[2]]);
  if (state.settings.projection === "3d") addAxisLabel(labels[2], [b.min[0], b.min[1], b.max[2]]);
}

function axisLabelDefinitions() {
  const proj = state.settings.projection;
  if (proj === "radec") return ["RA [deg]", "Dec [deg]", ""];
  if (proj === "raz") return ["RA [deg]", "z", ""];
  if (proj === "decz") return ["Dec [deg]", "z", ""];
  if (proj === "xy") return ["x [Mpc/h]", "y [Mpc/h]", ""];
  if (proj === "xz") return ["x [Mpc/h]", "z [Mpc/h]", ""];
  if (proj === "yz") return ["y [Mpc/h]", "z [Mpc/h]", ""];
  const mode = state.manifest.coordinate_modes.find(m => m.id === state.settings.coordinateMode);
  return mode?.axes || ["x", "y", "z"];
}

function addAxisLabel(text, pos) {
  if (!text) return;
  const s = projectToScreen(pos, state.lastViewProj);
  if (!s) return;
  const el = document.createElement("div");
  el.className = "axis-label";
  el.textContent = text;
  el.style.left = `${s[0]}px`;
  el.style.top = `${s[1]}px`;
  els.labels.append(el);
}

function projectToScreen(pos, m) {
  const clip = transformPoint(m, pos);
  if (clip[3] <= 0) return null;
  const x = clip[0] / clip[3];
  const y = clip[1] / clip[3];
  if (x < -1.2 || x > 1.2 || y < -1.2 || y > 1.2) return null;
  return [(x * 0.5 + 0.5) * els.scene.clientWidth, (-y * 0.5 + 0.5) * els.scene.clientHeight];
}

function onPointerDown(ev) {
  els.scene.setPointerCapture(ev.pointerId);
  state.interaction.pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  state.interaction.lastPointer = { x: ev.clientX, y: ev.clientY };
  updateGestureMemory();
}

function onPointerMove(ev) {
  if (!state.interaction.pointers.has(ev.pointerId)) return;
  state.interaction.pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  const pointers = [...state.interaction.pointers.values()];
  if (pointers.length >= 2) {
    const centroid = midpoint(pointers[0], pointers[1]);
    const dist = distance2(pointers[0], pointers[1]);
    if (state.interaction.lastCentroid && state.interaction.lastDistance) {
      panCamera(centroid.x - state.interaction.lastCentroid.x, centroid.y - state.interaction.lastCentroid.y);
      zoomCamera(state.interaction.lastDistance / Math.max(1, dist));
    }
    state.interaction.lastCentroid = centroid;
    state.interaction.lastDistance = dist;
  } else {
    const last = state.interaction.lastPointer || { x: ev.clientX, y: ev.clientY };
    const dx = ev.clientX - last.x;
    const dy = ev.clientY - last.y;
    if (ev.shiftKey || state.settings.projection !== "3d") panCamera(dx, dy);
    else {
      state.camera.yaw -= dx * 0.006;
      state.camera.pitch -= dy * 0.006;
    }
    state.interaction.lastPointer = { x: ev.clientX, y: ev.clientY };
  }
  state.needsRender = true;
}

function onPointerUp(ev) {
  state.interaction.pointers.delete(ev.pointerId);
  state.interaction.lastPointer = null;
  updateGestureMemory();
}

function updateGestureMemory() {
  const pointers = [...state.interaction.pointers.values()];
  if (pointers.length >= 2) {
    state.interaction.lastCentroid = midpoint(pointers[0], pointers[1]);
    state.interaction.lastDistance = distance2(pointers[0], pointers[1]);
  } else {
    state.interaction.lastCentroid = null;
    state.interaction.lastDistance = null;
  }
}

function midpoint(a, b) {
  return { x: 0.5 * (a.x + b.x), y: 0.5 * (a.y + b.y) };
}

function distance2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function onWheel(ev) {
  ev.preventDefault();
  zoomCamera(Math.exp(ev.deltaY * 0.001));
  state.needsRender = true;
}

function zoomCamera(factor) {
  if (state.settings.projection === "3d") {
    state.camera.distance = Math.max(1, state.camera.distance * factor);
  } else {
    state.camera.orthoScale = Math.max(0.1, state.camera.orthoScale * factor);
  }
}

function panCamera(dx, dy) {
  const scale = state.settings.projection === "3d"
    ? state.camera.distance / Math.max(els.scene.clientHeight, 1)
    : state.camera.orthoScale * 2 / Math.max(els.scene.clientHeight, 1);
  state.camera.target[0] -= dx * scale;
  state.camera.target[1] += dy * scale;
}

function toggleBlink() {
  state.blinking = !state.blinking;
  els.blinkBtn.classList.toggle("active", state.blinking);
  if (state.blinkTimer) clearInterval(state.blinkTimer);
  if (state.blinking) {
    const rate = () => Math.max(150, Number(els.blinkRate.value) || 850);
    state.blinkTimer = setInterval(() => {
      state.blinkPhase = state.blinkPhase ? 0 : 1;
      rebuildScene(false);
    }, rate());
  } else {
    state.blinkPhase = 1;
    rebuildScene(false);
  }
}

function downloadScreenshot() {
  state.needsRender = true;
  render();
  setTimeout(() => {
    els.scene.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `echoes-${state.method.id}-${state.realization.id}-${state.settings.projection}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    }, "image/png");
  }, 40);
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else els.app.requestFullscreen?.();
}

function toggleUi() {
  els.app.classList.toggle("ui-hidden");
}

function setStatus(text) {
  els.status.innerHTML = text;
}

function statusText(catalog) {
  const r = state.realization;
  return `<strong>${catalog.count.toLocaleString()}</strong> visible of ${r.total_count.toLocaleString()} in ${state.method.label}, ${r.label}.`;
}

function updateCosmologyText() {
  const c = state.manifest.cosmology;
  els.cosmo.textContent = `${c.label}: Om=${c.Om}, h=${c.h}, w0=${c.w0}, wa=${c.wa}.`;
}

function renderLegend() {
  const codes = state.manifest?.provenance_codes || {};
  els.legend.innerHTML = "";
  for (const [code, meta] of Object.entries(codes)) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="swatch" style="background:${meta.color}"></span>${code}: ${meta.short_label || meta.label}`;
    els.legend.append(item);
  }
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function mat4Ortho(left, right, bottom, top, near, far) {
  const out = new Float32Array(16);
  out[0] = 2 / (right - left);
  out[5] = 2 / (top - bottom);
  out[10] = -2 / (far - near);
  out[12] = -(right + left) / (right - left);
  out[13] = -(top + bottom) / (top - bottom);
  out[14] = -(far + near) / (far - near);
  out[15] = 1;
  return out;
}

function mat4LookAt(eye, center, up) {
  const z = normalize3([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -dot3(x, eye); out[13] = -dot3(y, eye); out[14] = -dot3(z, eye); out[15] = 1;
  return out;
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function transformPoint(m, p) {
  const x = p[0], y = p[1], z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}

function normalize3(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function pointShader() {
  return `
struct Camera {
  viewProj: mat4x4<f32>,
  viewport: vec2<f32>,
  pointScale: f32,
  pad: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VsOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
  );
  let p = points[instanceIndex];
  let local = corners[vertexIndex];
  var clip = camera.viewProj * vec4<f32>(p.xyz, 1.0);
  let px = local * p.w * camera.pointScale;
  clip.xy = clip.xy + px * vec2<f32>(2.0 / camera.viewport.x, 2.0 / camera.viewport.y) * clip.w;
  var out: VsOut;
  out.pos = clip;
  out.color = colors[instanceIndex];
  out.local = local;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let r = dot(in.local, in.local);
  if (r > 1.0) {
    discard;
  }
  let edge = smoothstep(1.0, 0.64, r);
  return vec4<f32>(in.color.rgb, in.color.a * edge);
}`;
}

function lineShaderCode() {
  return `
struct Camera {
  viewProj: mat4x4<f32>,
  viewport: vec2<f32>,
  pointScale: f32,
  pad: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
  var out: VsOut;
  out.pos = camera.viewProj * vec4<f32>(position, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  return in.color;
}`;
}

main();
