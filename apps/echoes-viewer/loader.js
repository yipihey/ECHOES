/*
 * loader.js — progressive extension-pack loader (DOM-free, Node-importable for fetch-order tests).
 *
 * Splits an `echoes.pack.v1` manifest into renderable layers grouped by tier (core → refinement →
 * texture), loads the core tier first (so the viewer paints in ~1-2 s), then streams the rest in
 * priority order with a small concurrency cap. The host (app.js) injects `fetchArray` (which fetches
 * + validates bytes/count/sha256) and receives a callback per assembled layer so it can append a
 * GPU segment without reloading what is already on screen.
 *
 * No window/document/DOM at module scope → `require('./loader.js')` works under Node for the
 * fetch-order assertions in tests/.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api; // Node
  else Object.assign(root, api);                                             // browser globals
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const TIER_PRIORITY = { core: 0, refinement: 10, texture: 20 };

  // Is this a v2 pack manifest (declares renderable `layers`)? v1 viewer manifests are handled by
  // the legacy base/methods path in app.js.
  function isPackManifest(m) {
    return !!(m && m.schema_version === "echoes.pack.v1" && Array.isArray(m.layers));
  }

  // Ordered layer-load tasks: core first, then by tier priority, then by manifest order. Stable.
  function planLayerTasks(manifest) {
    const layers = (manifest.layers || []).map((layer, idx) => ({
      layerId: layer.id,
      tier: layer.tier || "core",
      priority: (TIER_PRIORITY[layer.tier] != null ? TIER_PRIORITY[layer.tier] : 0) * 1000 + idx,
      layer,
    }));
    layers.sort((a, b) => a.priority - b.priority);
    return layers;
  }

  // Assemble one layer's column arrays via the injected fetchArray. Returns
  // {id, tier, color, count, columns:{name: TypedArray}} ready for a GPU segment.
  async function loadLayer(layer, fetchArray) {
    const names = Object.keys(layer.columns);
    const arrays = await Promise.all(names.map((n) => fetchArray(layer.columns[n])));
    const columns = {};
    names.forEach((n, i) => { columns[n] = arrays[i]; });
    return { id: layer.id, tier: layer.tier, color: layer.color, count: layer.count,
             bbox: layer.bbox || null, valueRange: layer.value_range || null, columns };
  }

  // Orchestrates progressive loading. `await loadCore()` resolves once every core-tier layer is in;
  // call `streamRest()` afterwards to drain refinement/texture in the background (non-blocking).
  class PackLoader {
    constructor(opts) {
      this.fetchArray = opts.fetchArray;
      this.onLayer = opts.onLayer || (() => {});      // (assembledLayer) => void  (append a segment)
      this.onError = opts.onError || (() => {});      // (err, task) => void        (isolate failures)
      this.maxConcurrent = opts.maxConcurrent || 6;
      this.tasks = planLayerTasks(opts.manifest);
      this.coreTasks = this.tasks.filter((t) => t.tier === "core");
      this.restTasks = this.tasks.filter((t) => t.tier !== "core");
      this.loaded = [];
    }

    async _runTask(t) {
      try {
        const assembled = await loadLayer(t.layer, this.fetchArray);
        this.loaded.push(assembled);
        this.onLayer(assembled);                       // host appends incrementally
      } catch (err) {
        this.onError(err, t);                          // one bad layer must not abort the rest
      }
    }

    // Blocking: load every core layer (bounded concurrency), then resolve → host can first-paint.
    async loadCore() {
      await this._drain(this.coreTasks.slice());
      return this.loaded.filter((l) => l.tier === "core");
    }

    // Non-blocking-ish: drain refinement/texture in priority order. Await it to know all is in.
    async streamRest() {
      await this._drain(this.restTasks.slice());
    }

    async _drain(queue) {
      const workers = [];
      const next = async () => {
        while (queue.length) { await this._runTask(queue.shift()); }
      };
      for (let i = 0; i < Math.min(this.maxConcurrent, queue.length || 1); i++) workers.push(next());
      await Promise.all(workers);
    }
  }

  return { isPackManifest, planLayerTasks, loadLayer, PackLoader, TIER_PRIORITY };
});
