/*
 * contextmenu.js — right-click pick + galaxy context menu (UMD).
 *
 * `pickNearest` and `menuModel` are DOM-free (Node-testable); `mountMenu`/`dismissMenu` are the thin
 * browser DOM layer. Shared by the WebGPU viewer (app.js) and inlined into the k3d fork snapshot, so
 * both viewers pick + present galaxies identically. Depends on astrolinks.js (AstroLinks).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ContextMenu = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function getAstroLinks() {
    if (typeof module !== "undefined" && module.exports) return require("./astrolinks.js");
    return (typeof self !== "undefined" ? self : this).AstroLinks;
  }

  // Nearest rendered point to a cursor, in screen space. `project(pos)` → [px,py] (CSS px) or null
  // (behind camera / off-screen). DOM-free: the caller injects `project`. Returns {index, px}|null.
  function pickNearest(opts) {
    const { positions, count, project, cssX, cssY } = opts;
    const maxPx = opts.maxPx || 18;
    const maxD2 = maxPx * maxPx;
    let best = -1, bestD2 = Infinity, bestPx = null;
    for (let i = 0; i < count; i++) {
      const s = project([positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2]]);
      if (!s) continue;
      const dx = s[0] - cssX, dy = s[1] - cssY;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = i; bestPx = s; }
    }
    if (best < 0 || bestD2 > maxD2) return null;
    return { index: best, px: bestPx };
  }

  // Pure presentation model for the menu (Node-testable). `o` = {ra,dec,z?,distMpc?,ksMag?,name?,pgc?,
  // provenance?,sourceLabel?,datasetLabel?}. Returns {header, info:[{label,value}], links:[...], actions:[...]}.
  function menuModel(o) {
    const AL = getAstroLinks();
    const c = AL.formatCoord(o.ra, o.dec);
    const info = [
      { label: "RA, Dec", value: c.sexagesimalStr },
      { label: "", value: c.decimalStr },
    ];
    if (o.z != null && isFinite(o.z)) info.push({ label: "redshift z", value: o.z.toFixed(4) });
    if (o.distMpc != null && isFinite(o.distMpc)) info.push({ label: "distance", value: `${o.distMpc.toFixed(1)} Mpc` });
    if (o.ksMag != null && isFinite(o.ksMag)) info.push({ label: "K_s", value: o.ksMag.toFixed(2) });
    if (o.pgc && o.pgc > 0) info.push({ label: "id", value: `PGC ${o.pgc}` });
    if (o.provenanceLabel) info.push({ label: "provenance", value: o.provenanceLabel });
    const name = o.name || (o.pgc && o.pgc > 0 ? `PGC ${o.pgc}` : null);
    return {
      header: o.datasetLabel ? `Galaxy — ${o.datasetLabel}` : "Galaxy",
      info,
      thumbnail: AL.thumbnailUrl(((o.ra % 360) + 360) % 360, o.dec, o.distMpc),
      links: AL.astroLinks({ ra: o.ra, dec: o.dec, z: o.z, distMpc: o.distMpc, name, pgc: o.pgc }),
      actions: [
        { id: "copy-coords", label: "Copy coordinates", text: c.sexagesimalStr },
        { id: "copy-row", label: "Copy data row", text: AL.dataRow({ ...o, name }) },
      ],
    };
  }

  // ---- browser DOM layer ----
  let openMenu = null;

  function dismissMenu() {
    if (openMenu && openMenu.parentNode) openMenu.parentNode.removeChild(openMenu);
    openMenu = null;
  }

  function mountMenu(model, clientX, clientY) {
    dismissMenu();
    const doc = document;
    const el = doc.createElement("div");
    el.className = "panel context-menu";
    el.setAttribute("role", "menu");
    const h = doc.createElement("div");
    h.className = "context-menu-header";
    h.textContent = model.header;
    el.appendChild(h);

    if (model.thumbnail) {
      const img = doc.createElement("img");
      img.className = "context-menu-thumb";
      img.loading = "lazy"; img.alt = "DSS2 thumbnail"; img.src = model.thumbnail;
      img.onerror = () => { img.style.display = "none"; };
      el.appendChild(img);
    }

    const info = doc.createElement("div");
    info.className = "context-menu-info";
    model.info.forEach((row) => {
      const r = doc.createElement("div");
      r.className = "context-menu-info-row";
      r.innerHTML = `<span class="k">${row.label}</span><span class="v"></span>`;
      r.querySelector(".v").textContent = row.value;
      info.appendChild(r);
    });
    el.appendChild(info);

    let lastGroup = null;
    model.links.forEach((lnk) => {
      if (lnk.group !== lastGroup) {
        const g = doc.createElement("div");
        g.className = "context-menu-group";
        g.textContent = lnk.group;
        el.appendChild(g); lastGroup = lnk.group;
      }
      const a = doc.createElement("a");
      a.href = lnk.href; a.target = "_blank"; a.rel = "noopener noreferrer";
      a.textContent = `${lnk.label} ↗`;
      el.appendChild(a);
    });

    const acts = doc.createElement("div");
    acts.className = "context-menu-actions";
    model.actions.forEach((act) => {
      const b = doc.createElement("button");
      b.textContent = act.label;
      b.onclick = () => {
        try {
          if (navigator.clipboard) navigator.clipboard.writeText(act.text);
          b.textContent = "Copied"; setTimeout(() => { b.textContent = act.label; }, 1200);
        } catch (e) { /* clipboard unavailable */ }
      };
      acts.appendChild(b);
    });
    el.appendChild(acts);

    // position, clamped to viewport
    el.style.left = "0px"; el.style.top = "0px"; el.style.visibility = "hidden";
    doc.body.appendChild(el);
    const w = el.offsetWidth, hgt = el.offsetHeight;
    const x = Math.min(clientX, window.innerWidth - w - 8);
    const y = Math.min(clientY, window.innerHeight - hgt - 8);
    el.style.left = `${Math.max(4, x)}px`; el.style.top = `${Math.max(4, y)}px`;
    el.style.visibility = "visible";
    openMenu = el;

    // one-shot dismissal
    const onAway = (ev) => { if (!el.contains(ev.target)) { cleanup(); dismissMenu(); } };
    const onKey = (ev) => { if (ev.key === "Escape") { cleanup(); dismissMenu(); } };
    const onScroll = () => { cleanup(); dismissMenu(); };
    function cleanup() {
      doc.removeEventListener("pointerdown", onAway, true);
      doc.removeEventListener("keydown", onKey, true);
      window.removeEventListener("wheel", onScroll, true);
    }
    setTimeout(() => {
      doc.addEventListener("pointerdown", onAway, true);
      doc.addEventListener("keydown", onKey, true);
      window.addEventListener("wheel", onScroll, true);
    }, 0);
    return el;
  }

  return { pickNearest, menuModel, mountMenu, dismissMenu };
});
