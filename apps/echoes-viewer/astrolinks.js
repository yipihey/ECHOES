/*
 * astrolinks.js — build external astronomy-service links for a sky position (DOM-free, UMD).
 *
 * Shared by both ECHOES viewers (the WebGPU explorer and the k3d textured/fork snapshot) so there is
 * one source of truth for the URL templates and coordinate formatting. All coordinates are J2000
 * decimal degrees. `require('./astrolinks.js')` works under Node (tests); in the browser it assigns to
 * the global (`AstroLinks`).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AstroLinks = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const enc = encodeURIComponent;
  const pad = (n, w) => String(Math.floor(n)).padStart(w, "0");

  function raToHMS(ra) {
    let h = (((ra % 360) + 360) % 360) / 15.0;        // 0..24 h
    const hh = Math.floor(h);
    const mm = Math.floor((h - hh) * 60);
    const ss = ((h - hh) * 60 - mm) * 60;
    return `${pad(hh, 2)}:${pad(mm, 2)}:${ss.toFixed(2).padStart(5, "0")}`;
  }

  function decToDMS(dec) {
    const sign = dec < 0 ? "-" : "+";
    const a = Math.abs(dec);
    const dd = Math.floor(a);
    const mm = Math.floor((a - dd) * 60);
    const ss = ((a - dd) * 60 - mm) * 60;
    return `${sign}${pad(dd, 2)}:${pad(mm, 2)}:${ss.toFixed(1).padStart(4, "0")}`;
  }

  function formatCoord(ra, dec) {
    const raDeg = ((ra % 360) + 360) % 360;
    return {
      raDeg, decDeg: dec,
      raHMS: raToHMS(raDeg), decDMS: decToDMS(dec),
      decimalStr: `${raDeg.toFixed(5)}, ${dec >= 0 ? "+" : ""}${dec.toFixed(5)} (J2000)`,
      sexagesimalStr: `${raToHMS(raDeg)} ${decToDMS(dec)} (J2000)`,
    };
  }

  // Field-of-view [deg] and cone radius [arcmin] scaled (loosely) to the galaxy's angular extent so
  // the imaging/atlas links frame the object. Defaults when no distance is known.
  function fovDegFromDistance(distMpc) {
    if (!distMpc || !isFinite(distMpc) || distMpc <= 0) return 0.25;
    const physMpc = 0.1;                              // ~100 kpc framing aperture
    const fov = (physMpc / distMpc) * (180 / Math.PI) * 2.0;
    return Math.min(2.0, Math.max(0.02, fov));
  }
  function coneRadiusArcmin() { return 2; }           // catalog cone-search radius (galaxies are sub-2')
  function legacyZoom(fovDeg) {
    return Math.max(10, Math.min(16, Math.round(16 - Math.log2(fovDeg / 0.05))));
  }

  // DSS2 colour thumbnail at the position (reuses the CDS hips2fits shape from build_texture_atlas.py).
  function thumbnailUrl(ra, dec, distMpc, px) {
    const fov = fovDegFromDistance(distMpc);
    px = px || 256;
    return `https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips=${enc("CDS/P/DSS2/color")}` +
      `&ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}&fov=${fov.toFixed(5)}&width=${px}&height=${px}` +
      `&projection=TAN&format=jpg`;
  }

  // Ordered external links for a sky position. Name-only links (NED-by-name, Wikipedia) are emitted only
  // when `name` is supplied (e.g. a resolved PGC). Coordinate links work for any object.
  function astroLinks(o) {
    const ra = o.ra, dec = o.dec;
    const R = coneRadiusArcmin();
    const fov = fovDegFromDistance(o.distMpc);
    const links = [];

    links.push({ group: "Object databases", label: "NED (cone search)",
      href: `https://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search` +
        `&in_csys=Equatorial&in_equinox=J2000.0&lon=${ra.toFixed(6)}d&lat=${dec.toFixed(6)}d` +
        `&radius=${R}&out_csys=Equatorial&out_equinox=J2000.0` });
    if (o.name) {
      links.push({ group: "Object databases", label: "NED (by name)",
        href: `https://ned.ipac.caltech.edu/byname?objname=${enc(o.name)}` });
    }
    links.push({ group: "Object databases", label: "SIMBAD (cone search)",
      href: `https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=${ra.toFixed(6)}+${dec.toFixed(6)}` +
        `&CooFrame=ICRS&CooEpoch=2000&CooEqui=2000&Radius=${R}&Radius.unit=arcmin&submit=submit+query` });

    links.push({ group: "Sky & imaging", label: "Aladin Lite",
      href: `https://aladin.cds.unistra.fr/AladinLite/?target=${ra.toFixed(6)}%20${dec.toFixed(6)}` +
        `&fov=${fov.toFixed(4)}&survey=${enc("P/DSS2/color")}` });
    links.push({ group: "Sky & imaging", label: "Legacy Survey viewer",
      href: `https://www.legacysurvey.org/viewer?ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}` +
        `&layer=ls-dr10&zoom=${legacyZoom(fov)}` });

    if (o.name) {
      links.push({ group: "Reference", label: "Wikipedia (search)",
        href: `https://en.wikipedia.org/wiki/Special:Search?search=${enc(o.name)}` });
    }
    return links;
  }

  // One tab-separated clipboard row of everything known about the object.
  function dataRow(o) {
    const c = formatCoord(o.ra, o.dec);
    const cells = [
      o.name || (o.pgc && o.pgc > 0 ? `PGC ${o.pgc}` : "anon"),
      c.raHMS, c.decDMS, o.ra.toFixed(6), o.dec.toFixed(6),
    ];
    if (o.z != null && isFinite(o.z)) cells.push(`z=${o.z.toFixed(4)}`);
    if (o.distMpc != null && isFinite(o.distMpc)) cells.push(`${o.distMpc.toFixed(1)}Mpc`);
    if (o.ksMag != null && isFinite(o.ksMag)) cells.push(`Ks=${o.ksMag.toFixed(2)}`);
    return cells.join("\t");
  }

  return { formatCoord, raToHMS, decToDMS, fovDegFromDistance, coneRadiusArcmin, legacyZoom,
           thumbnailUrl, astroLinks, dataRow };
});
