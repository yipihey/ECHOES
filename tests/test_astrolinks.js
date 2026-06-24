/*
 * astrolinks.js unit tests (GPU-free; `node tests/test_astrolinks.js`).
 * Asserts coordinate formatting and the EXACT external-service URL templates.
 */
const assert = require("assert");
const AL = require("../apps/echoes-viewer/astrolinks.js");

// M87-ish position for legible HMS/DMS
const ra = 187.70593, dec = 12.39112;

// ---- formatCoord ----
const c = AL.formatCoord(ra, dec);
assert.strictEqual(c.raHMS, "12:30:49.42", "RA HMS");          // 187.70593/15 h
assert.strictEqual(c.decDMS, "+12:23:28.0", "Dec DMS");
assert.ok(c.decimalStr.includes("187.70593") && c.decimalStr.includes("+12.39112"), "decimal");
assert.ok(c.sexagesimalStr.includes("J2000"));
// negative dec sign + wrap
assert.ok(AL.formatCoord(0, -5.5).decDMS.startsWith("-05:"), "negative Dec sign");
assert.strictEqual(AL.formatCoord(-1, 0).raDeg.toFixed(0), "359", "RA wrap to [0,360)");

// ---- astroLinks: exact hrefs, anonymous object (no name) ----
const links = AL.astroLinks({ ra, dec });
const by = (label) => { const l = links.find((x) => x.label === label); assert.ok(l, `missing ${label}`); return l.href; };

assert.strictEqual(by("NED (cone search)"),
  "https://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search" +
  "&in_csys=Equatorial&in_equinox=J2000.0&lon=187.705930d&lat=12.391120d" +
  "&radius=2&out_csys=Equatorial&out_equinox=J2000.0");
assert.strictEqual(by("SIMBAD (cone search)"),
  "https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=187.705930+12.391120" +
  "&CooFrame=ICRS&CooEpoch=2000&CooEqui=2000&Radius=2&Radius.unit=arcmin&submit=submit+query");
assert.ok(by("Aladin Lite").startsWith(
  "https://aladin.cds.unistra.fr/AladinLite/?target=187.705930%2012.391120&fov="), "Aladin");
assert.ok(by("Aladin Lite").includes("survey=P%2FDSS2%2Fcolor"), "Aladin survey encoded");
assert.ok(by("Legacy Survey viewer").startsWith(
  "https://www.legacysurvey.org/viewer?ra=187.705930&dec=12.391120&layer=ls-dr10&zoom="), "Legacy");

// name-only links absent when anonymous
assert.ok(!links.find((l) => l.label.includes("by name")), "no NED-by-name when anonymous");
assert.ok(!links.find((l) => l.label.includes("Wikipedia")), "no Wikipedia when anonymous");

// ---- with a name/PGC: NED-by-name + Wikipedia appear, encoded ----
const named = AL.astroLinks({ ra, dec, name: "PGC 41361", distMpc: 16.5 });
const ned2 = named.find((l) => l.label.includes("by name"));
assert.ok(ned2 && ned2.href === "https://ned.ipac.caltech.edu/byname?objname=PGC%2041361", "NED by name encoded");
const wiki = named.find((l) => l.label.includes("Wikipedia"));
assert.ok(wiki && wiki.href === "https://en.wikipedia.org/wiki/Special:Search?search=PGC%2041361", "Wikipedia search encoded");

// ---- thumbnail + dataRow ----
assert.ok(AL.thumbnailUrl(ra, dec, 100).includes("hips2fits") &&
          AL.thumbnailUrl(ra, dec, 100).includes("CDS%2FP%2FDSS2%2Fcolor"), "thumbnail url");
const row = AL.dataRow({ ra, dec, z: 0.004, distMpc: 16.5, ksMag: 5.81, pgc: 41361 });
assert.ok(row.split("\t").length >= 6 && row.includes("PGC 41361") && row.includes("z=0.0040"), "dataRow");

// legacyZoom monotonic in FOV
assert.ok(AL.legacyZoom(0.05) > AL.legacyZoom(0.5), "zoom larger for smaller FOV");

console.log("astrolinks: ALL ASSERTIONS PASSED");
