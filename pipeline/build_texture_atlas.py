"""Pre-bake a galaxy-image texture atlas for the local-neighborhood viewer.

For the real (PROV=0) galaxies in a brightness/distance tier, fetch a color image cutout per
galaxy through a per-galaxy *best-available survey* waterfall (DESI Legacy DR10 → Pan-STARRS1 →
DSS2 color / 2MASS, via the CDS ``hips2fits`` service), pack the validated cutouts into a few
fixed-grid texture-atlas sheets, and emit the per-galaxy atlas index + geometry the viewer needs.

Only ``data_release/local/local_2mpp_observed.npz`` (all PROV=0) is read, so the synthetic
PROV=5 galaxies are *structurally* excluded — they never get a real image (they have no real
counterpart). Galaxies for which every survey returns a blank/no-coverage tile (e.g. deep Zone of
Avoidance) get atlas index −1 and stay a sprite in the viewer; no broken textures.

The fetch is **resumable**: each accepted cutout is cached under ``data/local/cutouts/`` and a
JSON fetch-log records the survey used / failures, so re-runs only fetch what is missing. A
browser User-Agent + retry/backoff + a global rate limit (the proven pattern from
``pipeline/build_report.py::fetch_gallery_cutouts``) keep the one-time bulk fetch polite.

Outputs (default ``docs/visualizer-local/data/``):
  atlas/sheet_NNN.webp        the packed atlas sheets (sheet_px², tile_px tiles, 2px gutter)
  atlas_manifest.json         sheet list, tile/sheet geometry, per-sheet sha256, counts
  atlas_galaxies.npz          selected galaxies: ra,dec,dist_mpc,ksmag,xyz, ang_size_arcmin,
                              b_a, pa_deg, survey_code, atlas_tile_index, obs_index

    JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 pipeline/build_texture_atlas.py \
        --kmax 11.5 --workers 8 --out-dir docs/visualizer-local/data
    # quick smoke run: --max-galaxies 200 (brightest first)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.galaxy_geometry import (enrich_geometry, SURVEY_HIPS, SURVEY_NAME,
                                            SURVEY_NONE)

OBS = os.path.join("data_release", "local", "local_2mpp_observed.npz")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"


def _xyz(ra, dec, dist):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    r = np.asarray(dist, float); cd = np.cos(dec)
    return np.column_stack([r * cd * np.cos(ra), r * cd * np.sin(ra), r * np.sin(dec)]).astype(np.float32)


class RateLimiter:
    """Token-bucket-ish global throttle: at least ``min_interval`` s between any two requests."""
    def __init__(self, min_interval):
        self._lock = threading.Lock()
        self._next = 0.0
        self.min_interval = float(min_interval)

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.min_interval


def _hips_url(hips, ra, dec, fov_deg, px):
    return (f"{HIPS2FITS}?hips={urllib.parse.quote(hips)}&ra={ra:.6f}&dec={dec:.6f}"
            f"&fov={fov_deg:.5f}&width={px}&height={px}&projection=TAN&format=jpg")


def _fetch_jpeg(url, limiter, retries=4, timeout=45):
    """Fetch a JPEG (validated by magic bytes) with retry/backoff; bytes or None."""
    last = None
    for attempt in range(retries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            data = urllib.request.urlopen(req, timeout=timeout).read()
            if len(data) > 1200 and data[:2] == b"\xff\xd8":
                return data
            last = f"short/invalid ({len(data)} bytes)"
        except Exception as e:                                  # noqa: BLE001 (network)
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    return None


def _is_blank(jpeg_bytes, *, mean_floor=6.0, bright_frac_floor=0.015, bright_level=18,
             sat_frac=0.55, detail_floor=6.0):
    """True if the cutout is unusable, so the survey waterfall should fall through.

    Two failure modes: (1) **no coverage** — a HiPS cutout outside a survey footprint comes back
    near-black rather than failing (dark mean AND almost no bright pixels); (2) **saturated /
    no-detail** — the brightest nearby galaxies blow out to a flat white (or a single extreme hue)
    with no structure, which reads worse than a shallower survey would. Reject when most pixels are
    near-white OR the image has almost no spatial contrast (std below ``detail_floor``)."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
    except Exception:
        return True
    a = np.asarray(im, dtype=np.float32)
    if a.size == 0:
        return True
    no_coverage = (a.mean() < mean_floor) and ((a > bright_level).mean() < bright_frac_floor)
    saturated = (a > 245).mean() > sat_frac                  # blown-out white
    no_detail = a.std() < detail_floor                       # flat (no structure either extreme)
    return bool(no_coverage or saturated or no_detail)


def _select(obs, kmax, dmax, max_galaxies):
    """PROV=0 galaxies in the tier, brightest-first; returns column dict + obs row indices."""
    k = obs["ksmag"]; prov = obs["prov"]; d = obs["dist_mpc"]
    m = (prov == 0) & (k < kmax)
    if dmax is not None:
        m &= d < dmax
    idx = np.where(m)[0]
    idx = idx[np.argsort(k[idx])]                              # brightest first
    if max_galaxies and len(idx) > max_galaxies:
        idx = idx[:max_galaxies]
    return idx


def _load_xmatch_tables():
    """Load optional SGA-2020 / HyperLEDA cross-match tables if a fetcher cached them."""
    out = {}
    for tag, path in (("sga", os.path.join("data", "local", "sga", "sga_geometry.npz")),
                      ("leda", os.path.join("data", "local", "hyperleda", "leda_geometry.npz"))):
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            out[tag] = {k: z[k] for k in z.files}
    return out.get("sga"), out.get("leda")


def build(args):
    obs = np.load(OBS)
    sel = _select(obs, args.kmax, args.dmax, args.max_galaxies)
    ra = obs["ra"][sel].astype(float); dec = obs["dec"][sel].astype(float)
    dist = obs["dist_mpc"][sel].astype(float); ksmag = obs["ksmag"][sel].astype(float)
    n = len(sel)
    print(f"[atlas] tier K<{args.kmax}"
          f"{f' d<{args.dmax}' if args.dmax else ''}: {n} PROV=0 galaxies"
          f"{f' (capped from selection)' if args.max_galaxies else ''}")

    sga, leda = _load_xmatch_tables()
    geom = enrich_geometry(ra, dec, dist, ksmag, sga=sga, leda=leda)
    ang = geom.ang_size_arcmin
    print(f"[atlas] geometry: ang_size med={np.median(ang):.2f}' "
          f"(sga/leda matched {int((geom.geom_source!='estimated').sum())}/{n})")

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    log_path = os.path.join(cache_dir, "_fetch_log.json")
    fetch_log = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            fetch_log = json.load(f)

    limiter = RateLimiter(args.min_interval)

    def cutout_path(oidx):
        return os.path.join(cache_dir, f"{int(oidx):06d}.jpg")

    if args.revalidate:                                       # re-check cached tiles vs the
        bad = 0                                               # current quality test; drop failures
        for i in range(n):                                    # so the fetch loop re-tries them
            p = cutout_path(int(sel[i]))
            if os.path.exists(p) and _is_blank(open(p, "rb").read()):
                os.remove(p); fetch_log.pop(str(int(sel[i])), None); bad += 1
        print(f"[atlas] revalidate: {bad} cached tiles failed the quality test → re-fetch")

    def fetch_one(i):
        oidx = int(sel[i]); key = str(oidx)
        out = cutout_path(oidx)
        if key in fetch_log and (fetch_log[key].get("survey", -1) >= 0) and os.path.exists(out):
            return i, fetch_log[key]["survey"]                 # cached success
        if key in fetch_log and fetch_log[key].get("survey", -1) < 0 and not args.refetch_failed:
            return i, SURVEY_NONE                              # cached failure
        fov = float(np.clip(2.5 * ang[i] / 60.0, args.fov_min, args.fov_max))
        for code in geom.survey_pref[i]:
            code = int(code)
            if code < 0:
                continue
            data = _fetch_jpeg(_hips_url(SURVEY_HIPS[code], ra[i], dec[i], fov, args.fetch_px),
                               limiter, retries=args.retries)
            if data is not None and not _is_blank(data):
                with open(out, "wb") as f:
                    f.write(data)
                return i, code
        return i, SURVEY_NONE

    survey_used = np.full(n, SURVEY_NONE, np.int16)
    t0 = time.monotonic(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, code in ex.map(fetch_one, range(n)):
            survey_used[i] = code
            fetch_log[str(int(sel[i]))] = {"survey": int(code)}
            done += 1
            if done % 200 == 0 or done == n:
                ok = int((survey_used >= 0).sum())
                rate = done / max(time.monotonic() - t0, 1e-6)
                print(f"  [{done}/{n}] fetched, {ok} with imagery, {rate:.1f}/s")
                with open(log_path, "w") as f:                 # checkpoint (resumable)
                    json.dump(fetch_log, f)
    with open(log_path, "w") as f:
        json.dump(fetch_log, f)

    # ---- pack accepted cutouts into atlas sheets ----
    from PIL import Image
    tile = args.tile_px; gutter = args.gutter; sheet = args.sheet_px
    inner = tile - 2 * gutter
    per_row = sheet // tile
    per_sheet = per_row * per_row
    have = np.where(survey_used >= 0)[0]
    print(f"[atlas] packing {len(have)} tiles into {tile}px cells "
          f"({per_sheet}/sheet) on {sheet}px sheets")

    atlas_index = np.full(n, -1, np.int32)
    sheets = []
    cur = None
    for slot, i in enumerate(have):
        s, within = divmod(slot, per_sheet)
        if within == 0:
            if cur is not None:
                sheets.append(cur)
            cur = Image.new("RGB", (sheet, sheet), (0, 0, 0))
        row, col = divmod(within, per_row)
        try:
            im = Image.open(cutout_path(int(sel[i]))).convert("RGB").resize((inner, inner),
                                                                            Image.LANCZOS)
            cur.paste(im, (col * tile + gutter, row * tile + gutter))
            atlas_index[i] = slot
        except Exception as e:                                 # corrupt cache → leave as -1
            print(f"  [pack warn] obs{int(sel[i])}: {e}")
    if cur is not None:
        sheets.append(cur)

    out_dir = args.out_dir
    atlas_dir = os.path.join(out_dir, "atlas")
    os.makedirs(atlas_dir, exist_ok=True)
    sheet_meta = []
    for s, img in enumerate(sheets):
        p = os.path.join(atlas_dir, f"sheet_{s:03d}.webp")
        img.save(p, "WEBP", quality=args.webp_quality, method=4)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        sheet_meta.append({"file": f"atlas/sheet_{s:03d}.webp", "sha256": h,
                           "bytes": os.path.getsize(p)})

    manifest = {
        "product": "local_2mpp_texture_atlas",
        "tier": {"kmax": args.kmax, "dmax": args.dmax, "n_selected": int(n),
                 "n_textured": int((atlas_index >= 0).sum())},
        "tile_px": tile, "gutter_px": gutter, "sheet_px": sheet,
        "tiles_per_row": per_row, "tiles_per_sheet": per_sheet, "n_sheets": len(sheets),
        "survey_hips": {SURVEY_NAME[c]: h for c, h in SURVEY_HIPS.items()},
        "sheets": sheet_meta,
        "note": "atlas_tile_index in atlas_galaxies.npz: -1=sprite-only; else sheet=idx//tiles_"
                "per_sheet, within=idx%tiles_per_sheet, row=within//tiles_per_row, col=within%"
                "tiles_per_row; uv0=(col*tile_px/sheet_px, row*tile_px/sheet_px) + gutter.",
    }
    with open(os.path.join(out_dir, "atlas_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    np.savez(os.path.join(out_dir, "atlas_galaxies.npz"),
             obs_index=sel.astype(np.int32), ra=ra.astype(np.float32),
             dec=dec.astype(np.float32), dist_mpc=dist.astype(np.float32),
             ksmag=ksmag.astype(np.float32), xyz=_xyz(ra, dec, dist),
             ang_size_arcmin=ang.astype(np.float32), b_a=geom.b_a.astype(np.float32),
             pa_deg=geom.pa_deg.astype(np.float32),
             survey_code=survey_used.astype(np.int16), atlas_tile_index=atlas_index)

    by = {SURVEY_NAME[c]: int((survey_used == c).sum()) for c in SURVEY_HIPS}
    print(f"[atlas] wrote {len(sheets)} sheet(s) + manifest + atlas_galaxies.npz to {out_dir}")
    print(f"[atlas] textured {int((atlas_index>=0).sum())}/{n}  by survey: {by}  "
          f"no-image {int((survey_used<0).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=float, default=11.5, help="K_s brightness tier (broad=11.5)")
    ap.add_argument("--dmax", type=float, default=None, help="max distance [Mpc] (optional)")
    ap.add_argument("--max-galaxies", type=int, default=None,
                    help="cap (brightest-first) for a quick smoke run")
    ap.add_argument("--out-dir", default=os.path.join("docs", "visualizer-local", "data"))
    ap.add_argument("--cache-dir", default=os.path.join("data", "local", "cutouts"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-interval", type=float, default=0.05, help="global s between requests")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--fetch-px", type=int, default=256, help="cutout fetch resolution")
    ap.add_argument("--tile-px", type=int, default=128)
    ap.add_argument("--gutter", type=int, default=2)
    ap.add_argument("--sheet-px", type=int, default=4096)
    ap.add_argument("--webp-quality", type=int, default=82)
    ap.add_argument("--fov-min", type=float, default=0.02, help="min cutout FOV [deg]")
    ap.add_argument("--fov-max", type=float, default=0.5, help="max cutout FOV [deg]")
    ap.add_argument("--refetch-failed", action="store_true",
                    help="retry galaxies previously logged as no-image")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check cached tiles against the quality test; re-fetch any that fail")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
