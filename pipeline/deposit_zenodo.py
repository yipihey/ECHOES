#!/usr/bin/env python3
"""Create a DRAFT Zenodo deposition for the ECHOES data products and upload them.

Leaves the deposition UNPUBLISHED so you can review and click "Publish" in the web
UI (which mints the DOI). Reserves the DOI immediately and prints it.

Requires a Zenodo personal access token with the ``deposit:write`` scope:
    export ZENODO_TOKEN=...          # from https://zenodo.org/account/settings/applications/tokens/new/
    python pipeline/deposit_zenodo.py            # production zenodo.org
    python pipeline/deposit_zenodo.py --sandbox  # test against sandbox.zenodo.org first

Uploads everything in data_release/ except the metadata file itself.
"""
import argparse, json, os, sys, urllib.request, urllib.error

FILES = ["cmass_south_posterior.npz", "cmass_south_randoms.npz",
         "draw_samples.py", "README.md", "SHA256SUMS"]


def _api(method, url, token, data=None, binary=False):
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}access_token={token}"
    headers = {}
    if data is not None and not binary:
        data = json.dumps(data).encode(); headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return json.loads(body) if body and not binary else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo API {method} {url.split('?')[0]} failed: {e.code}\n{e.read().decode()[:600]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    p.add_argument("--dir", default="data_release")
    args = p.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN (a token with deposit:write). See the module docstring.")
    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    meta = json.load(open(os.path.join(args.dir, "zenodo_metadata.json")))

    print(f"[zenodo] creating draft deposition on {base} ...")
    dep = _api("POST", f"{base}/api/deposit/depositions", token, data={})
    dep_id = dep["id"]; bucket = dep["links"]["bucket"]
    for fn in FILES:
        path = os.path.join(args.dir, fn)
        if not os.path.exists(path):
            print(f"  skip {fn} (not found)"); continue
        print(f"  uploading {fn} ({os.path.getsize(path)/1e6:.2f} MB) ...")
        with open(path, "rb") as fh:
            _api("PUT", f"{bucket}/{fn}", token, data=fh.read(), binary=True)
    print("[zenodo] setting metadata ...")
    _api("PUT", f"{base}/api/deposit/depositions/{dep_id}", token, data=meta)

    edit = f"{base}/deposit/{dep_id}"
    doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(reserved on publish)")
    print(f"\nDraft created (NOT published).\n  reserved DOI : {doi}\n  review/publish: {edit}\n"
          f"After publishing, run:  python tools/set_doi.py {doi}")


if __name__ == "__main__":
    main()
