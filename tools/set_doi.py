#!/usr/bin/env python3
"""Replace the Zenodo DOI placeholder across the repo once the deposit is published.

    python tools/set_doi.py 10.5281/zenodo.1234567
"""
import sys, glob, re

PLACEHOLDER = "10.5281/zenodo." + "XXXXXXX"
TARGETS = ["DATA.md", "README.md", "CITATION.cff", "data_release/README.md"]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/set_doi.py 10.5281/zenodo.NNNNNNN")
    doi = sys.argv[1].replace("https://doi.org/", "").strip()
    if not re.match(r"10\.5281/zenodo\.\d+$", doi):
        sys.exit(f"not a Zenodo DOI: {doi}")
    n = 0
    for f in TARGETS:
        try:
            s = open(f).read()
        except FileNotFoundError:
            continue
        if PLACEHOLDER in s:
            open(f, "w").write(s.replace(PLACEHOLDER, doi)); n += 1; print("updated", f)
    print(f"replaced the DOI placeholder in {n} files -> {doi}")


if __name__ == "__main__":
    main()
