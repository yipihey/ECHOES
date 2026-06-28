"""Guard: the clean package carries no leftover *module* names from the research repo.

NB: ``Quaia``/``quasar``/``QSO`` are no longer forbidden — Quaia and DESI DR1 QSO are
now first-class ECHOES datasets (``echoes/surveys/quaia.py``, ``echoes/surveys/desi.py``),
ported as clean loaders that satisfy the survey/registry contract. What this guard still
catches is direct imports of the *research-repo* implementation modules (``twopt_density``,
``observed_ls``, ``density_field``, …), which must never leak into the clean package."""
import glob, os, re

SRC = glob.glob(os.path.join(os.path.dirname(__file__), "..", "echoes", "**", "*.py"),
                recursive=True)
LEGACY = re.compile(r"\b(twopt_density|morton_cascade|from \.observed_ls|from \.density_field|"
                    r"from \.posterior_sampler|from \.clustering_corrfunc|from \.weights_graphgp|"
                    r"from \.cmass_targets|from \.twoMRS|load_2mrs|DATA_MODEL)\b")

def test_no_legacy_references():
    bad = []
    for f in SRC:
        for i, line in enumerate(open(f), 1):
            if LEGACY.search(line):
                bad.append(f"{os.path.relpath(f)}:{i}: {line.strip()}")
    assert not bad, "leftover legacy references:\n" + "\n".join(bad)
