"""Guard: the clean package carries no leftover names from the research repo."""
import glob, os, re

SRC = glob.glob(os.path.join(os.path.dirname(__file__), "..", "echoes", "**", "*.py"),
                recursive=True)
LEGACY = re.compile(r"\b(twopt_density|morton_cascade|from \.observed_ls|from \.density_field|"
                    r"from \.posterior_sampler|from \.clustering_corrfunc|from \.weights_graphgp|"
                    r"from \.quaia|from \.desi|from \.cmass_targets|from \.twoMRS|load_2mrs)\b")

def test_no_legacy_references():
    bad = []
    for f in SRC:
        for i, line in enumerate(open(f), 1):
            if LEGACY.search(line):
                bad.append(f"{os.path.relpath(f)}:{i}: {line.strip()}")
    assert not bad, "leftover legacy references:\n" + "\n".join(bad)
