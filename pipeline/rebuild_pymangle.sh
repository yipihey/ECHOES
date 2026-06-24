#!/usr/bin/env bash
# Rebuild pymangle for the ECHOES venv (~/.venv/k3d) under NumPy 2.x.
#
# The PyPI pymangle 0.9.4 wheel was built against the NumPy 1.x C-ABI (import fails with
# "numpy.core.multiarray failed to import"), and its sdist setup.py uses the removed
# numpy.distutils. The GitHub source already uses plain setuptools, so we build IT from source
# against the venv's NumPy 2.x headers. The only missing piece is Python.h (no system
# python3.11-devel); we borrow the matching cp311 headers from the anaconda install (same ABI).
set -euo pipefail

VENV="${VENV:-$HOME/.venv/k3d}"
PY="$VENV/bin/python3"
PYINC="${PYINC:-$HOME/.local/share/anaconda3/include/python3.11}"   # Python.h (cp311)
SRC="${SRC:-/tmp/pymangle_src}"

[ -f "$PYINC/Python.h" ] || { echo "Python.h not found at $PYINC; set PYINC to a cp311 include dir"; exit 1; }

rm -rf "$SRC"
git clone --depth 1 https://github.com/esheldon/pymangle.git "$SRC"
cd "$SRC"
# build_ext --inplace compiles _mangle against the venv NumPy 2.x (setup.py adds numpy.get_include()).
C_INCLUDE_PATH="$PYINC" "$PY" setup.py build_ext --inplace

# install ONLY the freshly-built .so over the venv's broken one (avoid `setup.py install`, which
# easy_installs deps and clobbers NumPy). Copy from a CWD outside $SRC so it isn't shadowed.
SO=$(ls "$SRC"/pymangle/_mangle.cpython-*.so)
for d in "$VENV"/lib64/python3.11/site-packages/pymangle "$VENV"/lib/python3.11/site-packages/pymangle; do
    [ -d "$d" ] && cp -v "$SO" "$d/"
done

cd "$HOME" && "$PY" -c "import pymangle, numpy; print('pymangle OK under numpy', numpy.__version__)"
