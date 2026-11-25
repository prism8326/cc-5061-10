#!/usr/bin/env bash
# Build the bkflow C extension against the system CPython without python3-dev.
# Headers are downloaded automatically on first build into .pyheaders.
set -euo pipefail
cd "$(dirname "$0")/.."

PYV="$(python3 -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
HDR_ROOT="${PYHEADER_DIR:-$PWD/.pyheaders}"
SRC="$HDR_ROOT/Python-$PYV"

if [ ! -f "$SRC/Include/Python.h" ]; then
  mkdir -p "$HDR_ROOT"
  echo "[build] downloading CPython $PYV headers..."
  curl -sSL "https://www.python.org/ftp/python/$PYV/Python-$PYV.tgz" -o "$HDR_ROOT/py.tgz"
  tar xzf "$HDR_ROOT/py.tgz" -C "$HDR_ROOT"
fi
if [ ! -f "$SRC/pyconfig.h" ]; then
  echo "[build] generating pyconfig.h (./configure)..."
  (cd "$SRC" && ./configure --quiet)
fi
NUMPY_INC="$(python3 -c 'import numpy;print(numpy.get_include())')"

gcc -shared -fPIC -O3 -Wall -Wextra \
  -I"$SRC/Include" -I"$SRC" -I"$NUMPY_INC" \
  seamcarver/ext/bkflow.c -o seamcarver/bkflow.so
echo "[build] seamcarver/bkflow.so ready"
python3 -c "import sys;sys.path.insert(0,'.');from seamcarver import bkflow;print('import ok:',bkflow.grid_mincut.__doc__)"
