#!/bin/sh
# Dependency install for the API and Space images (run at build time from /app).
#
#   ml_cpu.sh sync <uv sync args>   uv sync --frozen --no-dev <args>; with INSTALL_ML=1 also
#                                   --extra ml --inexact, minus torch, torchaudio and the CUDA
#                                   packages (docker/ml-cpu-skip.txt)
#   ml_cpu.sh torch                 with INSTALL_ML=1: install the CPU-only torch and
#                                   torchaudio pinned in docker/torch-cpu.txt (hash-checked),
#                                   then check the result
#                                   (docker/check_ml_cpu.py); otherwise do nothing
#   ml_cpu.sh check                 with INSTALL_ML=1: run that check again (after the last sync)
#
# Without INSTALL_ML=1 the sync is exactly the stub image's previous `uv sync` command.
set -eu
here=$(dirname "$0")
cmd=${1:?usage: ml_cpu.sh sync ARGS... | torch | check}
shift

case "$cmd" in
    sync)
        if [ "${INSTALL_ML:-0}" = "1" ]; then
            skip=$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' -e 's/^/--no-install-package /' \
                "$here/ml-cpu-skip.txt")
            # --inexact: keep the pinned CPU torch/torchaudio, which uv.lock does not describe
            # (an exact sync removes them, --no-install-package notwithstanding).
            # shellcheck disable=SC2086  # one flag per word
            set -- "$@" --extra ml --inexact $skip
        fi
        exec uv sync --frozen --no-dev "$@"
        ;;
    torch)
        [ "${INSTALL_ML:-0}" = "1" ] || exit 0
        uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --require-hashes \
            --index-url https://download.pytorch.org/whl/cpu -r "$here/torch-cpu.txt"
        exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$here/check_ml_cpu.py" uv.lock "$here/torch-cpu.txt"
        ;;
    check)
        [ "${INSTALL_ML:-0}" = "1" ] || exit 0
        exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$here/check_ml_cpu.py" uv.lock "$here/torch-cpu.txt"
        ;;
    *)
        echo "ml_cpu.sh: unknown command $cmd" >&2
        exit 2
        ;;
esac
