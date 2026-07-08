#!/bin/bash
# setup_vae_bo.sh — OURS. Install the vae_bo baseline generator for validation.
#
# The vae_bo adapter in validation/generators/vae_bo/ is a THIN wrapper; the
# heavy upstream code is installed here (not vendored into the repo). See
# validation/generators/README.md for the entrant table and the boundary rule.
#
# STATUS: placeholder stub — not implemented yet. Fill in the upstream clone /
# pip install / weights download below, mirroring setup_reinvent.sh.

if [ $# -eq 0 ]; then
    echo "Usage: $0 <workspace_directory>"
    exit 1
fi

WORKSPACE=$1
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

echo "[setup_vae_bo] TODO: clone + install the vae_bo baseline into $WORKSPACE"
echo "[setup_vae_bo] not implemented yet — see validation/generators/README.md"
exit 1
