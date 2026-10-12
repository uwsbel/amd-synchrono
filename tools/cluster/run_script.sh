#!/usr/bin/env bash
# run_script.sh SCRIPT.py [args...]
#
# Runs a Blender-python script from tools/ either way:
#   * with a Blender binary, if one is available ($BLENDER overrides the name), or
#   * with the pip `bpy` module in the current python (the cluster/conda path).
# Set USE_BPY=1 to force the bpy-module path even when a blender binary exists.
set -euo pipefail
script=$1; shift
if [[ "${USE_BPY:-0}" != 1 ]] && command -v "${BLENDER:-blender}" >/dev/null 2>&1; then
  exec "${BLENDER:-blender}" --background --factory-startup --python "$script" -- "$@"
fi
exec python3 "$script" -- "$@"
