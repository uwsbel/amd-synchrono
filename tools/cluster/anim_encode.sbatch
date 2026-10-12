#!/usr/bin/env bash
# Encode the finished frame directories into mp4s (per camera + combined).
# Submitted by submit_anim.sh with a dependency on the chunk array; can also be
# run by hand:  OUTDIR=... FPS=30 sbatch tools/cluster/anim_encode.sbatch
#SBATCH --job-name=amd0831-encode
#SBATCH --output=renders/slurm-encode-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=sbel,research
set -euo pipefail
: "${OUTDIR:?}"
# Euler's documented conda init. `conda` IS on PATH on a compute node but only as a
# binary, and `conda activate` needs the shell function -- so the module's
# bootstrap-conda (which evals `conda shell.bash hook`) has to run unconditionally,
# not only when `command -v conda` fails.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/miniforge 2>/dev/null || true
declare -F bootstrap-conda >/dev/null && bootstrap-conda || true
if ! declare -F conda >/dev/null; then
  for _c in /opt/apps/miniforge/x86_64/*/bin/conda "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [[ -x "$_c" ]] && eval "$("$_c" shell.bash hook)" && break
  done
fi
conda activate "${CONDA_ENV:-amdrender}"
python3 tools/render_site_anim.py --dataset data --outdir "$OUTDIR" \
    --fps "${FPS:-30}" ${VIEWS:+--views "$VIEWS"} --encode-only
ls -la "$OUTDIR"/*.mp4
