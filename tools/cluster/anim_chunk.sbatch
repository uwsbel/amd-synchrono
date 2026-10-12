#!/usr/bin/env bash
# One array task = one time chunk of the site animation, rendered through
# tools/render_site_anim_patched.py (heightmap fix + hull lift + lunar-manip
# textures + SCM ruts). Submitted by tools/cluster/submit_anim.sh.
#
# Frames are numbered from the sample time, so concurrent chunks tile one shared
# --outdir with no clashes, and frames already on disk are skipped (safe re-runs).
#SBATCH --job-name=amd0831-chunk
#SBATCH --output=renders/slurm-chunk-%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=sbel,research
#SBATCH --gres=gpu:1
set -euo pipefail
: "${START:?}" "${END:?}" "${CHUNK:?}" "${OUTDIR:?}"
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

t0=$(python3 -c "print($START + $SLURM_ARRAY_TASK_ID * $CHUNK)")
t1=$(python3 -c "print(min($START + ($SLURM_ARRAY_TASK_ID + 1) * $CHUNK, $END))")
echo "chunk $SLURM_ARRAY_TASK_ID: t=$t0..$t1 -> $OUTDIR  (host $(hostname))"
nvidia-smi -L 2>/dev/null || echo "no GPU on this node (CPU render)"

mkdir -p "$OUTDIR/blend"
# shellcheck disable=SC2086  # EXTRA is a flat flag list by design
USE_BPY=1 bash tools/cluster/run_script.sh tools/render_site_anim_patched.py \
    --dataset data --meshes meshes --mesh-map meshmap.json \
    --heightmap assets/data/terrain/terrain2_graded.png \
    --camera-plan renders/camera_plan.json \
    --start "$t0" --end "$t1" \
    --outdir "$OUTDIR" --out "$OUTDIR/blend/chunk_${SLURM_ARRAY_TASK_ID}.blend" \
    --no-encode ${EXTRA:-}
