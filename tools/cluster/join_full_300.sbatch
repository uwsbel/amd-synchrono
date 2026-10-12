#!/usr/bin/env bash
# Join the five contiguous 60 s windows (t=0..300 s, frames 1..9000) into one
# 5-minute mp4. Same one-pass-from-PNGs approach as join_full.sbatch: encoding
# from the sources avoids the generation loss of re-compressing five finished
# mp4s, and symlinking each frame number once removes the frames the clips share
# at their seams (1801, 3601, and any other overlap), which would otherwise show
# as a stutter.
#SBATCH --job-name=amd0831-join300
#SBATCH --output=renders/slurm-join300-%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=sbel,research
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda/miniforge 2>/dev/null || true
declare -F bootstrap-conda >/dev/null && bootstrap-conda || true
conda activate "${CONDA_ENV:-amdrender}"

cd "$SLURM_SUBMIT_DIR"
OUT=renders/full_0-300
rm -rf "$OUT/frames"; mkdir -p "$OUT/frames"

python3 - <<"PY"
import os
OUT = "renders/full_0-300/frames"
# later windows win where two clips share a seam frame; the shared frame is the
# same instant rendered by both, so either copy is correct
SRC = ["renders/anim_0-60/overview", "renders/anim_60-120/overview",
       "renders/anim_120-180_zoom/overview", "renders/anim_180-240_follow/overview",
       "renders/anim_240-300_follow/overview"]
seen = {}
for d in SRC:
    for f in os.listdir(d):
        if f.startswith("frame_") and f.endswith(".png"):
            seen[int(f[6:-4])] = os.path.abspath(os.path.join(d, f))
nums = sorted(seen)
missing = [n for n in range(1, 9001) if n not in seen]
print(f"frames {nums[0]}..{nums[-1]}: {len(nums)} unique, {len(missing)} missing vs 1..9000")
if missing:
    raise SystemExit(f"gaps in the joined timeline: {missing[:10]}")
for n in nums:
    os.symlink(seen[n], os.path.join(OUT, f"frame_{n:04d}.png"))
print(f"linked {len(nums)} frames")
PY

ffmpeg -y -loglevel error -framerate 30 -start_number 1 \
  -i "$OUT/frames/frame_%04d.png" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart \
  "$OUT/full_0-300.mp4"
rm -rf "$OUT/frames"
ffprobe -v error -show_entries format=duration -show_entries stream=nb_frames,width,height \
  -of default=noprint_wrappers=1 "$OUT/full_0-300.mp4"
ls -la "$OUT/full_0-300.mp4"
