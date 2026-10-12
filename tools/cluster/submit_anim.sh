#!/usr/bin/env bash
# Submit the site animation as a SLURM array + a dependent encode.
#
#   tools/cluster/submit_anim.sh START END CHUNK OUTDIR [render_site_anim_patched.py flags...]
#   e.g.  tools/cluster/submit_anim.sh 0 120 20 renders/anim_0-120 \
#             --fps 30 --res 1280x720 --samples 64 --views overview --scm
#
# One array task per CHUNK seconds of sim time, all writing into the shared OUTDIR;
# when the array succeeds, an encode job builds the mp4s. Re-running the same command
# later only renders the frames that are missing.
#
# MAXPAR=N throttles how many array tasks run at once; CONDA_ENV overrides the env.
set -euo pipefail
START=${1:?start seconds}; END=${2:?end seconds}; CHUNK=${3:?chunk seconds}; OUTDIR=${4:?outdir}
shift 4
EXTRA="$*"
N=$(python3 -c "import math; print(math.ceil(($END - $START) / $CHUNK))")
FPS=30
args=($EXTRA); VIEWS=""
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ "${args[i]}" == "--fps" ]] && FPS=${args[i+1]}
  [[ "${args[i]}" == "--views" ]] && VIEWS=${args[i+1]}
done
mkdir -p "$OUTDIR" renders

# The overview camera path is a whole-run precompute shared by every chunk, so the
# zoom is seamless across chunk boundaries -- generate it once, here.
if [[ ! -f renders/camera_plan.json ]]; then
  echo "generating renders/camera_plan.json (one full read of the recording) ..."
  python3 tools/make_camera_plan.py --dataset data --out renders/camera_plan.json
fi

echo "submitting $N chunks of ${CHUNK}s ($START..$END s) -> $OUTDIR (fps $FPS)"
jid=$(sbatch --parsable \
      --array=0-$((N - 1))${MAXPAR:+%$MAXPAR} \
      --export=ALL,START="$START",END="$END",CHUNK="$CHUNK",OUTDIR="$OUTDIR",EXTRA="$EXTRA" \
      tools/cluster/anim_chunk.sbatch)
echo "  chunk array job: $jid"
eid=$(sbatch --parsable --dependency=afterok:"$jid" \
      --export=ALL,OUTDIR="$OUTDIR",FPS="$FPS",VIEWS="$VIEWS" \
      tools/cluster/anim_encode.sbatch)
echo "  encode job:      $eid (runs after the array succeeds)"
echo "watch with: squeue -u \$USER ; results in $OUTDIR/<view>/ and $OUTDIR/*.mp4"
