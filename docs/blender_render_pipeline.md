# Blender render pipeline

This pipeline makes movies from a pose recording of the lunar construction site.
It renders in Blender Cycles on the euler cluster, one SLURM job for each 20 s of
simulation time.

The ground is real. The ruts come from the `rank_*_scm.bin` soil deformation data,
as geometry, not as a texture.

## 1. The scripts

| File | Function |
|---|---|
| `tools/blender_import.py` | Reads the recording: manifest, poses, mesh resolution |
| `tools/render_site_frame.py` | Builds the scene: terrain, bodies, meshes, materials |
| `tools/render_site_anim.py` | Keyframes the scene and gives the overview and chase cameras |
| `tools/render_bird.py` | Gives the movie look, fixes the heightmap and solves the camera framing |
| `tools/scm_terrain.py` | Makes the rut geometry from the soil deformation data |
| `tools/render_site_anim_patched.py` | The driver that you start. It joins the four files above |
| `tools/make_camera_plan.py` | Reads the full recording one time and writes the zoom schedule |
| `tools/cluster/submit_anim.sh` | Submits one SLURM array job for each 20 s chunk, and an encode job |
| `tools/cluster/anim_chunk.sbatch` | One array task renders one chunk |
| `tools/cluster/anim_encode.sbatch` | Makes the mp4 file from the PNG frames |
| `tools/cluster/join_full_300.sbatch` | Joins five 60 s clips into one 300 s movie |
| `tools/cluster/check_env.py` | Examines the environment before you render |
| `tools/cluster/meshmap.json` | Tells the two arms apart. Both use the same file names |

This branch adds no recording data. The vehicle meshes and the heightmap are
already in `data/` of this repository. Section 3 tells you how to use them.

## 2. What you must have

1. An account on euler, with access to the `sbel` or `research` partition.
2. A recording directory with 15 ranks. Each rank has `rank_N_frames.bin`,
   `rank_N_objects.jsonl`, `rank_N_meta.json` and `rank_N_scm.bin`, plus one
   `static_props.jsonl`.
3. A conda environment. Section 2.1 makes it.

The repository has all the assets:

| Asset | Place in the repository |
|---|---|
| Vehicle and arm meshes | `data/vehicle`, `data/lrv_robotarm`, `data/m113_builder_arm` |
| Rock meshes `rock1.obj`, `rock2.obj`, `rock3.obj` | `data/rocks` |
| Heightmap | `data/terrain` |
| Materials, key light and world (the "look") | `data/render_look/look.blend` |
| The 18 textures that the look uses | `data/render_look/textures` |

`look.blend` (97 KB) holds only the 4 materials, the light `Light.001` and the
`World` of the lunar-manip file `section2_mode2.blend` (132 MB). The look is the
same. To use a lunar-manip checkout, set the `LUNAR_MANIP` environment variable.

### 2.1 Make the conda environment

Euler needs `bootstrap-conda` before `conda activate`. The `conda` command is on
the PATH of a compute node, but only as a program. The shell function is not.

```bash
ssh euler
module load conda/miniforge
bootstrap-conda
conda create -y -n amdrender python=3.11
conda activate amdrender
pip install bpy==4.5.12 pillow numpy==1.26.4
conda install -y -c conda-forge ffmpeg
```

The `bpy` module is a pip wheel. The cluster has no Blender program.

Versions that work: Python 3.11.15, bpy 4.5.12 LTS, numpy 1.26.4, Pillow 12.3.0.

## 3. Make the working directory

Do not render inside the repository. The scripts read `data/` as the recording,
but `data/` in the repository is the Chrono asset directory. Make a separate
working directory for each recording.

```bash
REPO=~/amd-synchrono          # this branch, cloned on euler
WORK=~/render-run-NAME        # one directory for each recording

mkdir -p $WORK/renders $WORK/meshes $WORK/assets/data
cd $WORK

# the scripts
ln -s $REPO/tools tools
cp $REPO/tools/cluster/meshmap.json .

# the recording. Copy it or link it. Do not put it in git
ln -s /path/to/the/recording data

# the meshes. meshes/ must be one real directory tree.
# os.walk does not go into a symbolic link of a subdirectory
cp -r $REPO/data/vehicle $REPO/data/lrv_robotarm $REPO/data/m113_builder_arm meshes/
cp $REPO/data/rocks/rock[123].obj meshes/

# the heightmap. The name comes from rank_1_meta.json
ln -s $REPO/data/terrain assets/data/terrain
```

The meshes take approximately 190 MB.

A file name finds each mesh, not a path. The recording holds the paths of the
machine that made it, and the pipeline ignores them. So the meshes of this
repository work with a recording from any machine.

## 4. Examine the environment

```bash
cd $WORK
module load conda/miniforge && bootstrap-conda && conda activate amdrender
python3 tools/cluster/check_env.py
```

Exit code 0 means that the environment is ready.

## 5. Render the first minute

Always render 60 s first. One minute takes approximately 1.2 h, because the three
chunks run at the same time.

```bash
cd $WORK
module load conda/miniforge && bootstrap-conda && conda activate amdrender
tools/cluster/submit_anim.sh 0 60 20 renders/anim_0-60 \
  --fps 30 --res 1280x720 --samples 64 --views overview --scm
```

The arguments are: start second, end second, seconds for each chunk, output
directory. All other flags go to `render_site_anim_patched.py`.

The first call also writes `renders/camera_plan.json`. This is the zoom schedule,
and it reads the full recording one time. Each chunk uses the same file, thus the
zoom is smooth across the chunk limits.

**Make a new working directory for a new recording.** `submit_anim.sh` keeps an
existing `renders/camera_plan.json`. If you change `data/` but keep the old
directory, your new movie gets the zoom schedule of the old recording.

The result is `renders/anim_0-60/overview.mp4` and the PNG frames next to it.

## 6. The five-minute demo

These five commands make the 0 s to 300 s demo: an overview that moves back, a
push-in to three rovers, then a shot that follows one rover. Start them from `$WORK`.

```bash
C="--fps 30 --res 1280x720 --samples 64 --views overview --scm --scm-stride 3"

tools/cluster/submit_anim.sh 0 60 20 renders/anim_0-60 $C
tools/cluster/submit_anim.sh 60 120 20 renders/anim_60-120 $C
tools/cluster/submit_anim.sh 120 180 20 renders/anim_120-180_zoom $C \
  --zoom-targets 10:collector+11:collector+12:collector \
  --zoom-start 120 --zoom-end 180 --zoom-smooth 0
tools/cluster/submit_anim.sh 180 240 20 renders/anim_180-240_follow $C \
  --zoom-start-targets 10:collector+11:collector+12:collector \
  --zoom-start-radius 26 --zoom-start-elevation 28 \
  --zoom-targets 11:collector --zoom-radius 26 --zoom-elevation 28 \
  --zoom-start 180 --zoom-end 186
tools/cluster/submit_anim.sh 240 300 20 renders/anim_240-300_follow $C \
  --zoom-targets 11:collector --zoom-start-targets 11:collector \
  --zoom-radius 26 --zoom-elevation 28 \
  --zoom-start-radius 26 --zoom-start-elevation 28 \
  --zoom-start 240 --zoom-end 300

# after the five encode jobs are complete
sbatch --dependency=afterok:ID1:ID2:ID3:ID4 tools/cluster/join_full_300.sbatch
```

Use `+` and not `,` to separate the camera targets. `sbatch --export` splits its
own value at each comma, thus a comma does not arrive at the script.

The join writes `renders/full_0-300/full_0-300.mp4`, which has 9000 frames and is
exactly 300.000 s. It encodes one time from the PNG frames. It does not join the
mp4 files, because that makes a second compression and two double frames.

### 6.1 The camera targets

`--zoom-targets 11:collector` means the chassis of the collector of rank 11.
The ranks 10, 11 and 12 are correct for the two recordings that made this demo.
For a new recording, make sure that:

- The ranks 10 to 12 are back at the ring at approximately t = 180 s.
- The rank 11 moves during 180 s to 300 s.

If they do not, select other ranks. `tools/make_camera_plan.py` shows how to read
positions from a recording.

### 6.2 Why `--zoom-smooth 0`

The smoothing of the target position (0.4 s) came after the first 120 s to 180 s
clip. A value of 0 gives the camera motion of that clip. The clips of 180 s to
300 s use the default 0.4 s.

## 7. Watch the jobs

```bash
squeue -u $USER
sacct -j JOBID -X --format=JobID%14,State,Elapsed
```

**Do not read a log file while its job runs.** The logs are on the cephfs home
file system. If you read `renders/slurm-chunk-*.out` from the login node, the file
system makes each write of the job synchronous. Blender writes tens of lines for
each frame, thus the render becomes approximately 3 times slower. This was
measured: 15 s to 23 s for each frame, against 6.5 s for each frame.

Count the frames instead. This touches only the PNG files:

```bash
ls renders/anim_0-60/overview | wc -l
```

To read a log of a job that runs, read it on its own node:

```bash
srun --jobid=JOBID --overlap --ntasks=1 grep "re-framed" renders/slurm-chunk-JOBID_0.out
```

After the job is complete, read the logs from anywhere.

## 8. Examine the result

Each chunk log must give these lines:

| Line | Meaning |
|---|---|
| `[anim] rank N: ... 15 lines` | All 15 ranks are loaded |
| `[patched] hull lift 0.2 m` | The track shoes do not cut through the hull |
| `[bird] all appended material images resolve` | The textures are found |
| `[patched] overview re-framed: ...` | The camera plan is applied |
| `[scm] N deformed nodes -> M terrain cells` | The ruts are geometry |

Count the mesh problems. The result must be 0:

```bash
grep -c "drawn as a box" renders/slurm-chunk-*.out
```

A number more than 0 means that `meshes/` does not have a file. The pipeline draws
a box in its place, and this is easy to miss in a wide shot.

Errors with the text `cannot read from MTL file` are normal. The OBJ files have no
material files. The pipeline gives the materials.

## 9. Known problems

| Problem | Cause | What to do |
|---|---|---|
| The ground detail flickers one frame at each 20 s limit | Each chunk makes fine geometry for all cells that it deforms, from its first frame. The group of cells changes at each chunk limit | Not corrected. To correct it, calculate the group of cells one time for the full range and use it for each chunk |
| The render is 3 times too slow | A log file is read while the job runs | See section 7 |
| The terrain is 14 m too low | Blender reads an sRGB heightmap with a gamma correction | Corrected in `render_bird.py`. Do not remove `install_heightmap_fix()` |
| The arms are boxes | The mesh map has an absolute path of a different machine | Corrected. The paths in `meshmap.json` are relative to the file itself |
| `conda activate` does not work in a job | The shell function is not there | `module load conda/miniforge` and `bootstrap-conda`, each time |

## 10. The time and the size

| Item | Value |
|---|---|
| Time for each frame | 5 s to 13 s, for the GPU of the node |
| Time for one 20 s chunk | 1.0 h to 1.7 h |
| Time for 60 s (3 chunks together) | 1.2 h to 1.7 h |
| PNG frames for each 60 s | 1800 files, approximately 2.2 GB |
| Memory for each job | 64 GB. The rut mesh gets to approximately 3 M vertices at t = 300 s |
