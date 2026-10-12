#!/usr/bin/env python3
"""Sanity-check a (cluster) environment for this render pipeline.

    python3 tools/cluster/check_env.py

Verifies: bpy imports and reports its Cycles compute devices, Pillow/ffmpeg are
present, the dataset's 15 ranks + static props are readable, the heightmap is
staged, and every mesh the manifest names resolves to a file.
Run it from the working directory. Exit code 0 = ready to render.
"""
import os
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

failures = []


def status(name, good, detail=""):
    if not good:
        failures.append(name)
    print(f"  [{'ok' if good else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


print(f"python {sys.version.split()[0]}  ({sys.executable})")

# ---- bpy
bpy = None
try:
    import bpy  # noqa: F811
    status("bpy", True, f"Blender {bpy.app.version_string}")
except Exception as exc:  # noqa: BLE001
    status("bpy", False, f"{exc} (pip install bpy==4.5.* in a python 3.11 env)")

# ---- Cycles compute devices
if bpy is not None:
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        found = []
        for dt in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = dt
                prefs.get_devices()
                devs = [d.name for d in prefs.devices if d.type == dt]
                if devs:
                    found.append(f"{dt}: {', '.join(sorted(set(devs)))}")
            except TypeError:
                continue
        status("cycles devices", True, "; ".join(found) if found else "none (CPU rendering)")
    except Exception as exc:  # noqa: BLE001
        status("cycles devices", False, str(exc))

# ---- helpers
try:
    import PIL  # noqa: F401
    status("Pillow", True)
except ImportError:
    print("  [--] Pillow missing (ok: heightmap falls back to bpy image loading)")
status("ffmpeg", shutil.which("ffmpeg") is not None,
       shutil.which("ffmpeg") or "not on PATH (PNG frames still render; no mp4)")

# ---- dataset
data = os.path.join(ROOT, "data")
n_ok = 0
for rank in range(1, 16):
    path = os.path.join(data, f"rank_{rank}_frames.bin")
    try:
        with open(path, "rb") as f:
            magic, _v, r, rate, _s = struct.unpack("<8sIIdd", f.read(32))
        if magic == b"AMDUWTRJ" and r == rank:
            n_ok += 1
    except (OSError, struct.error):
        pass
status("frames.bin ranks 1-15", n_ok == 15, f"{n_ok}/15 readable")
status("static_props.jsonl", os.path.exists(os.path.join(data, "static_props.jsonl")))

from render_site_frame import find_heightmap  # noqa: E402
from blender_import import load_manifest, load_meta, resolve_mesh  # noqa: E402

meta = load_meta(data, 1)
name = os.path.basename((meta.get("terrain") or {}).get("heightmap", ""))
# anim_chunk.sbatch passes --heightmap assets/data/terrain/<name>, so look there
# first; find_heightmap's own search covers a layout that stages it elsewhere.
staged = os.path.join(ROOT, "assets", "data", "terrain", name) if name else ""
hm = staged if staged and os.path.exists(staged) else find_heightmap(meta)
status("terrain heightmap", hm is not None,
       hm or f"missing -- expected assets/data/terrain/{name or '<name from rank_1_meta.json>'}")

# ---- meshes. resolve_mesh() returns a mesh-map entry without checking that it
# exists, and an unresolved mesh is drawn as a box of the recorded aabb -- which
# looks plausible in a wide shot. Catch it here instead of in the finished movie.
import json  # noqa: E402

map_path = os.path.join(ROOT, "meshmap.json")
mesh_map = {}
if os.path.exists(map_path):
    with open(map_path) as f:
        mesh_map = json.load(f)
mesh_dir = os.path.join(ROOT, "meshes")
missing = set()
for obj in load_manifest(data, 1):
    gp = f"{obj['group']}/{obj['part']}"
    for shape in obj.get("shapes", []):
        if shape.get("type") != "trimesh" or not shape.get("file"):
            continue
        target = resolve_mesh(shape["file"], gp, mesh_dir, mesh_map)
        if target and not os.path.isabs(target):
            target = os.path.join(os.path.dirname(os.path.abspath(map_path)), target)
        if not target or not os.path.exists(target):
            missing.add(os.path.basename(shape["file"]))
status("meshes resolve (rank 1)", not missing,
       "every mesh found" if not missing else
       f"{len(missing)} not found ({', '.join(sorted(missing)[:6])}) -- these render as boxes")

print("READY" if not failures else f"NOT READY: {', '.join(failures)}")
sys.exit(1 if failures else 0)
