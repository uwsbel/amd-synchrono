#!/usr/bin/env python3
"""Animate the WHOLE AMD-UW site (all ranks) over a time window and render it from
three cameras:

  overview   fixed bird's-eye view of the whole site
  chase_a    camera attached to one vehicle (default rank 1 collector), following it
  chase_b    same for another vehicle (default rank 8 builder)

    blender --background --factory-startup --python tools/render_site_anim.py -- \
        --dataset data --meshes meshes --start 0 --end 5 --fps 30 \
        --follow-a 1:collector --follow-b 8:builder \
        --outdir renders/anim_0-5 --out renders/anim_0-5.blend

Output: renders/anim_0-5/<camera>/frame_0001.png ... plus, if ffmpeg is on PATH,
<camera>.mp4 for each camera and combined.mp4 (overview left, chase cams stacked right).
Re-running with the same --outdir skips frames already on disk (Blender's
use_overwrite=False), so a killed run can just be restarted.

The scene build (terrain, static props, meshes, aabb fitting, materials, sun) is shared
with tools/render_site_frame.py; this script adds keyframes for every body of every rank
(bulk-written through the f-curve API, so 15 ranks x 150 frames takes seconds) and the
camera rigs. Chase cameras sit at a fixed offset in the vehicle's yaw-only frame
(behind/beside/above), with the target position and heading low-pass filtered so the
camera glides instead of shaking with the hull.
"""

import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blender_import import load_manifest, load_meta, read_frames          # noqa: E402
from render_site_frame import (MeshLibrary, add_camera, build_terrain,      # noqa: E402
                               find_heightmap, load_static, parse_ranks,
                               place_bodies, setup_render, setup_world)


# ------------------------------------------------------------------ keyframing

def _fcurves_of(obj):
    """Return the object's f-curves in a Blender-version-agnostic way."""
    ad = obj.animation_data
    act = ad.action
    if hasattr(act, "fcurves"):                       # Blender < 4.4 (legacy actions)
        return act.fcurves
    slot = ad.action_slot                             # 4.4+ slotted / layered actions
    return act.layers[0].strips[0].channelbag(slot, ensure=True).fcurves


def keyframe_track(obj, frames, positions, quats):
    """Bulk-keyframe location + rotation_quaternion of `obj` at blender `frames`."""
    n = len(frames)
    if n == 0:
        return
    obj.location = positions[0]
    obj.rotation_quaternion = quats[0]
    obj.keyframe_insert("location", frame=frames[0])
    obj.keyframe_insert("rotation_quaternion", frame=frames[0])
    if n == 1:
        return
    fcs = _fcurves_of(obj)
    for path, comps, data in (("location", 3, positions), ("rotation_quaternion", 4, quats)):
        for i in range(comps):
            fc = fcs.find(path, index=i)
            fc.keyframe_points.add(n - 1)
            co = [0.0] * (2 * n)
            for k in range(n):
                co[2 * k] = float(frames[k])
                co[2 * k + 1] = float(data[k][i])
            fc.keyframe_points.foreach_set("co", co)
            fc.update()


def keyframe_visibility(obj, first_frame, scene_first=1):
    """Hidden before `first_frame` (bodies that spawn mid-window)."""
    if first_frame <= scene_first:
        return
    for attr in ("hide_viewport", "hide_render"):
        setattr(obj, attr, True)
        obj.keyframe_insert(attr, frame=first_frame - 1)
        setattr(obj, attr, False)
        obj.keyframe_insert(attr, frame=first_frame)


# ------------------------------------------------------------------ chase camera

def yaw_of(quat):
    """Heading (rad) of the body x-axis projected on the world XY plane."""
    w, x, y, z = quat
    # first column of the rotation matrix = body x-axis in world
    ax = 1 - 2 * (y * y + z * z)
    ay = 2 * (x * y + w * z)
    return math.atan2(ay, ax)


def chase_track(times, positions, quats, back, side, height, look_h, tau):
    """Smoothed chase-camera locations and look-at points for every frame."""
    px, py, pz = positions[0]
    yaw = yaw_of(quats[0])
    cams, looks = [], []
    prev_t = times[0]
    for k in range(len(times)):
        dt = max(times[k] - prev_t, 1e-6)
        prev_t = times[k]
        alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
        tx, ty, tz = positions[k]
        px += alpha * (tx - px)
        py += alpha * (ty - py)
        pz += alpha * (tz - pz)
        ty_ = yaw_of(quats[k])
        d = (ty_ - yaw + math.pi) % (2 * math.pi) - math.pi   # shortest angular step
        yaw += alpha * d
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = -back, side
        cams.append((px + c * ox - s * oy, py + s * ox + c * oy, pz + height))
        looks.append((px, py, pz + look_h))
    return cams, looks


def keyframe_camera(cam, frames, cams, looks):
    from mathutils import Vector
    quats = []
    for loc, look in zip(cams, looks):
        direction = Vector(look) - Vector(loc)
        quats.append(tuple(direction.to_track_quat("-Z", "Y")))
    cam.rotation_mode = "QUATERNION"
    keyframe_track(cam, frames, cams, quats)


# ------------------------------------------------------------ overview camera plan

def plan_interp(ts, vs, t):
    import bisect
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    i = bisect.bisect_right(ts, t)
    w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return vs[i - 1] * (1 - w) + vs[i] * w


def load_or_make_plan(path, dataset, ranks):
    if os.path.exists(path):
        with open(path) as f:
            plan = json.load(f)
        print(f"[anim] camera plan: {path} ({len(plan['t'])} samples)")
        return plan
    print(f"[anim] camera plan {path} missing -- generating (one full read of the recording)")
    from make_camera_plan import build_plan, save_plan
    plan = build_plan(dataset, ranks)
    save_plan(plan, path)
    return plan


# ------------------------------------------------------------------ driver

def parse_follow(spec):
    """'1:collector' -> (1, 'collector')."""
    rank, _, kind = spec.partition(":")
    kind = kind or "collector"
    if kind not in ("collector", "builder"):
        raise SystemExit(f"--follow: kind must be collector or builder, got {kind}")
    return int(rank), kind


def chassis_index(objects, kind):
    for o in objects:
        if kind == "builder" and o["group"] == "builder" and "Chassis" in o["part"]:
            return o["index"]
        if kind == "collector" and o["group"] == "collector" and "chassis" in o["part"].lower():
            return o["index"]
    return None


CHASE_PRESETS = {
    # back, side, height, look_h, smoothing tau (s)
    "collector": (10.0, 3.5, 4.0, 0.8, 0.6),
    "builder": (12.0, 4.5, 5.0, 1.0, 0.6),
}


def build(a):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"

    ranks = parse_ranks(a.ranks)
    meta = load_meta(a.dataset, ranks[0]) if ranks else {}
    rate = float(meta.get("rate_hz") or 60.0)
    stride = max(1, int(round(rate / a.fps)))
    fps = rate / stride
    scene.render.fps = int(round(fps))
    scene.render.fps_base = 1.0
    print(f"[anim] recording {rate:g} Hz, output {fps:g} fps (stride {stride}), "
          f"window {a.start}..{a.end} s")

    mesh_map = None
    if a.mesh_map:
        with open(a.mesh_map) as f:
            mesh_map = json.load(f)
    lib = MeshLibrary(a.meshes, mesh_map)
    groups = set(a.groups.split(",")) if a.groups else None
    fit = not a.no_fit

    # ---- terrain + static props
    if not a.no_terrain:
        hm = find_heightmap(meta, a.heightmap)
        if hm:
            build_terrain(hm, meta, subdiv_levels=a.terrain_subdiv)
        else:
            print("[anim] WARNING: no heightmap found (pass --heightmap); terrain skipped")
    if not a.no_static:
        coll = bpy.data.collections.new("static_props")
        scene.collection.children.link(coll)
        static = [o for o in load_static(a.dataset) if not o["part"].startswith("patch_")]
        poses = {o["index"]: (o["first_pos"], o["first_rot"]) for o in static}
        _, n = place_bodies(static, poses, "static", lib, fit, coll)
        print(f"[anim] static props: {len(static)} bodies, {n} shapes")

    # ---- ranks: read the window, place bodies, keyframe
    follow = {}
    for tag, spec in (("chase_a", a.follow_a), ("chase_b", a.follow_b)):
        if spec:
            follow[tag] = parse_follow(spec)
    tracks_for_cams = {}          # tag -> (times, positions, quats)
    first_frame, last_frame = 10**9, 0
    all_frames = set()            # master frame grid of this window
    for rank in ranks:
        try:
            objects = load_manifest(a.dataset, rank)
        except FileNotFoundError:
            print(f"[anim] rank {rank}: no manifest, skipped")
            continue
        times = []                # rendered samples (t >= start)
        frame_nos = []
        warm = []                 # (time, poses) before start, only for chase-cam filter warm-up
        per_body = {}             # index -> (frames, positions, quats)
        first_pose = {}
        for time, poses in read_frames(a.dataset, rank, start=a.start - a.warmup - 1e-6,
                                       end=a.end + 1e-6, stride=stride):
            if time < a.start - 1e-6:
                warm.append((time, poses))
                continue
            k = int(round(time * fps)) + 1    # blender/file frame number, from the sample time
            times.append(time)
            frame_nos.append(k)
            for idx, (pos, quat) in poses.items():
                tr = per_body.get(idx)
                if tr is None:
                    tr = per_body[idx] = ([], [], [])
                    first_pose[idx] = (pos, quat)
                tr[0].append(k)
                tr[1].append(pos)
                tr[2].append(quat)
        if not times:
            print(f"[anim] rank {rank}: no samples in window, skipped")
            continue
        first_frame = min(first_frame, frame_nos[0])
        last_frame = max(last_frame, frame_nos[-1])
        all_frames.update(frame_nos)
        coll = bpy.data.collections.new(f"rank_{rank}")
        scene.collection.children.link(coll)
        empties, n = place_bodies(objects, first_pose, f"r{rank}", lib, fit, coll, groups)
        for idx, e in empties.items():
            fr, ps, qs = per_body[idx]
            keyframe_track(e, fr, ps, qs)
            keyframe_visibility(e, fr[0], scene_first=frame_nos[0])
        for tag, (frank, kind) in follow.items():
            if frank == rank:
                ci = chassis_index(objects, kind)
                if ci is not None and ci in per_body:
                    fr, ps, qs = per_body[ci]
                    t_of = dict(zip(frame_nos, times))
                    ts = [t_of[f] for f in fr]
                    wt = [t for t, p in warm if ci in p]
                    wp = [p[ci][0] for t, p in warm if ci in p]
                    wq = [p[ci][1] for t, p in warm if ci in p]
                    tracks_for_cams[tag] = (fr, ts, ps, qs, kind, (wt, wp, wq))
        print(f"[anim] rank {rank}: {len(times)} samples t={times[0]:.3f}..{times[-1]:.3f}, "
              f"{len(empties)} bodies, {n} shapes")

    if last_frame == 0:
        raise SystemExit("[anim] nothing to render in this window")
    scene.frame_start = first_frame
    scene.frame_end = last_frame
    scene.frame_set(scene.frame_start)

    # ---- lights + cameras
    setup_world(a.sun_azimuth, a.sun_elevation, a.sun_strength,
                sky=tuple(float(v) for v in a.sky.split(",")), sky_strength=a.sky_strength)
    site = meta.get("site") or {}
    cx, cy = (site.get("center") or [0, 0])[:2]
    cameras = {}
    # Overview camera: animated from the precomputed camera plan (zooms out as the
    # collectors drive down their rays, back in as they return, with a slow orbit).
    # The plan is a whole-run precompute (tools/make_camera_plan.py), so every
    # parallel chunk derives the identical path -- seamless across chunk boundaries.
    plan = load_or_make_plan(a.camera_plan, a.dataset, ranks)
    rlo, rhi = min(plan["r"]), max(plan["r"])
    grid = sorted(all_frames)
    ov_cams, ov_looks = [], []
    for f in grid:
        t = (f - 1) / fps
        r_raw = max(plan_interp(plan["t"], plan["r"], t), 45.0)
        z = plan_interp(plan["t"], plan["z"], t)
        dist = r_raw * a.overview_margin * a.overview_dist
        # zoomed in -> steep look (like the close site view); zoomed out -> shallow
        # angle (like the wide t=700 still), interpolated on the zoom fraction
        frac = (r_raw - rlo) / max(rhi - rlo, 1.0)
        el_deg = a.overview_elevation + (a.overview_elevation_far - a.overview_elevation) * max(0.0, min(1.0, frac))
        az_deg = a.overview_azimuth + a.orbit_rate * t
        az, el = math.radians(az_deg), math.radians(el_deg)
        ov_cams.append((cx + dist * math.cos(el) * math.cos(az),
                        cy + dist * math.cos(el) * math.sin(az),
                        z + dist * math.sin(el)))
        ov_looks.append((cx, cy, z))
    cam_ov = add_camera("cam_overview", ov_cams[0], ov_looks[0], lens=a.lens)
    keyframe_camera(cam_ov, grid, ov_cams, ov_looks)
    cameras["overview"] = cam_ov

    for tag in ("chase_a", "chase_b"):
        if tag not in tracks_for_cams:
            if tag in follow:
                print(f"[anim] WARNING: {tag} target {follow[tag]} not found; camera skipped")
            continue
        fr, ts, ps, qs, kind, (wt, wp, wq) = tracks_for_cams[tag]
        back, side, height, look_h, tau = CHASE_PRESETS[kind]
        preset = getattr(a, f"{tag}_offset")
        if preset:
            back, side, height = (float(v) for v in preset.split(","))
        tau = a.chase_smoothing if a.chase_smoothing is not None else tau
        # run the filter through the warm-up samples first so a chunk that starts
        # mid-turn does not begin with an unsmoothed camera jump
        cams, looks = chase_track(wt + ts, wp + ps, wq + qs, back, side, height, look_h, tau)
        cams, looks = cams[len(wt):], looks[len(wt):]
        cam = add_camera(f"cam_{tag}_r{follow[tag][0]}_{kind}", cams[0], looks[0], lens=a.chase_lens)
        keyframe_camera(cam, fr, cams, looks)
        cameras[tag] = cam
    scene.camera = cameras["overview"]

    setup_render(a.engine, a.samples, tuple(int(v) for v in a.res.lower().split("x")), use_gpu=not a.cpu)
    scene.render.use_persistent_data = True   # keep the synced scene between frames (big win with ~8k objects)
    scene.render.use_overwrite = False        # resume: skip frames already on disk
    scene.render.use_placeholder = False
    scene.render.use_file_extension = True
    print(f"[anim] {len(bpy.data.objects)} objects, frames {scene.frame_start}..{scene.frame_end}, cameras {list(cameras)}")

    os.makedirs(a.outdir, exist_ok=True)
    blend = a.out or os.path.join(a.outdir, "scene.blend")
    os.makedirs(os.path.dirname(os.path.abspath(blend)) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend))
    print(f"[anim] saved {blend}")
    if a.no_render:
        return
    cam_names = {k: c.name for k, c in cameras.items()}
    if not a.no_reload:
        # Render from a freshly loaded copy of the file: identical to what a user would open,
        # and it drops the import-time scratch state (templates, operator leftovers).
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
        scene = bpy.context.scene
    views = [v for v in a.views.split(",") if v in cam_names]
    for v in views:
        scene.camera = bpy.data.objects[cam_names[v]]
        d = os.path.join(a.outdir, v)
        os.makedirs(d, exist_ok=True)
        scene.render.filepath = os.path.join(os.path.abspath(d), "frame_")
        print(f"[anim] rendering {v}: frames {scene.frame_start}..{scene.frame_end} -> {d}", flush=True)
        bpy.ops.render.render(animation=True)
    if a.encode:
        encode(a.outdir, views, int(round(fps)), keep_all=a.keep_all_mp4)


def encode(outdir, views, fps, keep_all=False):
    ff = shutil.which("ffmpeg")
    if not ff:
        print("[anim] ffmpeg not found; frames left as PNG")
        return
    mp4s = {}
    for v in views:
        d = os.path.join(outdir, v)
        if not os.path.isdir(d):
            print(f"[anim] {v}: no frame directory, skipped")
            continue
        nums = sorted(int(f[6:-4]) for f in os.listdir(d) if f.startswith("frame_") and f.endswith(".png"))
        if not nums:
            print(f"[anim] {v}: no frames, skipped")
            continue
        missing = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)]
        if missing:
            print(f"[anim] WARNING {v}: {len(missing)} missing frames (first {missing[:5]}); "
                  f"ffmpeg will stop at the first gap")
        src = os.path.join(d, "frame_%04d.png")
        dst = os.path.join(outdir, f"{v}.mp4")
        cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", str(nums[0]), "-i", src,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", dst]
        subprocess.run(cmd, check=True)
        mp4s[v] = dst
        print(f"[anim] wrote {dst}")
    combined = None
    if "overview" in mp4s and "chase_a" in mp4s and "chase_b" in mp4s:
        combined = os.path.join(outdir, "combined.mp4")
        # overview full size left; the two chase cams stacked at half size right
        flt = ("[1:v]scale=iw/2:ih/2[a];[2:v]scale=iw/2:ih/2[b];[a][b]vstack=inputs=2[r];"
               "[0:v][r]hstack=inputs=2[o]")
        inputs = [mp4s["overview"], mp4s["chase_a"], mp4s["chase_b"]]
    elif "overview" in mp4s and "chase_a" in mp4s:
        combined = os.path.join(outdir, "combined.mp4")
        flt = "[0:v][1:v]hstack=inputs=2[o]"        # overview | chase, side by side
        inputs = [mp4s["overview"], mp4s["chase_a"]]
    if combined:
        cmd = [ff, "-y", "-loglevel", "error"]
        for i in inputs:
            cmd += ["-i", i]
        cmd += ["-filter_complex", flt, "-map", "[o]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-movflags", "+faststart", combined]
        subprocess.run(cmd, check=True)
        print(f"[anim] wrote {combined}")
        if not keep_all and "overview" in mp4s:
            os.remove(mp4s["overview"])    # deliverables: combined + the chase cams
            print(f"[anim] removed {mp4s['overview']} (kept inside combined.mp4; --keep-all-mp4 retains it)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    import argparse
    ap = argparse.ArgumentParser(prog="render_site_anim.py")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ranks", default="1-15")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0, help="output fps; stride = rate/fps")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="seconds of samples before --start fed to the chase-camera filter only")
    ap.add_argument("--meshes")
    ap.add_argument("--mesh-map")
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--groups")
    ap.add_argument("--heightmap")
    ap.add_argument("--terrain-subdiv", type=int, default=0)
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--out", help=".blend to save")
    ap.add_argument("--outdir", default="renders/anim")
    ap.add_argument("--views", default="overview,chase_a")
    ap.add_argument("--keep-all-mp4", action="store_true", help="also keep the per-view overview.mp4")
    ap.add_argument("--camera-plan", default="renders/camera_plan.json",
                    help="whole-run overview camera plan (generated if missing)")
    ap.add_argument("--orbit-rate", type=float, default=0.25, help="overview orbit, deg per sim second")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-reload", action="store_true", help="render in-process instead of reloading the saved .blend")
    ap.add_argument("--no-encode", dest="encode", action="store_false")
    ap.add_argument("--follow-a", default="1:collector", help="RANK:collector|builder")
    ap.add_argument("--follow-b", default="8:builder")
    ap.add_argument("--chase-a-offset", help="back,side,height in metres (vehicle yaw frame)")
    ap.add_argument("--chase-b-offset")
    ap.add_argument("--chase-smoothing", type=float, help="seconds; 0 = rigidly attached")
    ap.add_argument("--chase-lens", type=float, default=32.0)
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--lens", type=float, default=35.0)
    ap.add_argument("--overview-azimuth", type=float, default=-125.0)
    ap.add_argument("--overview-elevation", type=float, default=55.0, help="deg when zoomed in")
    ap.add_argument("--overview-elevation-far", type=float, default=34.0, help="deg when fully zoomed out")
    ap.add_argument("--overview-dist", type=float, default=1.9, help="camera distance / site radius")
    ap.add_argument("--overview-margin", type=float, default=1.08, help="site radius multiplier")
    ap.add_argument("--sun-azimuth", type=float, default=-40.0)
    ap.add_argument("--sun-elevation", type=float, default=28.0)
    ap.add_argument("--sun-strength", type=float, default=5.5)
    ap.add_argument("--sky", default="0.015,0.015,0.02", help="near-black lunar sky")
    ap.add_argument("--sky-strength", type=float, default=1.0)
    ap.add_argument("--encode-only", action="store_true", help="just run ffmpeg over --outdir (no Blender needed)")
    a = ap.parse_args(argv)
    if a.encode_only:
        encode(a.outdir, a.views.split(","), int(round(a.fps)), keep_all=a.keep_all_mp4)
        return
    build(a)


if __name__ == "__main__":
    main()
