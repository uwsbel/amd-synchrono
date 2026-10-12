#!/usr/bin/env python3
"""tools/render_site_anim.py, plus everything render_bird.py adds.

render_site_anim.py builds the animated three-camera site scene -- including the
plan-driven overview camera that zooms out as the collectors drive down their
harvest rays and back in as they return (tools/make_camera_plan.py). This wrapper
runs that build unchanged and then applies, in order:

  1. the heightmap decode fix and the builder hull lift (see render_bird.py);
  2. the lunar-manip texture/light/world pass;
  3. the SCM ruts, growing over the window rather than fixed;

and renders the window itself, so it can update the rut geometry per frame.

Every flag render_site_anim.py takes works here; the extra ones are --scm,
--scm-stride, --hull-lift, --raw-heightmap and --no-texture.

    python3 render_site_anim_patched.py -- \
        --dataset data --meshes meshes --mesh-map meshmap.json \
        --heightmap assets/data/terrain/terrain2_graded.png \
        --start 0 --end 20 --fps 30 --outdir renders/anim_0-20 \
        --views overview --scm

RUTS OVER TIME. The rut mesh's topology is fixed at construction from the cells
deformed by the END of the window, and only vertex heights are rewritten per
frame. A cell nothing has driven over yet renders as undeformed ground -- which
is what the terrain quad it replaced would have shown -- so nothing pops into
existence mid-chunk. Across a chunk boundary the two chunks disagree only in how
finely an as-yet-undeformed cell is tessellated (0.106 m sub-grid against the
terrain's 4 m quad, a sub-centimetre difference in the interpolated surface),
which is not visible.
"""

import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_site_frame as site      # noqa: E402
import render_site_anim as anim       # noqa: E402
import render_bird as bird            # noqa: E402
import scm_terrain                    # noqa: E402


def log(msg):
    print(f"[patched] {msg}", flush=True)


def take_flag(argv, name):
    if name in argv:
        argv.remove(name)
        return True
    return False


def take_value(argv, name, cast, default):
    if name in argv:
        i = argv.index(name)
        v = cast(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: render_site_anim_patched.py -- <render_site_anim.py flags>")
    cut = sys.argv.index("--") + 1
    argv = sys.argv[cut:]

    # strip our own flags before render_site_anim's parser sees them
    extra = {
        "scm": take_flag(argv, "--scm"),
        "no_texture": take_flag(argv, "--no-texture"),
        "raw_heightmap": take_flag(argv, "--raw-heightmap"),
        "scm_stride": take_value(argv, "--scm-stride", int, 3),
        "hull_lift": take_value(argv, "--hull-lift", float, bird.HULL_LIFT_Z),
        "frame_bias": take_value(argv, "--frame-bias", float, 0.5),
        "zoom_targets": take_value(argv, "--zoom-targets", str, None),
        "zoom_radius": take_value(argv, "--zoom-radius", float, 26.0),
        "zoom_elevation": take_value(argv, "--zoom-elevation", float, 28.0),
        "zoom_start": take_value(argv, "--zoom-start", float, None),
        # must be given explicitly when the window is split into chunks: a.end is
        # the CHUNK's end, so defaulting to it would complete the zoom in chunk 0
        "zoom_end": take_value(argv, "--zoom-end", float, None),
        # where the move STARTS from. Defaults to the plan's own framing of the
        # whole site; give these to begin on the exact camera another clip ended
        # on, so the two join without a jump.
        "zoom_start_targets": take_value(argv, "--zoom-start-targets", str, None),
        "zoom_start_radius": take_value(argv, "--zoom-start-radius", float, None),
        "zoom_start_elevation": take_value(argv, "--zoom-start-elevation", float, None),
        "zoom_smooth": take_value(argv, "--zoom-smooth", float, 0.4),
    }
    sys.argv[cut:] = argv

    if not extra["raw_heightmap"]:
        bird.install_heightmap_fix()
    # render_site_anim imported load_manifest into its OWN namespace, so patching
    # render_site_frame's copy (what install_hull_lift does) would not reach it
    _orig_load_manifest = anim.load_manifest

    def load_manifest(dataset, rank):
        objects = _orig_load_manifest(dataset, rank)
        lift = extra["hull_lift"]
        if abs(lift) < 1e-9:
            return objects
        n = 0
        for obj in objects:
            for shape in obj.get("shapes", []):
                if shape.get("shape_name") == bird.HULL_SHAPE_NAME:
                    pos = list(shape.get("pos", [0.0, 0.0, 0.0]))
                    pos[2] += lift
                    shape["pos"] = pos
                    n += 1
        if rank == 1:
            log(f"hull lift {lift:g} m on {n} {bird.HULL_SHAPE_NAME} shape(s) per rank")
        return objects

    anim.load_manifest = load_manifest
    # render_site_anim shares render_site_frame's MeshLibrary, which calls
    # resolve_mesh out of render_site_frame's globals, so this reaches both
    if "--mesh-map" in argv:
        bird.install_meshmap_relative(argv[argv.index("--mesh-map") + 1])

    _orig_build = anim.build

    def build(a):
        no_render = a.no_render
        a.no_render = True                 # build + save only; we render below
        _orig_build(a)
        a.no_render = no_render

        scene = bpy.context.scene
        meta = anim.load_meta(a.dataset, anim.parse_ranks(a.ranks)[0])
        rate = float(meta.get("rate_hz") or 60.0)
        stride = max(1, int(round(rate / a.fps)))
        fps = rate / stride

        a.frame_bias = extra["frame_bias"]
        rekey_overview(a, fps, extra)

        terrain_mat, bounds = (None, None)
        if not extra["no_texture"]:
            terrain_mat, bounds = bird.patch_textures()

        rut = window = None
        if extra["scm"]:
            rut, window = add_scm_window(a, meta, terrain_mat, bounds,
                                          extra["scm_stride"])

        blend = a.out or os.path.join(a.outdir, "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend))
        log(f"saved {blend}")
        if no_render:
            return

        cameras = {}
        for tag, name in (("overview", "cam_overview"),):
            if name in bpy.data.objects:
                cameras[tag] = bpy.data.objects[name]
        for o in bpy.data.objects:
            if o.type == "CAMERA" and o.name.startswith("cam_chase_a"):
                cameras["chase_a"] = o
            if o.type == "CAMERA" and o.name.startswith("cam_chase_b"):
                cameras["chase_b"] = o
        views = [v for v in a.views.split(",") if v in cameras]
        log(f"rendering {views} frames {scene.frame_start}..{scene.frame_end} "
            f"({fps:g} fps) -> {a.outdir}")

        for v in views:
            os.makedirs(os.path.join(a.outdir, v), exist_ok=True)
        cursor = None
        for f in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(f)
            if rut is not None:
                z = window.heights_at((f - 1) / fps)
                # SCM is recorded at 10 Hz against a 30 fps render, so two frames in
                # three change nothing -- and rewriting the mesh costs Cycles a full
                # re-sync and BVH rebuild of ~0.5 M vertices, so skip it when the
                # replay cursor has not moved
                if cursor != window._cursor:
                    rut.set_heights(z)
                    cursor = window._cursor
            for v in views:
                path = os.path.join(os.path.abspath(a.outdir), v, f"frame_{f:04d}.png")
                if os.path.exists(path):        # resume, as use_overwrite=False did
                    continue
                scene.camera = cameras[v]
                scene.render.filepath = path
                bpy.ops.render.render(write_still=True)
        log("frames done")
        if a.encode:
            anim.encode(a.outdir, views, int(round(fps)), keep_all=a.keep_all_mp4)

    anim.build = build
    anim.main()


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def chassis_objects(spec):
    """--zoom-targets '10:collector+11:collector' -> the scene objects for those
    vehicles' chassis empties (place_bodies names them 'r{rank}.{group}.{part}')."""
    # '+' as well as ',' : EXTRA rides to the compute node through
    # `sbatch --export=ALL,EXTRA=...`, which is itself comma-separated, so a
    # comma inside the value gets eaten before the script ever sees it
    out = []
    for token in spec.replace("+", ",").split(","):
        token = token.strip()
        if not token:
            continue
        rank, _, kind = token.partition(":")
        kind = kind or "collector"
        prefix = f"r{int(rank)}.{kind}."
        match = [o for o in bpy.data.objects
                 if o.name.startswith(prefix) and "chassis" in o.name.lower()
                 and o.type == "EMPTY"]
        if not match:
            raise SystemExit(f"--zoom-targets: no chassis object for {token!r} "
                             f"(looked for {prefix}*chassis*)")
        out.append(min(match, key=lambda o: len(o.name)))
    return out


def sample_locations(objs, frames):
    """{frame: centroid} of `objs`, read straight off their baked location
    f-curves -- far cheaper than stepping the depsgraph 1800 times just to read
    a handful of positions."""
    curves = []
    for o in objs:
        fcs = anim._fcurves_of(o)
        curves.append([fcs.find("location", index=i) for i in range(3)])
    out = {}
    for f in frames:
        n = len(curves)
        out[f] = tuple(sum(c[i].evaluate(f) for c in curves) / n for i in range(3))
    return out


def smooth_track(track, frames, tau_frames):
    """Zero-lag smoothing (EMA forward, EMA backward, averaged) of a per-frame
    position track. A chassis pose carries suspension and terrain chatter that a
    26 m framing turns into visible camera shake."""
    if tau_frames <= 0:
        return track
    alpha = 1.0 - math.exp(-1.0 / tau_frames)
    fwd = {}
    cur = track[frames[0]]
    for f in frames:
        cur = tuple(cur[i] + alpha * (track[f][i] - cur[i]) for i in range(3))
        fwd[f] = cur
    bwd = {}
    cur = track[frames[-1]]
    for f in reversed(frames):
        cur = tuple(cur[i] + alpha * (track[f][i] - cur[i]) for i in range(3))
        bwd[f] = cur
    return {f: tuple((fwd[f][i] + bwd[f][i]) * 0.5 for i in range(3)) for f in frames}


def rekey_overview(a, fps, extra=None):
    """Re-lay the overview camera path with render_bird.frame_disc.

    render_site_anim.py already animates this camera off the same plan -- zooming
    out ahead of the collectors and back in as they return -- but it sizes each
    frame as distance = overview_dist * margin * r with the camera aimed at the site
    centre, and that cuts the near side off (see frame_disc's docstring: at its
    defaults the near edge of the visible ground stops 48.6 m short of the centre
    while the far edge runs 79 m past it). Same plan, same azimuth and elevation
    schedule; only the distance and the aim point are re-solved so the whole disc
    of radius r is actually inside the frame.
    """
    import json

    cam = bpy.data.objects.get("cam_overview")
    if cam is None:
        return
    if not os.path.exists(a.camera_plan):
        log(f"no camera plan at {a.camera_plan}; leaving the overview camera as built")
        return
    with open(a.camera_plan) as f:
        plan = json.load(f)
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))

    # optional zoom-in finish: hold the plan's framing at --zoom-start and ease
    # from there down onto a handful of named vehicles by the end of the window.
    # Coverage is deliberately given up -- that is the whole point of the move.
    zoom = None
    if extra and extra.get("zoom_targets"):
        tau = extra["zoom_smooth"] * fps
        objs = chassis_objects(extra["zoom_targets"])
        centroids = smooth_track(sample_locations(objs, frames), frames, tau)
        t0 = extra["zoom_start"] if extra["zoom_start"] is not None else a.start
        t1 = extra["zoom_end"] if extra["zoom_end"] is not None else a.end
        _, _, r_plan0, el_plan0, _ = bird.plan_camera(plan, t0, a)
        r0 = extra["zoom_start_radius"] if extra["zoom_start_radius"] is not None else r_plan0
        el0 = (extra["zoom_start_elevation"] if extra["zoom_start_elevation"] is not None
               else el_plan0)
        from_objs = from_centroids = None
        if extra.get("zoom_start_targets"):
            from_objs = chassis_objects(extra["zoom_start_targets"])
            from_centroids = smooth_track(sample_locations(from_objs, frames), frames, tau)
        zoom = dict(objs=objs, centroids=centroids, t0=t0, t1=t1, r0=r0, el0=el0,
                    r1=extra["zoom_radius"], el1=extra["zoom_elevation"],
                    from_centroids=from_centroids)
        log(f"zoom onto {[o.name for o in objs]}"
            + (f" from {[o.name for o in from_objs]}" if from_objs else " from the whole site")
            + f": radius {r0:.0f} -> {zoom['r1']:.0f} m, "
              f"elevation {el0:.1f} -> {zoom['el1']:.1f} deg, over t={t0:g}..{t1:g}"
              f" (smoothing {extra['zoom_smooth']:g} s)")

    res = tuple(int(v) for v in a.res.lower().split("x"))
    locs, looks, radii = [], [], []
    for fr in frames:
        t = (fr - 1) / fps
        loc, aim, r, el, az = bird.plan_camera(plan, t, a)
        if zoom is not None:
            u = smoothstep((t - zoom["t0"]) / max(zoom["t1"] - zoom["t0"], 1e-6))
            # what we come FROM: the plan's view of the whole site, or another
            # clip's subject when this one has to continue from its last frame
            if zoom["from_centroids"] is not None:
                site = zoom["from_centroids"][fr]
            else:
                site = (plan["center"][0], plan["center"][1], aim[2])
            tgt = zoom["centroids"][fr]
            centre = tuple(site[i] + (tgt[i] - site[i]) * u for i in range(3))
            # hold the plan's radius/elevation at t0 and ease down from there, so
            # the shot never pulls back on the way in
            r = zoom["r0"] + (zoom["r1"] - zoom["r0"]) * u
            el = zoom["el0"] + (zoom["el1"] - zoom["el0"]) * u
            loc, aim = bird.frame_disc(centre, r, az, el, lens=a.lens,
                                       aspect=res[0] / res[1], margin=a.overview_margin,
                                       bias=a.frame_bias)
        locs.append(loc)
        looks.append(aim)
        radii.append((r, el, math.dist(loc, aim)))
    cam.animation_data_clear()
    cam.data.lens = a.lens
    anim.keyframe_camera(cam, frames, locs, looks)
    log(f"overview re-framed: radius {radii[0][0]:.0f}->{radii[-1][0]:.0f} m, "
        f"elevation {radii[0][1]:.1f}->{radii[-1][1]:.1f} deg, "
        f"distance {radii[0][2]:.0f}->{radii[-1][2]:.0f} m over {len(frames)} frames")


def add_scm_window(a, meta, terrain_mat, bounds, stride):
    """Build the rut mesh for [a.start, a.end] and cut the cells it replaces."""
    ranks = anim.parse_ranks(a.ranks)
    hm_path = site.find_heightmap(meta, a.heightmap)
    if hm_path is None:
        raise SystemExit("--scm needs the heightmap (--heightmap) to fill the cut cells")
    window = scm_terrain.ScmWindow(a.dataset, ranks, a.start, a.end)
    if not len(window):
        log("no SCM deformation in this window; terrain left undeformed")
        return None, None

    terrain_obj = bpy.data.objects.get("world.terrain")
    if terrain_obj is None:
        raise SystemExit("world.terrain not found; --scm needs the heightmap terrain")
    if terrain_mat is None:
        terrain_mat = terrain_obj.data.materials[0]
        xs = [v.co.x for v in terrain_obj.data.vertices]
        ys = [v.co.y for v in terrain_obj.data.vertices]
        bounds = (min(xs), max(xs), min(ys), max(ys))

    rows, nx, ny = site.read_gray_image(hm_path)
    terr = meta.get("terrain") or {}
    hmap = scm_terrain.Heightmap(rows, nx, ny,
                                 float(terr.get("length", 1024.0)),
                                 float(terr.get("width", 1024.0)),
                                 float(terr.get("min_height", -25.0)),
                                 float(terr.get("max_height", 25.0)))
    rut = scm_terrain.RutMesh(hmap, window.delta, window.plane, window.keys, stride=stride)
    rut.set_heights(window.heights_at(a.start))
    bird.planar_uv(rut.mesh, bounds)
    rut.mesh.materials.append(terrain_mat)
    scm_terrain.cut_terrain_cells(terrain_obj, rut.covered, ny)
    return rut, window


if __name__ == "__main__":
    main()
