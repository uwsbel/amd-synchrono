#!/usr/bin/env python3
"""Bird's-eye stills of an AMD-UW pose recording, with the lunar-manip look.

Three pieces glued together:

  * tools/render_site_frame.py (the AMD-UW render pipeline) rebuilds the whole
    site -- terrain from the heightmap, static props, all 15 ranks -- at one
    instant, with untextured placeholder materials;
  * scripts/patch_section1_textures.py (lunar-manip) swaps in the PBR
    terrain/rock/metal/plastic materials, the sun and the world that
    data/section1.blend and data/section2_mode2.blend were finished with, so
    these stills match the movies already rendered from that repo;
  * tools/scm_terrain.py (here) rebuilds the ground the vehicles actually
    churned up, from the run's own rank_N_scm.bin, at the requested instant.

The overview camera is passed in explicitly (not auto-framed off the vehicles
like render_site_frame does) so a series of stills at different times share
one fixed framing.

Two corrections are applied to what render_site_frame builds; both are patched
in here rather than edited into it, so that file stays the copy it came as:

  HEIGHTMAP DECODE. read_gray_image() falls back to bpy.data.images when PIL is
  missing (Blender's bundled Python has no PIL), and reads the pixels through
  the image's sRGB colourspace -- so an 8-bit-in-16-bit greyscale heightmap
  comes back gamma-decoded. On terrain2_graded.png that put the whole
  landscape at z=-23.04..1.27 instead of -13.82..12.65, i.e. the site's ground
  ~2 m below where the sim had it. Reading it as Non-Color agrees with the run's
  own SCM node heights to the millimetre.

  BUILDER HULL LIFT. The M113 hull visual is squashed to 0.110x in z
  (BuilderRig::AddSquashedChassisVisual) so it does not bury the arm, which
  leaves the track shoes standing up to 0.166 m proud of the deck above them --
  they visibly saw through the hull. Same fix, and same 0.20 m, as
  AMD-UW_SBEL tools/replay_run.py's HULL_LIFT_Z: the hull VISUAL moves, nothing
  else, so the render disagrees with the sim by exactly that offset.

    blender --background --factory-startup --python render_bird.py -- \
        --dataset DIR --meshes DIR --mesh-map F.json --heightmap F.png \
        --time 10 --render out.png --scm
"""

import argparse
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_site_frame as site  # noqa: E402
import scm_terrain  # noqa: E402

# The PBR materials, the key light and the world come from the lunar-manip repo.
# Set LUNAR_MANIP if your copy is not at ~/lunar-manip.
LUNAR_MANIP = os.path.expanduser(os.environ.get("LUNAR_MANIP", "~/lunar-manip"))
SECTION2_MODE2 = os.path.join(LUNAR_MANIP, "data", "section2_mode2.blend")
TEXTURES_DIR = os.path.join(LUNAR_MANIP, "data", "textures")

TERRAIN_MAT_NAME = "terrain2_full.png_detail.001"
PLASTIC_MAT_NAME = "M113_TrackShoeLeft_shoe_M113_TrackShoeRight_shoe_pbr"
METAL_MAT_NAME = "robot_pbr"
ROCK_MAT_NAME = "rock_pbr"
LIGHT_OBJ_NAME = "Light.001"
WORLD_NAME = "World"

# see the module docstring
HULL_LIFT_Z = 0.20
HULL_SHAPE_NAME = "Builder_Chassis_Squashed_Z"


def log(msg):
    print(f"[bird] {msg}", flush=True)


# ------------------------------------------- corrections to render_site_frame

def install_heightmap_fix():
    """read_gray_image that does not gamma-decode the heightmap."""
    _orig = site.read_gray_image

    def read_gray_image(path):
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(path)
            a = np.array(im).astype(np.float64)
            if a.ndim > 2:
                a = a[:, :, 0]
            denom = 65535.0 if a.max() > 255.0 else 255.0
            a = a / denom * 255.0
            h, w = a.shape
            return a.tolist(), w, h
        except ImportError:
            pass
        import numpy as np
        img = bpy.data.images.load(path)
        img.colorspace_settings.name = "Non-Color"   # <- the fix
        img.reload()
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        rows = px.reshape(h, w, 4)[:, :, 0][::-1] * 255.0   # want the top row first
        bpy.data.images.remove(img)
        return rows.astype(np.float64).tolist(), w, h

    site.read_gray_image = read_gray_image
    return _orig


def install_meshmap_relative(mesh_map_path):
    """Let --mesh-map hold paths relative to itself, resolved against its own
    directory.

    resolve_mesh() returns a mesh-map replacement verbatim and never checks that
    it exists -- so an absolute path baked on one machine does not fail loudly on
    another, it just fails to import and make_shape() quietly falls back to a box
    of the recorded aabb. That is how both manipulator arms turned into blocks the
    first time this ran on the cluster. Relative paths keep one meshmap.json
    correct on every host.
    """
    if not mesh_map_path:
        return
    root = os.path.dirname(os.path.abspath(mesh_map_path))
    _orig = site.resolve_mesh

    def resolve_mesh(src_path, group_part, mesh_dir, mesh_map):
        target = _orig(src_path, group_part, mesh_dir, mesh_map)
        if target and not os.path.isabs(target):
            target = os.path.join(root, target)
        if target and not os.path.exists(target):
            log(f"WARNING: mesh {target} does not exist ({group_part}) -- "
                f"it will be drawn as a box")
            return None
        return target

    site.resolve_mesh = resolve_mesh


def install_hull_lift(lift):
    """load_manifest that raises the builder hull visual by `lift`, only."""
    _orig = site.load_manifest

    def load_manifest(dataset, rank):
        objects = _orig(dataset, rank)
        if abs(lift) < 1e-9:
            return objects
        n = 0
        for obj in objects:
            for shape in obj.get("shapes", []):
                if shape.get("shape_name") == HULL_SHAPE_NAME:
                    pos = list(shape.get("pos", [0.0, 0.0, 0.0]))
                    pos[2] += lift
                    shape["pos"] = pos
                    n += 1
        if rank == 1:
            log(f"hull lift {lift:g} m applied to {n} {HULL_SHAPE_NAME} shape(s) per rank")
        return objects

    site.load_manifest = load_manifest


# ------------------------------------------------------------------ texturing
# Ported from lunar-manip scripts/patch_section1_textures.py. The only change:
# the appended materials' images are re-pointed at an ABSOLUTE path into
# lunar-manip/data/textures instead of the "//textures/..." relative one, since
# these .blend files do not live next to that folder.

def append_materials():
    wanted = {TERRAIN_MAT_NAME, PLASTIC_MAT_NAME, METAL_MAT_NAME, ROCK_MAT_NAME}
    to_fetch = {n for n in wanted if bpy.data.materials.get(n) is None}
    if to_fetch:
        with bpy.data.libraries.load(SECTION2_MODE2, link=False) as (data_from, data_to):
            data_to.materials = [m for m in data_from.materials if m in to_fetch]
        log(f"appended materials: {[m.name for m in data_to.materials]}")

    materials = {n: bpy.data.materials[n] for n in wanted}
    bad = []
    for mat in materials.values():
        for n in mat.node_tree.nodes:
            if n.type != "TEX_IMAGE" or not n.image:
                continue
            fp = n.image.filepath
            if fp.startswith("//textures/"):
                abs_path = os.path.join(TEXTURES_DIR, fp[len("//textures/"):])
                if not os.path.isfile(abs_path):
                    bad.append((mat.name, n.image.name, f"missing: {abs_path}"))
                    continue
                n.image.filepath = abs_path
                n.image.reload()
            elif not os.path.isfile(bpy.path.abspath(fp)):
                bad.append((mat.name, n.image.name, fp))
    if bad:
        for m, img, reason in bad:
            log(f"BAD TEXTURE: material={m!r} image={img!r} -> {reason}")
        raise SystemExit(f"{len(bad)} texture path problem(s)")
    log("all appended material images resolve")
    return materials


def planar_uv(mesh, bounds):
    """Planar UV over `bounds` = (minx, maxx, miny, maxy). Passing the bounds in
    rather than measuring the mesh is what lets the rut mesh, which covers a
    handful of cells, share the terrain's 1024 m UV mapping exactly."""
    minx, maxx, miny, maxy = bounds
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        u = (co.x - minx) / (maxx - minx) if maxx > minx else 0.0
        v = (co.y - miny) / (maxy - miny) if maxy > miny else 0.0
        uv_layer.data[loop.index].uv = (u, v)


def patch_terrain(terrain_mat):
    """Returns the terrain's XY bounds, for the rut mesh to reuse."""
    terrain_obj = bpy.data.objects.get("world.terrain")
    if terrain_obj is None:
        raise SystemExit("world.terrain not found")
    mesh = terrain_obj.data
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    bounds = (min(xs), max(xs), min(ys), max(ys))
    if not mesh.uv_layers:
        planar_uv(mesh, bounds)
        log(f"planar UV on world.terrain ({len(mesh.vertices)} verts)")
    mesh.materials.clear()
    mesh.materials.append(terrain_mat)
    return bounds


def patch_plastic(plastic_mat):
    targets = [o for o in bpy.data.objects if o.type == "MESH"
               and ("trackshoe" in o.name.lower() or "spindle" in o.name.lower())]
    for o in targets:
        o.data.materials.clear()
        o.data.materials.append(plastic_mat)
    log(f"plastic -> {len(targets)} track-shoe/wheel object(s)")
    return targets


def patch_rocks(rock_mat):
    rock_targets = [o for o in bpy.data.objects if o.type == "MESH"
                    and "rock" in o.name.lower() and "sprocket" not in o.name.lower()]
    view_layer = bpy.context.view_layer
    seen_meshes = set()
    for o in rock_targets:
        if o.data.uv_layers or o.data.name in seen_meshes:
            continue
        seen_meshes.add(o.data.name)
        for ob in view_layer.objects:
            ob.select_set(False)
        o.select_set(True)
        view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0))
        bpy.ops.object.mode_set(mode="OBJECT")
    log(f"smart-UV-unwrapped {len(seen_meshes)} rock mesh datablock(s)")
    for o in rock_targets:
        o.data.materials.clear()
        o.data.materials.append(rock_mat)
    log(f"rock_pbr -> {len(rock_targets)} rock object(s)")
    return rock_targets


def is_achromatic(mat):
    if mat is None or mat.name == "rock":
        return True
    if not mat.use_nodes:
        c = mat.diffuse_color
    else:
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        c = bsdf.inputs["Base Color"].default_value if bsdf else (0.8, 0.8, 0.8, 1.0)
    return (max(c[0], c[1], c[2]) - min(c[0], c[1], c[2])) <= 0.05


def patch_metal(metal_mat, handled_names):
    targets = []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name in handled_names:
            continue
        mat = o.material_slots[0].material if o.material_slots else None
        if is_achromatic(mat):
            targets.append(o)
    for o in targets:
        o.data.materials.clear()
        o.data.materials.append(metal_mat)
    log(f"robot_pbr -> {len(targets)} untextured/achromatic object(s)")


def patch_light_and_world():
    for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data.users == 0:
            bpy.data.lights.remove(data)
    with bpy.data.libraries.load(SECTION2_MODE2, link=False) as (data_from, data_to):
        data_to.objects = [o for o in data_from.objects if o == LIGHT_OBJ_NAME]
    new_light = data_to.objects[0]
    bpy.context.scene.collection.objects.link(new_light)
    log(f"light {new_light.name!r} from section2_mode2.blend (energy {new_light.data.energy})")

    with bpy.data.libraries.load(SECTION2_MODE2, link=False) as (data_from, data_to):
        data_to.worlds = [w for w in data_from.worlds if w == WORLD_NAME]
    if data_to.worlds:
        old = bpy.context.scene.world
        bpy.context.scene.world = data_to.worlds[0]
        if old is not None and old.users == 0:
            bpy.data.worlds.remove(old)
    log(f"world -> {bpy.context.scene.world.name!r}")


def patch_textures():
    """Returns (terrain material, terrain XY bounds)."""
    materials = append_materials()
    bounds = patch_terrain(materials[TERRAIN_MAT_NAME])
    plastic = patch_plastic(materials[PLASTIC_MAT_NAME])
    rocks = patch_rocks(materials[ROCK_MAT_NAME])
    handled = {o.name for o in plastic} | {o.name for o in rocks} | {"world.terrain"}
    patch_metal(materials[METAL_MAT_NAME], handled)
    patch_light_and_world()
    return materials[TERRAIN_MAT_NAME], bounds


# ---------------------------------------------------------------------- ruts

def add_scm(dataset, ranks, t, heightmap, meta, stride, terrain_mat, bounds):
    """Rebuild the deformed terrain cells from the run's SCM recording and cut
    them out of the flat heightmap terrain. Must run AFTER patch_textures, so
    the ruts can take the terrain's material and its 1024 m UV mapping."""
    delta, plane, keys, z = scm_terrain.load_scm(dataset, ranks, t)
    if keys is None or not len(keys):
        log("no SCM deformation at this time; terrain left undeformed")
        return
    terrain_obj = bpy.data.objects.get("world.terrain")
    if terrain_obj is None:
        raise SystemExit("world.terrain not found; --scm needs the heightmap terrain")

    rows, nx, ny = site.read_gray_image(heightmap)
    terr = meta.get("terrain") or {}
    hm = scm_terrain.Heightmap(rows, nx, ny,
                               float(terr.get("length", 1024.0)),
                               float(terr.get("width", 1024.0)),
                               float(terr.get("min_height", -25.0)),
                               float(terr.get("max_height", 25.0)))
    rut = scm_terrain.RutMesh(hm, delta, plane, keys, stride=stride)
    rut.set_heights(z)
    planar_uv(rut.mesh, bounds)
    rut.mesh.materials.append(terrain_mat)
    scm_terrain.cut_terrain_cells(terrain_obj, rut.covered, ny)


# ---------------------------------------------------------------------- driver

def orbit_point(centre, dist, az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (centre[0] + dist * math.cos(el) * math.cos(az),
            centre[1] + dist * math.cos(el) * math.sin(az),
            centre[2] + dist * math.sin(el))


def frame_disc(centre, radius, az_deg, el_deg, lens=35.0, sensor=36.0, aspect=16.0 / 9.0,
               margin=1.06, bias=0.5):
    """Camera position and aim point that put a disc of `radius` about `centre`
    wholly inside the frame, looking down at `el_deg` from azimuth `az_deg`.

    render_site_anim.py sizes the overview camera as distance = k * radius with the
    camera aimed at the site centre, which does not hold: the vertical field of view
    is what binds on a circular site, and it is asymmetric about the aim point --
    the ground runs away from the camera much faster on the far side than on the
    near side. At its own defaults (35 mm, elevation 55 deg, distance 2.05 r) the
    near edge of the visible ground falls 48.6 m short of the centre while the far
    edge reaches 79 m past it, so the near half of a 56 m ring is cut off.

    So solve it instead. With the camera h above the ground plane and the optical
    axis depressed by el, the ground is in frame from h/tan(el + vhalf) to
    h/tan(el - vhalf) horizontally; ask that band to be 2*radius wide and put the
    disc in the middle of it. The aim point comes out short of the site centre by
    itself -- which is the correction render_site_frame.py applies by hand as
    0.18 * r -- and the height, not the distance, is what the elevation trades off.
    """
    R = radius * margin
    el = math.radians(el_deg)
    hhalf = math.atan(0.5 * sensor / lens)
    vhalf = math.atan(0.5 * sensor / aspect / lens)
    # below this the horizon enters frame and there is no far limit to solve for
    el = max(el, vhalf + math.radians(2.0))
    # ground distances from the camera, as multiples of its height: the bottom of
    # the frame, the optical axis, the top of the frame
    near, axis, far = 1.0 / math.tan(el + vhalf), 1.0 / math.tan(el), 1.0 / math.tan(el - vhalf)
    # `bias` trades distance against where the site sits on screen. The band is
    # asymmetric about the axis (far > axis - near), so the cheapest framing --
    # h = 2R/(far-near), the disc filling the band exactly -- puts the site centre
    # above the middle of the frame. Pushing it down to the frame centre costs
    # h = R/(axis-near) instead: 25% further out at 55 deg elevation, 75% at 34.
    h = R * ((1.0 - bias) * 2.0 / (far - near) + bias / (axis - near))
    d_c = min(max(h * axis, h * near + R), h * far - R)
    # ...and widen if it is the horizontal field of view that binds after all
    s = math.hypot(d_c, h)
    if s * math.tan(hhalf) < R:
        k = R / (s * math.tan(hhalf))
        h, d_c = h * k, d_c * k
    az = math.radians(az_deg)
    loc = (centre[0] + d_c * math.cos(az), centre[1] + d_c * math.sin(az), centre[2] + h)
    da = h / math.tan(el)                      # aim point along the optical axis
    aim = (loc[0] - da * math.cos(az), loc[1] - da * math.sin(az), centre[2])
    return loc, aim


def plan_camera(plan, t, a):
    """The overview camera at time `t`, straight off the camera plan -- the same
    arithmetic render_site_anim.py does, so a still lands exactly on the movie's
    camera path. See tools/make_camera_plan.py: plan["r"] is a forward-looking
    moving max of the site's vehicle radius, smoothed both ways, so the camera is
    already pulling back before a collector heads out and easing in as it returns.
    """
    import make_camera_plan as mcp

    cx, cy = plan["center"]
    rlo, rhi = min(plan["r"]), max(plan["r"])
    r_raw = max(mcp.interp(plan["t"], plan["r"], t), 45.0)
    z = mcp.interp(plan["t"], plan["z"], t)
    # zoomed in -> steep look; zoomed out -> shallow, interpolated on the zoom fraction
    frac = min(max((r_raw - rlo) / max(rhi - rlo, 1.0), 0.0), 1.0)
    el = a.overview_elevation + (a.overview_elevation_far - a.overview_elevation) * frac
    az = a.overview_azimuth + a.orbit_rate * t
    res = tuple(int(v) for v in a.res.lower().split("x"))
    loc, aim = frame_disc((cx, cy, z), r_raw, az, el, lens=a.lens,
                          aspect=res[0] / res[1], margin=a.overview_margin,
                          bias=getattr(a, "frame_bias", 0.5))
    return loc, aim, r_raw, el, az


def load_or_make_plan(path, dataset, ranks):
    import json

    import make_camera_plan as mcp
    if os.path.exists(path):
        with open(path) as f:
            plan = json.load(f)
        log(f"camera plan {path}: {len(plan['t'])} samples, "
            f"r {min(plan['r']):.0f}..{max(plan['r']):.0f} m")
        return plan
    log(f"camera plan {path} missing -- generating (one full read of the recording)")
    plan = mcp.build_plan(dataset, ranks)
    mcp.save_plan(plan, path)
    return plan


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="render_bird.py")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ranks", default="1-15")
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--meshes")
    ap.add_argument("--mesh-map")
    ap.add_argument("--heightmap")
    ap.add_argument("--render", required=True)
    ap.add_argument("--out", help=".blend to save as well")
    ap.add_argument("--engine", default="CYCLES")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--lens", type=float, default=35.0)
    # fixed overview framing, shared by every still in a series
    ap.add_argument("--site-radius", type=float, default=57.0,
                    help="m; what the camera distance is scaled off")
    ap.add_argument("--ground-z", type=float, default=3.1, help="m; site ground height")
    ap.add_argument("--cam-azimuth", type=float, default=-125.0)
    ap.add_argument("--cam-elevation", type=float, default=34.0)
    ap.add_argument("--cam-dist", type=float, default=2.45, help="x --site-radius")
    # ...or, instead of the fixed framing above, follow the movie's camera plan
    ap.add_argument("--camera-plan", help="camera_plan.json: frame this time off the "
                                          "plan instead of --site-radius (generated if missing)")
    ap.add_argument("--overview-margin", type=float, default=1.06,
                    help="how much bigger than the vehicle radius the framed disc is")
    ap.add_argument("--frame-bias", type=float, default=0.5,
                    help="0 = closest framing that still covers everything (site sits "
                         "high in frame), 1 = site centred on screen (further out)")
    ap.add_argument("--overview-elevation", type=float, default=55.0, help="deg when zoomed in")
    ap.add_argument("--overview-elevation-far", type=float, default=34.0, help="deg when zoomed out")
    ap.add_argument("--overview-azimuth", type=float, default=-125.0)
    ap.add_argument("--orbit-rate", type=float, default=0.25, help="deg per sim second")
    ap.add_argument("--scm", action="store_true", help="draw the real regolith deformation")
    ap.add_argument("--scm-stride", type=int, default=3,
                    help="rut sub-grid pitch in SCM nodes (3 = ~0.105 m)")
    ap.add_argument("--hull-lift", type=float, default=HULL_LIFT_Z,
                    help=f"m the builder hull VISUAL is raised (default {HULL_LIFT_Z:g})")
    ap.add_argument("--raw-heightmap", action="store_true",
                    help="keep render_site_frame's gamma-decoded heightmap read")
    ap.add_argument("--no-texture", action="store_true", help="skip the lunar-manip look")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    if not a.raw_heightmap:
        install_heightmap_fix()
    install_hull_lift(a.hull_lift)
    install_meshmap_relative(a.mesh_map)

    ranks = site.parse_ranks(a.ranks)
    if a.camera_plan:
        plan = load_or_make_plan(a.camera_plan, a.dataset, ranks)
        cam_loc, aim, r_raw, el, az = plan_camera(plan, a.time, a)
        log(f"plan camera at t={a.time:g}: vehicle radius {r_raw:.1f} m, "
            f"distance {math.dist(cam_loc, aim):.1f} m, elevation {el:.1f} deg, "
            f"azimuth {az:.1f} deg")
    else:
        r = a.site_radius
        centre = (0.0, 0.0, a.ground_z)
        cam_loc = orbit_point(centre, r * a.cam_dist, a.cam_azimuth, a.cam_elevation)
        # aim a little short of centre (towards the camera): in perspective the near
        # half of the ring projects larger than the far half
        caz = math.radians(a.cam_azimuth)
        aim = (0.18 * r * math.cos(caz), 0.18 * r * math.sin(caz), a.ground_z)

    build = argparse.Namespace(
        dataset=a.dataset, ranks=a.ranks, time=a.time, meshes=a.meshes,
        mesh_map=a.mesh_map, no_fit=False, groups=None, heightmap=a.heightmap,
        terrain_subdiv=0, no_terrain=False, no_static=False,
        out=None, render=None, views="custom",
        overview_dist=2.45, closeup_azimuth=50.0, sector_azimuth=38.0, closeup_rank=1,
        engine=a.engine, samples=a.samples, res=a.res, cpu=a.cpu, lens=a.lens,
        cam_azimuth=a.cam_azimuth, cam_elevation=a.cam_elevation,
        cam=f"{cam_loc[0]},{cam_loc[1]},{cam_loc[2]}:{aim[0]},{aim[1]},{aim[2]}",
        sun_azimuth=-40.0, sun_elevation=28.0, sun_strength=5.5,
        sky="0.015,0.015,0.02", sky_strength=1.0,
    )
    site.build_site(build)

    terrain_mat, bounds = (None, None)
    if not a.no_texture:
        terrain_mat, bounds = patch_textures()

    if a.scm:
        meta = site.load_meta(a.dataset, ranks[0]) if ranks else {}
        hm_path = site.find_heightmap(meta, a.heightmap)
        if hm_path is None:
            raise SystemExit("--scm needs the heightmap (--heightmap) to fill the cut cells")
        if terrain_mat is None:
            terrain_obj = bpy.data.objects["world.terrain"]
            terrain_mat = terrain_obj.data.materials[0]
            xs = [v.co.x for v in terrain_obj.data.vertices]
            ys = [v.co.y for v in terrain_obj.data.vertices]
            bounds = (min(xs), max(xs), min(ys), max(ys))
        add_scm(a.dataset, ranks, a.time, hm_path, meta, a.scm_stride, terrain_mat, bounds)

    scene = bpy.context.scene
    cam = bpy.data.objects["cam_custom"]
    scene.camera = cam
    log(f"camera at {tuple(round(v, 1) for v in cam_loc)} -> {tuple(round(v, 1) for v in aim)}, "
        f"{a.lens} mm")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
        log(f"saved {a.out}")

    os.makedirs(os.path.dirname(os.path.abspath(a.render)) or ".", exist_ok=True)
    scene.render.filepath = os.path.abspath(a.render)
    bpy.ops.render.render(write_still=True)
    log(f"wrote {a.render}")


if __name__ == "__main__":
    main()
