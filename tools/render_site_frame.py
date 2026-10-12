#!/usr/bin/env python3
"""Rebuild the WHOLE AMD-UW site (all ranks + static props + terrain) at one instant
and render it in Blender.

    blender --background --python tools/render_site_frame.py -- \
        --dataset data --meshes meshes --time 0.1 \
        --heightmap /path/to/terrain2.bmp \
        --out renders/site_t0.1.blend --render renders/site_t0.1.png

What this adds on top of tools/blender_import.py (which animates ONE rank):

  * loads every rank (default 1..15) and places every body at the sample nearest
    to --time -- a static snapshot, no keyframes, so a 15-rank scene stays light;
  * builds the terrain patch from the heightmap exactly as Chrono's
    RigidTerrain does (pixel (ix,iy) -> x = ix*dx - L/2, y = W/2 - iy*dy,
    z = hMin + gray*(hMax-hMin)/255), so vehicles sit on the ground they drove on;
  * loads static_props.jsonl (rings, pad, laid rocks);
  * proper primitives (box/cylinder) instead of wire placeholders, materials from
    the manifest colours, a sun + dark sky, and one or more cameras.

Conventions honoured (see README "Three conventions you must get right"):
  pose = body reference frame; world = body_pose * shape_frame; trimeshes are
  fitted to the recorded aabb_min/aabb_max. Chrono and Blender are both Z-up
  right-handed with [w,x,y,z] quaternions; OBJ is imported with forward=Y, up=Z
  (no axis remap).
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blender_import import load_manifest, load_meta, read_frames, resolve_mesh  # noqa: E402

DEFAULT_HEIGHTMAP_DIRS = [
    os.path.join(HERE, "..", "meshes", "amd-uw", "data", "terrain"),
    os.path.expanduser("~/AMD-UW_SBEL/data/terrain"),
]


# ------------------------------------------------------------------ data helpers

def frame_at(dataset, rank, t):
    """Return (time, poses) of the sample nearest to t (poses: {index: (pos, quat)})."""
    best = None
    for time, poses in read_frames(dataset, rank, start=max(0.0, t - 0.05), end=t + 0.05):
        if best is None or abs(time - t) < abs(best[0] - t):
            best = (time, poses)
    if best is None:  # t before the first sample or after the last: fall back to a scan
        for time, poses in read_frames(dataset, rank):
            if best is None or abs(time - t) < abs(best[0] - t):
                best = (time, poses)
            if time > t:
                break
    return best


def load_static(dataset):
    path = os.path.join(dataset, "static_props.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_heightmap(meta, explicit=None):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    rel = (meta.get("terrain") or {}).get("heightmap", "")
    base = os.path.basename(rel)
    for d in DEFAULT_HEIGHTMAP_DIRS:
        cand = os.path.join(d, base)
        if base and os.path.exists(cand):
            return cand
    return None


def read_gray_image(path):
    """Return (rows, width, height) of an 8-bit grayscale image. rows[iy][ix], iy=0 is top."""
    try:
        from PIL import Image
        im = Image.open(path).convert("L")
        w, h = im.size
        data = list(im.getdata())
        return [data[iy * w:(iy + 1) * w] for iy in range(h)], w, h
    except ImportError:
        pass
    # Blender always has bpy.data.images
    import bpy
    img = bpy.data.images.load(path)
    w, h = img.size
    px = list(img.pixels)  # RGBA float, bottom row first
    rows = []
    for iy in range(h):            # want top row first
        src = h - 1 - iy
        rows.append([int(round(px[(src * w + ix) * 4] * 255)) for ix in range(w)])
    bpy.data.images.remove(img)
    return rows, w, h


# ------------------------------------------------------------------ blender bits

def build_terrain(heightmap_path, meta, subdiv_levels=0):
    """Chrono RigidTerrain heightmap patch as a Blender mesh (see docstring)."""
    import bpy
    terr = meta.get("terrain") or {}
    L = float(terr.get("length", 1024.0))
    W = float(terr.get("width", 1024.0))
    hmin = float(terr.get("min_height", -25.0))
    hmax = float(terr.get("max_height", 25.0))
    rows, nx, ny = read_gray_image(heightmap_path)
    dx = L / (nx - 1)
    dy = W / (ny - 1)
    hs = (hmax - hmin) / 255.0
    verts = []
    for iy in range(ny - 1, -1, -1):          # Chrono orders from bottom-left row up
        y = 0.5 * W - iy * dy
        row = rows[iy]
        for ix in range(nx):
            verts.append((ix * dx - 0.5 * L, y, hmin + row[ix] * hs))
    # NOTE subdiv_levels defaults to 0: Chrono collides with this raw grid, and a
    # Catmull-Clark subsurf sags the rendered surface a few cm below it on this
    # terraced (0.196 m-quantised) heightmap -- enough to make resting rocks float.
    faces = []
    for jy in range(ny - 1):
        for jx in range(nx - 1):
            a = jy * nx + jx
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, b, d, c))
    me = bpy.data.meshes.new("terrain2")
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new("world.terrain", me)
    bpy.context.scene.collection.objects.link(ob)
    if subdiv_levels > 0:
        mod = ob.modifiers.new("subdiv", "SUBSURF")
        mod.levels = subdiv_levels
        mod.render_levels = subdiv_levels
    ob.data.materials.append(terrain_material())
    print(f"[site] terrain {nx}x{ny} px, {L}x{W} m, z {hmin + min(min(r) for r in rows) * hs:.2f}"
          f"..{hmin + max(max(r) for r in rows) * hs:.2f}")
    return ob


_mat_cache = {}


def terrain_material():
    import bpy
    if "regolith" in _mat_cache:
        return _mat_cache["regolith"]
    m = bpy.data.materials.new("regolith")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.30, 0.29, 0.28, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    # A little procedural grain so the flat-shaded 4 m/px terrain reads as soil.
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 0.8
    tex.inputs["Detail"].default_value = 8.0
    tex.inputs["Roughness"].default_value = 0.7
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.235, 0.23, 0.225, 1)
    ramp.color_ramp.elements[1].color = (0.38, 0.37, 0.36, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.2
    tex2 = nt.nodes.new("ShaderNodeTexNoise")
    tex2.inputs["Scale"].default_value = 6.0
    tex2.inputs["Detail"].default_value = 10.0
    nt.links.new(coord.outputs["Object"], tex2.inputs["Vector"])
    nt.links.new(tex2.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _mat_cache["regolith"] = m
    return m


def rock_material():
    """Shared rock material; Object Info > Random drives a 3-stop tint so the three
    rock meshes, used hundreds of times, do not read as copies."""
    import bpy
    if "rock" in _mat_cache:
        return _mat_cache["rock"]
    m = bpy.data.materials.new("rock")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    info = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.16, 0.13, 0.11, 1)
    ramp.color_ramp.elements[1].color = (0.36, 0.31, 0.26, 1)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.26, 0.22, 0.18, 1)
    nt.links.new(info.outputs["Random"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 60.0
    tex.inputs["Detail"].default_value = 6.0
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    _mat_cache["rock"] = m
    return m


def color_material(color, group=None, part=""):
    """One shared Principled material per (colour, group-class)."""
    import bpy
    if group == "rock" or "rock" in (part or "").lower():
        return rock_material()
    c = tuple(round(float(v), 3) for v in (color or [0.7, 0.7, 0.7])[:3])
    metal = group in ("builder", "collector", "builder_arm", "collector_arm", "collector_trailer")
    key = (c, metal)
    if key in _mat_cache:
        return _mat_cache[key]
    m = bpy.data.materials.new(f"col_{c[0]:.2f}_{c[1]:.2f}_{c[2]:.2f}{'_m' if metal else ''}")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*c, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.45 if metal else 0.85
    bsdf.inputs["Metallic"].default_value = 0.35 if metal else 0.0
    _mat_cache[key] = m
    return m


_ROCK_OBJ_RE = re.compile(r"^rock\d+\.obj$", re.IGNORECASE)


class MeshLibrary:
    """Import each source mesh once and hand out linked-data copies."""

    def __init__(self, mesh_dir, mesh_map):
        self.mesh_dir = mesh_dir
        self.mesh_map = mesh_map
        self.cache = {}

    def template(self, src_path, group_part):
        import bpy
        target = resolve_mesh(src_path, group_part, self.mesh_dir, self.mesh_map)
        if not target:
            return None
        if target in self.cache:
            return self.cache[target]
        before = set(bpy.data.objects)
        ext = os.path.splitext(target)[1].lower()
        try:
            if ext == ".obj":
                if hasattr(bpy.ops.wm, "obj_import"):
                    bpy.ops.wm.obj_import(filepath=target, forward_axis="Y", up_axis="Z")
                else:
                    bpy.ops.import_scene.obj(filepath=target, axis_forward="Y", axis_up="Z")
            elif ext in (".glb", ".gltf"):
                bpy.ops.import_scene.gltf(filepath=target)
            elif ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=target)
            elif ext == ".stl":
                if hasattr(bpy.ops.wm, "stl_import"):
                    bpy.ops.wm.stl_import(filepath=target)
                else:
                    bpy.ops.import_mesh.stl(filepath=target)
            else:
                self.cache[target] = None
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"[site] FAILED to import {target}: {exc}")
            self.cache[target] = None
            return None
        new = [o for o in bpy.data.objects if o not in before]
        if not new:
            self.cache[target] = None
            return None
        if len(new) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for o in new:
                o.select_set(True)
            bpy.context.view_layer.objects.active = new[0]
            bpy.ops.object.join()
            new = [bpy.context.view_layer.objects.active]
        tmpl = new[0]
        for coll in list(tmpl.users_collection):
            coll.objects.unlink(tmpl)
        # Chrono's LoadRockMesh (AMD-UW RockField.cpp) bakes mesh_scale, a +90 deg
        # rotation about X (the curiosity rock OBJs are Y-up) and a re-centring
        # (xy centre -> 0, min z -> 0) into the rock vertices BEFORE the AABB the
        # manifest records is measured. The AABB fit reproduces the scale and the
        # re-centring, but the rotation has to be baked here too -- without it the
        # fit squashes a side-lying mesh into the upright AABB (0.2,0.37,0.11
        # instead of uniform 0.2) and resting rocks appear to float / sink.
        if _ROCK_OBJ_RE.match(os.path.basename(target)):
            from mathutils import Matrix
            tmpl.data.transform(Matrix.Rotation(math.pi / 2.0, 4, "X"))
        # Smooth shading for the curved parts (wheels, rocks) -- the source OBJs are faceted.
        for p in tmpl.data.polygons:
            p.use_smooth = True
        self.cache[target] = tmpl
        return tmpl


def local_bbox(obj):
    """Object-local AABB from the MESH VERTICES, not obj.bound_box: bound_box is
    refreshed lazily by the depsgraph, so it is stale for template meshes that we
    transform in place after import (the rock Y-up rotation)."""
    from mathutils import Vector
    me = getattr(obj, "data", None)
    if me is not None and getattr(me, "vertices", None) and len(me.vertices):
        import numpy as np
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        return Vector(co.min(axis=0)), Vector(co.max(axis=0))
    cs = [Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    hi = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return lo, hi


def make_shape(shape, name, group_part, group, lib, fit_aabb, collection):
    """Create the Blender object for one visual shape, with its local (shape-frame) matrix set.

    Returns the object, or None. The object's matrix_basis = shape_frame * fit, so
    parenting it to the body empty (matrix_parent_inverse = identity) gives
    world = body_pose * shape_frame * fit.
    """
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    stype = shape.get("type")
    obj = None
    fit = Matrix.Identity(4)
    if stype == "trimesh" and shape.get("file"):
        tmpl = lib.template(shape["file"], group_part)
        if tmpl is not None:
            obj = tmpl.copy()
            obj.data = tmpl.data
            if fit_aabb and shape.get("aabb_min"):
                lo, hi = local_bbox(tmpl)
                tlo, thi = Vector(shape["aabb_min"]), Vector(shape["aabb_max"])
                cur, want = hi - lo, thi - tlo
                s = [(want[i] / cur[i]) if abs(cur[i]) > 1e-9 else 1.0 for i in range(3)]
                centre_now = Vector([(lo[i] + hi[i]) * 0.5 * s[i] for i in range(3)])
                off = (tlo + thi) * 0.5 - centre_now
                fit = Matrix.Translation(off) @ Matrix.Diagonal((s[0], s[1], s[2], 1.0))
            elif shape.get("scale"):
                sc = shape["scale"]
                fit = Matrix.Diagonal((sc[0], sc[1], sc[2], 1.0))
    if obj is None:
        if stype == "box":
            size = shape.get("size") or [0.1, 0.1, 0.1]
            me = _unit_cube()
            obj = bpy.data.objects.new(name, me)
            fit = Matrix.Diagonal((size[0], size[1], size[2], 1.0))
        elif stype == "cylinder":
            r = float(shape.get("radius", 0.05))
            h = float(shape.get("height", 0.1))
            me = _unit_cylinder()               # radius 1, height 1, axis Z (Chrono convention)
            obj = bpy.data.objects.new(name, me)
            fit = Matrix.Diagonal((r, r, h, 1.0))
        elif stype == "sphere":
            r = float(shape.get("radius", 0.05))
            me = _unit_sphere()
            obj = bpy.data.objects.new(name, me)
            fit = Matrix.Diagonal((r, r, r, 1.0))
        elif stype == "trimesh" and shape.get("aabb_min"):
            # unresolvable mesh: a solid box of the recorded extent
            tlo, thi = Vector(shape["aabb_min"]), Vector(shape["aabb_max"])
            me = _unit_cube()
            obj = bpy.data.objects.new(name, me)
            sz = thi - tlo
            fit = Matrix.Translation((tlo + thi) * 0.5) @ Matrix.Diagonal((sz.x, sz.y, sz.z, 1.0))
        else:
            return None
    obj.name = name
    collection.objects.link(obj)
    frame = Matrix.Translation(Vector(shape.get("pos", [0, 0, 0]))) @ \
        Quaternion(shape.get("rot", [1, 0, 0, 0])).to_matrix().to_4x4()
    obj.matrix_basis = frame @ fit
    mat = color_material(shape.get("color"), group, group_part)
    # Materials live on the mesh DATA, one datablock per (mesh, material) pair, so every
    # object is a true instance for Cycles. Object-linked materials make Cycles treat
    # each of the ~5,000 shapes as its own geometry, which costs seconds per frame.
    obj.data = _mesh_with_material(obj.data, mat)
    return obj


_mesh_mat_cache = {}


def _mesh_with_material(base, mat):
    key = (base.name, mat.name)
    me = _mesh_mat_cache.get(key)
    if me is not None:
        return me
    taken = any(k[0] == base.name for k in _mesh_mat_cache)
    me = base.copy() if taken else base
    if len(me.materials):
        for i in range(len(me.materials)):
            me.materials[i] = mat
    else:
        me.materials.append(mat)
    _mesh_mat_cache[key] = me
    return me


_prim_cache = {}


def _unit_cube():
    import bpy
    if "cube" not in _prim_cache:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        o = bpy.context.active_object
        me = o.data
        me.name = "prim_unit_cube"
        bpy.data.objects.remove(o)
        _prim_cache["cube"] = me
    return _prim_cache["cube"]


def _unit_cylinder():
    import bpy
    if "cyl" not in _prim_cache:
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=1.0, vertices=32)
        o = bpy.context.active_object
        me = o.data
        me.name = "prim_unit_cylinder"
        for p in me.polygons:
            p.use_smooth = True
        bpy.data.objects.remove(o)
        _prim_cache["cyl"] = me
    return _prim_cache["cyl"]


def _unit_sphere():
    import bpy
    if "sph" not in _prim_cache:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
        o = bpy.context.active_object
        me = o.data
        me.name = "prim_unit_sphere"
        for p in me.polygons:
            p.use_smooth = True
        bpy.data.objects.remove(o)
        _prim_cache["sph"] = me
    return _prim_cache["sph"]


def place_bodies(objects, poses, prefix, lib, fit_aabb, collection, groups=None):
    """One empty per body at its pose, shapes parented. Returns {index: empty}."""
    import bpy
    from mathutils import Quaternion
    empties = {}
    n_shapes = 0
    for obj in objects:
        if groups and obj["group"] not in groups:
            continue
        idx = obj["index"]
        pose = poses.get(idx)
        if pose is None:
            continue           # body did not exist yet at this instant
        pos, quat = pose
        gp = f"{obj['group']}/{obj['part']}"
        name = f"{prefix}.{obj['group']}.{obj['part']}"
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.25
        empty.rotation_mode = "QUATERNION"
        empty.location = pos
        empty.rotation_quaternion = Quaternion(quat)
        collection.objects.link(empty)
        empties[idx] = empty
        for si, shape in enumerate(obj.get("shapes", [])):
            child = make_shape(shape, f"{name}.s{si}", gp, obj["group"], lib, fit_aabb, collection)
            if child is not None:
                child.parent = empty  # matrix_parent_inverse stays identity
                n_shapes += 1
    return empties, n_shapes


def setup_world(sun_azimuth_deg, sun_elevation_deg, sun_strength, sky=(0.12, 0.13, 0.15), sky_strength=1.0):
    import bpy
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    # Lunar sky is black; keep a faint ambient so shadows are not pure black.
    bg.inputs["Color"].default_value = (*sky[:3], 1.0)
    bg.inputs["Strength"].default_value = sky_strength
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = sun_strength
    sun.angle = math.radians(0.53)
    sun.color = (1.0, 0.98, 0.94)
    so = bpy.data.objects.new("sun", sun)
    scene.collection.objects.link(so)
    az = math.radians(sun_azimuth_deg)
    el = math.radians(sun_elevation_deg)
    # point the -Z axis of the lamp along the light direction
    so.rotation_euler = (math.pi / 2 - el, 0.0, az + math.pi / 2)
    return so


def add_camera(name, location, target, lens=35.0, sensor=36.0):
    import bpy
    from mathutils import Vector
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.sensor_width = sensor
    cam.clip_end = 5000.0
    co = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(co)
    co.location = location
    direction = Vector(target) - Vector(location)
    co.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return co


def setup_render(engine, samples, res, use_gpu=True):
    import bpy
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    if engine.upper() == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        if use_gpu:
            prefs = bpy.context.preferences.addons.get("cycles")
            if prefs:
                cp = prefs.preferences
                chosen = None
                for dev_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        cp.compute_device_type = dev_type
                        cp.get_devices()
                        if any(d.type == dev_type for d in cp.devices):
                            chosen = dev_type
                            break
                    except Exception:  # noqa: BLE001
                        continue
                if chosen:
                    for d in cp.devices:
                        d.use = d.type == chosen        # GPU only: hybrid CPU+GPU crashed in Embree
                    scene.cycles.device = "GPU"
                    scene.cycles.device = "GPU"
                    print(f"[site] cycles on {chosen}: "
                          f"{[d.name for d in cp.devices if d.use]}")
                else:
                    print("[site] no GPU compute device found; CPU render")
    else:
        # EEVEE (Next in 4.2+, renamed BLENDER_EEVEE_NEXT / BLENDER_EEVEE)
        engs = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engs else "BLENDER_EEVEE"
        try:
            scene.eevee.taa_render_samples = samples
            scene.eevee.use_shadows = True
        except Exception:  # noqa: BLE001
            pass
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:  # noqa: BLE001
        scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"


# --------------------------------------------------------------------- driver

def build_site(a):
    import bpy

    # fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"

    ranks = parse_ranks(a.ranks)
    meta = load_meta(a.dataset, ranks[0]) if ranks else {}
    mesh_map = None
    if a.mesh_map:
        with open(a.mesh_map) as f:
            mesh_map = json.load(f)
    lib = MeshLibrary(a.meshes, mesh_map)
    groups = set(a.groups.split(",")) if a.groups else None
    fit = not a.no_fit

    # ---- terrain
    if not a.no_terrain:
        hm = find_heightmap(meta, a.heightmap)
        if hm:
            build_terrain(hm, meta, subdiv_levels=a.terrain_subdiv)
        else:
            print("[site] WARNING: no heightmap found (pass --heightmap); terrain skipped")

    # ---- static props
    if not a.no_static:
        coll = bpy.data.collections.new("static_props")
        scene.collection.children.link(coll)
        static = load_static(a.dataset)
        static = [o for o in static if not (o["part"].startswith("patch_"))]  # terrain done above
        poses = {o["index"]: (o["first_pos"], o["first_rot"]) for o in static}
        _, n = place_bodies(static, poses, "static", lib, fit, coll)
        print(f"[site] static props: {len(static)} bodies, {n} shapes")

    # ---- ranks
    VEHICLE_GROUPS = {"builder", "collector", "collector_trailer", "builder_arm", "collector_arm"}
    vehicle_positions = []
    anchors = {}
    for rank in ranks:
        try:
            objects = load_manifest(a.dataset, rank)
        except FileNotFoundError:
            print(f"[site] rank {rank}: no manifest, skipped")
            continue
        got = frame_at(a.dataset, rank, a.time)
        if got is None:
            print(f"[site] rank {rank}: no frames, skipped")
            continue
        t_actual, poses = got
        coll = bpy.data.collections.new(f"rank_{rank}")
        scene.collection.children.link(coll)
        empties, n = place_bodies(objects, poses, f"r{rank}", lib, fit, coll, groups)
        print(f"[site] rank {rank}: t={t_actual:.4f} ({len(poses)} poses) -> "
              f"{len(empties)} bodies, {n} shapes")
        by_index = {o["index"]: o for o in objects}
        for idx, e in empties.items():
            o = by_index[idx]
            if o["group"] in VEHICLE_GROUPS:
                vehicle_positions.append(tuple(e.location))
            if o["group"] == "builder" and "Chassis" in o["part"]:
                anchors[(rank, "builder")] = tuple(e.location)
            if o["group"] == "collector" and "chassis" in o["part"].lower():
                anchors[(rank, "collector")] = tuple(e.location)

    # ---- lights, cameras
    setup_world(a.sun_azimuth, a.sun_elevation, a.sun_strength,
                sky=tuple(float(v) for v in a.sky.split(",")), sky_strength=a.sky_strength)
    site = meta.get("site") or {}
    cx, cy = (site.get("center") or [0, 0])[:2]
    if vehicle_positions:
        zs = [p[2] for p in vehicle_positions]
        cz = sum(zs) / len(zs)
        rmax = max(math.hypot(p[0] - cx, p[1] - cy) for p in vehicle_positions)
    else:
        cz, rmax = 4.0, 45.0
    cameras = {}

    def orbit_point(centre, dist, az_deg, el_deg, dz=0.0):
        az, el = math.radians(az_deg), math.radians(el_deg)
        return (centre[0] + dist * math.cos(el) * math.cos(az),
                centre[1] + dist * math.cos(el) * math.sin(az),
                centre[2] + dist * math.sin(el) + dz)

    # overview: elevated 3/4 view of the whole site (framed on the vehicles, not the
    # rock lines, which run out to ~200 m along each ray)
    dist = rmax * a.overview_dist
    # In perspective the near half of the ring projects larger than the far half, so aim
    # a little short of the centre (towards the camera) to keep the whole ring in frame.
    caz = math.radians(a.cam_azimuth)
    aim = (cx + 0.18 * rmax * math.cos(caz), cy + 0.18 * rmax * math.sin(caz), cz)
    cameras["overview"] = add_camera("cam_overview", orbit_point((cx, cy, cz), dist, a.cam_azimuth, a.cam_elevation),
                                     aim, lens=a.lens)

    cr = a.closeup_rank
    if (cr, "builder") in anchors and (cr, "collector") in anchors:
        b, c = anchors[(cr, "builder")], anchors[(cr, "collector")]
        mid = tuple((b[i] + c[i]) * 0.5 for i in range(3))
        radial = math.degrees(math.atan2(mid[1] - cy, mid[0] - cx))
        span = math.dist(b, c)
        # closeup: one pair, seen from outside the ring, ~50 deg off the radial, from
        # slightly above so the tracks and wheels are visible on the ground
        d = max(span * 2.1, 18.0)
        cameras["closeup"] = add_camera(f"cam_closeup_r{cr}", orbit_point(mid, d, radial + a.closeup_azimuth, 24.0),
                                        (mid[0], mid[1], mid[2] + 0.3), lens=40.0)
        # sector: several neighbouring pairs, looking along the ring from outside it
        # (site centre -> pair, then swing tangentially so the next pairs recede in depth)
        look = anchors[(cr, "builder")]
        look = (look[0] * 0.8 + cx * 0.2, look[1] * 0.8 + cy * 0.2, look[2] + 0.5)
        cameras["sector"] = add_camera(f"cam_sector_r{cr}", orbit_point(look, 48.0, radial - a.sector_azimuth, 21.0),
                                       look, lens=35.0)
    if a.cam:
        p_s, t_s = a.cam.split(":")
        cameras["custom"] = add_camera("cam_custom",
                                       tuple(float(v) for v in p_s.split(",")),
                                       tuple(float(v) for v in t_s.split(",")),
                                       lens=a.lens)
    scene.camera = cameras["overview"]

    setup_render(a.engine, a.samples, tuple(int(v) for v in a.res.lower().split("x")), use_gpu=not a.cpu)
    n_obj = len(bpy.data.objects)
    print(f"[site] {n_obj} objects in scene")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
        print(f"[site] saved {a.out}")

    if a.render:
        base, ext = os.path.splitext(a.render)
        ext = ext or ".png"
        which = [w for w in a.views.split(",") if w in cameras]
        os.makedirs(os.path.dirname(os.path.abspath(a.render)) or ".", exist_ok=True)
        for w in which:
            scene.camera = cameras[w]
            path = a.render if (len(which) == 1) else f"{base}_{w}{ext}"
            scene.render.filepath = os.path.abspath(path)
            print(f"[site] rendering {w} -> {path}")
            bpy.ops.render.render(write_still=True)
            print(f"[site] wrote {path}")


def parse_ranks(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    import argparse
    ap = argparse.ArgumentParser(prog="render_site_frame.py")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ranks", default="1-15", help="e.g. 1-15 or 1,3,5")
    ap.add_argument("--time", type=float, default=0.1, help="instant to rebuild (nearest sample)")
    ap.add_argument("--meshes", help="directory of replacement meshes, matched by basename")
    ap.add_argument("--mesh-map", help="JSON {pattern: replacement path}")
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--groups", help="comma-separated subset of groups")
    ap.add_argument("--heightmap", help="terrain heightmap image (default: look in meshes/ or ~/AMD-UW_SBEL)")
    ap.add_argument("--terrain-subdiv", type=int, default=0)
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--out", help=".blend to save")
    ap.add_argument("--render", help="PNG to render (with --views a,b it becomes NAME_view.png)")
    ap.add_argument("--views", default="overview", help="any of overview,sector,closeup")
    ap.add_argument("--overview-dist", type=float, default=2.45, help="overview camera distance / site radius")
    ap.add_argument("--closeup-azimuth", type=float, default=50.0, help="deg off the pair's radial")
    ap.add_argument("--sector-azimuth", type=float, default=38.0, help="deg off the pair's radial")
    ap.add_argument("--closeup-rank", type=int, default=1)
    ap.add_argument("--engine", default="CYCLES", help="CYCLES or EEVEE")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--lens", type=float, default=35.0)
    ap.add_argument("--cam-azimuth", type=float, default=-125.0, help="deg, overview camera azimuth about site")
    ap.add_argument("--cam-elevation", type=float, default=32.0)
    ap.add_argument("--cam", help="extra camera 'px,py,pz:tx,ty,tz' (view name: custom)")
    ap.add_argument("--sun-azimuth", type=float, default=-40.0)
    ap.add_argument("--sun-elevation", type=float, default=28.0)
    ap.add_argument("--sun-strength", type=float, default=5.5)
    ap.add_argument("--sky", default="0.015,0.015,0.02", help="world background colour r,g,b (near-black lunar)")
    ap.add_argument("--sky-strength", type=float, default=1.0)
    a = ap.parse_args(argv)
    build_site(a)


if __name__ == "__main__":
    main()
