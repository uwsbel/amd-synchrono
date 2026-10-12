#!/usr/bin/env python3
"""Real SCM regolith deformation as Blender geometry, at one instant or over a
time window.

A run with Chrono's Soil Contact Model writes `rank_N_scm.bin`: the absolute
height of every grid node any vehicle disturbed, re-emitted at
`deformation_rate_hz`, later samples overwriting earlier ones for the same
node. That is the ground truth for the ruts -- the heightmap terrain the rest
of the pipeline draws is the ground as it STARTED.

The node pitch is 0.035 m over a 1024 m patch, so the deformation cannot just
be added to the terrain mesh (that grid is 4.0157 m). Instead:

  * find the terrain CELLS (not bounding boxes -- a rank's harvest lane is one
    long diagonal, whose bounding box is mostly not the lane) that contain any
    deformed node;
  * rebuild exactly those cells as a fine sub-grid, taking each vertex's height
    from the nearest deformed SCM node when there is one and from the heightmap
    otherwise;
  * DELETE those cells from the terrain mesh.

Deleting rather than overlaying is the point. Two surfaces a few centimetres
apart at 140 m read as a slab hovering over the ground, and a rut that ends up
under the terrain is simply invisible; after the cut there is one surface
everywhere. The seam is exact because a cell edge on the terrain mesh is the
straight line between two corner vertices, which is precisely what the bilinear
heightmap gives along that same edge -- and because an undisturbed SCM node's
height IS the bilinear heightmap (measured over rank 1 frame 0 of
run_20260831_021713: median difference 0.000 m).

For an animation the mesh is built ONCE, with the topology the window ends
with, and only vertex heights are rewritten per frame (`RutMesh.set_heights`):
a cell that has not been driven over yet simply renders as undeformed ground,
which is what the terrain quad it replaced would have shown anyway. See
`ScmWindow` for the replay.
"""

import os
import struct

import numpy as np

SCM_HEADER = struct.Struct("<8sIIdd7dii")
SCM_FRAME = struct.Struct("<IdI")
SCM_FILE_MAGIC = b"AMDUWSCM"
SCM_FRAME_MAGIC = 0x4D435353
NODE_DT = np.dtype([("i", "<i4"), ("j", "<i4"), ("z", "<f4")])
assert NODE_DT.itemsize == 12

# node indices fit in +/- 2**20 (the grid is 14629 across), so one int64 holds both
_SHIFT = 1 << 21
_BIAS = 1 << 20


def _pack(i, j):
    return (np.asarray(i, dtype=np.int64) + _BIAS) * _SHIFT + (np.asarray(j, dtype=np.int64) + _BIAS)


def _unpack(k):
    return (k // _SHIFT - _BIAS), (k % _SHIFT - _BIAS)


def _last_write_wins(keys, z):
    """(unique sorted keys, their final z) -- stable-sort by key carrying the
    original order as the tiebreak, then keep the last row of each key's run."""
    if not len(keys):
        return np.empty(0, np.int64), np.empty(0, np.float64)
    order = np.lexsort((np.arange(len(keys)), keys))
    ks, zs = keys[order], z[order]
    last = np.empty(len(ks), dtype=bool)
    last[-1] = True
    np.not_equal(ks[1:], ks[:-1], out=last[:-1])
    return ks[last], zs[last]


def read_scm_window(path, t_start, t_end):
    """One rank's deformation, split at `t_start`.

    Returns (delta, plane, before_keys, before_z, win_t, win_keys, win_z):
    everything at or before `t_start` collapsed to its final value (that is all
    a later frame can see of it), and every row in (t_start, t_end] kept in file
    order, which is time order. Collapsing the past is what keeps a chunk late
    in a long run from having to hold the whole recording in memory.
    """
    with open(path, "rb") as f:
        head = f.read(SCM_HEADER.size)
        if len(head) < SCM_HEADER.size:
            raise ValueError(f"{path}: shorter than its header")
        fields = SCM_HEADER.unpack(head)
        if fields[0] != SCM_FILE_MAGIC:
            raise ValueError(f"{path}: bad magic {fields[0]!r}")
        delta, plane = fields[4], fields[5:12]
        bk, bz = [], []
        wt, wk, wz = [], [], []
        while True:
            fh = f.read(SCM_FRAME.size)
            if len(fh) < SCM_FRAME.size:
                break
            magic, t, count = SCM_FRAME.unpack(fh)
            if magic != SCM_FRAME_MAGIC:
                raise ValueError(f"{path}: lost frame sync at t={t}")
            buf = f.read(NODE_DT.itemsize * count)
            if len(buf) < NODE_DT.itemsize * count:
                break                      # truncated final sample
            if t > t_end:
                break
            a = np.frombuffer(buf, dtype=NODE_DT)
            k = _pack(a["i"], a["j"])
            if t <= t_start:
                bk.append(k)
                bz.append(a["z"])
            else:
                wt.append(t)
                wk.append(k)
                wz.append(a["z"].astype(np.float64))
    bk = np.concatenate(bk) if bk else np.empty(0, np.int64)
    bz = np.concatenate(bz).astype(np.float64) if len(bk) else np.empty(0, np.float64)
    bk, bz = _last_write_wins(bk, bz)
    return delta, plane, bk, bz, wt, wk, wz


class ScmWindow:
    """Every rank's deformation over [t_start, t_end], replayable forward.

    `keys` is the sorted union of every node touched by t_end -- the topology
    the mesh is built for. `heights_at(t)` returns those nodes' heights at time
    t, NaN for a node nothing has touched yet.
    """

    def __init__(self, dataset, ranks, t_start, t_end):
        self.delta = self.plane = None
        base_k, base_z = [], []
        wt, wk, wz = [], [], []
        for rank in ranks:
            path = os.path.join(dataset, f"rank_{rank}_scm.bin")
            if not os.path.exists(path):
                continue
            d, p, bk, bz, rt, rk, rz = read_scm_window(path, t_start, t_end)
            if self.delta is None:
                self.delta, self.plane = d, p
            elif abs(d - self.delta) > 1e-9:
                raise SystemExit(f"{path}: node pitch {d} != {self.delta} of the first rank")
            base_k.append(bk)
            base_z.append(bz)
            wt.extend(rt)
            wk.extend(rk)
            wz.extend(rz)
        if not base_k and not wk:
            self.keys = np.empty(0, np.int64)
            return

        bk = np.concatenate(base_k) if base_k else np.empty(0, np.int64)
        bz = np.concatenate(base_z) if base_z else np.empty(0, np.float64)
        # two ranks can touch the same node; whoever wrote last still wins, and
        # within one time bucket the order between ranks is arbitrary anyway
        bk, bz = _last_write_wins(bk, bz)

        # the window's rows, in time order across ranks
        if wt:
            order = np.argsort(np.asarray(wt), kind="stable")
            self.win_t = np.asarray(wt, dtype=np.float64)[order]
            counts = np.array([len(k) for k in wk], dtype=np.int64)[order]
            self.win_off = np.concatenate(([0], np.cumsum(counts)))
            self.win_k = np.concatenate([wk[i] for i in order]) if len(order) else np.empty(0, np.int64)
            self.win_z = np.concatenate([wz[i] for i in order]) if len(order) else np.empty(0, np.float64)
        else:
            self.win_t = np.empty(0, np.float64)
            self.win_off = np.zeros(1, dtype=np.int64)
            self.win_k = np.empty(0, np.int64)
            self.win_z = np.empty(0, np.float64)

        self.keys = np.union1d(bk, np.unique(self.win_k))
        self.z0 = np.full(len(self.keys), np.nan)
        if len(bk):
            self.z0[np.searchsorted(self.keys, bk)] = bz
        self.win_slot = np.searchsorted(self.keys, self.win_k)
        self._z = self.z0.copy()
        self._cursor = 0          # how many window SAMPLES have been applied
        self.t_start, self.t_end = t_start, t_end

    def __len__(self):
        return len(self.keys)

    def heights_at(self, t):
        """Node heights at time t. Monotonic in t; a backward step replays from
        the start of the window, which only a rewind can cause."""
        want = int(np.searchsorted(self.win_t, t, side="right"))
        if want < self._cursor:
            self._z = self.z0.copy()
            self._cursor = 0
        if want > self._cursor:
            a, b = self.win_off[self._cursor], self.win_off[want]
            # later rows overwrite earlier ones simply by assigning in order
            self._z[self.win_slot[a:b]] = self.win_z[a:b]
            self._cursor = want
        return self._z


def load_scm(dataset, ranks, tmax):
    """Every node deformed by `tmax`, last write winning -- the single-instant
    case, i.e. ScmWindow(dataset, ranks, tmax, tmax)."""
    w = ScmWindow(dataset, ranks, tmax, tmax)
    if not len(w):
        return None, None, None, None
    return w.delta, w.plane, w.keys, w.heights_at(tmax)


class Heightmap:
    """The terrain as render_site_frame.build_terrain lays it out: vertex
    (jx, r) at x = jx*dx - L/2, y = r*dy - W/2, height from grey row ny-1-r.
    `grey` is rows[iy][ix] on 0..255 (float), iy=0 the top image row."""

    def __init__(self, grey, nx, ny, length, width, hmin, hmax):
        self.g = np.asarray(grey, dtype=np.float64)[::-1]   # index by r, not iy
        self.nx, self.ny = nx, ny
        self.L, self.W = length, width
        self.dx = length / (nx - 1)
        self.dy = width / (ny - 1)
        self.hmin, self.hs = hmin, (hmax - hmin) / 255.0

    def cell_of(self, x, y):
        jx = np.clip(np.floor((x + 0.5 * self.L) / self.dx).astype(np.int64), 0, self.nx - 2)
        r = np.clip(np.floor((y + 0.5 * self.W) / self.dy).astype(np.int64), 0, self.ny - 2)
        return jx, r

    def height(self, x, y):
        fx = np.clip((x + 0.5 * self.L) / self.dx, 0, self.nx - 1)
        fy = np.clip((y + 0.5 * self.W) / self.dy, 0, self.ny - 1)
        i0 = np.minimum(np.floor(fx).astype(np.int64), self.nx - 2)
        r0 = np.minimum(np.floor(fy).astype(np.int64), self.ny - 2)
        tx, ty = fx - i0, fy - r0
        g = self.g
        return self.hmin + self.hs * (
            g[r0, i0] * (1 - tx) * (1 - ty) + g[r0, i0 + 1] * tx * (1 - ty)
            + g[r0 + 1, i0] * (1 - tx) * ty + g[r0 + 1, i0 + 1] * tx * ty)


class RutMesh:
    """The deformed terrain cells, rebuilt fine. Topology is fixed at
    construction (from `keys`, the nodes touched by the END of the window);
    `set_heights` rewrites the vertex z for one instant."""

    def __init__(self, hm, node_delta, plane, keys, stride=3, name="scm_ruts"):
        import bpy

        if not np.allclose(plane[:3], 0) or not np.allclose(plane[3:], (1, 0, 0, 0)):
            raise SystemExit(f"SCM reference plane is not the identity: {plane}")

        self.keys = keys
        ni, nj = _unpack(keys)
        cjx, cr = hm.cell_of(ni * node_delta, nj * node_delta)
        self.covered = np.unique(cjx * (hm.ny - 1) + cr)
        cells_jx = self.covered // (hm.ny - 1)
        cells_r = self.covered % (hm.ny - 1)

        # n sub-quads per cell edge, the same for every cell so neighbouring
        # cells land on one global lattice and share their boundary vertices
        n = max(1, int(round(hm.dx / (stride * node_delta))))
        step_x, step_y = hm.dx / n, hm.dy / n

        gi = np.broadcast_to(cells_jx[:, None, None] * n + np.arange(n + 1)[None, :, None],
                             (len(self.covered), n + 1, n + 1)).ravel()
        gj = np.broadcast_to(cells_r[:, None, None] * n + np.arange(n + 1)[None, None, :],
                             (len(self.covered), n + 1, n + 1)).ravel()
        gkey, inv = np.unique(gi.astype(np.int64) * (1 << 32) + gj, return_inverse=True)
        ui, uj = gkey // (1 << 32), gkey % (1 << 32)

        vx = ui * step_x - 0.5 * hm.L
        vy = uj * step_y - 0.5 * hm.W
        self.co = np.empty((len(vx), 3), dtype=np.float64)
        self.co[:, 0], self.co[:, 1] = vx, vy
        self.base_z = hm.height(vx, vy)
        self.co[:, 2] = self.base_z

        # which SCM node, if any, each vertex sits on
        vkey = _pack(np.rint(vx / node_delta), np.rint(vy / node_delta))
        pos = np.clip(np.searchsorted(keys, vkey), 0, max(len(keys) - 1, 0))
        self.hit = (keys[pos] == vkey) if len(keys) else np.zeros(len(vx), bool)
        self.slot = pos

        idx = inv.reshape(len(self.covered), n + 1, n + 1)
        faces = np.stack([idx[:, :-1, :-1].ravel(), idx[:, 1:, :-1].ravel(),
                          idx[:, 1:, 1:].ravel(), idx[:, :-1, 1:].ravel()], axis=1)

        me = bpy.data.meshes.new(name)
        me.from_pydata(self.co.tolist(), [], faces.tolist())
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        self.mesh = me
        self.obj = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(self.obj)
        print(f"[scm] {len(keys)} deformed nodes -> {len(self.covered)} terrain cells, "
              f"{n}x{n} sub-quads each ({step_x:.3f} m), {len(self.co)} verts / "
              f"{len(faces)} faces, {int(self.hit.sum())} of them on a deformed node")

    def set_heights(self, z):
        """`z` is aligned to the `keys` the mesh was built from; NaN means that
        node has not been touched yet, so the vertex keeps the heightmap."""
        zv = np.where(self.hit, z[self.slot], np.nan)
        self.co[:, 2] = np.where(np.isnan(zv), self.base_z, zv)
        self.mesh.vertices.foreach_set("co", self.co.ravel())
        self.mesh.update()


def cut_terrain_cells(terrain_obj, covered, ny):
    """Delete the terrain faces the rut mesh replaces, so no two surfaces
    coincide. Face (jx, r) of build_terrain's grid is index r*(nx-1)+jx."""
    import bmesh

    me = terrain_obj.data
    nx = len(me.vertices) // ny
    faces = covered // (ny - 1) + (covered % (ny - 1)) * (nx - 1)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    doomed = [bm.faces[int(k)] for k in faces if 0 <= k < len(bm.faces)]
    bmesh.ops.delete(bm, geom=doomed, context="FACES_ONLY")
    bm.to_mesh(me)
    bm.free()
    me.update()
    print(f"[scm] cut {len(doomed)} of the terrain's {(nx - 1) * (ny - 1)} cells")
