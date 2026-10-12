#!/usr/bin/env python3
"""Precompute the overview camera plan for a recording: how far the camera must be,
over time, to keep every vehicle in frame.

    python3 tools/make_camera_plan.py --dataset data --out renders/camera_plan.json

Scans every rank's builder + collector chassis at ~1 Hz and produces a smoothed
time series of the site's "vehicle radius" (max distance of any chassis from the
site centre) and mean ground height. tools/render_site_anim.py turns that into an
animated overview camera: it zooms out as collectors drive down their harvest rays
and back in as they return, always pre-empting (the radius series is a forward-
looking moving max) so no vehicle ever leaves frame.

The plan is DETERMINISTIC for a given recording, which is the point: every SLURM
chunk task loads the same file, so the camera path is seamless across chunks.
Stdlib only; one sequential read of each rank_N_frames.bin (~8 GB total, minutes).
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blender_import import load_manifest, load_meta, read_frames  # noqa: E402


def chassis_indices(objects):
    out = []
    for o in objects:
        if o["group"] == "builder" and "Chassis" in o["part"]:
            out.append(o["index"])
        if o["group"] == "collector" and "chassis" in o["part"].lower():
            out.append(o["index"])
    return out


def _ema(vals, alpha):
    out = list(vals)
    for i in range(1, len(out)):
        out[i] = out[i - 1] + alpha * (out[i] - out[i - 1])
    return out


def smooth(vals, alpha):
    """Zero-lag smoothing: EMA forward, then backward, averaged."""
    f = _ema(vals, alpha)
    b = _ema(vals[::-1], alpha)[::-1]
    return [(x + y) * 0.5 for x, y in zip(f, b)]


def moving_max(vals, half_window):
    n = len(vals)
    return [max(vals[max(0, i - half_window):min(n, i + half_window + 1)]) for i in range(n)]


def percentile(vals, p):
    """The p-th percentile of `vals` (p in 0..100), linearly interpolated.
    p=100 is the max, which is what keeps EVERY vehicle in frame; a lower p
    frames the bulk of the fleet and lets the few farthest-flung collectors run
    off the edge, which makes the zoom breathe a lot more once the fleet has
    desynchronised and somebody is always far out."""
    if not vals:
        return 0.0
    xs = sorted(vals)
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_plan(dataset, ranks=range(1, 16), sample_hz=1.0, lookahead_s=25.0, tau_s=10.0,
               pct=100.0):
    meta = load_meta(dataset, list(ranks)[0])
    rate = float(meta.get("rate_hz") or 60.0)
    site = meta.get("site") or {}
    cx, cy = (site.get("center") or [0, 0])[:2]
    stride = max(1, int(round(rate / sample_hz)))
    acc = {}  # sample key -> [t, max_r, sum_z, n]
    for rank in ranks:
        try:
            objects = load_manifest(dataset, rank)
        except FileNotFoundError:
            continue
        idxs = set(chassis_indices(objects))
        n = 0
        for time, poses in read_frames(dataset, rank, stride=stride):
            key = int(round(time * sample_hz))
            slot = acc.setdefault(key, [time, [], 0.0, 0])
            for i in idxs:
                p = poses.get(i)
                if p is None:
                    continue
                (x, y, z), _q = p
                slot[1].append(math.hypot(x - cx, y - cy))
                slot[2] += z
                slot[3] += 1
            n += 1
        print(f"[plan] rank {rank}: {n} samples", flush=True)
    keys = sorted(acc)
    t = [acc[k][0] for k in keys]
    z = [acc[k][2] / max(acc[k][3], 1) for k in keys]
    # zoom out BEFORE a vehicle gets far: forward-looking moving max, then smooth
    half = int(round(lookahead_s * sample_hz))
    alpha = 1.0 - math.exp(-1.0 / (tau_s * sample_hz))

    def series(p):
        return smooth(moving_max([percentile(acc[k][1], p) for k in keys], half), alpha)

    r = series(pct)
    z = smooth(z, alpha)
    plan = {
        "version": 1,
        "center": [cx, cy],
        "sample_hz": sample_hz,
        "lookahead_s": lookahead_s,
        "tau_s": tau_s,
        "percentile": pct,
        "t": [round(v, 3) for v in t],
        "r": [round(v, 2) for v in r],
        "z": [round(v, 2) for v in z],
    }
    if pct < 100:
        # keep the every-vehicle series alongside, so it is obvious from the file
        # how much of the fleet the chosen framing gives up
        plan["r_max"] = [round(v, 2) for v in series(100.0)]
    return plan


def save_plan(plan, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(plan, f)
    os.replace(tmp, path)  # atomic: concurrent generators just overwrite with identical content
    print(f"[plan] wrote {path}: {len(plan['t'])} samples, "
          f"r {min(plan['r']):.0f}..{max(plan['r']):.0f} m")


def interp(ts, vs, t):
    """Piecewise-linear interpolation with clamping."""
    import bisect
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    i = bisect.bisect_right(ts, t)
    w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return vs[i - 1] * (1 - w) + vs[i] * w


def main():
    ap = argparse.ArgumentParser(prog="make_camera_plan.py")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="renders/camera_plan.json")
    ap.add_argument("--sample-hz", type=float, default=1.0)
    ap.add_argument("--lookahead", type=float, default=25.0, help="zoom-out anticipation (s)")
    ap.add_argument("--tau", type=float, default=10.0, help="smoothing time constant (s)")
    ap.add_argument("--percentile", type=float, default=100.0,
                    help="which vehicle radius the framing covers: 100 = every vehicle "
                         "(default), lower frames the bulk of the fleet and zooms more")
    a = ap.parse_args()
    plan = build_plan(a.dataset, sample_hz=a.sample_hz, lookahead_s=a.lookahead, tau_s=a.tau,
                      pct=a.percentile)
    save_plan(plan, a.out)


if __name__ == "__main__":
    main()
