#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_frame_id(name: str) -> int | None:
    # Collect any "Frame 0195" or "Frame-1" tokens and use the largest value.
    matches = re.findall(r"Frame[\s\-]*(\d+)", name)
    if not matches:
        return None
    return max(int(m) for m in matches)


def euclid(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def mean_axis(vals):
    return [statistics.mean([v[i] for v in vals]) for i in range(3)] if vals else None


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)

def clip_limits(vals, p_low=0.01, p_high=0.99):
    if not vals:
        return None
    lo = pct(vals, p_low)
    hi = pct(vals, p_high)
    if lo is None or hi is None:
        return None
    if lo == hi:
        return (lo - 0.1, hi + 0.1)
    return (lo, hi)

def set_ylim_tight(ax, vals, pad_ratio=0.1):
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    if vmin == vmax:
        pad = 0.1 if vmin == 0 else abs(vmin) * pad_ratio
        ax.set_ylim(vmin - pad, vmax + pad)
        return
    pad = (vmax - vmin) * pad_ratio
    ax.set_ylim(vmin - pad, vmax + pad)

def set_y_ticks_precision(ax, decimals=2, step=None, max_ticks=200):
    try:
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
    except Exception:
        return
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    if step is not None:
        y0, y1 = ax.get_ylim()
        span = abs(y1 - y0)
        if span > 0 and (span / step) > max_ticks:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(step))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def match_pairs(gt_list, pr_list):
    unused = list(range(len(pr_list)))
    pairs = []
    for g in gt_list:
        gcen = g["box_3d"]["center"]
        best = None
        bestd = 1e9
        for pi in unused:
            p = pr_list[pi]
            d = euclid(gcen, p["box_3d"]["center"])
            if d < bestd:
                bestd = d
                best = pi
        if best is None:
            continue
        unused.remove(best)
        pairs.append((g, pr_list[best], bestd))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze keypoint xyz error distribution and per-kp ranking."
    )
    parser.add_argument(
        "--gt_dir", default="labels", help="groundtruth folder with JSON files"
    )
    parser.add_argument(
        "--pred_dir",
        default="prediction_results",
        help="prediction folder with JSON files",
    )
    parser.add_argument(
        "--visible_only",
        action="store_true",
        help="compute xyz distribution using visible GT keypoints only",
    )
    parser.add_argument(
        "--out_dir",
        default="kp_error_report",
        help="output directory for CSV and plots",
    )
    parser.add_argument("--frame_min", type=int, default=None, help="min frame id to include")
    parser.add_argument("--frame_max", type=int, default=None, help="max frame id to include")
    parser.add_argument(
        "--name_prefix",
        default=None,
        help="only include files whose names start with this prefix",
    )
    parser.add_argument(
        "--y_tick_step",
        type=float,
        default=0.01,
        help="y-axis major tick step (meters) for plots",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=4.5,
        help="plot height in inches to improve tick readability",
    )
    parser.add_argument(
        "--clip_p99",
        action="store_true",
        help="also output clipped plots (1-99 percentile) to suppress outliers",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pr_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    gt_files = sorted(gt_dir.glob("*.json"))
    pr_files = sorted(pr_dir.glob("*.json"))

    if not gt_files:
        raise SystemExit(f"No gt files found in {gt_dir}")
    if not pr_files:
        raise SystemExit(f"No pred files found in {pr_dir}")

    pr_by_frame = {}
    for p in pr_files:
        if args.name_prefix and not p.name.startswith(args.name_prefix):
            continue
        fid = parse_frame_id(p.name)
        if fid is not None:
            if args.frame_min is not None and fid < args.frame_min:
                continue
            if args.frame_max is not None and fid > args.frame_max:
                continue
            pr_by_frame[fid] = p

    axis_signed = []
    axis_abs = []
    kp_err = defaultdict(list)
    kp_axis_abs = defaultdict(list)
    center_dists = []
    frame_l2 = defaultdict(list)
    frame_axis_abs = defaultdict(list)
    frame_axis_signed = defaultdict(list)
    kp_frame_axis_abs = defaultdict(lambda: defaultdict(list))
    kp_frame_axis_signed = defaultdict(lambda: defaultdict(list))

    total_pairs = 0
    missing_pred = 0

    for gpath in gt_files:
        if args.name_prefix and not gpath.name.startswith(args.name_prefix):
            continue
        fid = parse_frame_id(gpath.name)
        if fid is None:
            missing_pred += 1
            continue
        if args.frame_min is not None and fid < args.frame_min:
            continue
        if args.frame_max is not None and fid > args.frame_max:
            continue
        if fid not in pr_by_frame:
            missing_pred += 1
            continue
        gtd = json.loads(gpath.read_text())
        prd = json.loads(pr_by_frame[fid].read_text())

        for g, p, center_d in match_pairs(gtd, prd):
            gk = g["keypoints"]
            pk = p["keypoints"]
            vis = g.get("keypoints_visible")
            center_dists.append(center_d)
            for i in range(min(len(gk), len(pk))):
                if args.visible_only and vis and not vis[i]:
                    continue
                dx = pk[i][0] - gk[i][0]
                dy = pk[i][1] - gk[i][1]
                dz = pk[i][2] - gk[i][2]
                axis_signed.append((dx, dy, dz))
                axis_abs.append((abs(dx), abs(dy), abs(dz)))
                l2 = euclid(gk[i], pk[i])
                kp_err[i].append(l2)
                kp_axis_abs[i].append((abs(dx), abs(dy), abs(dz)))
                frame_l2[fid].append(l2)
                frame_axis_abs[fid].append((abs(dx), abs(dy), abs(dz)))
                frame_axis_signed[fid].append((dx, dy, dz))
                kp_frame_axis_abs[i][fid].append((abs(dx), abs(dy), abs(dz)))
                kp_frame_axis_signed[i][fid].append((dx, dy, dz))
            total_pairs += 1

    print(f"frames(gt)={len(gt_files)} frames(pred)={len(pr_files)} missing_pred={missing_pred}")
    print(f"matched_pairs={total_pairs} visible_only={args.visible_only}")

    if center_dists:
        print("box_center_dist: mean={:.4f} median={:.4f} p90={:.4f} max={:.4f}".format(
            statistics.mean(center_dists),
            statistics.median(center_dists),
            pct(center_dists, 0.9),
            max(center_dists),
        ))

    if axis_signed:
        mean_signed = mean_axis(axis_signed)
        mean_abs = mean_axis(axis_abs)
        std_abs = [statistics.pstdev([v[i] for v in axis_abs]) for i in range(3)]
        print("xyz_signed_mean: dx={:.4f} dy={:.4f} dz={:.4f}".format(*mean_signed))
        print("xyz_abs_mean:    dx={:.4f} dy={:.4f} dz={:.4f}".format(*mean_abs))
        print("xyz_abs_std:     dx={:.4f} dy={:.4f} dz={:.4f}".format(*std_abs))

        for axis, idx in (("dx", 0), ("dy", 1), ("dz", 2)):
            vals = [v[idx] for v in axis_abs]
            print(
                f"{axis}_abs_dist: mean={statistics.mean(vals):.4f} "
                f"median={statistics.median(vals):.4f} "
                f"p90={pct(vals, 0.9):.4f} "
                f"p95={pct(vals, 0.95):.4f} "
                f"max={max(vals):.4f}"
            )

    if kp_err:
        print("per_kp_rank: idx mean_l2 dx_abs dy_abs dz_abs count")
        ranked = sorted(kp_err.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
        for idx, errs in ranked:
            mean_l2 = statistics.mean(errs)
            mean_abs = mean_axis(kp_axis_abs[idx])
            print(
                f"{idx:02d} {mean_l2:.4f} "
                f"{mean_abs[0]:.4f} {mean_abs[1]:.4f} {mean_abs[2]:.4f} {len(errs)}"
            )

    if frame_l2:
        ranked_frames = sorted(
            frame_l2.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True
        )
        print("worst_frames: frame_id mean_l2 count")
        for fid, errs in ranked_frames[:5]:
            print(f"{fid} {statistics.mean(errs):.4f} {len(errs)}")

    # Write CSV outputs.
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_csv = out_dir / "axis_distribution.csv"
    with axis_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "mean_abs", "median_abs", "p90_abs", "p95_abs", "max_abs"])
        for axis, idx in (("dx", 0), ("dy", 1), ("dz", 2)):
            vals = [v[idx] for v in axis_abs] if axis_abs else []
            if vals:
                writer.writerow(
                    [
                        axis,
                        f"{statistics.mean(vals):.6f}",
                        f"{statistics.median(vals):.6f}",
                        f"{pct(vals, 0.9):.6f}",
                        f"{pct(vals, 0.95):.6f}",
                        f"{max(vals):.6f}",
                    ]
                )
            else:
                writer.writerow([axis, "", "", "", "", ""])

    kp_csv = out_dir / "per_kp_rank.csv"
    with kp_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kp_idx", "mean_l2", "mean_abs_dx", "mean_abs_dy", "mean_abs_dz", "count"])
        ranked = sorted(kp_err.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
        for idx, errs in ranked:
            mean_l2 = statistics.mean(errs)
            mean_abs = mean_axis(kp_axis_abs[idx])
            writer.writerow(
                [
                    idx,
                    f"{mean_l2:.6f}",
                    f"{mean_abs[0]:.6f}",
                    f"{mean_abs[1]:.6f}",
                    f"{mean_abs[2]:.6f}",
                    len(errs),
                ]
            )

    frame_csv = out_dir / "frame_trend.csv"
    with frame_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_id", "mean_l2", "mean_abs_dx", "mean_abs_dy", "mean_abs_dz", "count_points"]
        )
        for fid in sorted(frame_l2.keys()):
            mean_l2 = statistics.mean(frame_l2[fid])
            mean_abs = mean_axis(frame_axis_abs[fid])
            writer.writerow(
                [
                    fid,
                    f"{mean_l2:.6f}",
                    f"{mean_abs[0]:.6f}",
                    f"{mean_abs[1]:.6f}",
                    f"{mean_abs[2]:.6f}",
                    len(frame_l2[fid]),
                ]
            )

    # Plot outputs.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib not available, skip plots: {exc}")
        return

    # Histograms for axis abs errors.
    if axis_abs:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, axis, idx in zip(axes, ("dx", "dy", "dz"), (0, 1, 2)):
            vals = [v[idx] for v in axis_abs]
            ax.hist(vals, bins=30, color="#2d6cdf", alpha=0.8)
            ax.set_title(f"{axis} |abs| hist")
            ax.set_xlabel("meters")
            ax.set_ylabel("count")
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "axis_abs_hist.png", dpi=200)
        plt.close(fig)

    # Heatmap for per-kp mean abs error.
    if kp_err:
        kp_ids = sorted(kp_err.keys())
        heat = []
        for idx in kp_ids:
            mean_abs = mean_axis(kp_axis_abs[idx])
            heat.append(mean_abs)
        fig, ax = plt.subplots(figsize=(6, max(2, len(kp_ids) * 0.25)))
        im = ax.imshow(heat, aspect="auto", cmap="magma")
        ax.set_title("per-kp mean |dx| |dy| |dz|")
        ax.set_xlabel("axis")
        ax.set_ylabel("kp_idx")
        ax.set_xticks([0, 1, 2], labels=["dx", "dy", "dz"])
        ax.set_yticks(range(len(kp_ids)), labels=[str(i) for i in kp_ids])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "per_kp_axis_heatmap.png", dpi=200)
        plt.close(fig)

    if frame_l2:
        frames = sorted(frame_l2.keys())
        x_idx = list(range(len(frames)))
        l2_means = [statistics.mean(frame_l2[fid]) for fid in frames]
        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, l2_means, color="#2d6cdf")
        ax.set_title("frame mean L2 error")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, l2_means)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "frame_l2_trend.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(l2_means)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, l2_means, color="#2d6cdf")
            ax.set_title("frame mean L2 error (p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_l2_trend_p1p99.png", dpi=200)
            plt.close(fig)

        dx_means = [mean_axis(frame_axis_abs[fid])[0] for fid in frames]
        dy_means = [mean_axis(frame_axis_abs[fid])[1] for fid in frames]
        dz_means = [mean_axis(frame_axis_abs[fid])[2] for fid in frames]
        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dx_means, label="dx", color="#2d6cdf")
        ax.plot(x_idx, dy_means, label="dy", color="#e89f00")
        ax.plot(x_idx, dz_means, label="dz", color="#009e73")
        ax.set_title("frame mean |axis| error")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        ax.legend()
        set_ylim_tight(ax, dx_means + dy_means + dz_means)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "frame_axis_trend.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dx_means + dy_means + dz_means)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dx_means, label="dx", color="#2d6cdf")
            ax.plot(x_idx, dy_means, label="dy", color="#e89f00")
            ax.plot(x_idx, dz_means, label="dz", color="#009e73")
            ax.set_title("frame mean |axis| error (p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            ax.legend()
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_axis_trend_p1p99.png", dpi=200)
            plt.close(fig)

        dx_means_s = [mean_axis(frame_axis_signed[fid])[0] for fid in frames]
        dy_means_s = [mean_axis(frame_axis_signed[fid])[1] for fid in frames]
        dz_means_s = [mean_axis(frame_axis_signed[fid])[2] for fid in frames]
        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dx_means_s, label="dx", color="#2d6cdf")
        ax.plot(x_idx, dy_means_s, label="dy", color="#e89f00")
        ax.plot(x_idx, dz_means_s, label="dz", color="#009e73")
        ax.set_title("frame mean axis error (signed)")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        ax.legend()
        set_ylim_tight(ax, dx_means_s + dy_means_s + dz_means_s)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "frame_axis_trend_signed.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dx_means_s + dy_means_s + dz_means_s)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dx_means_s, label="dx", color="#2d6cdf")
            ax.plot(x_idx, dy_means_s, label="dy", color="#e89f00")
            ax.plot(x_idx, dz_means_s, label="dz", color="#009e73")
            ax.set_title("frame mean axis error (signed, p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            ax.legend()
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_axis_trend_signed_p1p99.png", dpi=200)
            plt.close(fig)

        # Separate per-axis plots.
        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dx_means, color="#2d6cdf")
        ax.set_title("frame mean |dx| error")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, dx_means)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "frame_dx_trend.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dx_means)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dx_means, color="#2d6cdf")
            ax.set_title("frame mean |dx| error (p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_dx_trend_p1p99.png", dpi=200)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dx_means_s, color="#2d6cdf")
        ax.set_title("frame mean dx error (signed)")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, dx_means_s)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "frame_dx_trend_signed.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dx_means_s)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dx_means_s, color="#2d6cdf")
            ax.set_title("frame mean dx error (signed, p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_dx_trend_signed_p1p99.png", dpi=200)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dy_means, color="#e89f00")
        ax.set_title("frame mean |dy| error")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, dy_means)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
        fig.tight_layout()
        fig.savefig(out_dir / "frame_dy_trend.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dy_means)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dy_means, color="#e89f00")
            ax.set_title("frame mean |dy| error (p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_dy_trend_p1p99.png", dpi=200)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dy_means_s, color="#e89f00")
        ax.set_title("frame mean dy error (signed)")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, dy_means_s)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        fig.tight_layout()
        fig.savefig(out_dir / "frame_dy_trend_signed.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dy_means_s)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dy_means_s, color="#e89f00")
            ax.set_title("frame mean dy error (signed, p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_dy_trend_signed_p1p99.png", dpi=200)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, args.fig_height))
        ax.plot(x_idx, dz_means, color="#009e73")
        ax.set_title("frame mean |dz| error")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, dz_means)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        fig.tight_layout()
        fig.savefig(out_dir / "frame_dz_trend.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dz_means)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dz_means, color="#009e73")
            ax.set_title("frame mean |dz| error (p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_dz_trend_p1p99.png", dpi=200)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x_idx, dz_means_s, color="#009e73")
        ax.set_title("frame mean dz error (signed)")
        ax.set_xlabel("frame_index")
        ax.set_ylabel("meters")
        set_ylim_tight(ax, dz_means_s)
        if frames:
            step = max(1, len(frames) // 10)
            ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
        fig.tight_layout()
        fig.savefig(out_dir / "frame_dz_trend_signed.png", dpi=200)
        plt.close(fig)

        if args.clip_p99:
            lim = clip_limits(dz_means_s)
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dz_means_s, color="#009e73")
            ax.set_title("frame mean dz error (signed, p1-p99)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            if lim:
                ax.set_ylim(*lim)
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            fig.tight_layout()
            fig.savefig(out_dir / "frame_dz_trend_signed_p1p99.png", dpi=200)
            plt.close(fig)

        # Per-keypoint frame trends.
        for kp_idx in sorted(kp_frame_axis_abs.keys()):
            frames = sorted(kp_frame_axis_abs[kp_idx].keys())
            x_idx = list(range(len(frames)))
            dx_means = [mean_axis(kp_frame_axis_abs[kp_idx][fid])[0] for fid in frames]
            dy_means = [mean_axis(kp_frame_axis_abs[kp_idx][fid])[1] for fid in frames]
            dz_means = [mean_axis(kp_frame_axis_abs[kp_idx][fid])[2] for fid in frames]
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dx_means, label="dx", color="#2d6cdf")
            ax.plot(x_idx, dy_means, label="dy", color="#e89f00")
            ax.plot(x_idx, dz_means, label="dz", color="#009e73")
            ax.set_title(f"kp{kp_idx:02d} frame mean |axis| error")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            ax.legend()
            set_ylim_tight(ax, dx_means + dy_means + dz_means)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            fig.tight_layout()
            fig.savefig(out_dir / f"frame_kp{kp_idx:02d}_axis_trend.png", dpi=200)
            plt.close(fig)

            dx_means_s = [mean_axis(kp_frame_axis_signed[kp_idx][fid])[0] for fid in frames]
            dy_means_s = [mean_axis(kp_frame_axis_signed[kp_idx][fid])[1] for fid in frames]
            dz_means_s = [mean_axis(kp_frame_axis_signed[kp_idx][fid])[2] for fid in frames]
            fig, ax = plt.subplots(figsize=(8, args.fig_height))
            ax.plot(x_idx, dx_means_s, label="dx", color="#2d6cdf")
            ax.plot(x_idx, dy_means_s, label="dy", color="#e89f00")
            ax.plot(x_idx, dz_means_s, label="dz", color="#009e73")
            ax.set_title(f"kp{kp_idx:02d} frame mean axis error (signed)")
            ax.set_xlabel("frame_index")
            ax.set_ylabel("meters")
            ax.legend()
            set_ylim_tight(ax, dx_means_s + dy_means_s + dz_means_s)
            if frames:
                step = max(1, len(frames) // 10)
                ax.set_xticks(x_idx[::step], [str(f) for f in frames[::step]])
            set_y_ticks_precision(ax, decimals=2, step=args.y_tick_step)
            fig.tight_layout()
            fig.savefig(out_dir / f"frame_kp{kp_idx:02d}_axis_trend_signed.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
