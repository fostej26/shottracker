"""Plot StatsBomb shots on a soccer goal, from the shooter's point of view.

Reads every per-match JSON in data/statsbomb/shots_cache/ and reuses the goal
dimensions and feature logic from pull_statsbomb.py (goal 7.32 m x 2.44 m).

Usage:
    python plot_shots_on_goal.py                 # modeling set (on target, no pens/shootout)
    python plot_shots_on_goal.py --all-shots     # every shot that has a goal-plane position
    python plot_shots_on_goal.py --out shots.png # save instead of showing
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from pull_statsbomb import CROSSBAR_M, HALF_GOAL_M, build_features, modeling_set


def load_shots(cache_dir):
    rows = []
    for f in sorted(Path(cache_dir).glob("*.json")):
        rows.extend(json.loads(f.read_text()))
    if not rows:
        raise SystemExit(f"No shots found in {cache_dir}")
    return build_features(pd.DataFrame(rows))


def draw_goal(ax):
    # Posts and crossbar (white, thick) on a pitch-green background with net hatching.
    ax.add_patch(Rectangle((-HALF_GOAL_M, 0), 2 * HALF_GOAL_M, CROSSBAR_M,
                           facecolor="#e8f0e8", edgecolor="none", hatch="++",
                           alpha=0.35, zorder=0))
    ax.plot([-HALF_GOAL_M, -HALF_GOAL_M, HALF_GOAL_M, HALF_GOAL_M],
            [0, CROSSBAR_M, CROSSBAR_M, 0], color="black", lw=5,
            solid_capstyle="projecting", zorder=2)
    ax.axhline(0, color="#2e7d32", lw=3, zorder=1)  # goal line / ground


# Zone boundaries for the 13 Kaggle "shot_place" bins. The dataset does not
# define these, so they are assumptions: change them freely.
THIRD = HALF_GOAL_M / 3          # goal split into 3 equal columns
MID_Z = CROSSBAR_M / 2           # goal split into 2 rows
BIT_TOO_HIGH_TOP = CROSSBAR_M + 0.5  # "bit too high" band above the bar; "too high" above it
BAR_BAND = 0.1                   # thickness of the "hits the bar" band (m)


def draw_zones(ax, x_lim, z_lim):
    """Draw the 13-bin placement grid. 'Blocked' has no location, so it is only noted."""
    line = dict(color="black", lw=1, ls="--", alpha=0.6, zorder=1)
    label = dict(ha="center", va="center", fontsize=8, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    h = HALF_GOAL_M

    # --- on-goal zones (6): top row = left/centre/right, bottom row = left/centre/right
    for x in (-THIRD, THIRD):
        ax.plot([x, x], [0, CROSSBAR_M], **line)
    ax.plot([-h, h], [MID_Z, MID_Z], **line)
    cols = {"left": -2 * THIRD, "centre": 0, "right": 2 * THIRD}
    for name, cx in cols.items():
        ax.text(cx, CROSSBAR_M * 0.75, f"Top {name}" if name != "centre" else "Top centre",
                **label)
        ax.text(cx, CROSSBAR_M * 0.25, f"Bottom {name}" if name != "centre" else "Centre of goal",
                **label)

    # --- outside zones (5) + bar (1)
    ax.plot([-h, h], [BIT_TOO_HIGH_TOP, BIT_TOO_HIGH_TOP], **line)
    ax.plot([-h, -h], [CROSSBAR_M, z_lim], **line)
    ax.plot([h, h], [CROSSBAR_M, z_lim], **line)
    ax.plot([-x_lim, -h], [CROSSBAR_M, CROSSBAR_M], **line)
    ax.plot([h, x_lim], [CROSSBAR_M, CROSSBAR_M], **line)
    ax.text(0, (CROSSBAR_M + BIT_TOO_HIGH_TOP) / 2, "Bit too high", **label)
    ax.text(0, (BIT_TOO_HIGH_TOP + z_lim) / 2, "Too high", **label)
    ax.text(-(h + x_lim) / 2, CROSSBAR_M / 2, "Misses\nto the left", **label)
    ax.text((h + x_lim) / 2, CROSSBAR_M / 2, "Misses\nto the right", **label)
    ax.text(-(h + x_lim) / 2, (CROSSBAR_M + z_lim) / 2, "High and\nwide", **label)
    ax.text((h + x_lim) / 2, (CROSSBAR_M + z_lim) / 2, "High and\nwide", **label)
    ax.add_patch(Rectangle((-h, CROSSBAR_M - BAR_BAND / 2), 2 * h, BAR_BAND,
                           facecolor="orange", alpha=0.4, zorder=1))
    ax.text(0, CROSSBAR_M - BAR_BAND - 0.05, "Hits the bar", ha="center", va="top",
            fontsize=7, alpha=0.8)
    ax.text(x_lim, -0.25, "(13th bin, 'Blocked', has no location)", ha="right",
            va="center", fontsize=7, alpha=0.6)


def plot(df, title, out=None, zones=False):
    fig, ax = plt.subplots(figsize=(11, 5) if not zones else (13, 7))
    ax.set_facecolor("#cfe8cf")
    draw_goal(ax)
    margin = 1.5 if zones else 1.0
    x_lim = HALF_GOAL_M + margin
    z_lim = CROSSBAR_M + (1.5 if zones else 1.0)
    if zones:
        draw_zones(ax, x_lim, z_lim)

    goals = df[df["is_goal"]]
    other = df[~df["is_goal"]]
    ax.scatter(other["y_m"], other["z_m"], s=60, c="#d62728", edgecolor="white",
               alpha=0.85, label=f"No goal ({len(other)})", zorder=3)
    ax.scatter(goals["y_m"], goals["z_m"], s=80, c="#1f77b4", edgecolor="white",
               marker="*", label=f"Goal ({len(goals)})", zorder=4)

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-0.4, z_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal position from goal centre (m)  [shooter's view: right is +]")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=here / "data/statsbomb/shots_cache")
    ap.add_argument("--all-shots", action="store_true",
                    help="include off-target shots and penalties (any shot with an end position)")
    ap.add_argument("--zones", action="store_true",
                    help="overlay the 13 placement bins (6 on-goal zones + miss regions)")
    ap.add_argument("--out", default=None, help="save to this image path instead of showing")
    args = ap.parse_args()

    df = load_shots(args.cache)
    if args.all_shots:
        df = df[df["y_m"].notna() & df["z_m"].notna()]
        title = f"All shots with goal-plane position (n={len(df)})"
    else:
        df = modeling_set(df)
        title = f"On-target shots, no penalties (n={len(df)})"
    plot(df, title, args.out, zones=args.zones)


if __name__ == "__main__":
    main()
