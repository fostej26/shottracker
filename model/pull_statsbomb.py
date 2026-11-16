"""Pull StatsBomb open-data shots and build the placement-based xGOT modeling table.

Usage:
    python pull_statsbomb.py --max-matches 5          # quick smoke test
    python pull_statsbomb.py                          # everything (large, slow, cached)
    python pull_statsbomb.py --competition-ids 43 55  # World Cup + Euro, say

Outputs (under --out, default data/statsbomb):
    shots_all.csv        every shot, with flags
    shots_modeling.csv   on-target, non-penalty shots with goal-plane features
    shots_cache/         one small JSON per match, so re-runs skip downloads

Only the standard library and pandas are needed. Network is required to run it.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# ---------------------------------------------------------------------------
# Decisions. Fixed before looking at results; change them deliberately.
# ---------------------------------------------------------------------------
# On-target outcomes used for the modeling set. "Post" (hit the post, no goal)
# is kept in shots_all.csv but is NOT in the modeling set. Revisit on purpose
# if you want it, and write the reason in your notes.
ON_TARGET = {"Goal", "Saved", "Saved to Post"}
EXCLUDE_SHOT_TYPES = {"Penalty"}
EXCLUDE_PERIODS = {5}  # penalty shootout

# StatsBomb pitch is a 120 x 80 grid. Posts at y=36 and y=44, crossbar at
# z=2.67 grid units. The goal is 7.32 m x 2.44 m, so one unit is 0.915 m.
GOAL_Y_MIN, GOAL_Y_MAX, GOAL_Z_MAX = 36.0, 44.0, 2.67
GOAL_Y_CENTER = 40.0
M_PER_UNIT = 7.32 / 8.0
HALF_GOAL_M = 3.66
CROSSBAR_M = 2.44


def fetch_json(url, retries=4):
    """GET a JSON file with retries. Returns None on a 404."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "shottracker-xgot"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    return None


def _name(d, key):
    """d[key]['name'] if present, else None."""
    sub = d.get(key)
    return sub.get("name") if isinstance(sub, dict) else None


def extract_shots(events, meta):
    """Turn one match's raw events into a list of flat shot rows."""
    rows = []
    for ev in events:
        if _name(ev, "type") != "Shot":
            continue
        s = ev.get("shot") or {}
        end = s.get("end_location") or []
        loc = ev.get("location") or []
        keeper = None
        for p in s.get("freeze_frame") or []:
            if p.get("keeper") and not p.get("teammate"):
                keeper = p.get("location")
                break
        rows.append({
            **meta,
            "event_id": ev.get("id"),
            "period": ev.get("period"),
            "minute": ev.get("minute"),
            "team": _name(ev, "team"),
            "player": _name(ev, "player"),
            "shot_type": _name(s, "type"),
            "outcome": _name(s, "outcome"),
            "body_part": _name(s, "body_part"),
            "technique": _name(s, "technique"),
            "first_time": bool(s.get("first_time", False)),
            "x": loc[0] if len(loc) > 0 else None,
            "y": loc[1] if len(loc) > 1 else None,
            "end_x": end[0] if len(end) > 0 else None,
            "end_y": end[1] if len(end) > 1 else None,
            "end_z": end[2] if len(end) > 2 else None,
            "keeper_x": keeper[0] if keeper else None,
            "keeper_y": keeper[1] if keeper else None,
            "sb_xg": s.get("statsbomb_xg"),
        })
    return rows


def build_features(df):
    """Add flags and goal-plane features (meters) to the raw shot table."""
    df = df.copy()
    df["is_goal"] = df["outcome"].eq("Goal")
    df["is_on_target"] = df["outcome"].isin(ON_TARGET)
    df["is_penalty"] = df["shot_type"].isin(EXCLUDE_SHOT_TYPES)
    df["is_shootout"] = df["period"].isin(EXCLUDE_PERIODS)
    df["is_free_kick"] = df["shot_type"].eq("Free Kick")
    df["is_header"] = df["body_part"].eq("Head")

    # Goal-plane position in meters: y_m is horizontal offset from the goal
    # centre, z_m is height above the ground.
    df["y_m"] = (df["end_y"] - GOAL_Y_CENTER) * M_PER_UNIT
    df["z_m"] = df["end_z"] * M_PER_UNIT
    df["abs_y_m"] = df["y_m"].abs()
    df["dist_post_m"] = HALF_GOAL_M - df["abs_y_m"]
    df["dist_crossbar_m"] = CROSSBAR_M - df["z_m"]
    df["end_in_frame"] = (
        df["end_y"].between(GOAL_Y_MIN, GOAL_Y_MAX)
        & df["end_z"].between(0.0, GOAL_Z_MAX)
    )
    return df


def modeling_set(df):
    """On-target, non-penalty, non-shootout shots that have a goal-plane position."""
    keep = (
        df["is_on_target"]
        & ~df["is_penalty"]
        & ~df["is_shootout"]
        & df["y_m"].notna()
        & df["z_m"].notna()
    )
    return df[keep].copy()


def collect(out_dir, competition_ids, max_matches, sleep):
    cache = out_dir / "shots_cache"
    cache.mkdir(parents=True, exist_ok=True)

    competitions = fetch_json(f"{BASE}/competitions.json") or []
    rows, n_matches = [], 0
    for comp in competitions:
        if competition_ids and comp["competition_id"] not in competition_ids:
            continue
        matches = fetch_json(f"{BASE}/matches/{comp['competition_id']}/{comp['season_id']}.json")
        if matches is None:
            print(f"  skipped (no match list): {comp['competition_name']} {comp['season_name']}")
            continue
        for m in matches:
            if max_matches and n_matches >= max_matches:
                return rows
            mid = m["match_id"]
            meta = {
                "competition_id": comp["competition_id"],
                "season_id": comp["season_id"],
                "competition_name": comp["competition_name"],
                "season_name": comp["season_name"],
                "competition_gender": comp.get("competition_gender"),
                "match_id": mid,
                "match_date": m.get("match_date"),
            }
            cache_file = cache / f"{mid}.json"
            if cache_file.exists():
                shots = json.loads(cache_file.read_text())
            else:
                events = fetch_json(f"{BASE}/events/{mid}.json")
                if events is None:
                    print(f"  skipped (no events): match {mid}")
                    continue
                shots = extract_shots(events, meta)
                cache_file.write_text(json.dumps(shots))
                if sleep:
                    time.sleep(sleep)
            rows.extend(shots)
            n_matches += 1
            if n_matches % 25 == 0:
                print(f"  {n_matches} matches, {len(rows)} shots so far")
    return rows


def summarize(df, model):
    n_goal = int(model["is_goal"].sum())
    print("\n=== Summary ===")
    print(f"All shots:               {len(df)}")
    print(f"  penalties:             {int(df['is_penalty'].sum())}")
    print(f"  shootout:              {int(df['is_shootout'].sum())}")
    on = df[df["is_on_target"] & ~df["is_penalty"] & ~df["is_shootout"]]
    print(f"On target (no pens):     {len(on)}")
    print(f"  missing end y/z:       {int((on['y_m'].isna() | on['z_m'].isna()).sum())}")
    print(f"  outside goal frame:    {int((~model['end_in_frame']).sum())} (kept; flagged in end_in_frame)")
    print(f"Modeling set:            {len(model)}")
    print(f"  goals:                 {n_goal} ({n_goal / max(len(model), 1):.1%})")
    print("\nModeling shots by competition/season:")
    by = (model.groupby(["competition_name", "season_name"]).size()
          .sort_values(ascending=False).head(15))
    print(by.to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/statsbomb")
    ap.add_argument("--competition-ids", type=int, nargs="*", default=None)
    ap.add_argument("--max-matches", type=int, default=0, help="stop after N matches (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between downloads")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(out_dir, set(args.competition_ids or []), args.max_matches, args.sleep)
    if not rows:
        print("No shots collected.", file=sys.stderr)
        return 1

    df = build_features(pd.DataFrame(rows))
    model = modeling_set(df)
    df.to_csv(out_dir / "shots_all.csv", index=False)
    model.to_csv(out_dir / "shots_modeling.csv", index=False)
    summarize(df, model)
    print(f"\nWrote {out_dir / 'shots_all.csv'} and {out_dir / 'shots_modeling.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())