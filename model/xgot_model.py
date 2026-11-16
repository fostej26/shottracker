"""Placement-based xGOT model: learning skeleton.

Data handling is DONE. Everything marked `TODO(you)` is for you to implement,
roughly in the order below. Each TODO has hints but no solution. Run
`python xgot_model.py` at any point: it executes the steps in order and stops
at the first TODO that is still unimplemented, so you always know where you are.

Assumption (simplification, see xgot_plan.md): the keeper is 6'2" (1.88 m) and
always stands in the centre of the goal mouth (y_m = 0, on the goal line).
That means keeper position is a constant, so it is not a feature itself, but
it lets us build features that describe how hard a placement is for THAT keeper
(distance from the keeper, whether it is within his reach, etc).

Learning path:
    Step 1  load + inspect data            (done)
    Step 2  engineer features              (done)
    Step 3  baselines                      (TODO)
    Step 4  fit a model                    (TODO)
    Step 5  validate with grouped CV       (splitter done, loop is TODO)
    Step 6  calibration                    (TODO)
    Step 7  predict function for tracker   (TODO)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pull_statsbomb import CROSSBAR_M, HALF_GOAL_M

HERE = Path(__file__).parent
DATA_PATH = HERE / "data/statsbomb/shots_modeling.csv"
MODEL_PATH = HERE / "xgot_model.joblib"

# --- Keeper assumption ------------------------------------------------------
KEEPER_HEIGHT_M = 1.88            # 6'2"
KEEPER_Y_M = 0.0                  # always centre of the goal mouth
# Rough standing reach with arms overhead. Assumption: ~1.2x height is a
# common rule of thumb; treat it as a tunable constant, not a fact.
KEEPER_STANDING_REACH_M = KEEPER_HEIGHT_M * 1.2

RANDOM_SEED = 42


# ===========================================================================
# STEP 1: data handling (done)
# ===========================================================================
def load_data(path=DATA_PATH):
    """Load the modeling table and drop rows the model cannot use.

    Keeps shots that are in-frame only if `in_frame_only` logic is applied later;
    here we only guarantee that the label and placement columns are present.
    """
    df = pd.read_csv(path)
    needed = ["y_m", "z_m", "is_goal", "match_id"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    df["is_goal"] = df["is_goal"].astype(int)
    return df


def inspect(df):
    """Print the things you should always look at before modeling."""
    print(f"rows: {len(df)}  matches: {df['match_id'].nunique()}")
    print(f"goal rate: {df['is_goal'].mean():.3f}")
    print(f"outside goal frame: {(~df['end_in_frame']).sum()}")
    print(df[["y_m", "z_m"]].describe().round(2).to_string())
    print("\ngoal rate by competition:")
    print(df.groupby("competition_name")["is_goal"].agg(["mean", "size"]).round(3).to_string())


def split_xy(df, feature_cols):
    """Return X (features), y (label), groups (match ids for grouped CV)."""
    return df[feature_cols].to_numpy(float), df["is_goal"].to_numpy(int), df["match_id"].to_numpy()


def grouped_folds(groups, n_splits=5):
    """Yield (train_idx, test_idx) so all shots from one match stay together.

    Why: shots from the same match share conditions (keeper, weather, pitch).
    A random split would leak that into the test set and flatter your score.
    """
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    yield from gkf.split(dummy, dummy, groups)


# ===========================================================================
# STEP 2: feature engineering (done)
# ===========================================================================
# build_features() in pull_statsbomb.py already gave you: y_m, z_m, abs_y_m,
# dist_post_m, dist_crossbar_m. Below are the ones to add. Every feature must
# be computable from (y_m, z_m) alone, because that is all the tracker gives us.
#
# Look at each feature with a plot (goal rate vs the feature) BEFORE using it.

def feat_dist_from_keeper(df):
    """TODO(you): straight-line distance (m) from the keeper's body centre to the
    shot's goal-plane point.

    Keeper stands at (KEEPER_Y_M, 0) on the plane; his body centre is roughly at
    height KEEPER_HEIGHT_M / 2. Use Pythagoras on (y_m - KEEPER_Y_M) and
    (z_m - KEEPER_HEIGHT_M / 2).
    Why it should matter: farther from the keeper = harder to save.
    """

    dist = np.sqrt((df["y_m"] - KEEPER_Y_M) ** 2 + (df["z_m"] - KEEPER_HEIGHT_M / 2) ** 2)
    return dist
    

def feat_horizontal_dive_distance(df):
    dive = np.abs(df["y_m"] - KEEPER_Y_M)
    return dive


def feat_within_standing_reach(df):
    circle_center = KEEPER_HEIGHT_M * 0.8
    dist = np.sqrt((df["y_m"] - KEEPER_Y_M) ** 2 + (df["z_m"] - circle_center) ** 2)
    return (dist <= KEEPER_STANDING_REACH_M).astype(int)


def feat_height_zone(df):
    height = df["z_m"]
    if height < 0.5:
        height = 0
    elif height < 1.6:
        height = 1
    else:
        height = 2
    return height


def feat_corner_proximity(df):
    """TODO(you): distance to the NEAREST top corner (post/crossbar junction)
    at (+-HALF_GOAL_M, CROSSBAR_M). The plan says corners should be rated above
    dead centre, so check the data agrees.
    """
    dist_nw = np.sqrt((df["y_m"] + HALF_GOAL_M) ** 2 + (df["z_m"] - CROSSBAR_M) ** 2)
    dist_ne = np.sqrt((df["y_m"] - HALF_GOAL_M) ** 2 + (df["z_m"] - CROSSBAR_M) ** 2)
    return np.minimum(dist_nw, dist_ne)


def feat_polynomial_terms(df):
    """squared and interaction terms: abs_y_m**2, z_m**2,
    abs_y_m * z_m. Goal rate is not linear in position, so a plain logistic
    regression needs these to bend. (Or use sklearn PolynomialFeatures.)
    """
    df["abs_y_m_squared"] = df["abs_y_m"] ** 2
    df["z_m_squared"] = df["z_m"] ** 2
    df["abs_y_m_z_m"] = df["abs_y_m"] * df["z_m"]
    return df


def add_features(df):
    """(you): call the feat_* functions above and attach their results as
    columns. Return the new frame and the list of feature column names you
    want to feed the model.

    Start small: [abs_y_m, z_m] only. Add one feature at a time and check
    whether validation log loss improves (step 5).
    """
    df = df.copy()
    df["dist_from_keeper"] = feat_dist_from_keeper(df)
    df["horizontal_dive_distance"] = feat_horizontal_dive_distance(df)
    df["within_standing_reach"] = feat_within_standing_reach(df)
    df["height_zone"] = feat_height_zone(df)
    df["corner_proximity"] = feat_corner_proximity(df)
    df = feat_polynomial_terms(df)
    feature_cols = [
        "abs_y_m",
        "z_m",
        "dist_from_keeper",
        "horizontal_dive_distance",
        "within_standing_reach",
        "height_zone",
        "corner_proximity",
        "abs_y_m_squared",
        "z_m_squared",
        "abs_y_m_z_m",
    ]
    return df, feature_cols


# ===========================================================================
# STEP 3: baselines (TODO)
# ===========================================================================
def baseline_constant(y_train, n_test):
    """TODO(you): predict the training goal rate for every test shot.
    Any model that cannot beat this has learned nothing.
    """

    return np.full(n_test, y_train.mean())


def baseline_grid(train_df, test_df, n_y=3, n_z=3):
    """TODO(you): cut the goal face into an n_y x n_z grid (use pd.cut on
    abs_y_m or y_m, and z_m), compute the goal rate per cell on train, and look
    up each test shot's cell. Handle empty cells (fall back to overall rate).
    """
    
    # Create bins for y_m and z_m
    y_bins = pd.cut(train_df["y_m"], bins=n_y)
    z_bins = pd.cut(train_df["z_m"], bins=n_z)

    # Create a DataFrame with the goal rates for each cell
    grid = train_df.groupby([y_bins, z_bins])["is_goal"].mean().reset_index()
    grid.columns = ["y_bin", "z_bin", "goal_rate"]

    # Merge the test DataFrame with the grid to get the goal rates for each test shot
    test_df = test_df.copy()
    test_df["y_bin"] = pd.cut(test_df["y_m"], bins=n_y)
    test_df["z_bin"] = pd.cut(test_df["z_m"], bins=n_z)
    test_df = test_df.merge(grid, on=["y_bin", "z_bin"], how="left")

    # Fill NaN values (empty cells) with the overall goal rate from the training set
    overall_goal_rate = train_df["is_goal"].mean()
    test_df["goal_rate"].fillna(overall_goal_rate, inplace=True)

    return test_df["goal_rate"].to_numpy()


# ===========================================================================
# STEP 4: models (TODO)
# ===========================================================================
def make_model():
    """TODO(you): return an unfitted sklearn estimator.

    Order to try:
      1. LogisticRegression (wrap with a StandardScaler in a Pipeline)
      2. add polynomial features
      3. GradientBoostingClassifier(max_depth=2 or 3) -- only if it clearly wins

    With only a few hundred rows, expect the simple model to win. Overfitting is
    the main enemy: strong regularisation (small C) is your friend.
    """

    def logistic_regression_model():
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_SEED, C=1.0, max_iter=1000)
        )

    def gradient_boosting_model():
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(max_depth=3, random_state=RANDOM_SEED)

    def polynomial_features_model():
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LogisticRegression

        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_SEED, C=1.0, max_iter=1000)
        )
    
    return logistic_regression_model()  # Change this to try different models


# ===========================================================================
# STEP 5: validation (TODO)
# ===========================================================================
def cross_validate(df, feature_cols, model_factory):
    """TODO(you): for each (train_idx, test_idx) from grouped_folds(),
    fit a fresh model on train, predict probabilities on test
    (model.predict_proba(X)[:, 1]), and collect out-of-fold predictions.
    Return an array the same length as df.

    Then compute with sklearn.metrics: log_loss, brier_score_loss, roc_auc_score.
    Do the same for each baseline and compare. Metrics to look at:
      - log loss / Brier: lower is better (primary)
      - AUC: ranking quality (secondary)
    Warning: with ~300 shots the differences are noisy. Repeat with different
    GroupKFold shuffles or bootstrap the gap before believing a win.
    """
    raise NotImplementedError


# ===========================================================================
# STEP 6: calibration (TODO)
# ===========================================================================
def calibration_plot(y_true, y_prob, n_bins=5):
    """TODO(you): reliability diagram. Use sklearn.calibration.calibration_curve
    with few bins (small data!). Plot predicted vs observed rate plus the
    diagonal. A model can have good AUC and still be badly calibrated; since we
    want real probabilities, calibration matters.
    """
    raise NotImplementedError


# ===========================================================================
# STEP 7: tracker-facing function (TODO)
# ===========================================================================
def xgot(y_m, z_m):
    """TODO(you): load the saved model (joblib) and return P(goal) for one shot
    crossing the goal plane at (y_m, z_m) metres from the tracker.

    Requirements from xgot_plan.md:
      - build the same features you trained with (reuse add_features!)
      - return 0.0 if the point is outside the frame (a miss)
      - output must be within [0, 1]
      - symmetric: xgot(-y, z) == xgot(y, z)
    Also save the fitted model at the end of training with joblib.dump.
    """
    raise NotImplementedError


# ===========================================================================
def main():
    df = load_data()
    inspect(df)

    df, feature_cols = add_features(df)          # Step 2
    X, y, groups = split_xy(df, feature_cols)
    print(f"\nfeatures: {feature_cols}  X shape: {X.shape}")

    oof = cross_validate(df, feature_cols, make_model)   # Steps 4-5
    print("out-of-fold predictions:", oof[:5])
    # TODO(you): print log loss / Brier / AUC for model vs baselines,
    # then call calibration_plot(y, oof).


if __name__ == "__main__":
    main()
