"""Feature-importance helpers used by the predictor page."""

import pandas as pd


def as_ranked_frame(feature_names, importance_values):
    """Return a descending feature-importance DataFrame."""
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def top_features(feature_names, importance_values, k=5):
    """Return the top-k features as a small DataFrame."""
    return as_ranked_frame(feature_names, importance_values).head(k)
