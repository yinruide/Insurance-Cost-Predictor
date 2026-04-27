"""Optional XGBoost model entrypoint."""


def fit_xgboost():
    """Fail clearly when the optional xgboost dependency is unavailable."""
    try:
        import xgboost  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is not installed in this environment. "
            "Install the package before running the XGBoost experiment."
        ) from exc

    raise NotImplementedError(
        "XGBoost is optional in this repo and has not been wired into the presentation app."
    )


if __name__ == "__main__":
    fit_xgboost()
