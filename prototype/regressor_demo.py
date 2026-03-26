"""Probabilistic regression demo using skpro wrappers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split as sk_split


def _ensure_local_skpro_on_path():
    """Make local repo package importable when app runs from prototype/."""
    repo_root = Path(__file__).resolve().parent.parent
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

BASE_ESTIMATOR_OPTIONS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(),
    "Random Forest (50 trees)": lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting (50 trees)": lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
}

REGRESSOR_DESCRIPTIONS = {
    "Bootstrap": (
        "Trains N clones of the base model on bootstrap (resampled) datasets. "
        "Uncertainty comes from the spread of predictions across all N models — "
        "no distributional assumption needed."
    ),
    "Residual Double": (
        "Uses two models: one predicts the mean, a second predicts the residual "
        "magnitude. Combines them into a Normal(mean, residual) distribution. "
        "Simple, fast, and interpretable."
    ),
}

# Empirical base-fit time constants (seconds per 1000 rows per feature)
_BASE_SPF = {
    "Linear Regression":          0.00005,
    "Ridge":                       0.00005,
    "Random Forest (50 trees)":    0.002,
    "Gradient Boosting (50 trees)": 0.008,
}

# Wrapper multipliers (relative to a single fit)
_WRAPPER_MULT = {
    "Bootstrap":      None,   # depends on n_bootstrap; handled below
    "Residual Double": 2.2,
}


def estimate_complexity(
    n_rows: int,
    n_features: int,
    regressor_name: str,
    base_est_name: str,
    n_bootstrap: int = 50,
    test_size: float = 0.2,
) -> dict:
    """
    Return a human-readable complexity estimate before fitting.

    Returns a dict with keys:
      tier      : "Low" | "Medium" | "High" | "Very High"
      color     : streamlit-compatible colour string
      fits      : number of model fits that will happen
      est_sec   : estimated wall-clock seconds (float)
      est_label : formatted time string e.g. "< 1 s" / "~3 s" / "~45 s"
      detail    : one-line explanation
    """
    n_train = int(n_rows * (1 - test_size))
    spf = _BASE_SPF[base_est_name]          # seconds / (1000 rows × feature)
    base_time = spf * (n_train / 1000) * n_features

    if regressor_name == "Bootstrap":
        fits = n_bootstrap
        mult = n_bootstrap
        detail = (
            f"{n_bootstrap} bootstrap fits × {base_est_name} on "
            f"{n_train:,} rows × {n_features} features"
        )
    else:
        fits = 2
        mult = _WRAPPER_MULT[regressor_name]
        detail = (
            f"2 fits (mean + residual) × {base_est_name} on "
            f"{n_train:,} rows × {n_features} features"
        )

    est_sec = base_time * mult

    if est_sec < 2:
        tier, color, est_label = "Low", "green", "< 2 s"
    elif est_sec < 15:
        tier, color, est_label = "Medium", "orange", f"~{int(est_sec)} s"
    elif est_sec < 60:
        tier, color, est_label = "High", "red", f"~{int(est_sec)} s"
    else:
        mins = est_sec / 60
        tier, color, est_label = "Very High", "red", f"~{mins:.1f} min"

    return {
        "tier": tier,
        "color": color,
        "fits": fits,
        "est_sec": est_sec,
        "est_label": est_label,
        "detail": detail,
    }


def build_regressor(regressor_name: str, base_est_name: str, n_bootstrap: int = 50):
    base_est = BASE_ESTIMATOR_OPTIONS[base_est_name]()
    _ensure_local_skpro_on_path()

    if regressor_name == "Bootstrap":
        try:
            from skpro.regression.bootstrap import BootstrapRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import skpro. Run the app from the repository root "
                "or install local package with `pip install -e .`"
            ) from exc
        return BootstrapRegressor(base_est, n_bootstrap_samples=n_bootstrap, random_state=42)

    elif regressor_name == "Residual Double":
        try:
            from skpro.regression.residual import ResidualDouble
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import skpro. Run the app from the repository root "
                "or install local package with `pip install -e .`"
            ) from exc
        base_est2 = BASE_ESTIMATOR_OPTIONS[base_est_name]()
        return ResidualDouble(base_est, base_est2)

    raise ValueError(f"Unknown regressor: {regressor_name}")


def fit_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    regressor_name: str,
    base_est_name: str,
    test_size: float = 0.2,
    n_bootstrap: int = 50,
    coverage: float = 0.9,
) -> dict:
    """Fit a probabilistic regressor and return predictions + metrics."""
    X_train, X_test, y_train, y_test = sk_split(
        X, y, test_size=test_size, random_state=42
    )

    reg = build_regressor(regressor_name, base_est_name, n_bootstrap)
    reg.fit(X_train, y_train)

    y_pred_mean = reg.predict(X_test)
    y_pred_interval = reg.predict_interval(X_test, coverage=coverage)

    # iloc[:, 0] = lower, iloc[:, 1] = upper regardless of MultiIndex column names
    lower = y_pred_interval.iloc[:, 0].to_numpy(dtype=float)
    upper = y_pred_interval.iloc[:, 1].to_numpy(dtype=float)

    actual = y_test.to_numpy(dtype=float)
    mean_pred = (
        y_pred_mean.to_numpy().flatten()
        if hasattr(y_pred_mean, "to_numpy")
        else np.asarray(y_pred_mean, dtype=float).flatten()
    )

    rmse = float(np.sqrt(np.mean((actual - mean_pred) ** 2)))
    empirical_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    mean_interval_width = float(np.mean(upper - lower))

    # Try full CRPS (requires skpro + scipy installed together)
    crps_val = None
    try:
        y_pred_proba = reg.predict_proba(X_test)
        from skpro.metrics import CRPS
        crps_val = round(float(CRPS()(y_test, y_pred_proba)), 4)
    except Exception:
        pass

    # Sort by actual value for an intuitive coverage plot
    sort_idx = np.argsort(actual)
    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "actual": actual[sort_idx],
        "mean_pred": mean_pred[sort_idx],
        "lower": lower[sort_idx],
        "upper": upper[sort_idx],
        "rmse": round(rmse, 4),
        "empirical_coverage": round(empirical_coverage, 4),
        "mean_interval_width": round(mean_interval_width, 4),
        "crps": crps_val,
        "code": _generate_code(regressor_name, base_est_name, n_bootstrap, coverage),
    }


def _generate_code(
    regressor_name: str,
    base_est_name: str,
    n_bootstrap: int,
    coverage: float,
) -> str:
    base_snippets = {
        "Linear Regression": (
            "from sklearn.linear_model import LinearRegression\n"
            "base_est = LinearRegression()"
        ),
        "Ridge": (
            "from sklearn.linear_model import Ridge\n"
            "base_est = Ridge()"
        ),
        "Random Forest (50 trees)": (
            "from sklearn.ensemble import RandomForestRegressor\n"
            "base_est = RandomForestRegressor(n_estimators=50, random_state=42)"
        ),
        "Gradient Boosting (50 trees)": (
            "from sklearn.ensemble import GradientBoostingRegressor\n"
            "base_est = GradientBoostingRegressor(n_estimators=50, random_state=42)"
        ),
    }

    reg_snippets = {
        "Bootstrap": (
            "from skpro.regression.bootstrap import BootstrapRegressor\n"
            f"reg = BootstrapRegressor(base_est, n_bootstrap_samples={n_bootstrap})"
        ),
        "Residual Double": (
            "from skpro.regression.residual import ResidualDouble\n"
            f"reg = ResidualDouble(base_est, base_est)"
        ),
    }

    return (
        f"{base_snippets[base_est_name]}\n"
        f"{reg_snippets[regressor_name]}\n\n"
        "# Fit\n"
        "reg.fit(X_train, y_train)\n\n"
        "# Predictions\n"
        "y_pred_mean     = reg.predict(X_test)                           # point prediction\n"
        f"y_pred_interval = reg.predict_interval(X_test, coverage={coverage})  # prediction interval\n"
        "y_pred_proba    = reg.predict_proba(X_test)                    # full distribution\n\n"
        "# Evaluate\n"
        "from skpro.metrics import CRPS, PinballLoss\n"
        "crps = CRPS()(y_test, y_pred_proba)\n"
        'print(f"CRPS: {crps:.4f}")\n'
    )
