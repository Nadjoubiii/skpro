"""Distribution fitting logic using scipy.stats."""

import warnings
import numpy as np
import scipy.stats as st

# Candidate continuous distributions to try
CANDIDATE_DISTRIBUTIONS = [
    ("Normal",          st.norm),
    ("Log-Normal",      st.lognorm),
    ("Gamma",           st.gamma),
    ("Beta",            st.beta),
    ("Exponential",     st.expon),
    ("Weibull",         st.weibull_min),
    ("Pareto",          st.pareto),
    ("Cauchy",          st.cauchy),
    ("Laplace",         st.laplace),
    ("Logistic",        st.logistic),
    ("Student-t",       st.t),
    ("Chi-Squared",     st.chi2),
    ("Inverse Gamma",   st.invgamma),
    ("Half-Normal",     st.halfnorm),
    ("Uniform",         st.uniform),
]

# Map scipy name -> skpro class name
SKPRO_CLASS_MAP = {
    "norm":       "Normal",
    "lognorm":    "LogNormal",
    "gamma":      "Gamma",
    "beta":       "Beta",
    "expon":      "Exponential",
    "weibull_min": "Weibull",
    "pareto":     "Pareto",
    "cauchy":     "Cauchy",
    "laplace":    "Laplace",
    "logistic":   "Logistic",
    "t":          "TDistribution",
    "chi2":       "ChiSquared",
    "invgamma":   "InverseGamma",
    "halfnorm":   "HalfNormal",
    "uniform":    "Uniform",
}


def get_support_type(data: np.ndarray) -> str:
    """Infer support type from data."""
    if data.min() >= 0 and data.max() <= 1:
        return "bounded [0,1]"
    elif data.min() >= 0:
        return "non-negative [0, ∞)"
    else:
        return "real-valued (-∞, ∞)"


def compute_summary(data: np.ndarray) -> dict:
    """Compute descriptive statistics."""
    return {
        "n": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "skewness": float(st.skew(data)),
        "kurtosis": float(st.kurtosis(data)),
        "support": get_support_type(data),
    }


def fit_distributions(data: np.ndarray) -> list[dict]:
    """
    Fit all candidate distributions and rank by AIC.

    Returns list of dicts sorted by AIC (ascending).
    """
    results = []
    support = get_support_type(data)

    for name, dist in CANDIDATE_DISTRIBUTIONS:
        # Skip distributions incompatible with data support
        if support == "real-valued (-∞, ∞)" and name in (
            "Log-Normal", "Gamma", "Exponential", "Weibull",
            "Pareto", "Chi-Squared", "Inverse Gamma", "Half-Normal",
        ):
            continue
        if support == "bounded [0,1]" and name not in ("Beta", "Uniform"):
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(data)
                log_ll = np.sum(dist.logpdf(data, *params))

                if not np.isfinite(log_ll):
                    continue

                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * log_ll
                bic = k * np.log(n) - 2 * log_ll

                ks_stat, ks_p = st.kstest(data, dist.name, params)

                results.append({
                    "distribution": name,
                    "scipy_name": dist.name,
                    "params": params,
                    "log_likelihood": round(log_ll, 4),
                    "aic": round(aic, 4),
                    "bic": round(bic, 4),
                    "ks_stat": round(ks_stat, 4),
                    "ks_p_value": round(ks_p, 4),
                    "good_fit": ks_p > 0.05,
                })
        except Exception:
            continue

    results.sort(key=lambda x: x["aic"])
    return results


def generate_skpro_snippet(best: dict, col_name: str) -> str:
    """Generate a skpro code snippet for the best-fit distribution."""
    skpro_name = SKPRO_CLASS_MAP.get(best["scipy_name"], best["distribution"])
    params = best["params"]

    # Format params as keyword args (loc/scale/shape)
    dist_obj = next(
        (d for name, d in CANDIDATE_DISTRIBUTIONS if d.name == best["scipy_name"]),
        None,
    )
    if dist_obj is None:
        param_str = ", ".join(f"{p:.4f}" for p in params)
        param_snippet = f"# params: {param_str}"
    else:
        shapes = (dist_obj.shapes.split(", ") if dist_obj.shapes else [])
        all_keys = shapes + ["loc", "scale"]
        param_kv = {k: round(v, 4) for k, v in zip(all_keys, params)}
        param_snippet = ", ".join(f"{k}={v}" for k, v in param_kv.items())

    return f"""\
import pandas as pd
from skpro.distributions import {skpro_name}

# Best-fit distribution for column: '{col_name}'
dist = {skpro_name}({param_snippet})

# Use in a probabilistic regressor, e.g.:
# from skpro.regression import ProbaGradientBoosting
# regressor = ProbaGradientBoosting()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict_proba(X_test)  # returns distribution

# Evaluate probability of a value:
# prob = dist.cdf(1.5)
# log_pdf = dist.log_pdf(1.5)
"""
