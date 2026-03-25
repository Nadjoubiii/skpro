"""
skpro Distribution Recommender
================================
Upload a CSV, pick a column, and get:
  - Descriptive statistics
  - Top distribution fits ranked by AIC
  - Interactive histogram + fitted PDF overlays
  - AI-powered plain-English explanation
  - Auto-generated skpro code snippet
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import streamlit as st_app  # alias to avoid clash with scipy.stats

from fitter import compute_summary, fit_distributions, generate_skpro_snippet
from ai_suggest import get_ai_suggestion


SCIPY_DISTS = {
    "norm": st.norm,
    "lognorm": st.lognorm,
    "gamma": st.gamma,
    "beta": st.beta,
    "expon": st.expon,
    "weibull_min": st.weibull_min,
    "pareto": st.pareto,
    "cauchy": st.cauchy,
    "laplace": st.laplace,
    "logistic": st.logistic,
    "t": st.t,
    "chi2": st.chi2,
    "invgamma": st.invgamma,
    "halfnorm": st.halfnorm,
    "uniform": st.uniform,
}

# ── Page config ──────────────────────────────────────────────────────────────
st_app.set_page_config(
    page_title="skpro Distribution Recommender",
    page_icon="📊",
    layout="wide",
)

st_app.title("📊 skpro Distribution Recommender")
st_app.caption(
    "Upload any CSV, select a numeric column, and get the best-fit probability "
    "distribution — with AI reasoning and a ready-to-use skpro code snippet."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st_app.sidebar:
    st_app.header("Settings")
    top_n = st_app.slider("Number of distributions to show", 3, 10, 5)
    show_ai = st_app.toggle("Enable AI explanation", value=True)
    st_app.markdown("---")
    st_app.markdown(
        "**API keys** (optional for AI):\n"
        "Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` as environment variables "
        "or in [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)."
    )
    st_app.markdown("---")
    st_app.markdown("Built with [skpro](https://github.com/sktime/skpro) · [scipy](https://scipy.org)")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st_app.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st_app.info("Upload a CSV file to get started.")
    st_app.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st_app.error(f"Could not read CSV: {e}")
    st_app.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st_app.error("No numeric columns found in the uploaded file.")
    st_app.stop()

col = st_app.selectbox("Select a numeric column to analyse", numeric_cols)
series = pd.to_numeric(df[col], errors="coerce")
data = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

if len(data) == 0:
    st_app.error("Selected column has no finite numeric values after cleaning.")
    st_app.stop()

rounded = np.rint(data)
is_integer_like = bool(np.all(np.isclose(data, rounded, atol=1e-9)))
unique_vals = np.unique(data)
n_unique = len(unique_vals)
unique_ratio = n_unique / max(len(data), 1)
is_discrete = is_integer_like and (n_unique <= 120 or unique_ratio <= 0.2)

if len(data) < 10:
    st_app.warning("Column has fewer than 10 non-null values — results may be unreliable.")

# ── Summary statistics ────────────────────────────────────────────────────────
summary = compute_summary(data)

st_app.subheader("Data Summary")
c1, c2, c3, c4, c5, c6 = st_app.columns(6)
c1.metric("N", summary["n"])
c2.metric("Mean", f"{summary['mean']:.4f}")
c3.metric("Std", f"{summary['std']:.4f}")
c4.metric("Skewness", f"{summary['skewness']:.4f}")
c5.metric("Kurtosis", f"{summary['kurtosis']:.4f}")
c6.metric("Data Type", "Discrete" if is_discrete else "Continuous")
st_app.caption(f"Support detected: {summary['support']}")

# ── Fit distributions ─────────────────────────────────────────────────────────
with st_app.spinner("Fitting distributions..."):
    results = fit_distributions(data)

if not results:
    st_app.error("No distributions could be fit to this data.")
    st_app.stop()

# ── Results table ─────────────────────────────────────────────────────────────
st_app.subheader("Fit Results (ranked by AIC)")

table_data = [
    {
        "Rank": i + 1,
        "Distribution": r["distribution"],
        "AIC": r["aic"],
        "BIC": r["bic"],
        "Log-Likelihood": r["log_likelihood"],
        "KS p-value": r["ks_p_value"],
        "Good Fit": "✅" if r["good_fit"] else "❌",
    }
    for i, r in enumerate(results[:top_n])
]

st_app.dataframe(
    pd.DataFrame(table_data).set_index("Rank"),
    use_container_width=True,
)

# ── Plotly overlay chart ───────────────────────────────────────────────────────
st_app.subheader("Histogram + Fitted PDFs")

fig = go.Figure()

# Histogram/bar mode based on detected data type
if is_discrete:
    val_counts = pd.Series(data).value_counts().sort_index()
    probs = val_counts.values / max(val_counts.values.sum(), 1)
    fig.add_trace(go.Bar(
        x=val_counts.index.to_numpy(),
        y=probs,
        name="Data (empirical probability)",
        marker_color="steelblue",
        opacity=0.75,
    ))
else:
    fig.add_trace(go.Histogram(
        x=data,
        histnorm="probability density",
        name="Data",
        marker_color="steelblue",
        opacity=0.5,
        nbinsx=40,
    ))

# Overlay top distributions
colors = ["crimson", "darkorange", "green", "purple", "brown"]
data_min = float(np.min(data))
data_max = float(np.max(data))
data_span = max(data_max - data_min, 1e-8)
data_pad = 0.05 * data_span

# Auto-focus chart on the selected column using robust quantiles.
q_low, q_high = np.quantile(data, [0.01, 0.99])
if np.isfinite(q_low) and np.isfinite(q_high) and q_low < q_high:
    focus_span = max(float(q_high - q_low), 1e-8)
    focus_pad = 0.08 * focus_span
    x_focus_lo = float(q_low - focus_pad)
    x_focus_hi = float(q_high + focus_pad)
else:
    x_focus_lo = data_min - data_pad
    x_focus_hi = data_max + data_pad

if is_discrete and n_unique > 0:
    sorted_unique = np.sort(unique_vals)
    if len(sorted_unique) > 1:
        diffs = np.diff(sorted_unique)
        positive_diffs = diffs[diffs > 0]
        step = float(np.min(positive_diffs)) if len(positive_diffs) else 1.0
    else:
        step = 1.0
    x_focus_lo = float(sorted_unique[0] - 0.5 * step)
    x_focus_hi = float(sorted_unique[-1] + 0.5 * step)

if is_discrete:
    st_app.info(
        "Detected integer-like discrete data: showing empirical probability bars. "
        "Continuous PDF overlays are hidden for this view."
    )
else:
    for i, r in enumerate(results[:min(top_n, 5)]):
        dist_obj = SCIPY_DISTS.get(r["scipy_name"])
        if dist_obj is None:
            continue
        try:
            # Plot each PDF on a support-aware x-range to avoid misleading overlays.
            q_lo = dist_obj.ppf(0.001, *r["params"])
            q_hi = dist_obj.ppf(0.999, *r["params"])
            if np.isfinite(q_lo) and np.isfinite(q_hi) and q_lo < q_hi:
                x_lo = max(data_min - data_pad, float(q_lo))
                x_hi = min(data_max + data_pad, float(q_hi))
            else:
                x_lo = data_min - data_pad
                x_hi = data_max + data_pad

            # Keep overlays aligned with the focused data view.
            x_lo = max(x_lo, x_focus_lo)
            x_hi = min(x_hi, x_focus_hi)

            if x_lo >= x_hi:
                continue

            x_range = np.linspace(x_lo, x_hi, 400)
            pdf_vals = dist_obj.pdf(x_range, *r["params"])
            mask = np.isfinite(pdf_vals) & (pdf_vals >= 0)
            if np.sum(mask) < 2:
                continue

            fig.add_trace(go.Scatter(
                x=x_range[mask],
                y=pdf_vals[mask],
                mode="lines",
                name=f"{r['distribution']} (AIC={r['aic']})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        except Exception:
            continue

fig.update_layout(
    xaxis_title=col,
    yaxis_title="Probability" if is_discrete else "Density",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
    height=420,
)
fig.update_xaxes(range=[x_focus_lo, x_focus_hi], autorange=False)
fig.update_yaxes(autorange=True)
st_app.plotly_chart(fig, use_container_width=True)

# ── AI explanation ────────────────────────────────────────────────────────────
if show_ai:
    st_app.subheader("AI Distribution Explanation")
    with st_app.spinner("Asking AI..."):
        explanation = get_ai_suggestion(summary, results[:top_n])
    st_app.info(explanation)

# ── skpro code export ─────────────────────────────────────────────────────────
st_app.subheader("skpro Code Snippet")
best = results[0]
snippet = generate_skpro_snippet(best, col)
st_app.code(snippet, language="python")

st_app.download_button(
    label="Download snippet as .py",
    data=snippet,
    file_name=f"skpro_{best['scipy_name']}_fit.py",
    mime="text/plain",
)
