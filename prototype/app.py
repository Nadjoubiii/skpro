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

st_app.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        letter-spacing: -0.015em;
    }

    .hero-card {
        border: 1px solid #e2e8f0;
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
    }

    .hero-subtitle {
        margin: 0.35rem 0 0 0;
        color: #334155;
        font-size: 0.97rem;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st_app.markdown(
    """
    <div class="hero-card">
      <p class="hero-title">skpro Probabilistic Modeling Studio</p>
      <p class="hero-subtitle">
        Explore candidate distributions and benchmark skpro probabilistic wrappers
        on your own tabular data.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st_app.sidebar:
    st_app.subheader("Configuration")
    top_n = st_app.slider("Number of distributions to show", 3, 10, 5)
    show_ai = st_app.toggle("Enable AI explanation", value=True)
    st_app.markdown("---")
    st_app.caption("AI integration")
    st_app.markdown(
        "**API keys** (optional for AI):\n"
        "Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` as environment variables "
        "or in [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)."
    )
    st_app.markdown("---")
    st_app.caption("Built with skpro and scipy")

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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st_app.tabs(["Distribution Explorer", "Probabilistic Regressor"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st_app.markdown("### Distribution Explorer")
    st_app.caption("Fit candidate distributions and inspect support-aware diagnostics.")

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

    # ── Summary statistics ────────────────────────────────────────────────────
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

    # ── Fit distributions ─────────────────────────────────────────────────────
    with st_app.spinner("Fitting distributions..."):
        results = fit_distributions(data)

    if not results:
        st_app.error("No distributions could be fit to this data.")
        st_app.stop()

    # ── Results table ─────────────────────────────────────────────────────────
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

    # ── Plotly overlay chart ──────────────────────────────────────────────────
    st_app.subheader("Histogram + Fitted PDFs")

    fig = go.Figure()

    if is_discrete:
        val_counts = pd.Series(data).value_counts().sort_index()
        probs = val_counts.values / max(val_counts.values.sum(), 1)
        fig.add_trace(go.Bar(
            x=val_counts.index.to_numpy(),
            y=probs,
            name="Data (empirical probability)",
            marker_color="#0f4c81",
            opacity=0.75,
        ))
    else:
        fig.add_trace(go.Histogram(
            x=data,
            histnorm="probability density",
            name="Data",
            marker_color="#0f4c81",
            opacity=0.5,
            nbinsx=40,
        ))

    colors = ["#0f766e", "#0f4c81", "#7c3aed", "#b45309", "#be123c"]
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_span = max(data_max - data_min, 1e-8)
    data_pad = 0.05 * data_span

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
                q_lo = dist_obj.ppf(0.001, *r["params"])
                q_hi = dist_obj.ppf(0.999, *r["params"])
                if np.isfinite(q_lo) and np.isfinite(q_hi) and q_lo < q_hi:
                    x_lo = max(data_min - data_pad, float(q_lo))
                    x_hi = min(data_max + data_pad, float(q_hi))
                else:
                    x_lo = data_min - data_pad
                    x_hi = data_max + data_pad

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
        template="plotly_white",
        xaxis_title=col,
        yaxis_title="Probability" if is_discrete else "Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[x_focus_lo, x_focus_hi], autorange=False)
    fig.update_yaxes(autorange=True)
    st_app.plotly_chart(fig, use_container_width=True)

    # ── AI explanation ────────────────────────────────────────────────────────
    if show_ai:
        st_app.subheader("AI Distribution Explanation")
        with st_app.spinner("Asking AI..."):
            explanation = get_ai_suggestion(summary, results[:top_n])
        st_app.info(explanation)

    # ── skpro code export ─────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Probabilistic Regressor
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from regressor_demo import REGRESSOR_DESCRIPTIONS, estimate_complexity, fit_and_predict

    st_app.markdown("### Probabilistic Regressor")
    st_app.caption(
        "Apply a skpro probabilistic wrapper to your data. "
        "Select features (X) and a target (y), pick a wrapper, and see "
        "prediction intervals + CRPS evaluation."
    )

    # ── Column selection ──────────────────────────────────────────────────────
    cl, cr = st_app.columns(2)
    with cl:
        y_col = st_app.selectbox("Target column (y)", numeric_cols, key="y_col")
    with cr:
        x_default = [c for c in numeric_cols if c != y_col]
        x_cols = st_app.multiselect(
            "Feature columns (X)",
            [c for c in numeric_cols if c != y_col],
            default=x_default[:min(5, len(x_default))],
            key="x_cols",
        )

    if not x_cols:
        st_app.warning("Select at least one feature column.")
        st_app.stop()

    # ── Regressor config ──────────────────────────────────────────────────────
    cr1, cr2 = st_app.columns(2)
    with cr1:
        regressor_name = st_app.selectbox(
            "Probabilistic wrapper",
            list(REGRESSOR_DESCRIPTIONS.keys()),
            key="reg_name",
        )
    with cr2:
        base_est_name = st_app.selectbox(
            "Base sklearn estimator",
            ["Linear Regression", "Ridge", "Random Forest (50 trees)", "Gradient Boosting (50 trees)"],
            key="base_est",
        )

    st_app.info(f"**{regressor_name}**: {REGRESSOR_DESCRIPTIONS[regressor_name]}")

    cs1, cs2, cs3 = st_app.columns(3)
    with cs1:
        test_size = st_app.slider("Test set size", 0.1, 0.4, 0.2, 0.05, key="test_size")
    with cs2:
        n_bootstrap = st_app.slider("Bootstrap samples", 10, 200, 50, 10, key="n_bootstrap",
                                    disabled=(regressor_name != "Bootstrap"))
    with cs3:
        coverage = st_app.select_slider(
            "Interval coverage",
            options=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            value=0.9,
            key="coverage",
        )

    # ── Complexity estimate (live, before Run) ────────────────────────────────
    n_valid_rows = int(df[x_cols + [y_col]].dropna().shape[0]) if x_cols else 0
    if n_valid_rows > 0:
        cx = estimate_complexity(
            n_rows=n_valid_rows,
            n_features=len(x_cols),
            regressor_name=regressor_name,
            base_est_name=base_est_name,
            n_bootstrap=n_bootstrap,
            test_size=test_size,
        )
        tier_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Very High": "🔴"}
        st_app.markdown(
            f"**Complexity estimate:** {tier_icons[cx['tier']]} **{cx['tier']}** — "
            f"estimated time **{cx['est_label']}** &nbsp;|&nbsp; "
            f"**{cx['fits']}** model fit(s) &nbsp;|&nbsp; {cx['detail']}"
        )
        if cx["tier"] in ("High", "Very High"):
            st_app.warning(
                f"This configuration may take {cx['est_label']}. "
                "Consider reducing bootstrap samples, using a faster base model "
                "(Linear Regression or Ridge), or reducing the dataset size."
            )

    # ── Run ───────────────────────────────────────────────────────────────────
    if st_app.button("▶ Run Probabilistic Regression", type="primary"):
        X_df = df[x_cols].apply(pd.to_numeric, errors="coerce")
        y_series = pd.to_numeric(df[y_col], errors="coerce")
        valid_mask = X_df.notna().all(axis=1) & y_series.notna()
        X_clean = X_df[valid_mask].astype(float)
        y_clean = y_series[valid_mask].astype(float)

        if len(y_clean) < 20:
            st_app.error("Need at least 20 valid rows to fit a regressor.")
            st_app.stop()

        with st_app.spinner(f"Fitting {regressor_name} on {len(y_clean)} rows..."):
            result = fit_and_predict(
                X_clean, y_clean, regressor_name, base_est_name,
                test_size=test_size, n_bootstrap=n_bootstrap, coverage=coverage,
            )

        # ── Metrics ───────────────────────────────────────────────────────────
        st_app.subheader("Results")
        m1, m2, m3, m4 = st_app.columns(4)
        m1.metric("Train rows", result["n_train"])
        m2.metric("Test rows", result["n_test"])
        m3.metric("RMSE", result["rmse"])
        m4.metric(f"Coverage @ {int(coverage * 100)}%", f"{result['empirical_coverage']:.1%}")

        if result["crps"] is not None:
            mc1, mc2 = st_app.columns(2)
            mc1.metric("CRPS (lower = better)", result["crps"])
            mc2.metric("Mean interval width", result["mean_interval_width"])
        else:
            st_app.metric("Mean interval width", result["mean_interval_width"])

        # ── Plot: actual vs predicted + CI band ───────────────────────────────
        st_app.subheader("Actual vs Predicted (sorted by actual value)")
        idx = np.arange(len(result["actual"]))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=idx, y=result["actual"],
            mode="markers", name="Actual",
            marker=dict(color="#0f4c81", size=5, opacity=0.75),
        ))
        fig2.add_trace(go.Scatter(
            x=idx, y=result["mean_pred"],
            mode="lines", name="Predicted mean",
            line=dict(color="#be123c", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=np.concatenate([idx, idx[::-1]]),
            y=np.concatenate([result["upper"], result["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(190,18,60,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{int(coverage * 100)}% prediction interval",
        ))
        fig2.update_layout(
            template="plotly_white",
            xaxis_title="Test sample index (sorted by actual)",
            yaxis_title=y_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(t=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st_app.plotly_chart(fig2, use_container_width=True)

        # ── Code snippet ──────────────────────────────────────────────────────
        st_app.subheader("skpro Code Snippet")
        st_app.code(result["code"], language="python")
        st_app.download_button(
            label="Download snippet as .py",
            data=result["code"],
            file_name=f"skpro_{regressor_name.lower().replace(' ', '_')}.py",
            mime="text/plain",
            key="dl_reg",
        )
