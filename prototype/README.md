# skpro Distribution Recommender — Prototype

A Streamlit app that helps you choose the best probability distribution for your data, with AI-powered reasoning and ready-to-use skpro code.

## Features
- CSV upload + column selector
- Descriptive statistics summary (mean, std, skewness, kurtosis, support)
- Fits 15 candidate distributions via `scipy.stats`, ranks by AIC/BIC
- Interactive histogram with fitted PDF overlays (Plotly)
- AI explanation via OpenAI GPT-4o-mini or Google Gemini
- Auto-generated skpro code snippet + download

## Quick Start

```bash
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

## AI Setup (optional)

Set one of these environment variables before running:

```bash
# Option A — OpenAI
export OPENAI_API_KEY=sk-...

# Option B — Google Gemini (has a free tier)
export GOOGLE_API_KEY=AIza...
```

Or add them to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

The app works without an API key — the AI explanation panel will simply be disabled.

## Deploy to Streamlit Community Cloud

1. Push this `prototype/` folder to your GitHub fork
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point it to `prototype/app.py`
4. Add API keys under **Secrets** in the dashboard

## File Structure

```
prototype/
├── app.py            # Main Streamlit UI
├── fitter.py         # scipy distribution fitting + skpro snippet generation
├── ai_suggest.py     # OpenAI / Gemini LLM integration
└── requirements.txt
```
