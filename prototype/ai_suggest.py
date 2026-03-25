"""LLM-powered distribution suggestion using OpenAI or Gemini."""

import os


def build_prompt(summary: dict, top_fits: list[dict]) -> str:
    """Build the LLM prompt from data summary and top fit results."""
    top3 = top_fits[:3]
    fits_text = "\n".join(
        f"  {i+1}. {f['distribution']} (AIC={f['aic']}, KS p={f['ks_p_value']}, "
        f"good_fit={'Yes' if f['good_fit'] else 'No'})"
        for i, f in enumerate(top3)
    )
    return f"""\
You are a statistics expert helping a data scientist choose a probability distribution.

Data summary:
  - N = {summary['n']}
  - Mean = {summary['mean']:.4f}
  - Std = {summary['std']:.4f}
  - Skewness = {summary['skewness']:.4f}
  - Kurtosis (excess) = {summary['kurtosis']:.4f}
  - Min = {summary['min']:.4f}, Max = {summary['max']:.4f}
  - Support: {summary['support']}

Top fits by AIC:
{fits_text}

In 3-4 concise sentences:
1. Explain why the top-ranked distribution is the best fit for this data.
2. Mention when you might prefer the 2nd or 3rd option instead.
3. Note any data characteristics (skewness, support) driving the choice.
Keep it practical and avoid heavy mathematical notation.
"""


def get_ai_suggestion(summary: dict, top_fits: list[dict]) -> str:
    """
    Call the LLM API and return a plain-English explanation.

    Tries OpenAI first (OPENAI_API_KEY), then Google Gemini (GOOGLE_API_KEY).
    Returns a fallback message if neither key is set.
    """
    prompt = build_prompt(summary, top_fits)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    if openai_key:
        return _call_openai(prompt, openai_key)
    elif google_key:
        return _call_gemini(prompt, google_key)
    else:
        return (
            "**AI suggestion unavailable** — set `OPENAI_API_KEY` or `GOOGLE_API_KEY` "
            "as an environment variable (or in Streamlit secrets) to enable this feature."
        )


def _call_openai(prompt: str, api_key: str) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}"


def _call_gemini(prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"
