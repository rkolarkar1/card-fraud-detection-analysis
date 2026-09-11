"""
Shared GenAI explanation logic: turns a single transaction's SHAP attribution
into an analyst-readable explanation via a local LLM. Used by explain.py
(demo on known test-set cases) and score_new_transactions.py (real inference
on new, unlabeled transactions).
"""
import os
import re

import numpy as np

LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TOP_K_SIGNALS = 5

SYSTEM_PROMPT = (
    "You are a fraud-analyst copilot for a bank's Cards Fraud and Scams team. "
    "You are given a transaction's fraud model score and the specific signals "
    "that most influenced it, ranked by contribution. Explain plainly why the "
    "transaction received this score, referencing only the signals provided -- "
    "do not invent details that are not given. Then recommend one concrete next "
    "action for the analyst. Keep the tone factual and actionable.\n\n"
    "Respond in exactly this format, with no extra commentary:\n"
    "SUMMARY: <2-3 sentence explanation>\n"
    "KEY SIGNALS:\n"
    "- <signal 1>\n"
    "- <signal 2>\n"
    "- <signal 3>\n"
    "RECOMMENDED ACTION: <one sentence>"
)


def load_generator():
    """Loads the local LLM pipeline. Import is deferred to the call site so
    modules that only need the constants/helpers above don't pay the cost of
    importing torch/transformers."""
    from transformers import pipeline
    return pipeline("text-generation", model=LLM_MODEL, torch_dtype="auto", device_map="cpu")


def to_jsonable(value):
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)


def parse_explanation(text: str) -> dict:
    """Best-effort parse of the model's response into summary / key signals /
    recommended action. Small local models don't reliably echo exact section
    headers the way a hosted frontier model does -- in practice this one
    writes a lead sentence, a bullet or numbered list, then a bolded closing
    recommendation, without ever emitting a literal 'SUMMARY:' label. So this
    parses on structure (bullets vs. prose) rather than requiring the exact
    headers asked for in the prompt."""
    cleaned = text.replace("**", "")

    action_split = re.split(r"Recommended Action:\s*", cleaned, maxsplit=1, flags=re.I)
    if len(action_split) == 2:
        body, recommended_action = action_split
        recommended_action = recommended_action.strip()
    else:
        body, recommended_action = cleaned, ""

    body = re.sub(r"^\s*Summary:\s*", "", body, flags=re.I)

    key_signals = []
    summary_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet_match = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", stripped)
        if bullet_match:
            key_signals.append(bullet_match.group(1).strip())
        else:
            summary_lines.append(stripped)

    summary = " ".join(summary_lines).strip().rstrip(":").strip()
    if not summary and not key_signals and not recommended_action:
        summary = cleaned.strip()

    return {"summary": summary, "key_signals": key_signals, "recommended_action": recommended_action}


def top_signals(explainer, feature_cols, row, top_k: int = TOP_K_SIGNALS) -> list[dict]:
    """Computes SHAP values for a single-row DataFrame and returns the top_k
    contributing features as JSON-safe dicts."""
    shap_values = explainer.shap_values(row)[0]
    top_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
    return [
        {
            "feature": feature_cols[i],
            "value": to_jsonable(row.iloc[0, i]),
            "shap_contribution": round(float(shap_values[i]), 4),
            "direction": "increases risk" if shap_values[i] > 0 else "decreases risk",
        }
        for i in top_indices
    ]


def explain_transaction(generator, fraud_probability_pct: float, signals: list[dict]) -> dict:
    """Calls the local LLM with a transaction's score + top SHAP signals and
    returns the parsed explanation dict."""
    import json

    user_content = json.dumps({
        "fraud_probability_pct": fraud_probability_pct,
        "top_contributing_signals": signals,
    }, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    output = generator(messages, max_new_tokens=350, do_sample=False)
    raw_text = output[0]["generated_text"][-1]["content"]
    return parse_explanation(raw_text)
