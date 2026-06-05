"""
Helper for calling Google's Gemini directly via the Gemini API.

Handles BOTH Gemini 2.5 family (uses thinking_budget integer) and Gemini 3.x
family (uses thinking_level categorical, can't disable thinking entirely on
3.1 Pro). The helper auto-detects which API parameter to use based on the
model name.

Gemini 2.5 mapping (`thinking_budget`):
    reasoning_effort = "none"   -> thinking_budget = 0      (disabled)
    reasoning_effort = "low"    -> thinking_budget = 1024
    reasoning_effort = "medium" -> thinking_budget = 8192
    reasoning_effort = "high"   -> thinking_budget = -1     (dynamic/full)

Gemini 3.x mapping (`thinking_level`):
    reasoning_effort = "none"   -> thinking_level = "minimal"  (closest to off;
                                   3.1 Pro doesn't support "minimal" -- omitted
                                   in that case so model uses default = "high")
    reasoning_effort = "low"    -> thinking_level = "low"
    reasoning_effort = "medium" -> thinking_level = "medium"
    reasoning_effort = "high"   -> thinking_level = "high"

CRITICAL: Gemini 3.1 Pro always thinks (cannot be disabled). The total token
budget (max_output_tokens) is SHARED across thinking + visible output, so even
calls that don't request reasoning need a generous budget. We therefore set a
high default budget for all Gemini 3.x calls regardless of reasoning_effort.

Usage:
    Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env, then have your settings
    JSON specify a player_models entry of the form "gemini:gemini-3.1-pro-preview"
    or "gemini:gemini-2.5-flash". The dispatcher in replicate_helper.py will
    route to this module automatically based on the prefix.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# Gemini 2.5 mapping (integer token budget)
_REASONING_EFFORT_TO_THINKING_BUDGET_2_5 = {
    "none": 0,
    "low": 1024,
    "medium": 8192,
    "high": -1,
}

# Gemini 3.x mapping (categorical level)
_REASONING_EFFORT_TO_THINKING_LEVEL_3 = {
    "none": "minimal",   # closest to "off"; 3.1 Pro will reject this and we'll omit it
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# Generous default budget for all Gemini 3.x calls, since thinking is always on
# and max_output_tokens is the shared budget.
_GEMINI_3_DEFAULT_BUDGET = 16384

# Models in the Gemini 3.x family that do NOT support thinking_level="minimal"
# (per Google's docs: "You cannot disable thinking for Gemini 3.1 Pro").
_GEMINI_3_PRO_MODELS = {
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro",
}


def _is_gemini_3(model_id: str) -> bool:
    """Detect Gemini 3.x family (3, 3.1, 3.5, ...) based on model id."""
    return "gemini-3" in model_id


def _is_gemini_3_pro(model_id: str) -> bool:
    """Detect Gemini 3.x Pro variants which cannot disable thinking."""
    return model_id in _GEMINI_3_PRO_MODELS or (
        "gemini-3" in model_id and "pro" in model_id
    )


def call_gemini(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    reasoning_effort: str = None,
) -> str:
    """
    Call a Gemini model directly through Google's Gemini API.

    Args:
        model_name: Gemini model id, with or without the "gemini:" prefix
            (e.g., "gemini:gemini-3.1-pro-preview" or "gemini-2.5-flash").
        prompt: Prompt string to send.
        max_tokens: Caller's hint for visible output tokens. For Gemini 3.x this
            is automatically bumped to 16384 to accommodate always-on thinking;
            for Gemini 2.5 it is honored as-is when reasoning is off, and bumped
            to 16384 when reasoning is on.
        reasoning_effort: One of {"none", "low", "medium", "high"} or None.
            Mapping depends on model family (see module docstring).

    Returns:
        Model's text response, stripped.
    """
    try:
        from google import genai
        from google.genai import types

        # Strip the "gemini:" prefix if present
        model_id = (
            model_name.split(":", 1)[1] if model_name.startswith("gemini:") else model_name
        )

        print(f"\n[Direct Gemini API call to {model_id}]")
        print(f"Prompt preview: {prompt[:200]}...")

        client = genai.Client()  # picks up GEMINI_API_KEY or GOOGLE_API_KEY from env

        is_g3 = _is_gemini_3(model_id)
        is_g3_pro = _is_gemini_3_pro(model_id)

        # Choose output budget. Gemini 3.x always thinks (can't disable on Pro),
        # so the shared budget must always be generous, regardless of whether
        # reasoning_effort was explicitly set by the caller.
        # Sub-task calls (no reasoning_effort -> forced thinking_level='low' below)
        # need much less budget than full agent-decision calls.
        if is_g3:
            if reasoning_effort and reasoning_effort.lower() in ("high", "medium"):
                visible_budget = max(max_tokens, _GEMINI_3_DEFAULT_BUDGET)  # 16K
            else:
                # 'low' thinking_level: ~500 thinking tokens + a few hundred visible
                visible_budget = max(max_tokens, 4096)
        elif reasoning_effort and reasoning_effort.lower() != "none":
            visible_budget = max(max_tokens, _GEMINI_3_DEFAULT_BUDGET)
        else:
            visible_budget = max_tokens

        config_kwargs = {"max_output_tokens": visible_budget}

        # Translate reasoning_effort -> Gemini's per-family thinking parameter
        if is_g3:
            # Gemini 3.x: use thinking_level
            level = None
            if reasoning_effort:
                key = reasoning_effort.lower()
                if key in _REASONING_EFFORT_TO_THINKING_LEVEL_3:
                    candidate = _REASONING_EFFORT_TO_THINKING_LEVEL_3[key]
                    # 3.1 Pro rejects "minimal" -- fall back to "low" instead
                    # (the cheapest level Pro supports).
                    if candidate == "minimal" and is_g3_pro:
                        print(
                            f"[INFO] Gemini 3.x Pro does not support thinking_level='minimal'; "
                            f"using 'low' instead (cheapest available on Pro)."
                        )
                        level = "low"
                    else:
                        level = candidate
                else:
                    print(f"[WARN] unknown reasoning_effort={reasoning_effort!r}; using 'low'.")
                    level = "low"
            else:
                # No reasoning_effort passed = caller is a sub-task (puzzle-solving or
                # verification). These are trivial arithmetic problems that don't need
                # the model's default 'high' thinking. Set 'low' explicitly to save cost.
                level = "low"
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            print(
                f"[DEBUG] gemini-3.x: reasoning_effort={reasoning_effort or '(unset)'} -> "
                f"thinking_level={level}, max_output_tokens={visible_budget}"
            )
        else:
            # Gemini 2.5: use thinking_budget (integer)
            budget = None
            if reasoning_effort:
                key = reasoning_effort.lower()
                if key in _REASONING_EFFORT_TO_THINKING_BUDGET_2_5:
                    budget = _REASONING_EFFORT_TO_THINKING_BUDGET_2_5[key]
                else:
                    print(f"[WARN] unknown reasoning_effort={reasoning_effort!r}; using model default.")
            else:
                # No reasoning_effort passed = caller is a sub-task (puzzle-solving or
                # verification). Gemini 2.5 Flash's default is DYNAMIC thinking, which
                # for tiny max_output_tokens budgets (e.g. 50 for verify calls) burns
                # nearly the entire shared budget on hidden thoughts and truncates the
                # visible answer mid-digit (e.g. "86" -> "8"). Disable thinking
                # entirely for sub-tasks so the full visible budget is available.
                # Mirrors the 3.x branch above which forces thinking_level='low'.
                budget = 0
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            print(
                f"[DEBUG] gemini-2.5: reasoning_effort={reasoning_effort or '(unset->sub-task)'} -> "
                f"thinking_budget={budget}, max_output_tokens={visible_budget}"
            )

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )

        # Aggregate all non-thought text parts
        text_parts = []
        finish_reason = None
        try:
            candidates = response.candidates or []
            if candidates:
                cand = candidates[0]
                finish_reason = getattr(cand, "finish_reason", None)
                parts = getattr(getattr(cand, "content", None), "parts", None) or []
                n_thought = sum(1 for p in parts if getattr(p, "thought", False))
                n_text = sum(
                    1 for p in parts
                    if getattr(p, "text", None) and not getattr(p, "thought", False)
                )
                print(
                    f"[DEBUG] finish_reason={finish_reason}, "
                    f"parts: total={len(parts)} text={n_text} thought={n_thought}"
                )
                for p in parts:
                    if getattr(p, "thought", False):
                        continue
                    pt = getattr(p, "text", None)
                    if pt:
                        text_parts.append(pt)
        except Exception as parse_err:
            print(f"[WARN] could not iterate response parts ({parse_err}); falling back to response.text")
            text_parts = [response.text or ""]

        # Token usage diagnostics
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                candidates_tokens = getattr(usage, "candidates_token_count", None)
                thoughts_tokens = getattr(usage, "thoughts_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)
                print(
                    f"[DEBUG] tokens: prompt={prompt_tokens}, "
                    f"visible_output={candidates_tokens}, "
                    f"thoughts={thoughts_tokens}, total={total_tokens}"
                )
        except Exception as usage_err:
            print(f"[WARN] could not read usage_metadata ({usage_err})")

        text = "".join(text_parts) if text_parts else (response.text or "")
        print(f"Response (len={len(text)}): {text[:200]}...")

        if finish_reason and str(finish_reason).upper().endswith("MAX_TOKENS"):
            print(
                "⚠️ Response truncated by MAX_TOKENS limit. Consider raising max_output_tokens."
            )

        return text.strip()

    except Exception as e:
        error_msg = f"ERROR: Gemini API call failed - {type(e).__name__}: {str(e)}"
        print(f"⚠️ Error calling Gemini API: {e}")
        return error_msg
