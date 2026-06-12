"""
Helper for calling OpenAI's GPT-5.1 directly via the OpenAI API.

This module is used in the v2 pilot to verify that the Replicate-routed
GPT-5.1 results in the paper reproduce when the model is called directly
through OpenAI's own API. Both paths should produce equivalent behavior;
the pilot's purpose is to confirm this empirically.

Usage:
    Set OPENAI_API_KEY in your .env, then have your settings JSON specify
    a player_models entry of the form "openai:gpt-5.1-2025-11-13". The
    dispatcher in replicate_helper.py will route to this module
    automatically based on that prefix.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # .env is source of truth; ignore stale shell-exported tokens


def call_openai(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    reasoning_effort: str = None,
) -> str:
    """
    Call an OpenAI model directly through OpenAI's API.

    Args:
        model_name: OpenAI model snapshot id, with or without the "openai:" prefix
            (e.g., "openai:gpt-5.1-2025-11-13" or "gpt-5.1-2025-11-13").
        prompt: Prompt string to send.
        max_tokens: Maximum output token budget (mapped to `max_completion_tokens`
            for reasoning models, which require this name instead of `max_tokens`).
        reasoning_effort: For GPT-5.1, one of {"none", "low", "medium", "high"}.
            Other reasoning models accept the subset they support; see OpenAI docs.

    Returns:
        Model's text response, stripped.
    """
    try:
        # Import here so a missing `openai` package doesn't break the rest of the codebase
        from openai import OpenAI

        # Strip the "openai:" prefix if present
        model_id = model_name.split(":", 1)[1] if model_name.startswith("openai:") else model_name

        print(f"\n[Direct OpenAI API call to {model_id}]")
        print(f"Prompt preview: {prompt[:200]}...")

        client = OpenAI()  # picks up OPENAI_API_KEY from env

        # Build the request kwargs.
        # NOTE on reasoning models (gpt-5.x, o-series): they ignore `temperature`,
        # `top_p`, and `presence_penalty`. They also require `max_completion_tokens`
        # rather than the legacy `max_tokens`. Importantly, `max_completion_tokens`
        # is the TOTAL output budget INCLUDING hidden reasoning tokens. GPT-5.1 with
        # `reasoning_effort="high"` typically spends 2-8k tokens on reasoning before
        # producing any visible text; if the budget is too tight, the call returns
        # an empty response (reasoning exhausts the budget). To keep the same caller
        # interface as the Replicate path (which only counts visible output), we
        # automatically scale up the budget when reasoning is requested.
        if reasoning_effort and reasoning_effort.lower() not in ("none", ""):
            # Reasoning model: budget for both reasoning + visible output.
            # 16k is a safe ceiling for GPT-5.1 high reasoning + a few hundred
            # tokens of visible response. Caller's `max_tokens` is treated as
            # a floor for the visible output portion.
            completion_budget = max(max_tokens, 16384)
        else:
            completion_budget = max_tokens

        kwargs = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": completion_budget,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            print(f"[DEBUG] reasoning_effort={reasoning_effort}, "
                  f"max_completion_tokens={completion_budget}")

        completion = client.chat.completions.create(**kwargs)
        response = completion.choices[0].message.content or ""

        print(f"Response: {response[:200]}...")
        return response.strip()

    except Exception as e:
        error_msg = f"ERROR: OpenAI API call failed - {type(e).__name__}: {str(e)}"
        print(f"⚠️ Error calling OpenAI API: {e}")
        return error_msg
