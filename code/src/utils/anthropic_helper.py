"""
Helper for calling Anthropic's Claude models directly via the Anthropic API.

Used in V2 for the cross-model main study (Claude Sonnet 4.6, Claude Opus 4.6).

Per Anthropic's docs, the recommended way to use thinking on Claude 4.6+ models
is **adaptive mode** (`thinking: {type: "adaptive"}`), where Claude evaluates
each request and decides whether to think and how much. Manual mode
(`type: "enabled"` + `budget_tokens`) is deprecated on 4.6 and rejected on 4.7.

Mapping from the paper's `reasoning_effort` categories to Claude's API:

    reasoning_effort = "none"     -> thinking parameter OMITTED (Disabled mode,
                                     standard chat; zero thinking tokens)
    reasoning_effort = "adaptive" -> thinking = {type: "adaptive"} (default
                                     effort="high"; Claude almost always thinks)
    reasoning_effort = "low"      -> thinking = {type: "adaptive"} with
                                     effort="low" (skips thinking for simple tasks)
    reasoning_effort = "medium"   -> thinking = {type: "adaptive"} with effort="medium"
    reasoning_effort = "high"     -> thinking = {type: "adaptive"} with effort="high"
                                     (equivalent to "adaptive" alone)

For the V2 main cross-model study we use "adaptive" (Anthropic's
recommended-default for production use). For the reasoning-effort ablation we
use "high".

Usage:
    Set ANTHROPIC_API_KEY in your .env, then have your settings JSON specify a
    player_models entry of the form "anthropic:claude-sonnet-4-6" or
    "anthropic:claude-opus-4-6". The dispatcher in replicate_helper.py routes
    here based on the prefix.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # .env is source of truth; ignore stale shell-exported tokens


# Mapping from reasoning_effort categories to Anthropic API thinking config.
# Each value is a dict with "type" and optional "effort", or None to omit thinking.
_REASONING_EFFORT_TO_THINKING_CONFIG = {
    "none":     None,
    "adaptive": {"type": "adaptive"},                       # default effort = "high"
    "low":      {"type": "adaptive", "effort": "low"},
    "medium":   {"type": "adaptive", "effort": "medium"},
    "high":     {"type": "adaptive", "effort": "high"},
}


def call_anthropic(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    reasoning_effort: str = None,
) -> str:
    """
    Call a Claude model directly through Anthropic's API.

    Args:
        model_name: Anthropic model id, with or without the "anthropic:" prefix
            (e.g., "anthropic:claude-sonnet-4-6").
        prompt: Prompt string to send.
        max_tokens: Hard limit on total output tokens (thinking + visible text
            combined per Anthropic's docs). Auto-bumped to 16K when thinking
            is enabled to leave room for both.
        reasoning_effort: One of {"none", "adaptive", "low", "medium", "high"};
            mapped per the table above.

    Returns:
        Model's text response, stripped.
    """
    try:
        # Import here so a missing anthropic package doesn't break the rest of the codebase
        import anthropic

        # Strip the "anthropic:" prefix if present
        model_id = (
            model_name.split(":", 1)[1] if model_name.startswith("anthropic:") else model_name
        )

        print(f"\n[Direct Anthropic API call to {model_id}]")
        print(f"Prompt preview: {prompt[:200]}...")

        client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

        # Translate reasoning_effort -> Anthropic thinking config
        thinking_config = None
        if reasoning_effort:
            key = reasoning_effort.lower()
            if key in _REASONING_EFFORT_TO_THINKING_CONFIG:
                thinking_config = _REASONING_EFFORT_TO_THINKING_CONFIG[key]
            else:
                print(
                    f"[WARN] unknown reasoning_effort={reasoning_effort!r}; "
                    f"defaulting to disabled (no extended thinking)."
                )

        # Output budget. max_tokens covers thinking + visible text combined on
        # Claude 4.6+. When thinking is enabled (especially adaptive at default
        # effort=high), thinking can consume thousands of tokens before any
        # visible output appears. Give a generous budget then.
        if thinking_config is not None:
            request_max_tokens = max(max_tokens, 16384)
        else:
            request_max_tokens = max_tokens

        kwargs = {
            "model": model_id,
            "max_tokens": request_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if thinking_config is not None:
            kwargs["thinking"] = thinking_config
            effort = thinking_config.get("effort", "high (default)")
            print(
                f"[DEBUG] adaptive mode: effort={effort}, "
                f"max_tokens={request_max_tokens}"
            )
        else:
            print(f"[DEBUG] thinking disabled (standard chat), max_tokens={max_tokens}")

        response = client.messages.create(**kwargs)

        # Aggregate all text blocks (Claude can return thinking + text blocks).
        # In adaptive mode, thinking blocks contain summaries; visible answer is
        # in "text" blocks. We only want the visible text.
        text_parts = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            # "thinking" blocks are skipped; the agent loop should never see
            # the internal reasoning summary.

        # Log usage for diagnostics
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)
                print(f"[DEBUG] tokens: input={in_tok}, output={out_tok}")
        except Exception:
            pass

        text = "".join(text_parts)
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "max_tokens":
            print(
                "⚠️ Response truncated by max_tokens limit; "
                "consider raising max_tokens in the caller."
            )
        print(f"Response (len={len(text)}): {text[:200]}...")
        return text.strip()

    except Exception as e:
        error_msg = f"ERROR: Anthropic API call failed - {type(e).__name__}: {str(e)}"
        print(f"⚠️ Error calling Anthropic API: {e}")
        return error_msg
