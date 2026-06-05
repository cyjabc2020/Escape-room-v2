"""
Helper functions for calling Replicate API
"""
import os
from dotenv import load_dotenv

load_dotenv()

def call_replicate(model_name: str, prompt: str, max_tokens: int = 1024, reasoning_effort: str = None) -> str:
    """
    Dispatch an LLM call by model name.

    Backwards-compatible router:
      - Model names starting with "openai:" are routed to the direct OpenAI API
        (see openai_helper.call_openai). Used for the v2 pilot that reruns the
        same GPT-5.1 experiments without Replicate in the loop.
      - All other model names (e.g., "openai/gpt-5.1", "anthropic/claude-...")
        are sent to Replicate as before.

    Args:
        model_name: Provider/model identifier. Examples:
            - 'openai/gpt-5.1' (Replicate's official OpenAI integration)
            - 'openai:gpt-5.1-2025-11-13' (direct OpenAI API)
            - 'dummy' (no API call; handled by the caller)
        prompt: The prompt to send.
        max_tokens: Maximum response length.
        reasoning_effort: Reasoning effort level for reasoning models
            (e.g., 'none', 'low', 'medium', 'high' for GPT-5.1).

    Returns:
        Model's response as a string.
    """
    # Direct OpenAI API path (v2 pilot)
    if model_name.startswith("openai:"):
        from utils.openai_helper import call_openai
        return call_openai(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    # Direct Gemini API path (v2 pilot, cross-family comparison)
    if model_name.startswith("gemini:"):
        from utils.gemini_helper import call_gemini
        return call_gemini(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    # Direct Anthropic API path (V2 cross-family comparison)
    if model_name.startswith("anthropic:"):
        from utils.anthropic_helper import call_anthropic
        return call_anthropic(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    # Default: Replicate path
    # Also handles V2 models hosted on Replicate: DeepSeek-R1, Llama 3.3, Qwen, etc.
    try:
        import replicate

        print(f"\n[API Call to {model_name}]")
        print(f"Prompt preview: {prompt}...")

        # Use the exact model name from settings
        model_id = model_name

        # Build input parameters.
        # NOTE: OpenAI's GPT-5.1 (gpt-5.1-2025-11-13) does NOT expose `temperature`
        # to API callers. Replicate's wrapper accepts the parameter for compatibility
        # with other models, but for gpt-5.1 it is silently ignored and the model's
        # internal sampling settings are used. We omit `temperature` for gpt-5.1 to
        # avoid the implication that it has any effect; pass it explicitly only for
        # other (non-reasoning) models that do honor it.
        input_params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

        # Add model-specific reasoning/thinking parameters
        is_gpt51 = "gpt-5.1" in model_name or "openai/gpt-5.1" in model_id
        is_deepseek_r1 = "deepseek" in model_name.lower() and ("r1" in model_name.lower() or "reasoner" in model_name.lower())

        if is_gpt51:
            input_params["reasoning_effort"] = reasoning_effort if reasoning_effort else "none"
            print(f"[DEBUG] reasoning_effort parameter set to: {input_params['reasoning_effort']}")
        elif is_deepseek_r1:
            # DeepSeek-R1 is intrinsically a reasoning model and does not expose a
            # per-call thinking toggle on Replicate. We pass temperature=0.6 (the
            # recommended setting from DeepSeek's docs for the reasoner model).
            input_params["temperature"] = 0.6
            print(f"[DEBUG] DeepSeek-R1: temperature=0.6 (intrinsic reasoning, no toggle)")
        else:
            # Non-reasoning models (Llama, Qwen, etc.): include temperature
            input_params["temperature"] = 0.7

        # Run the model
        output = replicate.run(
            model_id,
            input=input_params
        )

        # Collect response (output is a generator)
        response = "".join(output)

        print(f"Response: {response}...")
        return response.strip()

    except Exception as e:
        error_msg = f"ERROR: API call failed - {type(e).__name__}: {str(e)}"
        print(f"⚠️ Error calling Replicate API: {e}")
        print(f"Falling back to default response")
        return error_msg

