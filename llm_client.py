"""
Thin LLM client supporting Anthropic, OpenAI, and Gemini.
Handles multi-turn conversations for interrogation experiments.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Lazy-load clients to avoid import errors if a package isn't installed
_anthropic_client = None
_openai_client = None
_gemini_configured = False


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _configure_gemini():
    global _gemini_configured
    if not _gemini_configured:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _gemini_configured = True


def call_llm(model, system_prompt, messages, max_tokens=1000):
    """
    Universal LLM call supporting multi-turn conversations.

    Args:
        model: Model name (e.g., "claude-sonnet-4-6", "gpt-4o", "gemini-2.5-flash-lite")
        system_prompt: System-level instructions
        messages: Either a string (single turn) or list of {"role": ..., "content": ...} dicts
        max_tokens: Max response length

    Returns:
        str: The model's response text
    """
    # Normalize messages to list format
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        if model.startswith("claude"):
            client = _get_anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages
            )
            return response.content[0].text

        elif model.startswith("gpt"):
            client = _get_openai()
            oai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=oai_messages
            )
            return response.choices[0].message.content

        elif model.startswith("gemini"):
            _configure_gemini()
            import google.generativeai as genai
            gmodel = genai.GenerativeModel(model, system_instruction=system_prompt)

            # Convert messages to Gemini format
            gemini_history = []
            for msg in messages[:-1]:
                gemini_role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": gemini_role, "parts": [msg["content"]]})

            chat = gmodel.start_chat(history=gemini_history)
            response = chat.send_message(messages[-1]["content"])
            return response.text

        else:
            raise ValueError(f"Unknown model: {model}")

    except Exception as e:
        print(f"  LLM Error ({model}): {e}")
        return f"API_ERROR: {e}"