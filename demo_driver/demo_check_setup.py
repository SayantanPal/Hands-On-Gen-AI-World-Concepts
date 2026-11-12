from __future__ import annotations

import platform
import sys
import time

from gen_ai_core import Prompt, RawLLM, LangChainLLM, llm

OK, BAD, WARN = "[ok]  ", "[FIX] ", "[note]"


def _check(label: str, ok: bool, fix: str = "") -> bool:
    print(f"{OK if ok else BAD}{label}" + ("" if ok else f"\n        -> {fix}"))
    return ok

def check_python_version() -> None:
    print(f"Python {platform.python_version()} on {platform.system()}")
    _check("Python 3.10 or newer", sys.version_info >= (3, 10),
           "Install Python 3.11 or 3.12 from python.org")

def check_raw_llm_setup() -> None:

    llm = RawLLM()
    if not llm.is_sdk_client_available:
        print(f"{WARN} No LLM configured. Labs still run (extractive answers).")
        print("        To use a real LLM, copy .env.example to .env and set GROQ_API_KEY or OPENAI_API_KEY,")
        print("        or install Ollama (https://ollama.com), run `ollama pull llama3.1:8b`, and set LLM_PROVIDER=ollama.")
    else:
        try:
            r = llm.generate_response_content(Prompt(role="a concise assistant", user_query="Say: ready"))
            _check(f"Raw LLM {llm._provider}/{llm._model} replied '{r.text[:30]}' in {r.latency_ms:.0f} ms", True)
        except Exception as e:
            _check(f"Raw LLM {llm._provider}/{llm._model}", False,
                   f"{e.__class__.__name__}: {str(e)[:200]}  "
                   "(check the key, the model name in LLM_MODEL, and your billing)")

def check_langchain_llm_setup() -> None:
    lc_llm = LangChainLLM()
    try:
        t0 = time.perf_counter()
        r = lc_llm.invoke(Prompt(role="a concise assistant", user_query="Say: ready"))
        ms = (time.perf_counter() - t0) * 1000
        _check(f"Langchain LLM {lc_llm._provider}/{lc_llm._model} replied '{r[:30]}' in {ms:.0f} ms", True)
    except Exception as e:
        _check(f"Langchain LLM {lc_llm._provider}/{lc_llm._model}", False,
            f"{e.__class__.__name__}: {str(e)[:200]}  "
            "(check the key, the model name in LLM_MODEL, and your billing)")



def check_setup() -> None:
    check_python_version()
    print("\n")
    check_raw_llm_setup()
    print("\n")
    check_langchain_llm_setup()
    print("\n")


# To Prevent auto-execution on importing method name in other classes
# unless explicitly called for consumption/usage via manual method call trigger or using python command on direct file execution
if __name__ == "__main__":
    check_setup()
