from __future__ import annotations

import platform
import sys
from tabnanny import check
import time

from gen_ai_core import Prompt, RawLLM, LangChainLLM, llm

OK, BAD, WARN = "[ok]  ", "[FIX] ", "[note]"


def _check(label: str, ok: bool, fix: str = "") -> bool:
    print(f"{OK if ok else BAD}{label}" + ("" if ok else f"\n        -> {fix}"))
    return ok

def check_python_version() -> None:
    print(f"Python {platform.python_version()} on {platform.system()}")
    _check("Python 3.10 or newer", sys.version_info >= (3, 10),
           "Install Python 3.11 or 3.12 or 3.13 from python.org")


def check_package_imports(pkgs: list[str]) -> None:
    import importlib
    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
            _check(f"package {pkg}", True)
        except Exception:
            _check(f"package {pkg}", False, "pip install -r requirements.txt   (inside your virtual environment)")


def check_raw_llm_setup() -> None:

    llm = RawLLM()
    if not llm.is_sdk_client_available:
        print(f"{WARN} No LLM configured. Labs still run (extractive answers).")
        print("        To use a real LLM, copy .env.example to .env and set GROQ_API_KEY or OPENAI_API_KEY,")
        print("        or install Ollama (https://ollama.com), run `ollama pull llama3.1:8b` or `ollama pull llama3.2`, and set LLM_PROVIDER=ollama.")
    else:
        try:
            r = llm.generate_response_content(Prompt(
                                                        role="a concise assistant",
                                                        task="Reply concisely with one word",
                                                        user_query="Say: ready"
                                                    )
                                            )
            _check(f"Raw LLM {llm._provider}/{llm._model} replied '{r.text[:30]}' in {r.latency_ms:.0f} ms", True)
        except Exception as e:
            _check(f"Raw LLM {llm._provider}/{llm._model}", False,
                   f"{e.__class__.__name__}: {str(e)[:200]}  "
                   "(check the key, the model name in LLM_MODEL, and your billing)")

def check_langchain_llm_setup() -> None:
    lc_llm = LangChainLLM()
    try:
        t0 = time.perf_counter()
        r = lc_llm.invoke(Prompt(
                                    role="a concise assistant", 
                                    task="Reply concisely with one word",
                                    user_query="Say: ready"
                                )
                        )
        ms = (time.perf_counter() - t0) * 1000
        _check(f"Langchain LLM {lc_llm._provider}/{lc_llm._model} replied '{r[:30]}' in {ms:.0f} ms", True)
    except Exception as e:
        _check(f"Langchain LLM {lc_llm._provider}/{lc_llm._model}", False,
            f"{e.__class__.__name__}: {str(e)[:200]}  "
            "(check the key, the model name in LLM_MODEL, and your billing)")

def check_ocr_setup() -> None:
    import shutil
    _check("Tesseract OCR installed (needed for scanned PDFs in Lab 02)", 
           shutil.which("tesseract") is not None,
            "macOS: brew install tesseract | Ubuntu: sudo apt install tesseract-ocr | "
            "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH. "
            "(Optional: without it, scanned pages are skipped.)"
    )

def check_embedding_models_setup() -> None:
    print("\nLoading the embedding model (first time downloads ~70 MB) ...")
    try:
        from rag.rag_core.embeddings import get_embedder_model
        emb = get_embedder_model()
        v = emb.embed_query("hello")
        _check(f"embedding model {emb.name} works, {emb.dim} dimensions", len(v) == emb.dim)
    except Exception as e:
        _check("embedding model", 
               False, 
                f"{e.__class__.__name__}: {str(e)[:120]}. If huggingface.co is blocked, "
                "set EMBEDDER=wordllama in .env (fully offline)."
            )

def check_reranker_setup() -> None:
    print("\nLoading the reranker (first time downloads ~80 MB) ...")
    from rag.rag_core.rerank import get_reranker
    reranker = get_reranker()
    _check(f"reranker: {reranker.name}", 
           "fallback" not in reranker.name,
            "The cross-encoder could not be downloaded; Lab 08 will use a weaker keyword fallback."
            )

def check_setup() -> None:
    check_python_version()
    print("\n")
    check_raw_llm_setup()
    print("\n")
    check_langchain_llm_setup()
    print("\n")
    check_package_imports(["dotenv", "openai", "anthropic"])
    print("\n")

    check_package_imports(["numpy", "fastembed", "pymupdf", "faiss"])
    print("\n")
    check_ocr_setup()
    print("\n")
    check_embedding_models_setup()
    print("\n")
    check_reranker_setup()
    print("\n")
    print("Setup check complete.")


# To Prevent auto-execution on importing method name in other classes
# unless explicitly called for consumption/usage via manual method call trigger or using python command on direct file execution
# Run with:  uv run --active python demo_driver/demo_check_setup.py
if __name__ == "__main__":
    check_setup()
