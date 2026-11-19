from rag.rag_core import loaders
from rag.rag_core import embedder
from config import event_banner
from rag.rag_core.chunking import (chunk_documents, fixed_size, parent_child, recursive, semantic, sliding_window,
                               structure_aware)

# At file load, 
# Load list of documents from cache or extracted as document objects, each containing individual metadata in global metadata file along withlist of pages objects from the corpus
docs = loaders.load_corpus()

def get_policy_chunks_from_single_doc(top_k_chuncks: int = 3, doc_id: str = "refund_policy_2025.md"):
    event_banner("1. One document, six strategies (first 3 chunks default each)")

    # with matching document ID, find the policy document with matching id
    policy = next(d for d in docs if d.doc_id == doc_id)
    demo = {
        "fixed_size(80)": fixed_size(policy.text, 80),
        "sliding_window(80, overlap 20)": sliding_window(policy.text, 80, 20),
        "recursive(120)": recursive(policy.text, 120),
        "structure_aware(200)": structure_aware(policy.text, 200, title=policy.metadata["title"]),
        "semantic": semantic(policy.text, embedder.get_embedder_model()),
    }
    for name, chunks in demo.items():
        print(f"\n### {name}: {len(chunks)} chunks")
        for c in chunks[:top_k_chuncks]:
            print("   >", " ".join(c.split())[:150], "...")

def demo_chunking():
    get_policy_chunks_from_single_doc()

# Run with:  uv run --active python demo_driver/demo_rag_concepts/demo_text_extraction_from_doc/demo_chunking.py
if __name__ == "__main__":
    demo_chunking()