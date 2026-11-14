from rag.rag_core import embeddings
from gen_ai_core import Prompt, RawLLM
from config import data_dir

def index_src_documents_from_kb(files: list[str] = [data_dir / "refund_policy_2025.md", data_dir / "delivery_and_shipping.md"]):
    chunks = []
    for f in files:
        text = open(f, encoding="utf-8").read()
        # naive chunking: one chunk per paragraph (blank-line separated)
        chunks += [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

    embedder = embeddings.get_embedder_model()
    embedding_vectors = embedder.embed_documents(chunks)           # shape: (number_of_chunks, dimensions)
    print(f"Indexed {len(chunks)} chunks with {embedder.name} -> matrix {embedding_vectors.shape}")
    return embedding_vectors, chunks


def retrieve_doc_chunks_for_query(query: str, embedding_vectors, chunks, top_k:int = 3):
    import numpy as np
    embedder = embeddings.get_embedder_model()
    query_embedding_vector = embedder.embed_query(query) # same model as the documents!
    scores = embedding_vectors @ query_embedding_vector            # cosine similarity (vectors have length 1)
    top_k_retrievals = np.argsort(-scores)[:top_k]
    print(f"Query vector for '{query}': {query_embedding_vector[:5]}... (length {len(query_embedding_vector)})")
    print(f"Top {top_k} matching chunks: {top_k_retrievals}")
    top_k_chunks = [chunks[i] for i in top_k_retrievals]
    print(f"\nQuestion: {query}\n\nTop {top_k} chunks:")
    for rank, idx in enumerate(top_k_retrievals, start=1):
        print(f"  {rank}. score={scores[idx]:.3f} {' '.join(chunks[idx].split())[:110]}...")
    return top_k_chunks


def get_augmented_prompt_from_retrieved_chunks(question_query: str, top_k_chunks: list[str]):
    context = "\n\n".join(f"[S{r}] {chunk}" for r, chunk in enumerate(top_k_chunks, start=1))
    retrieved_context = f"Question: {question_query}\n\nContext:\n{context}\n\nAnswer:"
    return Prompt(
                    role="a helpful assistant",
                    task = ("Answer the question using ONLY the sources below. "
                            "Cite sources like [S1]. "
                            "If the answer is not in the sources, say you don't know."),
                    rag_context=f"If the answer is not in the given context, say you don't know. {retrieved_context}",
                    user_query=question_query
            )

def generate_llm_ans_response_from_augmented_prompt(llm: RawLLM, augmented_system_prompt: Prompt):
    # Here you would typically call a language model API to generate the answer
    # For demonstration purposes, we'll just return a placeholder response
    print("\n----- PROMPT SENT TO THE LLM -----\n" + str(augmented_system_prompt) + "\n----------------------------------")
    ans_response = llm.generate_response_content(augmented_system_prompt)
    print(f"[Unhardened][Raw SDK]  {ans_response.text}\n  ({ans_response.latency_ms:.0f} ms, {ans_response.input_tokens} in / {ans_response.output_tokens} out)")
    return f"\n----- LLM Generated answer based on the prompt:-----\n{augmented_system_prompt}\n\nLLM Response:\n{ans_response.text}"


def retrieve_augment_generate():
    embedding_vectors, chunks = index_src_documents_from_kb(files=[data_dir / "refund_policy_2025.md", data_dir / "delivery_and_shipping.md"])
    import sys
    question_query = sys.argv[1] if len(sys.argv) > 1 else "How many days do I have to return a mobile phone?"
    top_k_chunks = retrieve_doc_chunks_for_query(question_query, embedding_vectors, chunks)
    augmented_prompt = get_augmented_prompt_from_retrieved_chunks(question_query, top_k_chunks)
    response = generate_llm_ans_response_from_augmented_prompt(RawLLM(), augmented_prompt)

# Run with:  uv run --active python rag/demo_rag.py
if __name__ == "__main__":
    retrieve_augment_generate()