from gen_ai_core.llm.local_transformers_llm import TransformersLLM

# sentiment analysis
def witness_small_label_classifier_speed_over_a_full_text_generating_llm() -> None:
    finbert = TransformersLLM().get_simple_text_label_classifier_model()
    headlines = [
        "Apple reports record-breaking revenue in Q4 despite market slowdown", # Positive
        "Tesla stock falls after earnings miss expectations", # Negative
        "Amazon maintains steady growth amid economic slowdown", # Neutral
        "Nvidia shares surge after strong AI chip demand drives record quarterly revenue" # Positive
    ]
    ]
    for h in headlines:
        print(h, "→", finbert(h)[0])     # e.g. {'label': 'positive', 'score': 0.93}


def demo_classifier():
    # Notice the speed: no prompt, no JSON — just a label and a confidence score. 
    # That is why a specialised small model often beats a giant chat model for one narrow, high-volume job.
    witness_small_label_classifier_speed_over_a_full_text_generating_llm()

# Run File using:
# uv run --active python gen_ai_core/classifier/demo_classifier.py
'''
Note it will download couple of files like below(so internet connectivity is important):

W1113 00:43:20.725000 52254 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.
config.json: 100%|████████████████████████████████████████████████████████████████| 758/758 [00:00<00:00, 156kB/s]
model.safetensors: downloading bytes: ████████████████████████████████████████████████████████|  413MB, 15.8MB/s  
model.safetensors: reconstructing file: 100%|████████████████████████████████████████|  438MB /  438MB, 28.3MB/s  
Loading weights: 100%|████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 8495.71it/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████| 252/252 [00:00<00:00, 1.20MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 35.8MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████| 112/112 [00:00<00:00, 196kB/s]
'''
if __name__ == "__main__":
    demo_classifier()