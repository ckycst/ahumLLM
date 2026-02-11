import sys
import os
from transformers import AutoTokenizer
from pi_down_load_models import load_embeddings_model_from_ollama
from pi_log import *

def main():
    model_id = "bge-m3"
    text = "Hello, world! I'm using NLP."
    embeddings = load_embeddings_model_from_ollama(model_id)
    log_info(f"Embeddings model loaded from {model_id}")
    texts = ["Hello, world! I'm using NLP."]
    embeddings = embeddings.embed_documents(texts)
    log_info(f"Embeddings: {embeddings}")

if __name__ == "__main__":
    print(main())
