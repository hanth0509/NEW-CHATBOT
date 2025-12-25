#CHAT AI
import json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# Load model & data (load 1 lần)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = np.load("embeddings.npy")

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def find_most_similar(
    query_embedding: np.ndarray,
    user_id: str = None,
    max_results: int = 5
) -> List[str]:

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        doc = metadata["documents"][idx]

        if user_id and str(doc.get("metadata", {}).get("user_id")) != user_id:
            continue

        results.append(doc["text"])
        if len(results) >= max_results:
            break

    return results

def generate_response(user_message: str, context: List[str]) -> str:
    context_text = "\n".join(f"- {c}" for c in context)

    prompt = f"""
Based on the following transaction information, please answer the user's question concisely and clearly.
Transaction Information:
{context_text}

Question:
{user_message}

Answer:
"""

    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
