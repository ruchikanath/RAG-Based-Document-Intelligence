from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Machine learning is a subset of artificial intelligence.",
    "ML is a branch of AI.",
    "I like eating pizza."
]

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))

similarity = cosine_similarity(embeddings)

print("\nSimilarity Matrix:")
print(similarity)