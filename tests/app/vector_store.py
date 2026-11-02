import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Extract text from PDF
# -----------------------------

pdf_path = "data/documents/AI_Fundamentals.pdf"

document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"


# -----------------------------
# 2. Create chunks
# -----------------------------

def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = create_chunks(full_text)

print("Total characters:", len(full_text))
print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Create embeddings
# -----------------------------

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# -----------------------------
# 5. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# -----------------------------
# 6. Test a query
# -----------------------------

query = "What is artificial intelligence?"

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# -----------------------------
# 7. Search FAISS
# -----------------------------

k = 3

distances, indices = index.search(query_embedding, k)


# -----------------------------
# 8. Display results
# -----------------------------

print("\nQuery:")
print(query)

print("\nRetrieved chunks:")

for i, index_number in enumerate(indices[0]):

    print(f"\n--- Result {i + 1} ---")

    print("Distance:", distances[0][i])

    print(chunks[index_number])