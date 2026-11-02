import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Read PDF
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

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Create document embeddings
# -----------------------------

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")


# -----------------------------
# 5. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# -----------------------------
# 6. Retriever function
# -----------------------------

def retrieve(query, k=3):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for i, index_number in enumerate(indices[0]):

        results.append({
            "chunk": chunks[index_number],
            "distance": distances[0][i]
        })

    return results


# -----------------------------
# 7. Test Retriever
# -----------------------------

query = input("\nAsk a question about your document: ")

results = retrieve(query, k=3)


print("\n==============================")
print("RETRIEVED RESULTS")
print("==============================")


for i, result in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Distance:", result["distance"])

    print("Text:")

    print(result["chunk"])