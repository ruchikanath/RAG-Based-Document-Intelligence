import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# ==========================================
# 1. LOAD PDF
# ==========================================

pdf_path = "data/documents/AI_Fundamentals.pdf"

document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"


# ==========================================
# 2. CREATE CHUNKS
# ==========================================

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


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 4. CREATE DOCUMENT EMBEDDINGS
# ==========================================

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")


# ==========================================
# 5. CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored:", index.ntotal)


# ==========================================
# 6. RETRIEVER
# ==========================================

def retrieve(query, k=3):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for i, index_number in enumerate(indices[0]):

        results.append({
            "text": chunks[index_number],
            "distance": distances[0][i]
        })

    return results


# ==========================================
# 7. ASK USER QUESTION
# ==========================================

query = input("\nAsk a question about your AI material: ")


# ==========================================
# 8. RETRIEVE RELEVANT CHUNKS
# ==========================================

results = retrieve(query, k=3)


# ==========================================
# 9. BUILD CONTEXT
# ==========================================

context = ""

for result in results:

    context += result["text"]
    context += "\n\n"


# ==========================================
# 10. CREATE RAG PROMPT
# ==========================================

prompt = f"""
You are an AI Knowledge Assistant for an Education and Training department.

Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I could not find this information in the provided training material."

Do not make up information.

Context:
--------------------
{context}
--------------------

Question:
{query}

Answer:
"""


# ==========================================
# 11. SEND TO OLLAMA
# ==========================================

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# 12. DISPLAY ANSWER
# ==========================================

answer = response["message"]["content"]

print("\n==============================")
print("RAG ANSWER")
print("==============================")

print(answer)