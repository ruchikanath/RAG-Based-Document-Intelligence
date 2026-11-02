import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# PATHS
# ==========================================

INDEX_PATH = "data/vector_store/index.faiss"
CHUNKS_PATH = "data/vector_store/chunks.pkl"
EVALUATION_PATH = "evaluation/evaluation.json"


# ==========================================
# LOAD EVALUATION DATA
# ==========================================

with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)


# ==========================================
# LOAD FAISS
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print("Number of vectors:", index.ntotal)


# ==========================================
# LOAD CHUNKS
# ==========================================

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)

print("Number of chunks:", len(chunks))


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# RETRIEVAL FUNCTION
# ==========================================

def retrieve(question, k=3):

    query_embedding = model.encode([question])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for i, index_number in enumerate(indices[0]):

        results.append({
            "source": chunks[index_number]["source"],
            "page": chunks[index_number]["page"],
            "text": chunks[index_number]["text"],
            "distance": float(distances[0][i])
        })

    return results


# ==========================================
# EVALUATION
# ==========================================

print("\n================================")
print("RAG EVALUATION")
print("================================")

total = len(evaluation_data)
passed = 0


for number, item in enumerate(evaluation_data, start=1):

    question = item["question"]
    expected_source = item["expected_source"]
    should_answer = item["should_answer"]

    results = retrieve(question, k=3)

    # --------------------------------------
    # Check whether expected information exists
    # --------------------------------------

    if not should_answer:

        # For questions that should NOT be answerable,
        # we simply report the retrieved sources.

        print(f"\nQuestion {number}: {question}")

        print("Expected: No answer")

        print("Retrieved:")

        for result in results:
            print(
                f"  - {result['source']} "
                f"(Page {result['page']})"
            )

        print("Result: MANUAL CHECK")

        continue


    # --------------------------------------
    # Check expected source
    # --------------------------------------

    retrieved_sources = [
        result["source"]
        for result in results
    ]

    if expected_source in retrieved_sources:

        print(f"\nQuestion {number}: {question}")

        print(
            f"Expected source: {expected_source}"
        )

        print(
            "Retrieved source:",
            expected_source
        )

        print("Result: PASS")

        passed += 1

    else:

        print(f"\nQuestion {number}: {question}")

        print(
            f"Expected source: {expected_source}"
        )

        print(
            "Retrieved sources:",
            retrieved_sources
        )

        print("Result: FAIL")


# ==========================================
# FINAL RESULT
# ==========================================

print("\n================================")
print("EVALUATION RESULT")
print("================================")

print("Total answerable questions:", 
      sum(item["should_answer"] for item in evaluation_data))

print("Passed:", passed)

answerable_questions = sum(
    item["should_answer"]
    for item in evaluation_data
)

if answerable_questions > 0:

    accuracy = (
        passed / answerable_questions
    ) * 100

    print(
        f"Retrieval Accuracy: {accuracy:.2f}%"
    )