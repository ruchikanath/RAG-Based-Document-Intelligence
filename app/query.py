import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# ==========================================
# PATHS
# ==========================================

INDEX_PATH = "data/vector_store/index.faiss"
CHUNKS_PATH = "data/vector_store/chunks.pkl"


# ==========================================
# LOAD FAISS INDEX
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print("Number of vectors:", index.ntotal)


# ==========================================
# LOAD CHUNKS + METADATA
# ==========================================

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)

print("Number of chunks:", len(chunks))


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# RETRIEVER
# ==========================================

def retrieve(query, k=20):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    query_lower = query.lower()

    # Important words from the query
    query_words = set(
        word.strip(".,?!:;()[]{}")
        for word in query_lower.split()
        if len(word.strip(".,?!:;()[]{}")) > 2
    )

    for i, index_number in enumerate(indices[0]):

        if index_number == -1:
            continue

        result = chunks[index_number].copy()

        result["distance"] = float(
            distances[0][i]
        )

        # ==================================
        # KEYWORD MATCH SCORE
        # ==================================

        text_lower = result["text"].lower()

        matched_words = [
            word
            for word in query_words
            if word in text_lower
        ]

        keyword_score = len(matched_words)

        result["keyword_score"] = keyword_score

        # ==================================
        # COMBINED SCORE
        # ==================================
        #
        # Lower FAISS distance = better
        # More keyword matches = better
        #

        result["combined_score"] = (
            result["distance"]
            - (keyword_score * 0.08)
        )

        results.append(result)

    # Sort using combined score
    results.sort(
        key=lambda x: x["combined_score"]
    )

    return results


# ==========================================
# REMOVE DUPLICATE DOCUMENT/PAGE RESULTS
# ==========================================

def remove_duplicates(results):

    unique_results = []
    seen = set()

    for result in results:

        key = (
            result["source"],
            result["page"]
        )

        if key not in seen:

            seen.add(key)
            unique_results.append(result)

    return unique_results


# ==========================================
# GROUP RESULTS BY DOCUMENT
# ==========================================

def group_by_document(results):

    grouped = {}

    for result in results:

        source = result["source"]

        if source not in grouped:
            grouped[source] = []

        grouped[source].append(result)

    return grouped


# ==========================================
# ASK QUESTION
# ==========================================

query = input(
    "\nAsk a question about your AI materials: "
)


# ==========================================
# RETRIEVE
# ==========================================

results = retrieve(
    query,
    k=20
)

print("\n================================")
print("INITIAL RETRIEVAL")
print("================================")

print(
    f"Retrieved {len(results)} candidate chunks."
)


# ==========================================
# RELEVANCE THRESHOLD
# ==========================================

THRESHOLD = 1.20

relevant_results = [
    result
    for result in results
    if result["distance"] <= THRESHOLD
]


# ==========================================
# NO RELEVANT INFORMATION
# ==========================================

if not relevant_results:

    print("\n================================")
    print("NO RELEVANT INFORMATION FOUND")
    print("================================")

    print(
        "I could not find this information "
        "in the provided training materials."
    )

    exit()


# ==========================================
# REMOVE DUPLICATE PAGES
# ==========================================

relevant_results = remove_duplicates(
    relevant_results
)


# ==========================================
# LIMIT CONTEXT
# ==========================================
#
# We retrieved 20 candidates, but don't
# necessarily send all 20 to the LLM.
#

MAX_CONTEXT_RESULTS = 8

context_results = relevant_results[
    :MAX_CONTEXT_RESULTS
]


# ==========================================
# DISPLAY RETRIEVED DOCUMENTS
# ==========================================

print("\n================================")
print("RETRIEVED DOCUMENTS")
print("================================")


for i, result in enumerate(
    context_results
):

    print(
        f"\nResult {i + 1}"
    )

    print(
        f"Source: {result['source']}"
    )

    print(
        f"Page: {result['page']}"
    )

    print(
        f"Section: {result.get('section', 'Unknown')}"
    )

    print(
        f"Distance: {result['distance']:.4f}"
    )

    print(
        f"Keyword matches: "
        f"{result['keyword_score']}"
    )

    print(
        f"Text: {result['text'][:700]}"
    )


# ==========================================
# BUILD CONTEXT
# ==========================================

context = ""

for i, result in enumerate(
    context_results
):

    context += f"""
SOURCE {i + 1}

Document:
{result['source']}

Page:
{result['page']}

Section:
{result.get('section', 'Unknown')}

Content:
{result['text']}

--------------------------------
"""


# ==========================================
# CREATE PROMPT
# ==========================================

prompt = f"""
You are an AI Knowledge Assistant for
an Education and Training Department.

Your task is to answer questions using ONLY
the provided training-material context.

IMPORTANT RULES:

1. Do NOT use outside knowledge.

2. Do NOT invent information.

3. If the answer is not present in the context,
say exactly:

"I could not find this information in the
provided training materials."

4. Always identify where the information was found.

5. Mention the document name.

6. Mention the page number.

7. Mention the section when available.

8. If multiple documents contain the topic,
mention all genuinely relevant documents.

9. Do not treat a book/reference-list mention
as evidence that the topic is actually taught,
unless the context clearly shows that it is
part of the course content.

10. If the question asks "where", focus on:
document, page, section, session and topic.

11. If the question asks "which documents",
list the documents that actually contain
the requested topic.

12. If the question asks "what" or "explain",
provide a concise explanation based only
on the retrieved training material.

13. Do not add information that is not present
in the context.

14. Do not claim that a topic is present in a
document unless the retrieved content supports
that claim.

Return the answer in this format:

ANSWER:
<answer>

WHERE FOUND:
- Document: <document>
- Page: <page>
- Section: <section>

RELEVANT CONTENT:
<useful information from the context>

RELATED TOPICS:
- <topic>
- <topic>

TRAINING MATERIALS:
- <document> - Page <page>
- <document> - Page <page>


CONTEXT:
================================

{context}

================================

QUESTION:
{query}
"""


# ==========================================
# CALL OLLAMA
# ==========================================

print("\nGenerating answer...")

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
# DISPLAY ANSWER
# ==========================================

answer = response[
    "message"
]["content"]


print("\n================================")
print("RAG ANSWER")
print("================================")

print(answer)


# ==========================================
# DISPLAY SOURCES
# ==========================================

print("\n================================")
print("SOURCES")
print("================================")

shown_sources = set()

for result in context_results:

    key = (
        result["source"],
        result["page"]
    )

    if key not in shown_sources:

        shown_sources.add(key)

        print(
            f"- {result['source']} "
            f"(Page {result['page']})"
        )