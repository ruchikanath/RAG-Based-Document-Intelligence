import os
import pickle
import re
import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ==========================================
# PATHS
# ==========================================

DOCUMENT_FOLDER = "data/documents"
VECTOR_STORE_FOLDER = "data/vector_store"

INDEX_PATH = os.path.join(
    VECTOR_STORE_FOLDER,
    "index.faiss"
)

CHUNKS_PATH = os.path.join(
    VECTOR_STORE_FOLDER,
    "chunks.pkl"
)


# ==========================================
# CREATE VECTOR STORE FOLDER
# ==========================================

os.makedirs(
    VECTOR_STORE_FOLDER,
    exist_ok=True
)


# ==========================================
# CHUNKING CONFIGURATION
# ==========================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ==========================================
# CREATE TEXT SPLITTER
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

# ==========================================
# CHUNKING FUNCTION
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
# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# PROCESS ALL PDF DOCUMENTS
# ==========================================

all_chunks = []


for filename in os.listdir(DOCUMENT_FOLDER):

    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        DOCUMENT_FOLDER,
        filename
    )

    print(f"\nProcessing: {filename}")

    document = pymupdf.open(pdf_path)


    # ======================================
    # PROCESS EACH PAGE
    # ======================================

for page_number, page in enumerate(document):

    text = page.get_text()

    # ------------------------------------------
    # CLEAN TEXT
    # ------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # ------------------------------------------
    # DETECT SECTION / SESSION
    # ------------------------------------------

    section = "Unknown"

    for line in lines:

        # Detect Session 1, Session 1 & 2, Session 10, etc.
        if re.search(
            r"Session\s+\d+(\s*&\s*\d+)?",
            line,
            re.IGNORECASE
        ):
            section = line
            break

    # ------------------------------------------
    # CREATE CHUNKS
    # ------------------------------------------

    page_chunks = create_chunks(text)

    # ------------------------------------------
    # STORE METADATA
    # ------------------------------------------

    for chunk in page_chunks:

        if chunk.strip():

            all_chunks.append({
                "text": chunk,
                "source": filename,
                "page": page_number + 1,
                "course": "PGCP-AI",
                "year": "2026",
                "document_id": filename,
                "section": section
            })


    document.close()


# ==========================================
# PROCESSING SUMMARY
# ==========================================

print("\n================================")
print("DOCUMENT PROCESSING COMPLETE")
print("================================")

print(
    "Total chunks:",
    len(all_chunks)
)


# ==========================================
# CHECK IF DOCUMENTS WERE FOUND
# ==========================================

if not all_chunks:

    print("ERROR: No chunks were created.")

    exit()


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

texts = [
    chunk["text"]
    for chunk in all_chunks
]

print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

embeddings = np.array(
    embeddings
).astype("float32")


print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================
# CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)


print(
    "Vectors stored:",
    index.ntotal
)


# ==========================================
# SAVE FAISS INDEX
# ==========================================

faiss.write_index(
    index,
    INDEX_PATH
)


# ==========================================
# SAVE CHUNKS + METADATA
# ==========================================

with open(
    CHUNKS_PATH,
    "wb"
) as file:

    pickle.dump(
        all_chunks,
        file
    )


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n================================")
print("VECTOR STORE SAVED")
print("================================")

print(
    "FAISS index:",
    INDEX_PATH
)

print(
    "Chunks:",
    CHUNKS_PATH
)