import pymupdf


def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    full_text = ""

    for page in document:
        full_text += page.get_text() + "\n"

    return full_text


def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


pdf_path = "data/documents/AI_Fundamentals.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = create_chunks(text)

print("Total characters:", len(text))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)