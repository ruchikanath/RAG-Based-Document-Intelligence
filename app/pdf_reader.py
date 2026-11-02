import pymupdf


pdf_path = "data/documents/AI_Fundamentals.pdf"

document = pymupdf.open(pdf_path)

full_text = ""

for page_number, page in enumerate(document):
    text = page.get_text()

    full_text += text + "\n"

print("Number of pages:", len(document))
print("Total characters:", len(full_text))

print("\n--- FIRST 3000 CHARACTERS ---\n")
print(full_text[:3000])