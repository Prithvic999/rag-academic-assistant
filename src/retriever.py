import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# READ PDF

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        # REMOVE EXTRA LINE BREAKS
        extracted = extracted.replace("\n", " ")

        # REMOVE MULTIPLE SPACES
        extracted = re.sub(r'\s+', ' ', extracted)

        text += extracted

    return text

# CHUNK TEXT

def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks

# LOAD PDF

pdf_text = extract_text_from_pdf("data/sample.pdf")

# CREATE CHUNKS

chunks = chunk_text(pdf_text)

# LOAD EMBEDDING MODEL

model = SentenceTransformer('all-MiniLM-L6-v2')

# CREATE EMBEDDINGS

embeddings = model.encode(chunks)

# CREATE FAISS INDEX

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# USER QUERY

query = input("Enter your question: ")

# CONVERT QUERY TO EMBEDDING

query_embedding = model.encode([query])

# SEARCH TOP 3 SIMILAR CHUNKS

distances, indices = index.search(np.array(query_embedding), k=3)

# PRINT RESULTS

print("\nTop Relevant Chunks:\n")

for i in indices[0]:
    print(chunks[i])
    print("\n" + "="*80 + "\n")