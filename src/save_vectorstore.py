from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# -----------------------------------
# STEP 1: PDF TEXT EXTRACTION
# -----------------------------------

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            # Clean text
            clean_text = " ".join(page_text.split())

            text += clean_text + " "

    return text


# -----------------------------------
# STEP 2: LOAD PDF
# -----------------------------------

pdf_path = "data/sample.pdf"

pdf_text = extract_text_from_pdf(pdf_path)

print("PDF Loaded Successfully")


# -----------------------------------
# STEP 3: SMART CHUNKING
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_text(pdf_text)

print("Total Chunks:", len(chunks))


# -----------------------------------
# STEP 4: LOAD EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# STEP 5: CREATE EMBEDDINGS
# -----------------------------------

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embeddings Created")


# -----------------------------------
# STEP 6: CREATE VECTORSTORE FOLDER
# -----------------------------------

os.makedirs("vectorstore", exist_ok=True)


# -----------------------------------
# STEP 7: CREATE FAISS INDEX
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS Index Created")


# -----------------------------------
# STEP 8: SAVE FAISS INDEX
# -----------------------------------

faiss.write_index(
    index,
    "vectorstore/faiss_index.index"
)


# -----------------------------------
# STEP 9: SAVE CHUNKS
# -----------------------------------

with open("vectorstore/chunks.pkl", "wb") as f:

    pickle.dump(chunks, f)


print("\nVector Database Saved Successfully!")