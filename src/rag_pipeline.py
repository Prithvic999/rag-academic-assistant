from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# -------------------------------
# STEP 1: READ PDF
# -------------------------------

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            clean_text = page_text.replace("\n", " ")
            text += clean_text

    return text


# -------------------------------
# STEP 2: CHUNKING
# -------------------------------

pdf_text = extract_text_from_pdf("data/sample.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(pdf_text)

print("Total Chunks:", len(chunks))


# -------------------------------
# STEP 3: EMBEDDINGS
# -------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

embedding_array = np.array(embeddings)


# -------------------------------
# STEP 4: FAISS VECTOR STORE
# -------------------------------

dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_array)


# -------------------------------
# STEP 5: LOAD FLAN-T5
# -------------------------------

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("RAG System Ready!")


# -------------------------------
# STEP 6: QUESTION LOOP
# -------------------------------

while True:

    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        break

    # Convert question to embedding
    question_embedding = embedding_model.encode([question])

    # Search similar chunks
    distances, indices = index.search(
        np.array(question_embedding),
        k=3
    )

    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])

    context = " ".join(retrieved_chunks)

    # Prompt
    print("\nTOP RETRIEVED ANSWER:\n")

for idx, chunk in enumerate(retrieved_chunks, 1):

    print(f"\nResult {idx}:\n")
    print(chunk)
    print("\n" + "="*80)
    print(answer)