from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# STEP 1: READ PDF
# -------------------------------

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            clean_text = " ".join(page_text.split())
            text += clean_text

    return text


# -------------------------------
# STEP 2: LOAD PDF
# -------------------------------

pdf_text = extract_text_from_pdf("data/sample.pdf")


# -------------------------------
# STEP 3: CHUNKING
# -------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(pdf_text)

print("Total Chunks:", len(chunks))


# -------------------------------
# STEP 4: EMBEDDINGS
# -------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

embedding_array = np.array(embeddings)


# -------------------------------
# STEP 5: CREATE FAISS INDEX
# -------------------------------

dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_array)

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

    # Retrieve top 3 chunks
    distances, indices = index.search(
        np.array(question_embedding),
        k=2
    )

    print("\nTOP RELEVANT CHUNKS:\n")

    for i in indices[0]:

        print(chunks[i])

        print("\n" + "="*80 + "\n")