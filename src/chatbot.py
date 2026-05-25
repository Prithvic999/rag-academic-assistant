from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# -------------------------------
# LOAD CHUNKS
# -------------------------------

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Chunks Loaded:", len(chunks))

# -------------------------------
# LOAD FAISS INDEX
# -------------------------------

index = faiss.read_index("vectorstore/faiss_index.index")

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("RAG Chatbot Ready!")

# -------------------------------
# QUESTION LOOP
# -------------------------------

while True:

    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        break

    # -------------------------------
    # CREATE QUESTION EMBEDDING
    # -------------------------------

    question_embedding = embedding_model.encode([question])

    # -------------------------------
    # SEARCH TOP CHUNKS
    # -------------------------------

    distances, indices = index.search(
        np.array(question_embedding),
        k=2
    )

    # -------------------------------
    # COMBINE RETRIEVED CHUNKS
    # -------------------------------

    retrieved_text = ""

    for i in indices[0]:

        retrieved_text += chunks[i] + " "

    # -------------------------------
    # CLEAN TEXT
    # -------------------------------

    retrieved_text = " ".join(retrieved_text.split())

    # -------------------------------
    # PRINT ANSWER
    # -------------------------------

    print("\nFINAL ANSWER:\n")

    sentences = retrieved_text.split(".")

    short_answer = ""

    for sentence in sentences[:5]:
        short_answer += sentence + ". "

    print(short_answer)

    print("\n" + "=" * 80 + "\n")