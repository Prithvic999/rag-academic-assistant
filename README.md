# 📚 AI Academic Assistant

Research-Grade Retrieval-Augmented Generation (RAG) Academic Assistant built using FAISS, Sentence Transformers, Cross-Encoder Reranking, Gemini API, and Streamlit.

---

## 🚀 Features

- 📄 Multi-PDF Upload Support
- 🔍 Semantic Search using FAISS
- 🧠 Sentence Transformer Embeddings
- 🎯 Cross-Encoder Reranking
- 🤖 Gemini-based Answer Generation
- 📚 Citation-based Grounded Answers
- 💬 Chat History Support
- 💾 Chat Export & Save
- 🌐 Streamlit Interactive UI
- 📑 Research-Paper-Based RAG Architecture

---

## 🏗️ System Architecture

```text
User Question
      ↓
Sentence Embedding
      ↓
FAISS Semantic Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Context Augmentation
      ↓
Gemini LLM Generation
      ↓
Grounded Answer + Citations

🛠️ Tech Stack
Python
Streamlit
FAISS
Sentence Transformers
Cross-Encoder Models
Gemini API
NumPy
PyPDF

## 🔗 Live Demo
https://rag-academic-assistant-qtjuvy2p7rawvzkrq6unge.streamlit.app/

📂 Project Structure
rag-academic-assistant/
│
├── app/
│   └── app.py
│
├── src/
│   ├── pdf_reader.py
│   ├── text_chunker.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── chatbot.py
│   └── rag_pipeline.py
│
├── data/
│   └── sample.pdf
│
├── vectorstore/
│   ├── faiss_index.index
│   └── chunks.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
