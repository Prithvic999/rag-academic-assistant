import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from pypdf import PdfReader
import os
from google import genai
from datetime import datetime

# -----------------------------------
# PAGE CONFIG & STATE
# -----------------------------------
st.set_page_config(page_title="RAG Academic Assistant", page_icon="📚", layout="wide")

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks_metadata" not in st.session_state:
    st.session_state.chunks_metadata = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# CACHE MODELS
# -----------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

embedding_model, reranker_model = load_models()

# -----------------------------------
# CORE PROCESSING
# -----------------------------------
def get_pdf_pages_with_metadata(pdf_docs):
    pages_data = []
    for pdf in pdf_docs:
        filename = pdf.name
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_data.append({"text": text, "page_num": i + 1, "source": filename})
    return pages_data

def get_text_chunks_with_metadata(pages_data, chunk_size=500, chunk_overlap=100):
    chunks = []
    for data in pages_data:
        text = data["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "chunk_text": text[start:end],
                "page_num": data["page_num"],
                "source": data["source"]
            })
            start = end - chunk_overlap
    return chunks

def build_vector_store(chunks_metadata):
    text_list = [chunk["chunk_text"] for chunk in chunks_metadata]
    embeddings = embedding_model.encode(text_list)
    embeddings = np.array(embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# -----------------------------------
# NEW: EXPORT FUNCTION
# -----------------------------------
def format_chat_for_export(messages):
    """Formats the chat history into a clean, readable text string for downloading."""
    if not messages:
        return "No conversation history to export."
        
    export_text = f"--- Academic RAG Session Export ---\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for msg in messages:
        role = "Prithvi" if msg["role"] == "user" else "AI Assistant"
        export_text += f"{role}:\n{msg['content']}\n\n"
        
        # Include the citations in the export if they exist
        if "sources" in msg and msg["sources"]:
            export_text += "Sources Used:\n"
            for idx, src in enumerate(msg["sources"]):
                export_text += f"  [{idx+1}] {src['source']} (Page {src['page']}) - Score: {src['score']:.2f}\n"
            export_text += "\n"
            
        export_text += "-" * 40 + "\n\n"
        
    return export_text

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    # NEW: Export and Clear controls grouped together
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        # Create the downloadable text file dynamically
        chat_export_string = format_chat_for_export(st.session_state.messages)
        st.download_button(
            label="💾 Export Notes",
            data=chat_export_string,
            file_name=f"RAG_Research_Notes_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
    st.divider()
    st.header("📄 Upload Documents")
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Process Documents"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting pages, chunking, and indexing..."):
                pages_data = get_pdf_pages_with_metadata(pdf_docs)
                chunks_metadata = get_text_chunks_with_metadata(pages_data, chunk_size=500, chunk_overlap=100)
                st.session_state.index = build_vector_store(chunks_metadata)
                st.session_state.chunks_metadata = chunks_metadata
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks_metadata)} chunks.")

# -----------------------------------
# MAIN CHAT UI & RERANKING PIPELINE
# -----------------------------------
st.title("📚 AI Academic Assistant")
st.caption("Research-Grade RAG: Two-Stage Retrieval (FAISS + Cross-Encoder Reranking)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 View Reranked Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**{idx+1}. {src['source']} (Page {src['page']})** - *Cross-Encoder Score: {src['score']:.2f}*")
                    st.caption(src["text"])
                    st.divider()

if prompt_text := st.chat_input("Ask a question about your documents..."):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
        st.stop()
    if st.session_state.index is None:
        st.error("Please upload and process a PDF document first.")
        st.stop()
        
    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    with st.chat_message("assistant"):
        with st.spinner("Retrieving via FAISS and Reranking with Cross-Encoder..."):
            
            question_embedding = np.array(embedding_model.encode([prompt_text])).astype('float32')
            _, indices = st.session_state.index.search(question_embedding, k=10)
            
            broad_chunks = []
            for i in indices[0]:
                if i < len(st.session_state.chunks_metadata):
                    broad_chunks.append(st.session_state.chunks_metadata[i])
            
            cross_input = [[prompt_text, chunk["chunk_text"]] for chunk in broad_chunks]
            rerank_scores = reranker_model.predict(cross_input)
            
            for i in range(len(broad_chunks)):
                broad_chunks[i]["rerank_score"] = rerank_scores[i]
                
            ranked_chunks = sorted(broad_chunks, key=lambda x: x["rerank_score"], reverse=True)
            best_chunks = ranked_chunks[:4]
            
            retrieved_sources = []
            context_text = ""
            for chunk in best_chunks:
                retrieved_sources.append({
                    "source": chunk["source"],
                    "page": chunk["page_num"],
                    "text": chunk["chunk_text"],
                    "score": chunk["rerank_score"]
                })
                context_text += f"[DOCUMENT: {chunk['source']} | PAGE: {chunk['page_num']}]\n{chunk['chunk_text']}\n\n---\n\n"
            
            chat_history_str = ""
            for msg in st.session_state.messages[-5:-1]: 
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history_str += f"{role}: {msg['content']}\n\n"

            full_prompt = f"""
            You are an expert academic AI assistant. Answer the user's question using ONLY the provided context.
            
            CRITICAL RULES FOR CITATIONS:
            Every time you state a fact, you MUST include an inline citation showing exactly where it came from based on the brackets provided in the context. Format your citations like this: (Filename.pdf, Page X).
            
            RULES:
            1. Base your answer strictly on the CURRENT CONTEXT.
            2. Do not hallucinate.
            3. If the answer is not in the context, explicitly state: "I cannot find a complete answer to this in the provided documents."

            CURRENT CONTEXT:
            {context_text}
            
            PREVIOUS CONVERSATION HISTORY:
            {chat_history_str}

            LATEST QUESTION:
            {prompt_text}
            """

            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model='gemini-2.5-flash', contents=full_prompt)
                answer = response.text
            except Exception as api_err:
                answer = f"API Error: {api_err}"
                
            st.markdown(answer)
            
            with st.expander("🔍 View Reranked Sources"):
                for idx, src in enumerate(retrieved_sources):
                    st.markdown(f"**{idx+1}. {src['source']} (Page {src['page']})** - *Cross-Encoder Score: {src['score']:.2f}*")
                    st.caption(src["text"])
                    st.divider()
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_sources})