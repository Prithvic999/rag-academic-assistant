import os
from datetime import datetime
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv

# -----------------------------------
# INITIALIZATION & SECURITY
# -----------------------------------
# Load environment variables from .env file (for local testing)
# Streamlit Cloud will automatically bypass this and use the Secrets vault
load_dotenv()

# Page config MUST be the first Streamlit command
st.set_page_config(page_title="RAG Academic Assistant", page_icon="📚", layout="wide")

# -----------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks_metadata" not in st.session_state:
    st.session_state.chunks_metadata = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# CACHE AI MODELS
# -----------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

embedding_model, reranker_model = load_models()

# -----------------------------------
# CORE RAG PROCESSING FUNCTIONS
# -----------------------------------
def get_pdf_pages_with_metadata(pdf_docs):
    """Extracts text and page numbers from uploaded PDFs."""
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
    """Slices text into readable chunks with overlap for context."""
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
    """Embeds the chunks and loads them into a FAISS vector database."""
    text_list = [chunk["chunk_text"] for chunk in chunks_metadata]
    embeddings = embedding_model.encode(text_list)
    embeddings = np.array(embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def format_chat_for_export(messages):
    """Formats the chat history into a clean text string for downloading."""
    if not messages:
        return "No conversation history to export."
        
    export_text = f"--- Academic RAG Session Export ---\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for msg in messages:
        role = "Prithvi" if msg["role"] == "user" else "AI Assistant"
        export_text += f"{role}:\n{msg['content']}\n\n"
        
        # Include citations in the export if they exist
        if "sources" in msg and msg["sources"]:
            export_text += "Sources Used:\n"
            for idx, src in enumerate(msg["sources"]):
                export_text += f"  [{idx+1}] {src['source']} (Page {src['page']}) - Score: {src['score']:.2f}\n"
            export_text += "\n"
            
        export_text += "-" * 40 + "\n\n"
        
    return export_text

# -----------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Securely retrieve the API key without asking the user
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        st.success("✅ API Key Secured")
    else:
        st.error("❌ API Key Missing. Check .env or Secrets.")
    
    st.divider()
    
    # Export and Clear controls grouped together
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
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
                st.success(f"Successfully indexed {len(chunks_metadata)} chunks!")
# --- ADD THIS TO THE VERY BOTTOM OF YOUR SIDEBAR ---
    st.divider()
    st.header("🛠️ One-Click Tools")
    
    # Only show these tools IF a document has been uploaded and processed
    if st.session_state.chunks_metadata:
        
        # 4. PDF Summarizer
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Summarizing document..."):
                full_text = "\n".join([chunk["chunk_text"] for chunk in st.session_state.chunks_metadata])
                prompt = f"Analyze the following academic text. Provide: 1. An Abstract summary, 2. Key findings, 3. Conclusion.\n\nTEXT:\n{full_text}"
                
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                
                # Add the response to the chat interface
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
                
        # 5. Question Generator
        if st.button("❓ Generate Viva Questions", use_container_width=True):
            with st.spinner("Generating exam questions..."):
                full_text = "\n".join([chunk["chunk_text"] for chunk in st.session_state.chunks_metadata[:15]]) # Limit to first 15 chunks to save time
                prompt = f"Act as a university professor. Based on this text, generate: 5 Short Answer Questions, 5 Long Essay Questions, and 5 tricky Viva/Oral Exam Questions.\n\nTEXT:\n{full_text}"
                
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()

        # 6. Paper Analyzer
        if st.button("🔬 Analyze Research Paper", use_container_width=True):
            with st.spinner("Analyzing research structure..."):
                full_text = "\n".join([chunk["chunk_text"] for chunk in st.session_state.chunks_metadata])
                prompt = f"Analyze this research paper and extract the following exactly: Title, Objective, Methodology, Dataset (if any), Results, Limitations, and Future Work.\n\nTEXT:\n{full_text}"
                
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
# -----------------------------------
# MAIN CHAT UI & RERANKING PIPELINE
# -----------------------------------
st.title("📚 AI Academic Assistant")
st.caption("Research-Grade RAG: Two-Stage Retrieval (FAISS + Cross-Encoder Reranking)")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 View Reranked Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**{idx+1}. {src['source']} (Page {src['page']})** - *Score: {src['score']:.2f}*")
                    st.caption(src["text"])
                    st.divider()

# Handle new user input
if prompt_text := st.chat_input("Ask a question about your documents..."):
    
    # Safety Checks
    if not api_key:
        st.error("Please configure your Gemini API Key in the environment variables.")
        st.stop()
    if st.session_state.index is None:
        st.error("Please upload and process a PDF document first before asking questions.")
        st.stop()
        
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving via FAISS and Reranking with Cross-Encoder..."):
            
            # Step 1: Broad Retrieval (FAISS)
            question_embedding = np.array(embedding_model.encode([prompt_text])).astype('float32')
            _, indices = st.session_state.index.search(question_embedding, k=10)
            
            broad_chunks = []
            for i in indices[0]:
                if i < len(st.session_state.chunks_metadata):
                    broad_chunks.append(st.session_state.chunks_metadata[i])
            
            # Step 2: Precise Reranking (Cross-Encoder)
            cross_input = [[prompt_text, chunk["chunk_text"]] for chunk in broad_chunks]
            rerank_scores = reranker_model.predict(cross_input)
            
            for i in range(len(broad_chunks)):
                broad_chunks[i]["rerank_score"] = rerank_scores[i]
                
            ranked_chunks = sorted(broad_chunks, key=lambda x: x["rerank_score"], reverse=True)
            best_chunks = ranked_chunks[:4] # Take the top 4 most relevant chunks
            
            # Step 3: Format Context for Gemini
            retrieved_sources = []
            context_text = ""
            for chunk in best_chunks:
                retrieved_sources.append({
                    "source": chunk["source"],
                    "page": chunk["page_num"],
                    "text": chunk["chunk_text"],
                    "score": float(chunk["rerank_score"])
                })
                context_text += f"[DOCUMENT: {chunk['source']} | PAGE: {chunk['page_num']}]\n{chunk['chunk_text']}\n\n---\n\n"
            
            # Grab recent chat history for conversational memory
            chat_history_str = ""
            for msg in st.session_state.messages[-5:-1]: 
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history_str += f"{role}: {msg['content']}\n\n"

            # Step 4: Final Prompt Construction
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

            # Step 5: Call Google Gemini API
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=full_prompt
                )
                answer = response.text
            except Exception as api_err:
                answer = f"API Error: {api_err}"
                
            # Step 6: Render Output
            st.markdown(answer)
            
            with st.expander("🔍 View Reranked Sources"):
                for idx, src in enumerate(retrieved_sources):
                    st.markdown(f"**{idx+1}. {src['source']} (Page {src['page']})** - *Score: {src['score']:.2f}*")
                    st.caption(src["text"])
                    st.divider()
            
            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_sources})
