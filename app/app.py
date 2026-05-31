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
load_dotenv()

st.set_page_config(page_title="RAG Academic Assistant", page_icon="📚", layout="wide")

# Custom CSS for a professional, polished look
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1 { text-align: center; font-weight: 700; color: #1E3A8A; }
    div[data-testid="stMetric"] {
        background-color: #F3F4F6;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #E5E7EB;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0px 8px 8px 0px;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks_metadata" not in st.session_state:
    st.session_state.chunks_metadata = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# State variables for tabbed tools
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "viva_output" not in st.session_state:
    st.session_state.viva_output = ""
if "analysis_output" not in st.session_state:
    st.session_state.analysis_output = ""

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

def format_chat_for_export(messages):
    if not messages:
        return "No conversation history to export."
    export_text = f"--- Academic RAG Session Export ---\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    for msg in messages:
        role = "Kushagra" if msg["role"] == "user" else "AI Assistant"
        export_text += f"{role}:\n{msg['content']}\n\n"
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
    st.title("⚙️ AI Assistant")
    st.caption("Advanced Retrieval-Augmented Generation")
    
    st.markdown("""
    ### Features
    ✅ Multi-PDF Support  
    ✅ Semantic Vector Search  
    ✅ Cross-Encoder Reranking  
    ✅ Context-Grounded Answers  
    ✅ Auto-Paper Analysis  
    """)
    st.divider()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        chat_export_string = format_chat_for_export(st.session_state.messages)
        st.download_button(
            label="💾 Export",
            data=chat_export_string,
            file_name=f"RAG_Notes_{datetime.now().strftime('%Y%m%d')}.txt",
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
            # Replaced spinner with a modern status container
            with st.status("📚 Processing Academic Documents...", expanded=True) as status:
                st.write("Extracting text and formatting pages...")
                pages_data = get_pdf_pages_with_metadata(pdf_docs)
                
                st.write("Chunking text for context windows...")
                chunks_metadata = get_text_chunks_with_metadata(pages_data, chunk_size=500, chunk_overlap=100)
                
                st.write("Generating mathematical vector embeddings...")
                st.session_state.index = build_vector_store(chunks_metadata)
                st.session_state.chunks_metadata = chunks_metadata
                
                # Reset tool outputs on new document load
                st.session_state.summary_output = ""
                st.session_state.viva_output = ""
                st.session_state.analysis_output = ""
                st.session_state.messages = []
                
                status.update(label="✅ Documents successfully indexed!", state="complete", expanded=False)
                st.success(f"Processed {len(pdf_docs)} documents into {len(chunks_metadata)} searchable chunks.")

    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Built by Kushagra Chaudhary<br>
        Powered by FAISS • Gemini • Sentence Transformers
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# MAIN UI: METRICS & TABS
# -----------------------------------
st.title("📚 Academic RAG Assistant")

# Dashboard Metrics
if st.session_state.chunks_metadata:
    col1, col2, col3 = st.columns(3)
    # Get unique document names
    doc_names = list(set([chunk["source"] for chunk in st.session_state.chunks_metadata]))
    col1.metric("📄 PDFs Loaded", len(doc_names))
    col2.metric("🧩 Knowledge Chunks", len(st.session_state.chunks_metadata))
    col3.metric("🧠 Active Index", "FAISS Ready")
    st.divider()

# Create structured tabs for the workspace
tab1, tab2, tab3, tab4 = st.tabs(["💬 Interactive Chat", "📝 Document Summary", "❓ Viva & Exam Prep", "🔬 Paper Analysis"])

# --- TAB 1: CHAT INTERFACE ---
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, src in enumerate(message["sources"]):
                        # Upgraded Source Display
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {idx+1}:</strong> {src['source']} (Page {src['page']})<br>
                            <span style="color: #6B7280; font-size: 0.9em;">Relevance Score: {src['score']:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f'"{src["text"]}"')

    if prompt_text := st.chat_input("Ask a question about your uploaded documents..."):
        if not api_key:
            st.error("Please configure your Gemini API Key in the environment variables.")
            st.stop()
        if st.session_state.index is None:
            st.error("Please upload and process a PDF document first.")
            st.stop()
            
        with st.chat_message("user"):
            st.markdown(prompt_text)
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving via FAISS and Reranking..."):
                question_embedding = np.array(embedding_model.encode([prompt_text])).astype('float32')
                _, indices = st.session_state.index.search(question_embedding, k=10)
                
                broad_chunks = [st.session_state.chunks_metadata[i] for i in indices[0] if i < len(st.session_state.chunks_metadata)]
                
                cross_input = [[prompt_text, chunk["chunk_text"]] for chunk in broad_chunks]
                rerank_scores = reranker_model.predict(cross_input)
                
                for i in range(len(broad_chunks)):
                    broad_chunks[i]["rerank_score"] = rerank_scores[i]
                    
                best_chunks = sorted(broad_chunks, key=lambda x: x["rerank_score"], reverse=True)[:4]
                
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
                
                chat_history_str = "".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}\n\n" for msg in st.session_state.messages[-5:-1]])

                full_prompt = f"""
                You are an expert academic AI assistant. Answer the user's question using ONLY the provided context.
                Every time you state a fact, you MUST include an inline citation (Filename.pdf, Page X).
                If the answer is not in the context, explicitly state: "I cannot find a complete answer to this."
                
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
                
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, src in enumerate(retrieved_sources):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {idx+1}:</strong> {src['source']} (Page {src['page']})<br>
                            <span style="color: #6B7280; font-size: 0.9em;">Relevance Score: {src['score']:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f'"{src["text"]}"')
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved_sources})

# --- TAB 2: SUMMARIZER ---
with tab2:
    st.header("📝 Automated Document Summary")
    if st.session_state.chunks_metadata:
        if st.button("Generate Summary", key="btn_summary"):
            with st.spinner("Analyzing and summarizing..."):
                full_text = "\n".join([chunk["chunk_text"] for chunk in st.session_state.chunks_metadata])
                prompt = f"Analyze the following academic text. Provide: 1. An Abstract summary, 2. Key findings, 3. Conclusion.\n\nTEXT:\n{full_text}"
                client = genai.Client(api_key=api_key)
                st.session_state.summary_output = client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
        if st.session_state.summary_output:
            st.markdown(st.session_state.summary_output)
    else:
        st.info("Upload a document to generate a summary.")

# --- TAB 3: QUESTION GENERATOR ---
with tab3:
    st.header("❓ Viva & Exam Generator")
    if st.session_state.chunks_metadata:
        if st.button("Generate Practice Questions", key="btn_viva"):
            with st.spinner("Crafting academic questions..."):
                full_text = "\n".join([chunk["chunk_text"] for chunk in st.session_state.chunks_metadata[:15]])
                prompt = f"Act as a university professor. Based on this text, generate: 5 Short Answer Questions, 5 Long Essay Questions, and 5 tricky Viva/Oral Exam Questions.\n\nTEXT:\n{full_text}"
                client = genai.Client(api_key=api_key)
                st.session_state.viva_output = client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
        if st.session_state.viva_output:
            st.markdown(st.session_state.viva_output)
    else:
        st.info("Upload a document to generate practice questions.")

# --- TAB 4: PAPER ANALYSIS ---
with tab4:
    st.header("🔬 Deep Research Analysis")
    if st.session_state.chunks_metadata:
        if st.button("Analyze Research Structure", key="btn_analysis"):
            with st.spinner("Deconstructing research methodology..."):
                full_text = "\n".join([chunk["chunk_text"] for chunk in st.session_state.chunks_metadata])
                prompt = f"Analyze this research paper and extract the following exactly: Title, Objective, Methodology, Dataset (if any), Results, Limitations, and Future Work.\n\nTEXT:\n{full_text}"
                client = genai.Client(api_key=api_key)
                st.session_state.analysis_output = client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
        if st.session_state.analysis_output:
            st.markdown(st.session_state.analysis_output)
    else:
        st.info("Upload a document to analyze its structure.")
