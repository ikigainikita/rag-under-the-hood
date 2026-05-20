import streamlit as st
import time
import tempfile
import os
import concurrent.futures

# ==========================================
# 1. IMPORT YOUR BACKEND / LANGCHAIN APPS
# ==========================================
from custom import app as custom_app, process_new_document as custom_doc_processor
from standard import app as standard_app, process_new_document as standard_doc_processor
from evaluator import compare_chain

st.set_page_config(page_title="RAG Comparison Dashboard", layout="wide")

# Added custom scroll container styling with fixed height and explicit scrollbars
st.markdown("""
<style>
    .metric-card { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center; margin-bottom: 15px; }
    .metric-val { font-size: 24px; font-weight: bold; color: #1E88E5; }
    .metric-lbl { font-size: 12px; color: #6c757d; text-transform: uppercase; }
    .output-box { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; min-height: 200px; font-size: 16px; line-height: 1.8; color: #333;}
    
    /* Dedicated context box scrollbar styles */
    .context-scroll-container { 
        background-color: #f8f9fa; 
        border: 1px solid #dee2e6; 
        border-radius: 8px; 
        padding: 15px; 
        height: 250px; 
        overflow-y: scroll; 
        font-size: 14px; 
        line-height: 1.6; 
        color: #495057;
    }
    /* Elegant scrollbar styling for Webkit browsers (Chrome, Safari, Edge) */
    .context-scroll-container::-webkit-scrollbar {
        width: 8px;
    }
    .context-scroll-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .context-scroll-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    .context-scroll-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SINGLE PDF UPLOAD & PROCESSING HANDLER
# ==========================================
def handle_pdf_upload(uploaded_file):
    if uploaded_file is not None:
        tmp_file_path = None
        try:
            with st.spinner("Extracting file buffer from RAM to disk..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

            # 1. Custom Backend (No cache, fits TF-IDF from scratch)
            with st.spinner(f"Custom Backend: Vectorizing '{uploaded_file.name}' from scratch..."):
                custom_doc_processor(tmp_file_path)
                
            # 2. Standard Backend (Passes original filename for caching logic)
            with st.spinner(f"Standard Backend: Checking cache or building FAISS index for '{uploaded_file.name}'..."):
                standard_doc_processor(tmp_file_path, uploaded_file.name)
                
            st.success(f"Successfully processed '{uploaded_file.name}' in both pipelines! You can now query this document.")
            st.balloons()
            return True

        except Exception as e:
            st.error(f"Pipeline crashed during runtime: {e}")
            return False
            
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# ==========================================
# 3. EXECUTION WRAPPERS
# ==========================================
def run_custom_chain(query: str):
    start_time = time.time()
    
    try:
        res = custom_app.invoke({
            "question": query,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_docs": [],
            "answer": "",
        })
        final_text = res.get("answer", "No answer generated.")
        retrieved_docs = res.get("docs", [])
    except Exception as e:
        final_text = f"Error during custom execution: {str(e)}"
        retrieved_docs = []
    
    latency_ms = (time.time() - start_time) * 1000
    return final_text, retrieved_docs, latency_ms

def run_standard_chain(query: str):
    start_time = time.time()
    
    try:
        res = standard_app.invoke({
            "question": query,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_docs": [],
            "answer": "",
        })
        final_text = res.get("answer", "No answer generated.")
        retrieved_docs = res.get("docs", [])
    except Exception as e:
        final_text = f"Error during standard execution: {str(e)}"
        retrieved_docs = []
    
    latency_ms = (time.time() - start_time) * 1000
    return final_text, retrieved_docs, latency_ms

# Helper to flatten all chunk content into a single combined paragraph string
def get_flattened_context(docs_list) -> str:
    if not docs_list:
        return "No source context text was retrieved for this query."
        
    extracted_texts = []
    for doc in docs_list:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        cleaned_text = " ".join(text.split())
        extracted_texts.append(cleaned_text)
        
    return " ".join(extracted_texts)

# ==========================================
# 4. USER INTERFACE
# ==========================================
st.title("🤖 RAG Architecture Inspector")
st.markdown("Upload a document to update the vector stores, then query to compare retrieval and generation metrics.")

with st.sidebar:
    st.header("📄 Knowledge Base")
    st.write("Upload a single PDF to dynamically update the vector database.")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document", 
        type=["pdf"], 
        accept_multiple_files=False 
    )
    
    if uploaded_file is not None:
        if st.button("Process & Vectorize Document"):
            handle_pdf_upload(uploaded_file)

st.write("---")

user_query = st.text_input("Enter your query:", "Explain vectorization and cosine similarity")

if st.button("Generate & Compare"):
    with st.spinner("Invoking Langchain apps in parallel..."):
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_custom = executor.submit(run_custom_chain, user_query)
            future_std = executor.submit(run_standard_chain, user_query)
            
            custom_text, custom_docs, custom_time = future_custom.result()
            std_text, std_docs, std_time = future_std.result()
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🛠️ Custom Library")
            st.markdown(f'<div class="metric-card"><div class="metric-val">{custom_time:.1f} ms</div><div class="metric-lbl">Total Execution Latency</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{custom_text}</div>', unsafe_allow_html=True)
            
            # Subheader and dedicated scrolling context block
            st.markdown("### 📄 Retrieved Context")
            flattened_custom = get_flattened_context(custom_docs)
            st.markdown(f'<div class="context-scroll-container">{flattened_custom}</div>', unsafe_allow_html=True)

        with col2:
            st.header("📦 Standard Library")
            st.markdown(f'<div class="metric-card"><div class="metric-val">{std_time:.1f} ms</div><div class="metric-lbl">Total Execution Latency</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{std_text}</div>', unsafe_allow_html=True)
            
            # Subheader and dedicated scrolling context block
            st.markdown("### 📄 Retrieved Context")
            flattened_std = get_flattened_context(std_docs)
            st.markdown(f'<div class="context-scroll-container">{flattened_std}</div>', unsafe_allow_html=True)

        st.divider()
        
        st.subheader("🧠 LLM Output Analysis")
        with st.spinner("Analyzing similarity between outputs..."):
            try:
                evaluation = compare_chain.invoke({
                    "question": user_query,
                    "answer_1": custom_text,
                    "answer_2": std_text
                })
                
                score_color = "green" if evaluation.similarity_score > 75 else "orange" if evaluation.similarity_score > 40 else "red"
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                    <h3 style="margin-top: 0; color: {score_color};">Similarity Score: {evaluation.similarity_score}%</h3>
                    <p style="margin-bottom: 0; font-size: 16px;"><strong>Verdict:</strong> {evaluation.explanation}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to run LLM comparison: {e}")