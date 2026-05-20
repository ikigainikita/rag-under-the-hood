from typing import List,TypedDict
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START,END
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from tavily import TavilyClient

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda
import os 

llm = ChatOllama(model="qwen2.5-coder:3b", num_predict=256, model_kwargs={"keep_alive": "-1"})
def process_new_document(file_path: str, original_filename: str):
    """
    Checks for an existing FAISS cache based on the original filename.
    Loads from disk if available, otherwise embeds from scratch and saves.
    """
    global retriever
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text-v2-moe:latest"
    )

    # 1. Create a unique folder name using the ORIGINAL filename
    save_path = f"vectorstore/{original_filename.replace('.pdf', '')}"
    
    # Ensure the parent vectorstore directory exists
    os.makedirs("vectorstore", exist_ok=True)
    
    # 2. Check if cache exists
    if os.path.exists(save_path):
        print(f"[Standard Backend] Found cache for '{original_filename}'. Loading from disk...")
        vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return  # Exit function early, skipping the heavy embedding!
        
    print(f"[Standard Backend] No cache found for '{original_filename}'. Embedding from scratch...")
    
    # 3. Load and Chunk the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
    
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        
    if not chunks:
        print("[Standard Backend] No valid text found in document.")
        return

    # 4. Generate Vectors and Save to Disk
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    
    # 5. Update the global retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print(f"[Standard Backend] Pipeline cached and tracking {len(chunks)} chunks.")


class State(TypedDict):
    question:str
    docs:List[Document]

    good_docs:List[Document]
    verdict:str
    reason:str

    strips:List[str]
    kept_strips:List[str]
    refined_context:str
    answer:str
    web_docs:List[Document]
    web_query:str
retriever = None 

def retrieve(state: State) -> State:
    # Failsafe if no doc is uploaded
    return {"docs": retriever.invoke(state["question"])}


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict, analytical research assistant. You extract facts strictly from the provided documents."
        ),
        (
            "human",
            """Answer the user's question based ONLY and EXCLUSIVELY on the provided Context inside the <context> tags. 

CRITICAL RULES:
1. Do NOT use any outside knowledge, pre-trained data, or real-world assumptions.
2. If a specific strategy, concept, or point is not explicitly written within the <context> tags below, you MUST NOT include it in your response.
3. If the answer cannot be found inside the <context> tags, reply exactly with: "The provided context does not contain the answer."

<context>
{docs}
</context>

Question: {question}
Answer:"""
        )
    ]
)
def generate(state:State)->State:
    out=(answer_prompt|llm).invoke({"question":state["question"],"docs":state["docs"]})
    return {"answer":out.content}
def not_related(state: State) -> State:
    # FIX: Actually update the state with a fallback answer
    return {"answer": "The provided context does not contain information related to your question. Cannot generate an answer."}



from langchain_core.prompts import ChatPromptTemplate






# Modified route that skips the heavy LLM evaluation loop


# Modified route that skips the heavy LLM evaluation loop

def direct_route(state: State) -> State:
    # Fallback if no documents were returned by the vectorizer
    if not state.get("docs"):
        return "not_related"
    else:
     return "generate"

# Re-link the graph to be CPU-friendly
g = StateGraph(State)
g.add_node("retrieve", retrieve)      
g.add_node("generate", generate)
g.add_node("not_related", not_related)

g.add_edge(START, "retrieve")

# Route directly based on whether chunks were found or not
g.add_conditional_edges(
    "retrieve",
    direct_route,
    {"generate": "generate", "not_related": "not_related"}
)

g.add_edge("generate", END)
app = g.compile()


if __name__ == "__main__":
    res=app.invoke({
        "question":"Open source software are free , but their is whole industry growing which is also profitable , how companies based on open source systems profit  ",
        "docs":[],
        "good_docs":[],
        "verdict":"",
        "reason":"",
        "strips":[],
        "kept_strips":"",
        "refined_context":"",
        "web_docs":[],
        "answer":"",
    })   
    print(res['verdict'])
    print(res['answer'])