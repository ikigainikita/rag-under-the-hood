from typing import List,TypedDict,List, Dict, Any
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

llm = ChatOllama(model="qwen2.5-coder:3b", num_predict=256, model_kwargs={"keep_alive": "-1"})

from langchain_core.runnables import RunnableLambda
#loader = TextLoader("document.txt", encoding="utf-8")
from library import TextPreprocessor ,TfidfVectorizer ,calculate_cosine_similarity

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
    raw_retrieval_results: List[Dict[str, Any]]


import unicodedata
import re


document_vectors = []

preprocessor = TextPreprocessor(
        lemma_file="lemmas.txt", 
        contraction_file="contractions.txt", 
        stop_file="stopwords.txt"
    )
vectorizer = TfidfVectorizer(preprocessor)
def process_new_document(file_path: str):
    """
    Accepts a single local PDF path, processes it, trains the TF-IDF vectorizer,
    and REPLACES the global document database so queries apply ONLY to this file.
    """
    global document_vectors, vectorizer
    
    # 1. LOAD THE PDF FROM THE FILE PATH
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    preprocessor = TextPreprocessor(
        lemma_file="lemmas.txt", 
        contraction_file="contractions.txt", 
        stop_file="stopwords.txt"
    )

    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)

    final_chunks = []
    for d in chunks:
        # Clean string layout encoding issues
        raw_text = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        processed_tokens = preprocessor.preprocess(raw_text)
        
        # Filter out junk/empty chunks
        if len(processed_tokens) < 15:  
            continue
            
        junk_words = {'intentional', 'left', 'blank'}
        if set(processed_tokens).issubset(junk_words):
            continue

        d.page_content = raw_text 
        
        # Ensure metadata exists
        if not hasattr(d, 'metadata') or d.metadata is None:
            d.metadata = {}
            
        d.metadata["processed_text"] = " ".join(processed_tokens) 
        final_chunks.append(d)

    if not final_chunks:
        print("[Custom Backend] No valid text found in document.")
        return

    # ==========================================
    # 2. VECTORIZATION (Strictly for this single PDF)
    # ==========================================
    processed_texts_for_math = [doc.metadata["processed_text"] for doc in final_chunks]

    vectorizer = TfidfVectorizer(preprocessor) 
    vectorizer.fit(processed_texts_for_math) 

    # CLEAR previous document history! 
    # This ensures parallel files aren't mixed in.
    document_vectors = [] 

    for i, doc in enumerate(final_chunks):
        tfidf_map, norm = vectorizer.transform(doc.metadata["processed_text"])
        
        document_vectors.append({
            "doc_id": i,
            "vector": tfidf_map,
            "norm": norm,
            "raw_content": doc.page_content, 
            "metadata": doc.metadata
        })

    print(f"[Custom Backend] Context wiped. Pipeline now tracking {len(document_vectors)} vectors strictly for the uploaded file.")

def retrieve_relevant_chunks(query, vectorizer, document_vectors, preprocessor, top_k=5):
    
    query_tokens = preprocessor.preprocess(query)
    processed_query = " ".join(query_tokens)
    
    query_vector, query_norm = vectorizer.transform(processed_query)
    
    if query_norm == 0:
        return [] 

    # 2. Score all documents
    scores = []
    for doc in document_vectors:
        sim = calculate_cosine_similarity(
            query_vector, query_norm, 
            doc["vector"], doc["norm"]
        )
        
        scores.append({
            "score": sim,
            "metadata": doc.get("metadata", {}), 
            "raw_content": doc["raw_content"]
        })

    # Sort by similarity score in descending order
    scores.sort(key=lambda x: x["score"], reverse=True)

    #  Return the best res
    return {"raw_retrieval_results": scores[:top_k]}


# ==========================================
# 4. EXAMPLE USAGE
# =========================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

rewritten_query = ChatPromptTemplate.from_messages(
    [(
        "system",
        """You are an expert search query optimization assistant. 
The user will ask a conceptual question. Your job is to rewrite it into the EXACT phrasing a textbook author would use to define the concept in an introductory paragraph.
DO NOT provide a list of synonyms. DO NOT provide a comma-separated list.
Write a single, declarative sentence that looks like a formal definition.
Output ONLY the sentence, with no introductory filler.

Example User Query: "what is open source"
Example Output:  """
    ),
    ("human", "{question}")
    ]
)

# Adding StrOutputParser automatically extracts the string, so you don't need .content later
rewrite_chain = rewritten_query | llm | StrOutputParser()


from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

def fetch_docs(query:str):
    # ALL preprocessing and vector math is safely handled inside this function already.
    # No need to duplicate the work here!
    results = retrieve_relevant_chunks(
        query=query, 
        vectorizer=vectorizer, 
        document_vectors=document_vectors, 
        preprocessor=preprocessor, 
        top_k=3
    )
    return results

retriever = RunnableLambda(fetch_docs)

def retrieve(state: State) -> State:
    q = state["question"]
    
    # 1. Fetch raw results
    raw_results = retriever.invoke(q)
    
    langchain_docs = []
    
    # 2. Iterate over the list inside the dictionary
    for item in raw_results.get("raw_retrieval_results", []):
        content = item["raw_content"]
        
        # Ensure content is a string before calling string methods
        if isinstance(content, list):
            content = " ".join(content)
            
        # FIX: Removed .lower() so the LLM gets proper nouns, acronyms, and grammar
        cleaned_content = content.strip() 
        
        langchain_docs.append(
            Document(page_content=cleaned_content, metadata=item.get("metadata", {}))
        )
    
    # Return the dictionary with the updated keys. 
    return {
        "docs": langchain_docs,
        # Ensure your State's TypedDict actually includes "raw_retrieval_results" as a key!
        "raw_retrieval_results": raw_results 
    }

def not_related(state: State) -> State:
    # FIX: Actually update the state with a fallback answer
    return {"answer": "The provided context does not contain information related to your question. Cannot generate an answer."}



from langchain_core.prompts import ChatPromptTemplate

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
def rewrite_query(state: State):
    """
    Node that rewrites the user's question into a textbook definition 
    format to optimize vector retrieval.
    """
    print("---REWRITING QUERY---")
    
    # Extract the current question from the state
    current_question = state["question"]
    
    # Invoke the chain using the extracted question
    rewritten_question = rewrite_chain.invoke({"question": current_question})
    
    print(f"Original: {current_question}")
    print(f"Rewritten: {rewritten_question}")
    
    # 3. Return the updated state
    # This overwrites the existing 'question' key in the graph state
    return {"question": rewritten_question}

def generate(state:State)->State:
    out=(answer_prompt|llm).invoke({"question":state["question"],"docs":state["docs"]})
    print(out.content)
    return {"answer":out.content} 
# Modified route that skips the heavy LLM evaluation loop

def direct_route(state: State) -> State:
    # Fallback if no documents were returned by the vectorizer
    if not state.get("docs"):
        return "not_related"
    return "generate"

# Re-link the graph to be CPU-friendly
g = StateGraph(State)
g.add_node("retrieve", retrieve)  
#g.add_node("rewrite_query", rewrite_query)  
    
g.add_node("generate", generate)
g.add_node("not_related", not_related)

#g.add_edge(START, "rewrite_query")
#g.add_edge("rewrite_query", "retrieve")
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