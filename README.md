# RAG Under the Hood: Custom vs. Abstracted Retrieval 🔍

An intelligent Document Q&A system built to analyze and compare custom retrieval components against standard, pre-built NLP libraries. 

While most RAG (Retrieval-Augmented Generation) pipelines rely on abstracted "black box" functions, this project dives into the underlying architecture by building core components from scratch to evaluate performance, accuracy, and mechanics.

#🧠 Core Objectives
* **Demystify Abstractions:** Compare standard library implementations (like vectorization and retrieval logic) against custom-built solutions.
* **Custom NLP Processing:** Implement custom tokenization, stopword removal, and lemmatization without relying solely on heavy frameworks.
* **Algorithmic Transparency:** Understand the exact mathematical and architectural tradeoffs of search and retrieval mechanics.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM Integration:** Local LLMs via Ollama (e.g.qwen2.5-coder:3b)
* **Architecture:** RAG (LangGraph)
* **NLP Assets:** Custom text processing (Lemmas, Contractions, Stopwords)

## 📂 Repository Structure
* `custom.py` - Contains the ground-up implementations of retrieval and vectorization logic (e.g., custom TF-IDF).
* `standard.py` - Implements the same functionality using standard abstracted libraries for baseline comparison.
* `app2.py` - The intelligent routing and generation pipeline.
* `evaluator.py` - Logic to benchmark and compare the custom vs. abstract implementations.
* `*.txt` - Custom datasets for stopword filtering, lemmatization, and contraction expansion to support the custom NLP pipeline.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/ikigainikita/rag-under-the-hood.git](https://github.com/ikigainikita/rag-under-the-hood.git)
cd rag-under-the-hood
