# HR Policy Question Answering System using RAG (LangChain, Chroma, LLaMA-3.1)

Built an end-to-end RAG-based HR chatbot that answers policy-related queries in natural language. Preprocessed HR documents from CSV using pandas and LangChain’s DataFrameLoader, then chunked them using RecursiveCharacterTextSplitter. Generated embeddings via sentence-transformers/all-MiniLM-L6-v2 and stored them in a Chroma vector database. Used Chroma’s retriever to fetch relevant context and passed it to a quantized LLaMA-3.1 model (via HuggingFace pipeline) for generating precise, context-aware answers. Integrated everything using LangChain’s RetrievalQA for seamless document search and response generation.

---

## 🧠 Overview

The system combines a generative language model with vector-based document retrieval to accurately answer queries grounded in your HR policy documents. The architecture is built using:

- **LangChain** for document processing and chain orchestration
- **Chroma** for storing and retrieving dense vector embeddings
- **LLaMA-3.1** for generating coherent, relevant responses

---

## 🚀 Features

- **RAG-Based Question Answering**: Combines retrieval of relevant documents with generation for accurate results.
- **Natural Language Querying**: Users can ask HR questions conversationally.
- **Fast and Scalable Retrieval**: Uses Chroma for efficient vector-based search.
- **Context-Aware Responses**: Responses are generated using the retrieved context and LLaMA-3.1 model.
- **Modular & Extensible**: Easy to update or expand with more documents or domains.

---

## 🛠️ Tech Stack

- **LangChain** – for text splitting, document loading, and RetrievalQA
- **LLaMA-3.1 (Quantized)** – via HuggingFace pipeline for lightweight, efficient text generation
- **Chroma DB** – for storing and retrieving dense vector document embeddings
- **HuggingFace Transformers** – for model and tokenizer support
- **Sentence-Transformers (MiniLM-L6-v2)** – for generating document embeddings
- **Pandas** – for reading and processing CSV-based HR policy data

---

## 📂 Project Flow

1. **Load HR Policies** from a `.csv` file using `pandas` and `LangChain's DataFrameLoader`.
2. **Split Text** into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Generate Embeddings** using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Store Embeddings** in a persistent **Chroma** vector store.
5. **Set Up Retriever** using Chroma's similarity search.
6. **Generate Answers** using `RetrievalQA` chain with **LLaMA-3.1** as the language model.
7. **Query System** via simple function call and get real-time, accurate responses.

---

## 🧪 Example Query

```python
query = "What is the Diwali bonus policy at this company?"
get_answers(QnA, query)
