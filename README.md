# Design_RAG_for_LLM
# Retrieval Augmented Generation (RAG) for Medical Applications

This repository demonstrates how to implement a **Retrieval Augmented Generation (RAG)** system using a medical dataset. The RAG system combines the power of a vector database for retrieving relevant information and a Large Language Model (LLM) for generating contextual responses.

---

## Features
- Create embeddings from a medical dataset using `SentenceTransformers`.
- Store and manage embeddings in a vector database using `Qdrant`.
- Retrieve relevant context based on a query using cosine similarity.
- Use a local or cloud-based LLM to generate a grounded, accurate response.

---

## Prerequisites

Before running the code, ensure you have the following:
1. Python 3.8 or higher installed.
2. The following Python libraries:
   - `sentence-transformers`
   - `qdrant-client`
   - `transformers`
   - `pandas`

Install them using pip:
```bash
pip install sentence-transformers qdrant-client transformers pandas

