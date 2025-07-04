# Chat with Websites

This is a Streamlit app that allows you to chat with the content of any website using local LLMs and Retrieval-Augmented Generation (RAG).

Just enter a URL, and the app will extract, chunk, embed, and index the website content so you can ask questions about it. It works with both Arabic and English.

---

## Features

- Extracts and processes website content.
- Uses local LLMs for answering questions.
- Maintains chat history for context-aware responses.
- Uses FAISS for retrieval and multilingual embeddings.
- Built with a clean Streamlit interface.

---

## Installation

Install required packages:

```bash
pip install streamlit langchain langchain-community langchain-core faiss-cpu sentence-transformers
pip install transformers huggingface-hub
