# RAG Implementation Workspace

A modular collection of LangChain-based components and examples for building Retrieval-Augmented Generation (RAG) pipelines. This repository serves as a learning resource and a foundation for developing sophisticated document retrieval systems.

## 📂 Project Structure

- **`ingestion_pipeline.py`**: The primary end-to-end script that demonstrates loading, chunking, and indexing text files into a ChromaDB vector store.
- **`Document_Loader/`**: Contains modular scripts for ingesting various data formats:
  - `pdf_loader.py`: Using `PyPDFLoader`.
  - `csv_loader.py`: Using `CSVLoader`.
  - `webbase_loader.py`: Scrapes content from URLs.
  - `directory_loader.py`: Bulk loading from a folder.
- **`Text_Splitters/`**: Various strategies for breaking documents into manageable chunks:
  - `length_based.py`: Basic character-count splitting.
  - `DocumentStructured_Based.py`: Language-aware splitting (e.g., for Python code).
- **`Retrievers/`**: Advanced retrieval techniques beyond simple similarity search:
  - `MQR.py`: Multi-Query Retrieval to improve recall.
  - `MMR.py`: Max Marginal Relevance for diverse results.
  - `wikipedia_retriever.py`: Direct querying of the Wikipedia API.
- **`vectorStoreChromaDB/`**: Specific examples and utilities for managing the ChromaDB vector database.
- **`download_articles.py`**: A utility script to quickly download Wikipedia articles as local `.txt` files for testing.
- **`docs/`**: The default storage directory for input documents.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An API key for models (e.g., OpenAI, Groq, or Hugging Face) if you plan to use LLM-based components.

### Installation

1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Create a `.env` file in the root directory to store your API keys:
   ```env
   OPENAI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

### Usage

#### 1. Prepare Data
Use the Wikipedia downloader to fetch some test documents:
```bash
python download_articles.py
```

#### 2. Run the Ingestion Pipeline
Index the documents in the `docs/` folder into the vector store:
```bash
python ingestion_pipeline.py
```

#### 3. Explore Modules
You can run individual scripts in the subdirectories to see specific LangChain features in action:
```bash
python Document_Loader/pdf_loader.py
python Retrievers/MQR.py
```

## 🛠 Tech Stack

- **Framework**: [LangChain](https://github.com/langchain-ai/langchain)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) / [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLMs**: Support for OpenAI, Groq, and more.
