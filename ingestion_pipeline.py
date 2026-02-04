import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)


# here we are loading the documents
def load_documents(document_path="docs"):
    """ Load all text files from the docs folder """
    print(f"Loading documents from {document_path}")
    
    # check if docs exist
    if not os.path.exists(document_path):
        raise ValueError(f"Documents not found at {document_path}")

    # load all text files
    loader = DirectoryLoader(
        path=document_path,
        glob="*.txt",
        loader_cls=TextLoader, # only look for txt files
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError("No documents found in the specified directory")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
    
    return documents

# chunking the documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=0):
    """ Split documents into smaller chunks with overlap """
    print("Splitting documents into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap= chunk_overlap
        
    )
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---- Chunk {i+1} ----")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    return chunks

# vector storing

def create_vector_store(chunks, persist_directory="db/chromadb"):
    """ Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    # create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata = {"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    print(f"Vector store created at: {persist_directory}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding model: {embeddings.__class__.__name__}")
    print(f"Collection metadata: {vectorstore._collection.metadata}")
    return vectorstore
    
# 

def main():
    print('Main Function')

    #1. Loading the files
    documents = load_documents(document_path="docs")

    #2. Chunking the files
    chunks =chunk_documents(documents)

    #3. Embedding the files
    vectorstore = create_vector_store(chunks)

    #4. Storing the files in a vector store 
    

if __name__ == "__main__": 
    main()

