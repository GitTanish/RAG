from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import sentence_transformers

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(document_path="docs"):
    """ Load all text files from the docs folder """
    print(f"Loading documents from {document_path}")
    try:
        loader = DirectoryLoader(document_path,
        glob="*.txt", 
        loader_cls=TextLoader)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def main():
    print('Main Function')

    #1. Loading the files

    #2. Chunking the files
    #3. Embedding the files
    #4. Storing the files in a vector store 

if __name__ == "__main__":
    main()

