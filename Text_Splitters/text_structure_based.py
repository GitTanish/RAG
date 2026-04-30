from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader

loader = PyPDFLoader("docs/attention.pdf")

document = loader.lazy_load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=5 # must provide overlap else it will default to overlap of 200
    )
text_chunks = splitter.split_documents(document)
print(text_chunks[900].page_content)