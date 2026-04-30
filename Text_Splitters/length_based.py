from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader

loader = UnstructuredFileLoader("docs/attention.pdf", language="en")

document = loader.lazy_load()

splitter = CharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=5
    )
text_chunks = splitter.split_documents(document)
print(text_chunks[40].page_content)