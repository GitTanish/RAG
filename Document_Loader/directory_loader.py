from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

loader = DirectoryLoader(
    path = "docs",
    glob = "**/*.pdf",
    loader_cls=PyPDFLoader
)


docs = loader.lazy_load()

for document in docs:
    print(document.metadata)