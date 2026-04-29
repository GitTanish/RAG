from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/Attention.pdf")
docs = loader.load()
print(docs[0].page_content[:1000])
print(docs[1].metadata)


# simple pdf - PyPDFLoader
# Pdf with tables/columns -  PDFPlumberLoader
# scanned/image pdf - UnstructuredPDFLoader