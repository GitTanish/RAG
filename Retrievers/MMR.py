from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_PERSIST_DIR = "./faiss_index"

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore_faiss = FAISS.from_documents(docs, embeddings)
vectorstore_faiss.save_local(FAISS_PERSIST_DIR)

retriever = vectorstore_faiss.as_retriever(
    search_type="mmr", # <-- MMR enabled
    search_kwargs={"k": 3, "lambda_mult": 0.5} # k = top 2 results, lambda_mult = 0.5 for balanced relevance and diversity  
)

query = "What makes life easy?"
results = retriever.invoke(query)
for i , doc in enumerate(results):
    print(f"\n-- Result {i+1} --")
    print(f"Content: \n{doc.page_content}")
