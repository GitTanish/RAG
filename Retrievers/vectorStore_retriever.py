# pip install -U langchain langchain-chroma langchain-huggingface sentence-transformers

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIR = "./my_collection_db"
COLLECTION = "local_store"

# ---- Embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---- Data ----
docs = [
    Document(
        page_content="Neon Tetra is a small schooling fish with bright blue and red stripes. Peaceful, prefers groups and soft water.",
        metadata={"type": "Community", "difficulty": "Intermediate", "name": "Neon Tetra"},
    ),
    Document(
        page_content="Betta fish has long fins and strong colors. Males are territorial. Needs warm water, filter, and care. Can be beginner-friendly but requires specific conditions.",
        metadata={"type": "Specimen", "difficulty": "Beginner", "name": "Betta"},
    ),
    Document(
        page_content="Guppies are hardy, colorful, easy to breed, and adapt to many water conditions. Very beginner friendly.",
        metadata={"type": "Community", "difficulty": "Beginner", "name": "Guppy"},
    ),
    Document(
        page_content="Corydoras catfish are peaceful bottom dwellers. Clean leftover food and do best in groups.",
        metadata={"type": "Bottom-dweller", "difficulty": "Beginner", "name": "Corydoras"},
    ),
    Document(
        page_content="Angelfish are large cichlids with triangular bodies. Semi-aggressive and need tall tanks.",
        metadata={"type": "Semi-aggressive", "difficulty": "Advanced", "name": "Angelfish"},
    ),
]

ids = [f"fish_{i}" for i in range(len(docs))]

# ---- Vector Store (reset cleanly) ----
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# hard reset (prevents duplicates across runs)
try:
    vectorstore.delete_collection()
except Exception:
    pass

vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

vectorstore.add_documents(docs, ids=ids)

# ---- Retriever ----
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"difficulty": "Beginner"},  # Filter for beginner-friendly fish,
    }
)

# ---- Query ----
query = "Which fish is best for beginners?"
results = retriever.invoke(query)
print(results)
# Chroma often requires calling MMR directly
# ---- Step 1: MMR retrieval (no retriever wrapper) ----
# mmr_docs = vs.max_marginal_relevance_search(
#     query,
#     k=5,          # final candidate set after MMR
#     fetch_k=12,   # pool to diversify from
#     lambda_mult=0.5,
#     # filter={"difficulty": "Beginner"},  # optional hard filter
# )


# ---- Output ----
for i, doc in enumerate(results, 1):
    print(f"\n-- Result {i} --")
    print(doc.page_content)
    print(doc.metadata)
