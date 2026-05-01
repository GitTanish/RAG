from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



doc1 = Document(
    page_content="The Neon Tetra is a vibrant schooling fish famous for its glowing blue and red horizontal stripes. They are peaceful, thrive in groups, and prefer soft, slightly acidic water.",
    metadata={"type": "Community", "difficulty": "Beginner"}
)

doc2 = Document(
    page_content="The Betta Fish, or Siamese Fighting Fish, is known for its spectacular flowing fins and bold colors. Males are highly territorial and should be kept alone in a heated, filtered tank.",
    metadata={"type": "Specimen", "difficulty": "Intermediate"}
)

doc3 = Document(
    page_content="Guppies are incredibly popular due to their endless color variations and ease of breeding. They are active swimmers that adapt well to most water conditions, making them ideal for beginners.",
    metadata={"type": "Community", "difficulty": "Beginner"}
)

doc4 = Document(
    page_content="The Corydoras Catfish is a peaceful bottom-dweller that helps keep tanks clean by scavenging for leftover food. They are social creatures that are happiest when kept in groups of six or more.",
    metadata={"type": "Bottom-dweller", "difficulty": "Beginner"}
)

doc5 = Document(
    page_content="The Angelfish is a majestic cichlid with a unique triangular shape. While beautiful, they can be semi-aggressive as they grow larger and require a tall tank to accommodate their height.",
    metadata={"type": "Semi-aggressive", "difficulty": "Advanced"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vectorstore = Chroma(
    collection_name="local_store",
    embedding_function=embeddings,
    persist_directory="./my_local_db"
)  

vectorstore.add_documents(docs)

a=vectorstore.get(include=['embeddings','documents','metadatas'])
print(a)

# search documents
answer=vectorstore.similarity_search("Which fish is best for beginners?", k=2)
print(answer)

# search with similarity score
answer1=vectorstore.similarity_search_with_score("  Angelfish difficulty level?", k=2)
print(answer1)

# metadata filtering

answer3 = vectorstore.similarity_search(
    "Which fish is best?", 
    k=2, 
    filter={"difficulty": "Beginner"} # This is the magic line
)
print(answer3)