from langchain_community.retrievers import WikipediaRetriever
from typing import Any

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "Capital of Trinidad and Tobago?"

# `invoke` is provided at runtime by the library but lacks static typing.
# Tell linters/type-checkers to ignore the missing-member/attr-defined error.
docs = retriever.invoke(query)  # type: ignore[attr-defined]

for i, doc in enumerate(docs):
    print(f"\n-- Document {i+1} --")
    print(f"Content: \n{doc.page_content}")