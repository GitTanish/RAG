from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="docs/amazon_reviews.csv", encoding="utf-8")

data = loader.load()
print(data[0].page_content[:1000])