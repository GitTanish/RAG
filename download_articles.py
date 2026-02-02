# ai generated
import os
import wikipediaapi

def download_wikipedia_article():
    # Ensure docs folder exists
    DOCS_DIR = "docs"
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="rag-wiki-downloader/1.0"
    )

    # Ask user for article name
    title = input("Enter Wikipedia article name: ").strip()

    if not title:
        print("❌ Article name cannot be empty.")
        return

    page = wiki.page(title)

    if not page.exists():
        print(f"❌ Article '{title}' does not exist on Wikipedia.")
        return

    # Create safe filename
    filename = title.replace(" ", "_") + ".txt"
    filepath = os.path.join(DOCS_DIR, filename)

    # Write article text to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(page.text)

    print(f"✅ Article saved to {filepath}")


if __name__ == "__main__":
    download_wikipedia_article()
