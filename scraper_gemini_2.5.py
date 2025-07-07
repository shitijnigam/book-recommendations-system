import requests
from bs4 import BeautifulSoup
import openai
import sqlite3
import numpy as np
import os
import re

# --- 1. Configuration ---
# It's recommended to set your OpenAI API key as an environment variable
# for security reasons.
# In your terminal, you can do: export OPENAI_API_KEY='your_key_here'
openai.api_key = os.getenv("OPENAI_API_KEY")
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_best-selling_books"
DB_NAME = "books.db"


# --- 2. Scraping Logic ---
def scrape_books():
    """Scrapes book titles and their initial paragraphs from Wikipedia."""
    print("Scraping book data from Wikipedia...")
    response = requests.get(WIKIPEDIA_URL)
    soup = BeautifulSoup(response.content, "html.parser")

    books = []
    # This selector is specific to the provided Wikipedia page structure.
    # It might need adjustment if the page layout changes.
    book_list = soup.find_all("tr")

    for row in book_list:
        title_element = row.find("i")
        if title_element and title_element.find("a"):
            title = title_element.find("a").get_text(strip=True)

            # Find the description in the same row
            # This is a simplified approach; descriptions might be in different tags
            description_cell = row.find_all("td")
            description = ""
            if len(description_cell) > 1:
                # Combine text from relevant cells, cleaning it up
                raw_text = description_cell[2].get_text(strip=True)
                # Remove citations like [1] or [a]
                clean_text = re.sub(r"\[[a-zA-Z0-9]+\]", "", raw_text)
                description = clean_text

            if title and description:
                books.append({"title": title, "description": description})

    print(f"Found {len(books)} books.")
    return books


# --- 3. Embedding Logic ---
def get_embeddings(texts):
    """Generates embeddings for a list of texts using OpenAI."""
    print("Generating embeddings with OpenAI...")
    # Using a recommended model for text embeddings
    response = openai.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


# --- 4. Database Logic ---
def store_in_db(books_with_embeddings):
    """Stores book data and embeddings in an SQLite database."""
    print(f"Storing data in {DB_NAME}...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    # The embedding is stored as a BLOB (Binary Large Object)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            embedding BLOB NOT NULL
        )
    """
    )

    for book in books_with_embeddings:
        embedding_np = np.array(book["embedding"], dtype=np.float32)
        cursor.execute(
            "INSERT INTO books (title, description, embedding) VALUES (?, ?, ?)",
            (book["title"], book["description"], embedding_np.tobytes()),
        )

    conn.commit()
    conn.close()
    print("Data stored successfully!")


# --- Main Execution ---
if __name__ == "__main__":
    # Check for API key
    if not openai.api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )

    # Execute the steps
    scraped_books = scrape_books()

    if scraped_books:
        descriptions = [book["description"] for book in scraped_books]
        embeddings = get_embeddings(descriptions)

        # Combine the original book data with the new embeddings
        for i, book in enumerate(scraped_books):
            book["embedding"] = embeddings[i]

        store_in_db(scraped_books)
