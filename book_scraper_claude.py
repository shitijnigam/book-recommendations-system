import requests
import sqlite3
import json
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from urllib.parse import unquote
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm


# Configuration
@dataclass
class Config:
    # Database settings
    DB_NAME = "books_embeddings.db"

    # Wikipedia API settings
    WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"
    BOOKS_CATEGORY = "Category:Books"
    MAX_BOOKS = 1000  # Adjust based on your needs

    # Embedding settings
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
    EMBEDDING_DIMENSIONS = 1536
    BATCH_SIZE = 100

    # Rate limiting
    WIKIPEDIA_DELAY = 0.1  # seconds between requests
    OPENAI_DELAY = 0.1  # seconds between embedding requests


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Create books table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE NOT NULL,
                description TEXT,
                wikipedia_url TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding_created BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Create embeddings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER,
                embedding BLOB,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (book_id) REFERENCES books (id)
            )
        """
        )

        # Create index for faster searches
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_book_title ON books(title)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_book_id ON embeddings(book_id)"
        )

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_name}")

    def insert_book(self, title: str, description: str, wikipedia_url: str) -> int:
        """Insert a book into the database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO books (title, description, wikipedia_url)
                VALUES (?, ?, ?)
            """,
                (title, description, wikipedia_url),
            )

            book_id = cursor.lastrowid
            conn.commit()
            return book_id
        except sqlite3.Error as e:
            logger.error(f"Error inserting book {title}: {e}")
            return None
        finally:
            conn.close()

    def insert_embedding(self, book_id: int, embedding: np.ndarray, model_name: str):
        """Insert an embedding into the database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            # Convert numpy array to bytes
            embedding_bytes = embedding.tobytes()

            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (book_id, embedding, model_name)
                VALUES (?, ?, ?)
            """,
                (book_id, embedding_bytes, model_name),
            )

            # Update book to mark embedding as created
            cursor.execute(
                """
                UPDATE books SET embedding_created = TRUE WHERE id = ?
            """,
                (book_id,),
            )

            conn.commit()
            logger.debug(f"Embedding inserted for book_id: {book_id}")
        except sqlite3.Error as e:
            logger.error(f"Error inserting embedding for book_id {book_id}: {e}")
        finally:
            conn.close()

    def get_books_without_embeddings(self) -> List[Tuple[int, str, str]]:
        """Get books that don't have embeddings yet."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, title, description 
            FROM books 
            WHERE embedding_created = FALSE
        """
        )

        books = cursor.fetchall()
        conn.close()
        return books

    def get_book_count(self) -> int:
        """Get total number of books in database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM books")
        count = cursor.fetchone()[0]
        conn.close()
        return count


class WikipediaScraper:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "BookScraper/1.0 (Educational Purpose)"}
        )

    def get_books_from_category(
        self, category: str = None, limit: int = None
    ) -> List[str]:
        """Get list of book titles from Wikipedia category."""
        if category is None:
            category = self.config.BOOKS_CATEGORY
        if limit is None:
            limit = self.config.MAX_BOOKS

        books = []
        continue_token = None

        while len(books) < limit:
            url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": min(500, limit - len(books)),
                "format": "json",
                "cmtype": "page",
            }

            if continue_token:
                params["cmcontinue"] = continue_token

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if "query" in data and "categorymembers" in data["query"]:
                    for member in data["query"]["categorymembers"]:
                        if len(books) >= limit:
                            break
                        books.append(member["title"])

                # Check if there are more pages
                if "continue" in data and "cmcontinue" in data["continue"]:
                    continue_token = data["continue"]["cmcontinue"]
                else:
                    break

                time.sleep(self.config.WIKIPEDIA_DELAY)

            except requests.RequestException as e:
                logger.error(f"Error fetching category members: {e}")
                break

        logger.info(f"Found {len(books)} books in category: {category}")
        return books

    def get_book_description(self, title: str) -> Optional[str]:
        """Get book description from Wikipedia."""
        url = f"https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsectionformat": "plain",
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "query" in data and "pages" in data["query"]:
                for page_id, page_data in data["query"]["pages"].items():
                    if "extract" in page_data:
                        extract = page_data["extract"]
                        # Clean up the extract
                        extract = re.sub(r"\n+", " ", extract)
                        extract = re.sub(r"\s+", " ", extract)
                        return extract.strip()

            return None

        except requests.RequestException as e:
            logger.error(f"Error fetching description for {title}: {e}")
            return None

    def scrape_books(self, db_manager: DatabaseManager):
        """Scrape books and store in database."""
        logger.info("Starting book scraping process...")

        # Get list of book titles
        book_titles = self.get_books_from_category()

        logger.info(f"Scraping details for {len(book_titles)} books...")

        for i, title in enumerate(tqdm(book_titles, desc="Scraping books")):
            try:
                description = self.get_book_description(title)

                if description:
                    wikipedia_url = (
                        f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    )
                    book_id = db_manager.insert_book(title, description, wikipedia_url)

                    if book_id:
                        logger.debug(f"Saved book: {title}")

                time.sleep(self.config.WIKIPEDIA_DELAY)

            except Exception as e:
                logger.error(f"Error processing book {title}: {e}")
                continue

        logger.info("Book scraping completed!")


class EmbeddingGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        self.sentence_model = None

    def setup_openai(self, api_key: str):
        """Setup OpenAI client."""
        openai.api_key = api_key
        self.openai_client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")

    def setup_sentence_transformers(self, model_name: str = "all-MiniLM-L6-v2"):
        """Setup Sentence Transformers model (local)."""
        self.sentence_model = SentenceTransformer(model_name)
        logger.info(f"Sentence Transformers model loaded: {model_name}")

    def create_embedding_openai(self, text: str) -> np.ndarray:
        """Create embedding using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        try:
            response = self.openai_client.embeddings.create(
                input=text, model=self.config.EMBEDDING_MODEL
            )

            embedding = np.array(response.data[0].embedding)
            time.sleep(self.config.OPENAI_DELAY)
            return embedding

        except Exception as e:
            logger.error(f"Error creating OpenAI embedding: {e}")
            raise

    def create_embedding_local(self, text: str) -> np.ndarray:
        """Create embedding using local Sentence Transformers."""
        if not self.sentence_model:
            raise ValueError("Sentence Transformers model not initialized")

        try:
            embedding = self.sentence_model.encode(text)
            return embedding

        except Exception as e:
            logger.error(f"Error creating local embedding: {e}")
            raise

    def generate_embeddings(self, db_manager: DatabaseManager, use_openai: bool = True):
        """Generate embeddings for all books without embeddings."""
        if use_openai:
            if not self.openai_client:
                raise ValueError(
                    "OpenAI client not initialized. Call setup_openai() first."
                )
            model_name = self.config.EMBEDDING_MODEL
        else:
            if not self.sentence_model:
                self.setup_sentence_transformers()
            model_name = "sentence-transformers"

        books = db_manager.get_books_without_embeddings()
        logger.info(
            f"Generating embeddings for {len(books)} books using {'OpenAI' if use_openai else 'local'} model..."
        )

        for book_id, title, description in tqdm(books, desc="Creating embeddings"):
            try:
                # Combine title and description for embedding
                text = f"{title}. {description}" if description else title

                # Create embedding
                if use_openai:
                    embedding = self.create_embedding_openai(text)
                else:
                    embedding = self.create_embedding_local(text)

                # Store in database
                db_manager.insert_embedding(book_id, embedding, model_name)

            except Exception as e:
                logger.error(f"Error creating embedding for book {title}: {e}")
                continue

        logger.info("Embedding generation completed!")


def main():
    """Main function to run the complete pipeline."""
    config = Config()

    # Initialize database
    db_manager = DatabaseManager(config.DB_NAME)

    # Initialize scraper
    scraper = WikipediaScraper(config)

    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator(config)

    print("Book Scraping and Embedding System")
    print("=" * 40)

    # Step 1: Scrape books
    current_book_count = db_manager.get_book_count()
    print(f"Current books in database: {current_book_count}")

    if current_book_count < config.MAX_BOOKS:
        print("Starting book scraping...")
        scraper.scrape_books(db_manager)
    else:
        print("Sufficient books already in database. Skipping scraping.")

    # Step 2: Generate embeddings
    books_without_embeddings = len(db_manager.get_books_without_embeddings())
    print(f"Books without embeddings: {books_without_embeddings}")

    if books_without_embeddings > 0:
        print("\nChoose embedding method:")
        print("1. OpenAI API (requires API key)")
        print("2. Local Sentence Transformers (free)")

        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            api_key = input("Enter your OpenAI API key: ").strip()
            embedding_gen.setup_openai(api_key)
            embedding_gen.generate_embeddings(db_manager, use_openai=True)
        elif choice == "2":
            embedding_gen.generate_embeddings(db_manager, use_openai=False)
        else:
            print("Invalid choice. Exiting.")
            return
    else:
        print("All books already have embeddings!")

    print("\nProcess completed!")
    print(f"Total books: {db_manager.get_book_count()}")
    print(f"Database location: {config.DB_NAME}")


if __name__ == "__main__":
    main()
