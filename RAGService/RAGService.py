import os
import pandas as pd
from langchain_openai import ChatOpenAI
from typing import List
import requests
from bs4 import BeautifulSoup
# from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

class RAGService:
    def __init__(self) -> None:
        """
        Initializes the RAGService class.

        - Sets up the OpenAI embedding model.
        - Defines the local path to store the FAISS vector database.
        """
        self.embedding_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4.1-mini")
        self.faiss_path = "faiss_index"

    # def _ingestion(self, urls: List[str]) -> List[Document]:
    #     """
    #     Loads and chunks content from multiple website URLs into Langchain Documents.

    #     Args:
    #         urls (List[str]): A list of website URLs to load content from.

    #     Returns:
    #         List[Document]: A list of Langchain Document objects created from the websites' content.
    #     """
    #     all_documents = []

    #     for url in urls:
    #         try:
    #             loader = WebBaseLoader(url)
    #             raw_documents = loader.load()
    #             splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    #             split_docs = splitter.split_documents(raw_documents)
    #             all_documents.extend(split_docs)
    #             print(f"Ingested {len(split_docs)} chunks from {url}")
    #         except Exception as e:
    #             print(f"Failed to ingest from {url}: {e}")

    #     self.documents = all_documents
    #     return self.documents

    def _ingestion(self, urls: List[str]) -> List[Document]:
        all_documents = []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }

        MAX_CHUNKS_PER_URL = 300  # Limit per URL to prevent overload

        for url in urls:
            try:
                # ✅ Step 0: Skip PDF URLs
                if url.lower().endswith(".pdf"):
                    print(f"Skipping PDF URL: {url}")
                    continue

                # ✅ Step 1: Fetch webpage content
                response = requests.get(url, headers=headers, timeout=(5, 15))
                response.raise_for_status()

                # ✅ Step 2: Extract clean text
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)

                if not text or len(text) < 100:
                    print(f"Skipping empty or too short content from {url}")
                    continue

                # ✅ Step 3: Wrap in Langchain Document
                document = Document(page_content=text, metadata={"source": url})

                # ✅ Step 4: Chunk uniformly
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1800,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", " ", ""]
                )
                split_docs = splitter.split_documents([document])

                # ✅ Step 5: Truncate to max chunks if needed
                if len(split_docs) > MAX_CHUNKS_PER_URL:
                    split_docs = split_docs[:MAX_CHUNKS_PER_URL]
                    print(f"Truncated to {MAX_CHUNKS_PER_URL} chunks for {url}")

                all_documents.extend(split_docs)
                print(f"Ingested {len(split_docs)} chunks from {url}")

            except Exception as e:
                print(f"Failed to ingest from {url}: {e}")

        self.documents = all_documents
        return self.documents
    def _save(self, documents):
        """
        Embeds the ingested documents using the OpenAI embedding model and saves them
        in a FAISS vector store locally.
        """
        if not hasattr(self, "documents"):
            raise ValueError("Documents not ingested. Run _ingestion first.")
        vector_store = FAISS.from_documents(documents, self.embedding_model)
        vector_store.save_local(self.faiss_path)
        print(f"Saved FAISS index with {len(documents)} documents at {self.faiss_path}")

    def _load(self):
        """
        Loads the FAISS vector store from the local disk into memory.
        """
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError("FAISS index not found. Run _save to create it.")

        self.vector_store = FAISS.load_local(
            self.faiss_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded.")
        return self.vector_store

    def _search(self, query: str, top_k: int = 5):
        """
        Searches the FAISS vector store for the top_k most relevant chunks based on the query.

        Args:
            query (str): The natural language query to search for.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            List[str]: A list of the most relevant content chunks matching the query.
        """
        if not hasattr(self, "vector_store"):
            self._load()
        results = self.vector_store.similarity_search(query, k=top_k)
        return results

    def _delete(self):
        """
        Deletes the FAISS index directory and its contents if it exists.
        """
        if os.path.exists(self.faiss_path):
            for file in os.listdir(self.faiss_path):
                os.remove(os.path.join(self.faiss_path, file))
            os.rmdir(self.faiss_path)
            print(f"Deleted FAISS index at {self.faiss_path}")
        else:
            print("No FAISS index to delete.")

    def remove_duplicate(self, chunks: List[Document]) -> List[Document]:
        """
        Removes duplicate content chunks based on their page content.

        Args:
            chunks (List[Document]): The list of Document chunks to deduplicate.

        Returns:
            List[Document]: The deduplicated list of Documents.
        """
        seen = set()
        unique_docs = []
        for doc in chunks:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)
        print(f"Reduced {len(chunks)} chunks to {len(unique_docs)} unique chunks.")
        return unique_docs

    
    

