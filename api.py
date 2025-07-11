from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import json
import logging

from LLMService.llm_service import LLMService
from RAGService.RAGService import RAGService
from SearchService.SearchService import SearchExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Request model
class BrandUpdateRequest(BaseModel):
    competitor_name: str
    requesting_company: str

# Initialize services
llm_service = LLMService()
rag_service = RAGService()
search_service = SearchExtractor()

# Utility function to save output
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@app.post("/generate_brand_update")
def generate_brand_update(request: BrandUpdateRequest):
    try:
        logger.info(f"Generating queries for {request.competitor_name}")
        queries = llm_service.query_generation_for_compititor(
            compititor_name=request.competitor_name
        )

        urls = []
        for q in queries.queries:
            search_results = search_service._perform_google_search(q.query, 3)
            urls.extend(search_results)

        logger.info(f"Found {len(urls)} URLs")
        documents = rag_service._ingestion(urls=urls)
        rag_service._save(documents=documents)

        chunks = []
        for q in queries.queries:
            chunk = rag_service._search(query=q.query, top_k=6)
            chunks.extend(chunk)

        rag_service._delete()
        logger.info(f"Total chunks: {len(chunks)}")
        unique_chunks = rag_service.remove_duplicate(chunks=chunks)
        logger.info(f"Unique chunks: {len(unique_chunks)}")

        response = llm_service.extract_updates(
            chunks=unique_chunks,
            competitor_name=request.competitor_name,
            requesting_company=request.requesting_company,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brand_update_{request.competitor_name.lower()}_{timestamp}.json"
        save_to_json(response, filename)

        return {"status": "success", "file": filename, "data": response}

    except Exception as e:
        logger.error(f"Error generating brand update: {e}")
        raise HTTPException(status_code=500, detail=str(e))
