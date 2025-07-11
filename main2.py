from LLMService.llm_service import LLMService
from RAGService.RAGService import RAGService
from SearchService.SearchService import SearchExtractor
import json
from pathlib import Path
from typing import Any
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict, is_dataclass
import logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

llm_service=LLMService()
rag_service=RAGService()
search_service=SearchExtractor()

#query generation
queries=llm_service.query_generation_for_compititor(compititor_name="Adidas")
urls=[]
for i in queries.queries:
    url=search_service._perform_google_search(i.query,3)
    urls.extend(url)

documents=rag_service._ingestion(urls=urls)
rag_service._save(documents=documents)


chunks=[]
# Queries generation
for i in queries.queries:
    # Search for relevant chunks
    chunk = rag_service._search(query=i.query, top_k=6)
    chunks.extend(chunk)

rag_service._delete()
print("Lenght of Chunks : - ",len(chunks))
unique_chunks = rag_service.remove_duplicate(chunks=chunks)
print("Lenght of Unique Chunks : - ",len(chunks))

response=llm_service.extract_updates(chunks=unique_chunks,competitor_name="Adidas", requesting_company="Nike")
save_to_json(response, f"brand_update_adidas_{timestamp}.json")