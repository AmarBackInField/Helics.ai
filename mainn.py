import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from common.models import BrandAnalysisRequest
from LLMService.llm_service import LLMService
from RAGService.RAGService import RAGService
from SearchService.SearchService import SearchExtractor

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"brand_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

# def save_to_json(data: Any, filename: str, result_folder: str = "results") -> None:
#     """Save data to JSON file in results folder"""
#     try:
#         result_dir = Path(result_folder)
#         result_dir.mkdir(exist_ok=True)
        
#         filepath = result_dir / filename
#         with open(filepath, 'w', encoding='utf-8') as file:
#             json.dump(data, file, indent=2, ensure_ascii=False)
        
#         logger.info(f"Successfully saved data to {filepath}")
    
#     except Exception as e:
#         logger.error(f"Error saving data to {filename}: {str(e)}")
#         raise
import json
from pathlib import Path
from typing import Any
from dataclasses import asdict, is_dataclass
import logging

logger = logging.getLogger(__name__)
def convert_to_serializable(obj):
    """Recursively convert custom objects to JSON-safe dicts."""
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    else:
        return obj

def save_to_json(data: Any, filename: str, result_folder: str = "results") -> None:
    try:
        Path(result_folder).mkdir(parents=True, exist_ok=True)
        filepath = Path(result_folder) / filename

        serializable_data = convert_to_serializable(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filename}: {str(e)}")
        raise


def main():
    """Main function to run brand analysis"""
    global logger
    logger = setup_logging()
    
    try:
        logger.info("Starting brand analysis process")
        
        # Initialize services
        logger.info("Initializing services...")
        llm_service = LLMService()
        rag_service = RAGService()
        search_service = SearchExtractor()
        logger.info("Services initialized successfully")
        
        # Create initial request

        
        request = BrandAnalysisRequest(
            brand_name="nike",
            product="shoes",
            category="fashion",
            audience="genz",
            location="India"
        )
        logger.info(f"Created initial request for brand: {request.brand_name}")
        
        # Generate query and search
        logger.info("Generating initial query...")
        query = llm_service.query_generation(request)
        logger.info(f"Generated query: {query}")
        
        logger.info("Performing Google search...")
        urls = search_service._perform_google_search(query=query, num_results=3)
        logger.info(f"Found {len(urls)} URLs")
        
        # Process documents
        logger.info("Processing documents...")
        documents = rag_service._ingestion(urls=urls)
        rag_service._save(documents=documents)
        logger.info(f"Processed {len(documents)} documents")
        
        # Search for relevant chunks
        logger.info("Searching for relevant chunks...")
        chunks = rag_service._search(query=query, top_k=6)
        logger.info(f"Found {len(chunks)} relevant chunks")
        
        # Find competitors
        logger.info("Finding competitors...")
        competitors = llm_service.finding_competitors(query, chunks)
        rag_service._delete() # deleting older storage
        logger.info(f"Found competitors: {competitors}")
        
        # Save competitors data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_json(competitors, f"competitors_{timestamp}.json")
        
        # Extract competitor names
        names = []
        for comp in competitors.competitors:
            if hasattr(comp, 'name'):
                names.append(comp.name)
        logger.info(f"Extracted competitor names: {names}")
        
        # Create requests for all brands (original + competitors)
        logger.info("Creating brand analysis requests...")
        requests = []
        
        # Add original brand request
        requests.append(BrandAnalysisRequest(
            brand_name="nike",
            product="shoes",
            category="fashion",
            audience="genz",
            location="India"
        ))
        
        # Add competitor requests
        for name in names:
            requests.append(BrandAnalysisRequest(
                brand_name=name,
                product="shoes",
                category="fashion",
                audience="genz",
                location="India"
            ))
        
        logger.info(f"Created {len(requests)} brand analysis requests")
        
        # Generate DNA queries for all brands
        logger.info("Generating DNA queries...")
        queries = []
        for i, req in enumerate(requests):
            logger.info(f"Generating DNA query for brand {i+1}/{len(requests)}: {req.brand_name}")
            query = llm_service.query_generation_for_DNA(req)
            queries.append(query)
        
        logger.info(f"Generated {len(queries)} DNA queries")
        
        # Analyze each brand
        logger.info("Starting brand DNA analysis...")
        brand_analyses = []
        
        for j, query in enumerate(queries):
            brand_name = requests[j].brand_name
            logger.info(f"Analyzing brand {j+1}/{len(queries)}: {brand_name}")
            
            try:
                # Search for brand-specific content
                logger.info(f"Searching for {brand_name} content...")
                urls = search_service._perform_google_search(query=query, num_results=3)
                logger.info(f"Found {len(urls)} URLs for {brand_name}")
                
                # Process documents
                documents = rag_service._ingestion(urls=urls)
                rag_service._save(documents=documents)
                logger.info(f"Processed {len(documents)} documents for {brand_name}")
                
                # Enhanced query generation
                logger.info(f"Generating enhanced queries for {brand_name}...")
                response = llm_service.enhanced_query(query)
                multiple_queries = response.enhanced_queries  # list of queries
                logger.info(f"Generated {len(multiple_queries)} enhanced queries for {brand_name}")
                
                # Collect chunks from all enhanced queries
                chunks = []
                for i, enhanced_query in enumerate(multiple_queries):
                    logger.info(f"Processing enhanced query {i+1}/{len(multiple_queries)} for {brand_name}")
                    chunk = rag_service._search(query=enhanced_query, top_k=6)
                    chunks.extend(chunk)
                
                # Remove duplicates
                logger.info(f"Removing duplicates from {len(chunks)} chunks for {brand_name}")
                unique_chunks = rag_service.remove_duplicate(chunks=chunks)
                logger.info(f"After deduplication: {len(unique_chunks)} unique chunks for {brand_name}")
                
                # Analyze brand DNA
                logger.info(f"Analyzing DNA for {brand_name}...")
                final_response = llm_service.analyze_brand_DNA(requests[j], content=unique_chunks)
                
                # brand_analysis = {
                #     'brand_name': brand_name,
                #     'request': requests[j].__dict__,
                #     'query': query,
                #     'enhanced_queries': multiple_queries,
                #     'chunks_count': len(unique_chunks),
                #     'analysis': final_response
                # }
                
                # brand_analyses.append(brand_analysis)
                brand_analyses.append(final_response)
                logger.info(f"Successfully analyzed {final_response.brand_name}")
                
                # Save individual brand analysis
                save_to_json(final_response, f"brand_analysis_{brand_name.lower()}_{timestamp}.json")

                # CLean old storage
                rag_service._delete()
                
            except Exception as e:
                logger.error(f"Error analyzing brand {brand_name}: {str(e)}")
                # Continue with next brand instead of failing completely
                continue
        
        # Save complete analysis results
        logger.info("Saving complete analysis results...")
        complete_results = {
            'timestamp': timestamp,
            'original_request': request.__dict__,
            'competitors': competitors,
            'total_brands_analyzed': len(brand_analyses),
            'brand_analyses': brand_analyses
        }
        
        save_to_json(complete_results, f"complete_brand_analysis_{timestamp}.json")
        
        logger.info(f"Brand analysis completed successfully! Analyzed {len(brand_analyses)} brands")
        logger.info("Results saved in 'results' folder")
        
        return brand_analyses
        
    except Exception as e:
        logger.error(f"Fatal error in brand analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = main()
        print(f"\nAnalysis completed successfully! Processed {len(results)} brands.")
        print("Check the 'results' folder for detailed JSON files.")
        print("Check the 'logs' folder for execution logs.")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        print("Check the logs for detailed error information.")