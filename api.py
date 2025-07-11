from flask import Flask, request, jsonify,send_file
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
import traceback
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def save_to_json(data, filename):
    """Save data to JSON file with proper error handling"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving to JSON: {str(e)}")
        return False

def process_competitor_analysis(competitor_name: str, requesting_company: str = "Nike"):
    """Main processing function for competitor analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info(f"Starting competitor analysis for {competitor_name}")
        
        # Initialize services
        llm_service = LLMService()
        rag_service = RAGService()
        search_service = SearchExtractor()
        
        logger.info("Services initialized successfully")
        
        # Query generation
        logger.info("Generating queries for competitor analysis")
        queries = llm_service.query_generation_for_compititor(compititor_name=competitor_name)
        logger.info(f"Generated {len(queries.queries)} queries")
        
        # Search for URLs
        urls = []
        for i, query in enumerate(queries.queries):
            logger.info(f"Searching for query {i+1}: {query.query}")
            url = search_service._perform_google_search(query.query, 3)
            urls.extend(url)
        
        logger.info(f"Found {len(urls)} URLs total")
        
        # Document ingestion
        logger.info("Starting document ingestion")
        documents = rag_service._ingestion(urls=urls)
        rag_service._save(documents=documents)
        logger.info(f"Ingested {len(documents)} documents")
        
        # Search for relevant chunks
        chunks = []
        for i, query in enumerate(queries.queries):
            logger.info(f"Searching chunks for query {i+1}: {query.query}")
            chunk = rag_service._search(query=query.query, top_k=6)
            chunks.extend(chunk)
        
        # Clean up
        rag_service._delete()
        logger.info(f"Found {len(chunks)} chunks total")
        
        # Remove duplicates
        unique_chunks = rag_service.remove_duplicate(chunks=chunks)
        logger.info(f"Removed duplicates, left with {len(unique_chunks)} unique chunks")
        
        # Extract updates
        logger.info("Extracting competitor updates")
        response = llm_service.extract_updates(
            chunks=unique_chunks,
            competitor_name=competitor_name,
            requesting_company=requesting_company
        )
        
        # Save results
        filename = f"brand_update_{competitor_name.lower()}_{timestamp}.json"
        if save_to_json(response, filename):
            logger.info(f"Analysis completed successfully for {competitor_name}")
            return {
                "status": "success",
                "competitor": competitor_name,
                "requesting_company": requesting_company,
                "timestamp": timestamp,
                "total_queries": len(queries.queries),
                "total_urls": len(urls),
                "total_documents": len(documents),
                "total_chunks": len(chunks),
                "unique_chunks": len(unique_chunks),
                "output_file": filename,
                "message": "Competitor analysis completed successfully"
            }
        else:
            raise Exception("Failed to save results to JSON file")
            
    except Exception as e:
        logger.error(f"Error in competitor analysis: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated JSON file"""
    try:
        logger.info(f"Download requested for file: {filename}")
        
        # Security check - only allow JSON files with specific pattern
        if not filename.endswith('.json') or not filename.startswith('brand_update_'):
            logger.warning(f"Invalid file request: {filename}")
            return jsonify({
                "status": "error",
                "message": "Invalid file requested"
            }), 400
        
        # Check if file exists
        file_path = Path(filename)
        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            return jsonify({
                "status": "error",
                "message": "File not found"
            }), 404
        
        logger.info(f"Serving file: {filename}")
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error serving file",
            "error": str(e)
        }), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        logger.info("Health check requested")
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "API is running successfully"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/main', methods=['POST'])
def main_analysis():
    """Main competitor analysis endpoint"""
    try:
        logger.info("Main analysis endpoint called")
        
        # Get request data
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Extract parameters
        competitor_name = data.get('competitor_name')
        requesting_company = data.get('requesting_company', 'Nike')
        
        if not competitor_name:
            logger.warning("No competitor_name provided in request")
            return jsonify({
                "status": "error",
                "message": "competitor_name is required"
            }), 400
        
        logger.info(f"Processing analysis for competitor: {competitor_name}, requesting company: {requesting_company}")
        
        # Process the analysis
        result = process_competitor_analysis(competitor_name, requesting_company)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in main analysis endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(error)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, host='127.0.0.1', port=8001)