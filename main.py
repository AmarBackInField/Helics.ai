from common.models import BrandAnalysisRequest
from LLMService.llm_service import LLMService
from RAGService.RAGService import RAGService
from SearchService.SearchService import SearchExtractor

llm_service=LLMService()
rag_service=RAGService()
search_service=SearchExtractor()


request=BrandAnalysisRequest(brand_name="nike",product="shoes",category="fashion",audience="GenZ",location="India")


query=llm_service.query_generation(request)
urls=search_service._perform_google_search(query=query,num_results=3)
documents=rag_service._ingestion(urls=urls)
rag_service._save(documents=documents)
chunks=rag_service._search(query=query,top_k=6)
compititors=llm_service.finding_competitors(query,chunks) # List of compititors

with open("compitor.json","w") as file:
    file(compititors)

names=[]
for i in compititors.compititors:
    names.append(i['name'])

requests=[]
for i in range(len(names)+1):
    if(i==0):
        requests.append(BrandAnalysisRequest(brand_name="nike",product="shoes",category="fashion",audience="GenZ",location="India"))
    else:
        requests.append(BrandAnalysisRequest(brand_name=names[i],product="shoes",category="fashion",audience="GenZ",location="India"))

queries=[]
for i in range(len(requests)):
    query=llm_service.query_generation_for_DNA(requests[i])
    queries.append(query)

j=0
ans=[]
for query in queries:
    urls=search_service._perform_google_search(query=query,num_results=3)
    documents=rag_service._ingestion(urls=urls)
    rag_service._save(documents=documents)
    response=llm_service.enhanced_query(query)
    multiple_queries=response.enhanced_queries#list of queries
    chunks=[]
    for i in multiple_queries:
        chunk=rag_service._search(query=i,top_k=6)
        chunks.extend(chunk)
    unique_chunk=rag_service.remove_duplicate(chunks=chunks)
    final_response=llm_service.analyze_brand_DNA(requests[j],content=unique_chunk)
    ans.append(final_response)
