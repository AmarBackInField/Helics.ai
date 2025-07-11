# from langchain_openai import ChatOpenAI
# from common.config import MODEL
# from common.models import BrandAnalysisRequest,CompetitorList,BrandDNA,EnhancedQuery
# model_name=MODEL

# class LLMService:
#     def __init__(self) -> None:
#         self.llm=ChatOpenAI(model=model_name)

#     def query_generation(self,BrandAnalysisRequest)->CompetitorList:
#         prompt=f"""You are a market expert , a user will give the some information to find it's compitotrs
#         like his 
#         Brand Name- {BrandAnalysisRequest.brand_name}, 
#         Product - {BrandAnalysisRequest.product},
#         Category - {BrandAnalysisRequest.category}, 
#         Audience - {BrandAnalysisRequest.audience},
#         Location - {BrandAnalysisRequest.location}

#         ## Now, Generate a query to find its top compititors selling same product of same category

#         Example:
#         Brand Name - Nike,
#         Product - Shoes,
#         Category - Fashion,
#         Audience - Genz,Tier 2 people,
#         Location - India
        
#         Response:
#         {{
#             "query":"Find the top compititor of Nike Brand which are targetin Genz, Tier 2 peoples in India"
#         }}

# """
#         Prompt=prompt.format(BrandAnalysisRequest)
#         query=self.llm.invoke(Prompt)
#         return query
    
#     def anaylze_page_structure_for_compititors(self,query,chunk)->BrandDNA:
#         """
#         It will returns the List of compititors
#         arguments:
#             query
#             chunk
#         return:
#             CompetitorList
#         """





#     def query_generation_for_DNA(BrandAnalysisRequest):
#         prompt=f"""You are a market expert , a user will give the some information to generate query for regarding it can gain maximum information about that brand in specific location
#         like his 
#         Brand Name- {BrandAnalysisRequest.brand_name}, 
#         Product - {BrandAnalysisRequest.product},
#         Category - {BrandAnalysisRequest.category}, 
#         Audience - {BrandAnalysisRequest.audience},
#         Location - {BrandAnalysisRequest.location}

#         ## Now, Generate a query to find .

#         Example:
#         Brand Name - Nike,
#         Product - Shoes,
#         Category - Fashion,
#         Audience - Genz,Tier 2 people,
#         Location - India
        
#         Response:
#         {{
#             "query":"Find insights of Nike Shoes Brand which is Targeting Genz, in India"
#         }}

#         """
#         Prompt=prompt.format(BrandAnalysisRequest)
#         query=self.llm.invoke(Prompt)
#         return query
    
#     def enhanced_query(self,query)->EnhancedQuery:
#         """
#         It will generate multple queries originating from original by rephrasing it
#         argument: query
#         return : EnhancedQuery
#         """


from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from common.config import MODEL
from common.models import BrandAnalysisRequest, CompetitorList, BrandDNA, EnhancedQuery,SearchQueryList, AnswerFormat
from common.models import (
    AnswerFormat,
    CompetitorUpdate,
    CompetitorProfile,
    AnalyticsSummary,
    CompetitiveIntelligenceReport,
    Source,
    UpdateType,
    ImpactLevel,
    SentimentScore,
    GeographicScope,
    SourceType
)
import re
import json
import re
import json
from langchain_core.messages import HumanMessage
from datetime import datetime, timedelta
from utils.logger import logger
from datetime import datetime
from typing import List, Dict, Any
model_name = MODEL

class LLMService:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model=model_name)

    def query_generation(self, brand_request: BrandAnalysisRequest) -> str:
        """
        Generate a query to find competitors for a brand
        """
        prompt = f"""You are a market expert. A user will give you information to find competitors.
        
        Brand Name: {brand_request.brand_name}
        Product: {brand_request.product}
        Category: {brand_request.category}
        Audience: {brand_request.audience}
        Location: {brand_request.location}

        Generate a concise search query to find top competitors selling the same product in the same category.

        Example:
        Brand Name: Nike
        Product: Shoes
        Category: Fashion
        Audience: Genz
        Location: India
        
        Response: "top Nike competitors shoes fashion brands targeting Gen Z India"

        Now generate a similar query for the given brand information. Return only the query string.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().replace('"', '')
    
    def finding_competitors(self, query: str, chunks) -> CompetitorList:
        """
        Analyze search results to extract competitor information
        Arguments:
            query: The search query used
            chunk: The search result content
        Returns:
            CompetitorList: List of competitors with relevance scores
        """
        prompt = f"""You are a market research expert. Analyze the following search results to extract competitor brand information.

        Search Query: {query}
        Search Results Content: {chunks}

        Extract competitor brands mentioned in the content. For each competitor, provide:
        1. Brand name
        2. Relevance score (0-1) based on how well they match the search intent
        3. Market position description if available

        Return your analysis in JSON format following this structure:
        {{
            "competitors": [
                {{
                    "name": "Brand Name",
                    "relevance_score": 0.9,
                    "market_position": "Market leader in premium segment"
                }}
            ],
            "total_found": 5
        }}

        Focus on actual brand names, not generic terms. Only include brands that are clearly competitors in the same category.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Extract JSON from response
            response_text = response.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_data = json.loads(json_match.group())
                return CompetitorList(**json_data)
            else:
                # Fallback if JSON parsing fails
                return CompetitorList(competitors=[], total_found=0)
                
        except Exception as e:
            print(f"Error parsing competitor analysis: {e}")
            return CompetitorList(competitors=[], total_found=0)

    def query_generation_for_DNA(self, brand_request: BrandAnalysisRequest) -> str:
        """
        Generate a query to find brand DNA information
        """
        prompt = f"""You are a market expert. Generate a search query to find comprehensive brand insights.
        
        Brand Name: {brand_request.brand_name}
        Product: {brand_request.product}
        Category: {brand_request.category}
        Audience: {brand_request.audience}
        Location: {brand_request.location}

        Generate a query to find detailed brand information including values, positioning, and market presence.

        Example:
        Brand Name: Nike
        Product: Shoes
        Category: Fashion
        Audience: Genz
        Location: India
        
        Response: "Nike brand analysis values positioning target audience India market"

        Generate a similar query for the given brand. Return only the query string.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().replace('"', '')
        
    def clean_and_parse_response(self, response_content: str) -> str:
        """
        Cleans LLM output that may be wrapped in markdown ```json code blocks and returns a clean JSON string.
        """
        # Remove Markdown-style triple backticks
        if response_content.strip().startswith("```"):
            # Use regex to extract the content inside triple backticks
            match = re.search(r"```(?:json)?\s*(.*?)```", response_content, re.DOTALL)
            if match:
                response_content = match.group(1).strip()

        # Validate it's proper JSON
        try:
            json_obj = json.loads(response_content)  # Just to ensure it's valid JSON
            return json.dumps(json_obj)  # Return back as a string for model_validate_json
        except json.JSONDecodeError as e:
            print("❌ JSON parsing error:", e)
            print("🚨 Raw response:\n", response_content)
            return None
    
    

    def query_generation_for_compititor(self, compititor_name):
        """
        Generate 3 Google search queries for gathering top information about a competitor.
        """
        prompt = (
            f"You are a research assistant. Generate 3 distinct Google search queries "
            f"that help gather information about {compititor_name}, including their business model, "
            f"market strategy, recent news, and performance. Return the result as a JSON object matching this schema:\n\n"
            f"{{\n  \"queries\": [\n    {{\"query\": \"...\"}},\n    {{\"query\": \"...\"}},\n    {{\"query\": \"...\"}}\n  ]\n}}"
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        response=self.clean_and_parse_response(response.content)
        parsed = SearchQueryList.model_validate_json(response)
        return parsed
    

    def extract_updates(self, chunks: List, competitor_name: str = "", requesting_company: str = "") -> AnswerFormat:
        """
        Extracting the updates or tracking movement of competitor
        """
        # Combine all chunks into a single text for analysis
        try:
            combined_text = "\n\n".join([doc.page_content for doc in chunks])
        except AttributeError:
            logger.error("Chunks do not contain Document objects with .page_content")
            return self._create_default_answer_format(competitor_name, requesting_company, chunks)

        # Create a comprehensive prompt
        prompt = f"""
        You are a competitive intelligence analyst. Analyze the following text and extract competitor updates and insights.

        Competitor Company: {competitor_name}
        Requesting Company: {requesting_company}

        Text to analyze:
        {combined_text}

        Please provide a comprehensive competitive intelligence report in the following JSON format:

        {{
            "competitive_intelligence_report": {{
                "requesting_company": "{requesting_company}",
                "competitor_company": "{competitor_name}",
                "competitor_profile": {{
                    "company_name": "{competitor_name}",
                    "industry": "extract from text or infer",
                    "headquarters": "extract if mentioned",
                    "founded_year": null,
                    "employee_count": null,
                    "annual_revenue": null,
                    "market_cap": null,
                    "key_executives": [],
                    "primary_markets": [],
                    "main_product_lines": []
                }},
                "updates": [
                    {{
                        "date": "2024-01-01T00:00:00Z",
                        "update_type": "product_launch",
                        "title": "Update title",
                        "summary": "Brief summary",
                        "detailed_description": "Detailed description",
                        "impact_level": "high",
                        "relevance_score": 0.8,
                        "sentiment": "positive",
                        "geographic_scope": "global",
                        "target_audience": ["consumers"],
                        "product_categories": ["category1"],
                        "marketing_metrics": {{
                            "engagement_rate": null,
                            "reach": null,
                            "impressions": null,
                            "mentions": null,
                            "hashtag_performance": null,
                            "influencer_involvement": null
                        }},
                        "competitive_advantages": [],
                        "market_positioning": "description",
                        "sources": [
                            {{
                                "url": "https://example.com",
                                "title": "Source title",
                                "source_type": "news_article",
                                "publication_date": "2024-01-01T00:00:00Z",
                                "author": null,
                                "credibility_score": 0.8
                            }}
                        ],
                        "keywords": ["keyword1", "keyword2"],
                        "estimated_budget": null,
                        "revenue_impact": null,
                        "related_updates": [],
                        "action_items": []
                    }}
                ],
                "analytics_summary": {{
                    "total_updates": 1,
                    "date_range": {{
                        "start": "2024-01-01T00:00:00Z",
                        "end": "2024-12-31T23:59:59Z"
                    }},
                    "update_type_breakdown": {{}},
                    "sentiment_distribution": {{}},
                    "geographic_distribution": {{}},
                    "high_impact_updates_count": 0,
                    "trending_keywords": [],
                    "average_relevance_score": 0.0
                }},
                "search_parameters": {{
                    "competitor_name": "{competitor_name}",
                    "date_range": "last_year",
                    "sources": ["news", "social_media", "press_releases"]
                }},
                "data_sources": ["web_scraping", "news_feeds", "social_media"],
                "confidence_level": 0.8,
                "key_insights": [],
                "strategic_recommendations": [],
                "threat_assessment": "Assessment of competitive threats",
                "opportunity_analysis": "Analysis of market opportunities",
                "monitoring_keywords": [],
                "next_review_date": null
            }},
            "response_time_ms": null,
            "api_version": "v1.0",
            "status": "success"
        }}

        Instructions:
        1. Extract all relevant competitor updates from the text
        2. Classify each update by type (product_launch, marketing_campaign, partnership, etc.)
        3. Assess impact level (high, medium, low) and relevance score (0-1)
        4. Determine sentiment (very_positive, positive, neutral, negative, very_negative)
        5. Identify geographic scope and target audience
        6. Extract any financial information, metrics, or competitive advantages mentioned
        7. Provide actionable insights and recommendations
        8. Ensure all dates are in ISO format
        9. Only include actual information found in the text, use null for missing data
        10. Provide a comprehensive analytics summary

        Return only the JSON response, no additional text.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            parsed_dict = self.clean_and_parse_response_update(response.content)

            if parsed_dict:
                # answer = AnswerFormat.model_validate(parsed_dict)
                # return answer
                return parsed_dict
            response_text = response.content.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                answer_format = AnswerFormat(**parsed_data)

                logger.info(f"✅ Extracted {len(answer_format.competitive_intelligence_report.updates)} updates")
                return answer_format
            else:
                logger.warning("⚠️ No JSON found in LLM response. Returning default.")
                return self._create_default_answer_format(competitor_name, requesting_company, chunks)

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"LLM response: {response_text}")
            return self._create_default_answer_format(competitor_name, requesting_company, chunks)

        except Exception as e:
            logger.error(f"❌ Unexpected error in extract_updates: {e}")
            return self._create_default_answer_format(competitor_name, requesting_company, chunks)
    def clean_and_parse_response_update(self, response_content: str) -> dict:
        """
        Cleans LLM output that may be wrapped in markdown ```json code blocks and returns a valid Python dict.
        """
        # Remove Markdown-style triple backticks if present
        if response_content.strip().startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", response_content, re.DOTALL)
            if match:
                response_content = match.group(1).strip()

        # Parse and return as dict
        try:
            json_obj = json.loads(response_content)
            return json_obj  # This should be passed to model_validate_json
        except json.JSONDecodeError as e:
            print("❌ JSON parsing error:", e)
            print("🚨 Raw response:\n", response_content)
            return None
    def _create_default_answer_format(self, competitor_name: str, requesting_company: str, chunks: List) -> AnswerFormat:
        """
        Create a default AnswerFormat when LLM processing fails
        """
        current_time = datetime.now().isoformat()

        # Extract plain text from Document objects
        try:
            joined_chunks = "\n".join([doc.page_content for doc in chunks])
        except Exception:
            joined_chunks = "\n".join([str(doc) for doc in chunks])

        detailed_description = joined_chunks[:1000] + "..." if len(joined_chunks) > 1000 else joined_chunks

        default_update = CompetitorUpdate(
            date=current_time,
            update_type=UpdateType.OTHER,
            title=f"Competitor Intelligence Update - {competitor_name}",
            summary=f"Raw data extracted for {competitor_name}",
            detailed_description=detailed_description,
            impact_level=ImpactLevel.MEDIUM,
            relevance_score=0.5,
            sentiment=SentimentScore.NEUTRAL,
            geographic_scope=GeographicScope.GLOBAL,
            sources=[
                Source(
                    url="https://internal-analysis.com",
                    title="Internal Analysis",
                    source_type=SourceType.OTHER,
                    publication_date=current_time,
                    credibility_score=0.5
                )
            ],
            keywords=["competitor", "intelligence", "analysis"]
        )

        competitor_profile = CompetitorProfile(
            company_name=competitor_name or "Unknown Competitor",
            industry="Unknown"
        )

        analytics_summary = AnalyticsSummary(
            total_updates=1,
            date_range={"start": current_time, "end": current_time},
            update_type_breakdown={"other": 1},
            sentiment_distribution={"neutral": 1},
            geographic_distribution={"global": 1},
            high_impact_updates_count=0,
            trending_keywords=["competitor", "analysis"],
            average_relevance_score=0.5
        )

        report = CompetitiveIntelligenceReport(
            requesting_company=requesting_company or "Unknown Company",
            competitor_company=competitor_name or "Unknown Competitor",
            competitor_profile=competitor_profile,
            updates=[default_update],
            analytics_summary=analytics_summary,
            search_parameters={"competitor_name": competitor_name, "analysis_type": "text_extraction"},
            data_sources=["text_analysis"],
            confidence_level=0.3,
            key_insights=["Limited data available for analysis"],
            strategic_recommendations=["Gather more comprehensive competitor data"],
            threat_assessment="Unable to assess threat level with current data",
            opportunity_analysis="Insufficient data for opportunity analysis",
            monitoring_keywords=[competitor_name] if competitor_name else ["competitor"],
            next_review_date=(datetime.now() + timedelta(days=30)).isoformat()
        )

        return AnswerFormat(competitive_intelligence_report=report)
    
    def enhanced_query(self, query: str) -> EnhancedQuery:
        """
        Generate multiple enhanced queries from the original query
        Arguments:
            query: Original search query
        Returns:
            EnhancedQuery: Object containing original query, enhanced versions, and search intent
        """
        prompt = f"""You are a search optimization expert. Take the original query and generate multiple enhanced versions to improve search results.

        Original Query: {query}

        Generate 2-3 enhanced queries that:
        1. Rephrase the original query with different keywords
        2. Add context-specific terms
        3. Use different search patterns (long-tail, specific, broad)
        4. Include synonyms and related terms

        Also identify the search intent category from: informational, commercial, navigational, transactional, investigational

        Return in JSON format:
        {{
            "original_query": "{query}",
            "enhanced_queries": [
                "enhanced query 1",
                "enhanced query 2",
                "enhanced query 3",
                "enhanced query 4",
                "enhanced query 5"
            ],
            "search_intent": "informational"
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Extract JSON from response
            response_text = response.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_data = json.loads(json_match.group())
                return EnhancedQuery(**json_data)
            else:
                # Fallback if JSON parsing fails
                return EnhancedQuery(
                    original_query=query,
                    enhanced_queries=[query],
                    search_intent="informational"
                )
                
        except Exception as e:
            print(f"Error parsing enhanced query: {e}")
            return EnhancedQuery(
                original_query=query,
                enhanced_queries=[query],
                search_intent="informational"
            )

    def analyze_brand_DNA(self, brand_request: BrandAnalysisRequest, content) -> BrandDNA:
        """
        Analyze content to extract brand DNA information
        Arguments:
            brand_request: Brand analysis request with brand details
            content: Content to analyze for brand DNA
        Returns:
            BrandDNA: Comprehensive brand DNA analysis
        """
        prompt = f"""You are a brand strategist. Analyze the following content to extract comprehensive brand DNA for {brand_request.brand_name}.

        Brand: {brand_request.brand_name}
        Product: {brand_request.product}
        Category: {brand_request.category}
        Target Audience: {brand_request.audience}
        Location: {brand_request.location}

        Content to analyze: {content}

        Extract and analyze the following brand DNA elements:
        1. Core Values (list of 3-5 key values)
        2. Personality Traits (list of 3-5 traits)
        3. Positioning Statement (concise statement)
        4. Unique Selling Proposition (USP)
        5. Target Audience segments
        6. Brand Voice and Tone
        7. Visual Identity description
        8. Emotional Connection with audience
        9. Market Differentiation
        10. Brand Promise

        Return in JSON format following this structure:
        {{
            "brand_name": "{brand_request.brand_name}",
            "core_values": ["value1", "value2", "value3"],
            "personality_traits": ["trait1", "trait2", "trait3"],
            "positioning": "positioning statement",
            "unique_selling_proposition": "USP description",
            "target_audience": ["segment1", "segment2"],
            "brand_voice": "voice description",
            "visual_identity": "visual identity description",
            "emotional_connection": "emotional connection description",
            "market_differentiation": "differentiation description",
            "brand_promise": "brand promise description",
            "attributes": [
                {{
                    "attribute": "Innovation",
                    "value": "High focus on innovation",
                    "strength": 0.8,
                    "source": "website analysis"
                }}
            ]
        }}

        Base your analysis on the provided content and ensure all fields are filled with relevant information.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Extract JSON from response
            response_text = response.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_data = json.loads(json_match.group())
                return BrandDNA(**json_data)
            else:
                # Fallback if JSON parsing fails
                return BrandDNA(brand_name=brand_request.brand_name)
                
        except Exception as e:
            print(f"Error parsing brand DNA analysis: {e}")
            return BrandDNA(brand_name=brand_request.brand_name)
        
    





