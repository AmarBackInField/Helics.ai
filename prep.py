from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import json
import logging
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for structured data
class UpdateType(str, Enum):
    PRODUCT_LAUNCH = "product_launch"
    MARKETING_CAMPAIGN = "marketing_campaign"
    PARTNERSHIP = "partnership"
    SPONSORSHIP = "sponsorship"
    PRICE_CHANGE = "price_change"
    STORE_OPENING = "store_opening"
    FINANCIAL_NEWS = "financial_news"
    EXECUTIVE_CHANGE = "executive_change"
    SUSTAINABILITY_INITIATIVE = "sustainability_initiative"
    TECHNOLOGY_UPDATE = "technology_update"
    SOCIAL_MEDIA_CAMPAIGN = "social_media_campaign"
    ACQUISITION = "acquisition"
    RECALL = "recall"
    AWARD = "award"
    OTHER = "other"

class SentimentScore(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class SourceType(str, Enum):
    NEWS_ARTICLE = "news_article"
    PRESS_RELEASE = "press_release"
    SOCIAL_MEDIA = "social_media"
    BLOG_POST = "blog_post"
    FINANCIAL_REPORT = "financial_report"
    INDUSTRY_REPORT = "industry_report"
    COMPANY_WEBSITE = "company_website"
    REVIEW_SITE = "review_site"
    FORUM = "forum"
    PODCAST = "podcast"
    VIDEO = "video"
    OTHER = "other"

class GeographicScope(str, Enum):
    GLOBAL = "global"
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    COUNTRY_SPECIFIC = "country_specific"
    REGIONAL = "regional"

class ImpactLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Pydantic models for structured data
class Source(BaseModel):
    """Source information for the competitive intelligence data"""
    url: str = Field(..., description="URL of the source")
    title: str = Field(..., description="Title of the source article/content")
    source_type: SourceType = Field(..., description="Type of source")
    publication_date: str = Field(..., description="When the content was published (ISO format)")
    author: Optional[str] = Field(None, description="Author of the content")
    credibility_score: Optional[float] = Field(None, ge=0, le=1, description="Credibility score between 0-1")

class MarketingMetrics(BaseModel):
    """Marketing-specific metrics and KPIs"""
    engagement_rate: Optional[float] = Field(None, description="Social media engagement rate")
    reach: Optional[int] = Field(None, description="Estimated reach of the campaign/update")
    impressions: Optional[int] = Field(None, description="Number of impressions")
    mentions: Optional[int] = Field(None, description="Number of mentions across platforms")
    hashtag_performance: Optional[Dict[str, int]] = Field(None, description="Hashtag usage and performance")
    influencer_involvement: Optional[List[str]] = Field(None, description="List of influencers involved")

class CompetitorUpdate(BaseModel):
    """Individual competitor update/news item"""
    update_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the update")
    date: str = Field(..., description="Date when the update occurred (ISO format)")
    update_type: UpdateType = Field(..., description="Type of update")
    title: str = Field(..., description="Title/headline of the update")
    summary: str = Field(..., description="Brief summary of the update")
    detailed_description: str = Field(..., description="Detailed description of the update")
    
    # Impact and relevance
    impact_level: ImpactLevel = Field(..., description="Assessed impact level")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score for the requesting company")
    sentiment: SentimentScore = Field(..., description="Overall sentiment of the update")
    
    # Geographic and market context
    geographic_scope: GeographicScope = Field(..., description="Geographic scope of the update")
    target_audience: Optional[List[str]] = Field(None, description="Target audience segments")
    product_categories: Optional[List[str]] = Field(None, description="Relevant product categories")
    
    # Marketing insights
    marketing_metrics: Optional[MarketingMetrics] = Field(None, description="Marketing-related metrics")
    competitive_advantages: Optional[List[str]] = Field(None, description="Competitive advantages highlighted")
    market_positioning: Optional[str] = Field(None, description="Market positioning strategy")
    
    # Sources and verification
    sources: List[Source] = Field(..., description="Sources for this update")
    keywords: List[str] = Field(..., description="Relevant keywords and tags")
    
    # Financial impact (if applicable)
    estimated_budget: Optional[float] = Field(None, description="Estimated budget for the initiative")
    revenue_impact: Optional[str] = Field(None, description="Expected revenue impact")
    
    # Additional context
    related_updates: Optional[List[str]] = Field(None, description="IDs of related updates")
    action_items: Optional[List[str]] = Field(None, description="Suggested action items for the requesting company")

class CompetitorProfile(BaseModel):
    """Profile information about the competitor"""
    company_name: str = Field(..., description="Name of the competitor company")
    industry: str = Field(..., description="Primary industry")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    founded_year: Optional[int] = Field(None, description="Year founded")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    annual_revenue: Optional[float] = Field(None, description="Annual revenue in USD")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    key_executives: Optional[List[str]] = Field(None, description="Key executive names")
    primary_markets: Optional[List[str]] = Field(None, description="Primary geographic markets")
    main_product_lines: Optional[List[str]] = Field(None, description="Main product lines")

class AnalyticsSummary(BaseModel):
    """Summary analytics for the competitive intelligence report"""
    total_updates: int = Field(..., description="Total number of updates found")
    date_range: Dict[str, str] = Field(..., description="Start and end dates of the analysis (ISO format)")
    update_type_breakdown: Dict[str, int] = Field(..., description="Count by update type")
    sentiment_distribution: Dict[str, int] = Field(..., description="Distribution of sentiment scores")
    geographic_distribution: Dict[str, int] = Field(..., description="Geographic distribution")
    high_impact_updates_count: int = Field(..., description="Number of high-impact updates")
    trending_keywords: List[str] = Field(..., description="Most frequently mentioned keywords")
    average_relevance_score: float = Field(..., ge=0, le=1, description="Average relevance score")

class CompetitiveIntelligenceReport(BaseModel):
    """Complete competitive intelligence report"""
    report_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique report identifier")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Report generation timestamp")
    requesting_company: str = Field(..., description="Company requesting the intelligence")
    competitor_company: str = Field(..., description="Competitor being analyzed")
    
    # Core data
    competitor_profile: CompetitorProfile = Field(..., description="Profile of the competitor")
    updates: List[CompetitorUpdate] = Field(..., description="List of competitor updates")
    analytics_summary: AnalyticsSummary = Field(..., description="Summary analytics")
    
    # Report metadata
    search_parameters: Dict[str, Any] = Field(..., description="Parameters used for the search")
    data_sources: List[str] = Field(..., description="Data sources consulted")
    confidence_level: float = Field(..., ge=0, le=1, description="Overall confidence in the data")
    
    # Insights and recommendations
    key_insights: List[str] = Field(..., description="Key insights from the analysis")
    strategic_recommendations: List[str] = Field(..., description="Strategic recommendations")
    threat_assessment: str = Field(..., description="Assessment of competitive threats")
    opportunity_analysis: str = Field(..., description="Analysis of market opportunities")
    
    # Future monitoring
    monitoring_keywords: List[str] = Field(..., description="Keywords for future monitoring")
    next_review_date: Optional[str] = Field(None, description="Suggested next review date (ISO format)")

class AnswerFormat(BaseModel):
    """Main response format for the AI Marketing Intelligence API"""
    competitive_intelligence_report: CompetitiveIntelligenceReport = Field(
        ..., 
        description="Complete competitive intelligence report"
    )
    
    # API response metadata
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    api_version: str = Field(default="v1.0", description="API version")
    status: str = Field(default="success", description="Response status")

# Main LLM class
class PerplexityLLM(LLM):
    api_key: str 
    model: str = "sonar-pro"
    endpoint: str = "https://api.perplexity.ai/chat/completions"

    def _call(self, prompt: str, start_date: str = None, last_date: str = None, **kwargs) -> str:
        """Make API call to Perplexity AI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional competitive intelligence analyst. Provide comprehensive, accurate, and actionable insights."},
                {"role": "user", "content": prompt}
            ],
            # "response_format": {
            #     "type": "json_schema",
            #     "json_schema": {
            #         "schema": AnswerFormat.model_json_schema()
            #     }
            # },
            "web_search_options": {
                "search_context_size": "high"
            }
        }

        # Add date filters if provided
        # if start_date:
        #     payload["search_after_date_filter"] = start_date
        # if last_date:
        #     payload["search_before_date_filter"] = last_date

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise ValueError(f"API Error: {e}")
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            raise ValueError(f"Unexpected response format: {e}")

    @property
    def _llm_type(self) -> str:
        return "perplexity"

# Service class for competitive intelligence
class CompetitiveIntelligenceService:
    """Service class for handling competitive intelligence operations"""
    
    def __init__(self, perplexity_llm: PerplexityLLM):
        self.llm = perplexity_llm
    
    def generate_search_prompt(self, requesting_company: str, competitor: str, 
                             start_date: str, end_date: str) -> str:
        """Generate a structured prompt for competitive intelligence gathering"""
        return f"""
        Generate a comprehensive competitive intelligence report for {requesting_company} 
        about their competitor {competitor}.
        
        Search for all relevant updates, news, and developments about {competitor} 
        from {start_date} to {end_date}.
        
        Focus on gathering information about:
        - Product launches and updates
        - Marketing campaigns and strategies
        - Partnerships and collaborations
        - Financial performance and announcements
        - Strategic initiatives and business moves
        - Market positioning changes
        - Technology innovations
        - Sustainability and CSR initiatives
        - Executive changes and leadership updates
        - Store openings and expansion plans
        - Price changes and promotional activities
        - Awards and recognition received
        - Social media campaigns and engagement
        
        For each update found, provide:
        - Detailed description and context
        - Impact assessment for {requesting_company}
        - Relevance score and sentiment analysis
        - Source information and credibility
        - Geographic scope and target audience
        - Competitive advantages identified
        - Strategic implications
        
        Also include:
        - Competitive threats and opportunities analysis
        - Strategic recommendations for {requesting_company}
        - Market trend implications
        - Future monitoring suggestions
        
        Ensure the response follows the CompetitiveIntelligenceReport schema exactly.
        Be thorough, accurate, and provide actionable insights.
        """
    
    def get_competitor_intelligence(self, requesting_company: str, competitor: str, 
                                  start_date: str, end_date: str) -> CompetitiveIntelligenceReport:
        """Get competitive intelligence report"""
        try:
            prompt = self.generate_search_prompt(requesting_company, competitor, start_date, end_date)
            
            logger.info(f"Generating competitive intelligence report for {requesting_company} vs {competitor}")
            logger.info(f"Date range: {start_date} to {end_date}")
            
            response = self.llm._call(prompt, start_date=start_date, last_date=end_date)
            
            # Parse the JSON response
            parsed_response = json.loads(response)
            
            # Validate and return the structured report
            return CompetitiveIntelligenceReport.model_validate(
                parsed_response['competitive_intelligence_report']
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from API: {e}")
        except Exception as e:
            logger.error(f"Error generating competitive intelligence report: {e}")
            raise

    def export_report_to_json(self, report: CompetitiveIntelligenceReport, filename: str = None) -> str:
        """Export report to JSON file"""
        if not filename:
            filename = f"competitive_intelligence_{report.requesting_company}_{report.competitor_company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report exported to {filename}")
        return filename

# Usage example
if __name__ == "__main__":
    # Initialize your Perplexity LLM (replace with your actual API key)
    perplexity_llm = PerplexityLLM(api_key="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")
    
    # Create the service
    intel_service = CompetitiveIntelligenceService(perplexity_llm)
    
    try:
        # Get competitive intelligence
        report = intel_service.get_competitor_intelligence(
            requesting_company="Nike",
            competitor="Adidas",
            start_date="6/1/2025",
            end_date="7/11/2025"
        )
        
        # Display summary
        print(f"✅ Generated report successfully!")
        print(f"Report ID: {report.report_id}")
        print(f"Generated at: {report.generated_at}")
        print(f"Total updates found: {len(report.updates)}")
        print(f"High impact updates: {report.analytics_summary.high_impact_updates_count}")
        print(f"Average relevance score: {report.analytics_summary.average_relevance_score:.2f}")
        print(f"Confidence level: {report.confidence_level:.2f}")
        
        print("\n📊 Key Insights:")
        for i, insight in enumerate(report.key_insights, 1):
            print(f"{i}. {insight}")
        
        print("\n🎯 Strategic Recommendations:")
        for i, recommendation in enumerate(report.strategic_recommendations, 1):
            print(f"{i}. {recommendation}")
        
        # Export to JSON
        filename = intel_service.export_report_to_json(report)
        print(f"\n💾 Report exported to: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        print(f"❌ Error: {e}")