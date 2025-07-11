from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class CategoryEnum(str, Enum):
    FASHION = "fashion"
    TECHNOLOGY = "technology"
    FOOD = "food"
    AUTOMOTIVE = "automotive"
    BEAUTY = "beauty"
    SPORTS = "sports"
    ELECTRONICS = "electronics"
    LIFESTYLE = "lifestyle"

class AudienceEnum(str, Enum):
    GENZ = "genz"
    MILLENNIALS = "millennials"
    GENX = "genx"
    BOOMERS = "boomers"
    ALL = "all"

class BrandAnalysisRequest(BaseModel):
    brand_name: str = Field(..., description="Name of the brand to analyze")
    product: str = Field(..., description="Main product category")
    category: CategoryEnum = Field(..., description="Business category")
    audience: str = Field(..., description="Target audience")
    location: str=Field(...,description="the Location")

class CompetitorInfo(BaseModel):
    name: str = Field(..., description="Competitor brand name")
    relevance_score: float = Field(default=0.0, description="Relevance score (0-1)")
    market_position: Optional[str] = Field(None, description="Market position description")

class CompetitorList(BaseModel):
    competitors: List[CompetitorInfo] = Field(..., description="List of competitor brands")
    total_found: int = Field(..., description="Total number of competitors found")

class DNAAttribute(BaseModel):
    attribute: str = Field(..., description="DNA attribute name")
    value: str = Field(..., description="Attribute value or description")
    strength: float = Field(..., description="Strength score (0-1)")
    source: Optional[str] = Field(None, description="Source of information")

class BrandDNA(BaseModel):
    brand_name: str = Field(..., description="Brand name")
    core_values: List[str] = Field(default_factory=list, description="Core brand values")
    personality_traits: List[str] = Field(default_factory=list, description="Brand personality traits")
    positioning: str = Field(default="", description="Brand positioning statement")
    unique_selling_proposition: str = Field(default="", description="USP")
    target_audience: List[str] = Field(default_factory=list, description="Target audience segments")
    brand_voice: str = Field(default="", description="Brand voice and tone")
    visual_identity: str = Field(default="", description="Visual identity description")
    emotional_connection: str = Field(default="", description="Emotional connection with audience")
    market_differentiation: str = Field(default="", description="How brand differentiates from competitors")
    brand_promise: str = Field(default="", description="Brand promise to customers")
    attributes: List[DNAAttribute] = Field(default_factory=list, description="Additional DNA attributes")

class SearchResult(BaseModel):
    url: str = Field(..., description="URL of the search result")
    title: str = Field(..., description="Title of the page")
    content: str = Field(..., description="Extracted content")
    relevance_score: float = Field(default=0.0, description="Relevance score")

class EnhancedQuery(BaseModel):
    original_query: str = Field(..., description="Original query")
    enhanced_queries: List[str] = Field(..., description="List of enhanced queries")
    search_intent: str = Field(..., description="Search intent category")

class BrandComparison(BaseModel):
    target_brand: str = Field(..., description="Target brand name")
    competitor_brand: str = Field(..., description="Competitor brand name")
    strengths: List[str] = Field(default_factory=list, description="Areas where competitor is stronger")
    weaknesses: List[str] = Field(default_factory=list, description="Areas where target brand is stronger")
    opportunities: List[str] = Field(default_factory=list, description="Opportunities for improvement")
    similarity_score: float = Field(default=0.0, description="Similarity score (0-1)")

class BrandDNAAnalysis(BaseModel):
    target_brand: BrandDNA = Field(..., description="Target brand DNA")
    competitors: List[BrandDNA] = Field(..., description="Competitor brands DNA")
    comparisons: List[BrandComparison] = Field(..., description="Brand comparisons")
    insights: List[str] = Field(default_factory=list, description="Key insights")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    overall_score: float = Field(default=0.0, description="Overall brand strength score")

class RAGChunk(BaseModel):
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score")
    source: str = Field(default="", description="Source of the chunk")

class ProcessingStatus(BaseModel):
    stage: str = Field(..., description="Current processing stage")
    progress: float = Field(..., description="Progress percentage (0-100)")
    message: str = Field(..., description="Status message")
    completed: bool = Field(default=False, description="Whether processing is completed")

class ErrorResponse(BaseModel):
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# Constants for the system
COMPETITOR_SEARCH_QUERIES = {
    "fashion": [
        "{brand_name} competitors fashion",
        "{brand_name} vs similar brands",
        "top {category} brands like {brand_name}",
        "{brand_name} alternatives {audience}"
    ],
    "technology": [
        "{brand_name} competitors tech",
        "{brand_name} vs similar companies",
        "top {category} companies like {brand_name}",
        "{brand_name} alternatives market"
    ],
    "default": [
        "{brand_name} competitors",
        "{brand_name} vs similar brands",
        "top {category} brands like {brand_name}",
        "{brand_name} alternatives"
    ]
}

DNA_EXTRACTION_QUERIES = {
    "core_values": [
        "{brand_name} core values mission",
        "{brand_name} brand values principles",
        "{brand_name} company culture values"
    ],
    "personality": [
        "{brand_name} brand personality traits",
        "{brand_name} brand voice tone",
        "{brand_name} brand character"
    ],
    "positioning": [
        "{brand_name} brand positioning strategy",
        "{brand_name} market position",
        "{brand_name} brand statement"
    ],
    "audience": [
        "{brand_name} target audience demographics",
        "{brand_name} customer base",
        "{brand_name} consumer profile"
    ],
    "differentiation": [
        "{brand_name} unique selling proposition",
        "{brand_name} competitive advantage",
        "{brand_name} brand differentiation"
    ]
}

from pydantic import BaseModel, Field
from typing import List

class SearchQuery(BaseModel):
    query: str = Field(..., description="A Google search query for gathering information about the competitor.")
class SearchQueryList(BaseModel):
    queries: List[SearchQuery] = Field(..., description="A list of 3 Google search queries related to the competitor.")


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
