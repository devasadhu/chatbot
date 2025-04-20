import gradio as gr
import requests
import random
import re
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any, Tuple, Optional

# Configuration for optional LLM API integration
LLM_API_ENABLED = False  # Set to True when you have your LLM API ready
LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# Configuration for FinBERT (used as fallback for sentiment analysis)
FINBERT_ENABLED = False  # Toggle for using FinBERT API
API_TOKEN = os.environ.get("HF_API_TOKEN", "")
MODEL_NAME = "ProsusAI/finbert"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

# Enhanced stock data with more context and information
STOCK_INFO = {
    "tech": {
        "AAPL": {"name": "Apple Inc.", "description": "Consumer electronics, software, and services"},
        "MSFT": {"name": "Microsoft Corporation", "description": "Software, cloud computing, hardware"},
        "GOOGL": {"name": "Alphabet Inc.", "description": "Internet services, software, hardware"},
        "AMZN": {"name": "Amazon.com Inc.", "description": "E-commerce, cloud computing, digital streaming"},
        "NVDA": {"name": "NVIDIA Corporation", "description": "Graphics processing units, AI computing"}
    },
    "healthcare": {
        "JNJ": {"name": "Johnson & Johnson", "description": "Pharmaceuticals, medical devices, consumer goods"},
        "PFE": {"name": "Pfizer Inc.", "description": "Pharmaceuticals and biotechnology"},
        "UNH": {"name": "UnitedHealth Group", "description": "Health insurance and healthcare services"},
        "ABBV": {"name": "AbbVie Inc.", "description": "Biopharmaceuticals"},
        "MRK": {"name": "Merck & Co.", "description": "Pharmaceuticals and vaccines"}
    },
    "finance": {
        "JPM": {"name": "JPMorgan Chase & Co.", "description": "Banking and financial services"},
        "BAC": {"name": "Bank of America Corp.", "description": "Banking and financial services"},
        "WFC": {"name": "Wells Fargo & Company", "description": "Banking and financial services"},
        "GS": {"name": "Goldman Sachs Group", "description": "Investment banking and financial services"},
        "MS": {"name": "Morgan Stanley", "description": "Investment banking and financial services"}
    },
    "energy": {
        "XOM": {"name": "Exxon Mobil Corporation", "description": "Oil and gas exploration, production, refining"},
        "CVX": {"name": "Chevron Corporation", "description": "Oil and gas exploration, production, refining"},
        "COP": {"name": "ConocoPhillips", "description": "Oil and gas exploration and production"},
        "SLB": {"name": "Schlumberger Limited", "description": "Oilfield services and equipment"},
        "EOG": {"name": "EOG Resources", "description": "Oil and gas exploration and production"}
    },
    "consumer": {
        "PG": {"name": "Procter & Gamble", "description": "Consumer goods, personal care products"},
        "KO": {"name": "Coca-Cola Company", "description": "Beverages"},
        "PEP": {"name": "PepsiCo, Inc.", "description": "Beverages and snack foods"},
        "WMT": {"name": "Walmart Inc.", "description": "Retail, wholesale, and other services"},
        "MCD": {"name": "McDonald's Corporation", "description": "Fast food restaurants"}
    }
}

# Financial education content - expanded with more resources
FINANCIAL_EDUCATION = {
    "investing_basics": {
        "title": "Investing Basics",
        "content": "Investing involves allocating resources (usually money) with the expectation of generating income or profit. The main investment types include stocks, bonds, mutual funds, ETFs, real estate, and commodities.",
        "resources": [
            {"name": "Investment Fundamentals", "type": "guide"},
            {"name": "Risk vs. Return", "type": "concept"},
            {"name": "Asset Allocation", "type": "strategy"}
        ]
    },
    "stock_market": {
        "title": "Understanding the Stock Market",
        "content": "The stock market is where shares of publicly traded companies are bought and sold. It provides companies with capital while giving investors the opportunity to share in the profits of businesses.",
        "resources": [
            {"name": "How Stock Markets Work", "type": "guide"},
            {"name": "Bull vs. Bear Markets", "type": "concept"},
            {"name": "Market Indices Explained", "type": "concept"}
        ]
    },
    "personal_finance": {
        "title": "Personal Finance Management",
        "content": "Personal finance covers budgeting, saving, investing, debt management, and retirement planning. It's about making informed decisions to achieve your financial goals.",
        "resources": [
            {"name": "Budgeting Strategies", "type": "guide"},
            {"name": "Emergency Fund Planning", "type": "strategy"},
            {"name": "Debt Reduction Methods", "type": "strategy"}
        ]
    },
    "retirement_planning": {
        "title": "Retirement Planning",
        "content": "Retirement planning involves defining retirement income goals and the actions needed to achieve those goals. It includes identifying sources of income, estimating expenses, and implementing a savings program.",
        "resources": [
            {"name": "Retirement Accounts Explained", "type": "guide"},
            {"name": "The 4% Withdrawal Rule", "type": "concept"},
            {"name": "Social Security Benefits", "type": "guide"}
        ]
    }
}

# Financial concepts dictionary - expanded with more detailed explanations
FINANCIAL_CONCEPTS = {
    "inflation": {
        "short": "The rate at which the general level of prices for goods and services rises, causing purchasing power to fall.",
        "detailed": "Inflation is the gradual increase in prices and fall in the purchasing value of money. It affects everything from your grocery bill to investment returns. Central banks like the Federal Reserve typically target a moderate inflation rate of about 2% annually. Investments need to outpace inflation to generate real returns."
    },
    "compound_interest": {
        "short": "Interest calculated on the initial principal and also on the accumulated interest over previous periods.",
        "detailed": "Compound interest is essentially 'interest on interest' and is the reason why investing early is so powerful. For example, $1,000 invested at 5% annually will be worth $1,050 after one year. The next year, you earn interest on $1,050, not just the original $1,000. Over time, this effect snowballs dramatically."
    },
    "diversification": {
        "short": "Spreading investments across different assets to reduce risk.",
        "detailed": "Diversification means not putting all your eggs in one basket. By spreading investments across various asset classes (stocks, bonds, real estate), sectors, and geographic regions, you can reduce overall portfolio risk. When one investment performs poorly, others might perform well, helping stabilize your returns."
    },
    "etf": {
        "short": "Exchange-Traded Fund, an investment fund traded on stock exchanges that holds assets like stocks, bonds, or commodities.",
        "detailed": "ETFs combine features of individual stocks (they trade on exchanges) and mutual funds (they represent a basket of securities). They typically have lower expense ratios than mutual funds and offer liquidity, tax efficiency, and exposure to specific indices, sectors, or investing strategies."
    },
    "p_e_ratio": {
        "short": "Price-to-Earnings ratio, a valuation ratio of a company's current share price compared to its per-share earnings.",
        "detailed": "The P/E ratio helps investors evaluate if a stock is overvalued or undervalued. It's calculated by dividing the market price per share by the earnings per share. A high P/E might suggest investors expect higher growth in the future, while a low P/E might indicate an undervalued stock or concerns about future performance."
    },
    "dollar_cost_averaging": {
        "short": "Investing a fixed amount at regular intervals regardless of market conditions.",
        "detailed": "Dollar-cost averaging reduces the impact of volatility by spreading purchases over time. When prices are high, your fixed investment buys fewer shares; when prices are low, it buys more. This strategy removes the pressure of trying to time the market and can be particularly effective for long-term investors."
    },
    "liquidity": {
        "short": "The ease with which an asset can be converted to cash without affecting its market price.",
        "detailed": "Liquidity refers to how quickly you can sell an investment without losing value. Cash is the most liquid asset, while real estate is relatively illiquid. Stocks of large companies traded on major exchanges are quite liquid, while stocks of small companies or those traded on over-the-counter markets may be less liquid."
    },
    "rebalancing": {
        "short": "The process of realigning the weightings of a portfolio of assets to maintain the original desired level of asset allocation.",
        "detailed": "Rebalancing involves periodically buying or selling assets to maintain your target allocation. For example, if your strategy calls for 60% stocks and 40% bonds, but stock growth has pushed the ratio to 70/30, rebalancing would involve selling some stocks and buying bonds to return to 60/40."
    }
}

# Expanded investment strategies
INVESTMENT_STRATEGIES = {
    "value_investing": {
        "name": "Value Investing",
        "description": "Buying stocks that appear underpriced relative to their intrinsic value",
        "key_metrics": ["P/E Ratio", "P/B Ratio", "Dividend Yield"],
        "famous_proponents": ["Warren Buffett", "Benjamin Graham"],
        "ideal_for": "Patient investors focused on long-term growth",
        "risk_level": "Moderate"
    },
    "growth_investing": {
        "name": "Growth Investing",
        "description": "Focusing on companies with strong growth potential, often in expanding sectors",
        "key_metrics": ["Revenue Growth Rate", "Earnings Growth Rate", "Market Share Trends"],
        "famous_proponents": ["Peter Lynch", "Philip Fisher"],
        "ideal_for": "Investors seeking capital appreciation over dividends",
        "risk_level": "High"
    },
    "dividend_investing": {
        "name": "Dividend Investing",
        "description": "Investing in stable companies that regularly distribute earnings to shareholders",
        "key_metrics": ["Dividend Yield", "Dividend Growth Rate", "Payout Ratio"],
        "famous_proponents": ["John Bogle", "Jeremy Siegel"],
        "ideal_for": "Income-focused investors, particularly retirees",
        "risk_level": "Low to Moderate"
    },
    "index_investing": {
        "name": "Index Investing",
        "description": "Buying funds that track market indices to match market returns",
        "key_metrics": ["Expense Ratio", "Tracking Error", "Fund Size"],
        "famous_proponents": ["John Bogle", "Burton Malkiel"],
        "ideal_for": "Passive investors seeking market returns with minimal research",
        "risk_level": "Varies with index (generally Moderate)"
    }
}

# Enhanced policy and investment product information
FINANCIAL_PRODUCTS = {
    "term_insurance": {
        "type": "Insurance",
        "description": "Pure life insurance coverage for a specific period",
        "benefits": ["High coverage at affordable premiums", "Tax benefits on premiums", "Financial security for dependents"],
        "considerations": ["No maturity benefits", "Coverage ends with term", "Premiums increase with age"],
        "ideal_for": "Primary breadwinners with dependents"
    },
    "ulip": {
        "type": "Insurance + Investment",
        "description": "Unit Linked Insurance Plan combining insurance and investment",
        "benefits": ["Life coverage", "Market-linked returns", "Tax benefits", "Fund switching options"],
        "considerations": ["Higher charges than pure investments", "Lock-in period", "Market risk"],
        "ideal_for": "Those looking for insurance with investment potential"
    },
    "mutual_funds": {
        "type": "Investment",
        "description": "Professionally managed investment funds pooling money from many investors",
        "benefits": ["Professional management", "Diversification", "Liquidity", "Variety of options"],
        "considerations": ["Expense ratios", "Market risk", "No guaranteed returns"],
        "ideal_for": "Most investors seeking market exposure with professional management"
    },
    "fixed_deposits": {
        "type": "Investment",
        "description": "Time deposits with banks offering fixed interest rates",
        "benefits": ["Guaranteed returns", "Safety of principal", "Predictable income", "Various tenure options"],
        "considerations": ["Lower returns than market investments", "Interest rate risk", "Premature withdrawal penalties"],
        "ideal_for": "Conservative investors seeking capital preservation"
    },
    "etfs": {
        "type": "Investment",
        "description": "Exchange-traded funds that track indices, sectors, commodities, or other assets",
        "benefits": ["Low expense ratios", "Trading flexibility", "Tax efficiency", "Diversification"],
        "considerations": ["Brokerage fees", "Market risk", "Tracking errors"],
        "ideal_for": "Both beginner and sophisticated investors seeking specific market exposure"
    }
}

# Personalization profiles to tailor responses
PERSONA_PROFILES = {
    "beginner": {
        "knowledge_level": "basic",
        "terminology": "simplified",
        "depth": "introductory",
        "focus": ["education", "fundamentals", "risk management"]
    },
    "intermediate": {
        "knowledge_level": "moderate",
        "terminology": "standard",
        "depth": "balanced",
        "focus": ["strategies", "portfolio management", "market analysis"]
    },
    "advanced": {
        "knowledge_level": "sophisticated",
        "terminology": "technical",
        "depth": "detailed",
        "focus": ["advanced strategies", "technical analysis", "macroeconomic impacts"]
    },
    "retiree": {
        "knowledge_level": "varies",
        "terminology": "standard",
        "depth": "practical",
        "focus": ["income generation", "wealth preservation", "estate planning"]
    },
    "student": {
        "knowledge_level": "developing",
        "terminology": "educational",
        "depth": "foundational",
        "focus": ["basics", "learning resources", "gradual introduction"]
    }
}

# Conversation memory store to maintain context
class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages = []
        self.topics_discussed = set()
        self.user_interests = set()
        self.user_profile = {
            "persona": "beginner",
            "interests": [],
            "knowledge_areas": [],
            "goals": []
        }
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history:
            self.messages.pop(0)
        
        # Extract topics and interests
        if role == "user":
            self._extract_topics_and_interests(content)
    
    def _extract_topics_and_interests(self, content: str):
        """Extract topics and interests from user messages"""
        # Simple keyword-based extraction
        content_lower = content.lower()
        
        # Financial topics
        topics = {
            "stocks": ["stock", "equity", "shares", "nasdaq", "nyse"],
            "retirement": ["retire", "401k", "pension", "ira"],
            "budgeting": ["budget", "spending", "expense", "income"],
            "investing": ["invest", "portfolio", "asset", "allocation"],
            "taxes": ["tax", "deduction", "write-off", "filing"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in content_lower for keyword in keywords):
                self.topics_discussed.add(topic)
                
        # Risk tolerance indicators
        risk_keywords = {
            "conservative": ["safe", "secure", "low risk", "conservative", "preserve"],
            "moderate": ["balanced", "moderate", "middle ground"],
            "aggressive": ["aggressive", "growth", "high risk", "high return"]
        }
        
        for risk_level, keywords in risk_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                self.user_profile["risk_tolerance"] = risk_level
                break
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Return a summary of the conversation context"""
        return {
            "topics_discussed": list(self.topics_discussed),
            "user_profile": self.user_profile,
            "messages_count": len(self.messages)
        }
    
    def get_recent_messages(self, count: int = 3) -> List[Dict[str, str]]:
        """Get the most recent messages"""
        return self.messages[-count:] if len(self.messages) >= count else self.messages


def get_resource_recommendations(user_query: str, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate personalized resource recommendations based on query and user profile"""
    query_lower = user_query.lower()
    
    # Maps topics to relevant resources
    topic_resource_map = {
        "stocks": ["stock_market", "investing_basics"],
        "investing": ["investing_basics", "stock_market"],
        "retirement": ["retirement_planning", "personal_finance"],
        "budget": ["personal_finance"],
        "saving": ["personal_finance"],
        "finance": ["personal_finance", "investing_basics"]
    }
    
    # Find matching topics
    matched_topics = []
    for topic, keywords in topic_resource_map.items():
        if topic in query_lower or any(keyword in query_lower for keyword in [topic]):
            matched_topics.extend(keywords)
    
    # Get unique recommendations
    recommendations = []
    seen_titles = set()
    
    # Always add a general resource if available
    if "personal_finance" in FINANCIAL_EDUCATION and "personal_finance" not in seen_titles:
        recommendations.append(FINANCIAL_EDUCATION["personal_finance"])
        seen_titles.add("personal_finance")
    
    # Add topic-specific resources
    for topic in matched_topics:
        if topic in FINANCIAL_EDUCATION and topic not in seen_titles:
            recommendations.append(FINANCIAL_EDUCATION[topic])
            seen_titles.add(topic)
    
    # If we don't have enough recommendations, add some general ones
    if len(recommendations) < 2:
        for topic in ["investing_basics", "stock_market", "retirement_planning"]:
            if topic in FINANCIAL_EDUCATION and topic not in seen_titles:
                recommendations.append(FINANCIAL_EDUCATION[topic])
                seen_titles.add(topic)
                if len(recommendations) >= 2:
                    break
    
    return recommendations[:2]  # Limit to 2 recommendations

def generate_dynamic_stock_sentiment(stock_symbol: str) -> Dict[str, Any]:
    """Generate realistic but dynamic stock sentiment rather than using static data"""
    sentiments = ["positive", "neutral", "negative"]
    weights = [0.5, 0.3, 0.2]  # More likely to be positive or neutral
    
    # Find stock info
    stock_info = None
    stock_name = stock_symbol
    for sector, stocks in STOCK_INFO.items():
        if stock_symbol in stocks:
            stock_info = stocks[stock_symbol]
            stock_name = stock_info["name"]
            break
    
    # Generate sentiment data
    sentiment = random.choices(sentiments, weights=weights)[0]
    
    # Score will depend on sentiment
    score = 0
    if sentiment == "positive":
        score = round(random.uniform(0.65, 0.95), 2)
    elif sentiment == "neutral":
        score = round(random.uniform(0.45, 0.65), 2)
    else:
        score = round(random.uniform(0.15, 0.45), 2)
    
    # Generate dynamic news count
    news_count = random.randint(3, 25)
    
    # Generate trending direction
    trending = "up" if sentiment == "positive" else "down" if sentiment == "negative" else "steady"
    
    # Generate reason phrases based on sentiment
    reasons = []
    if sentiment == "positive":
        reasons = [
            "strong quarterly results",
            "new product announcements",
            "expanded market share",
            "strategic partnerships",
            "analyst upgrades"
        ]
    elif sentiment == "neutral":
        reasons = [
            "mixed earnings results",
            "pending regulatory decisions",
            "competitive market conditions",
            "sector rotation",
            "waiting for upcoming announcements"
        ]
    else:
        reasons = [
            "missed earnings expectations",
            "regulatory challenges",
            "increased competition",
            "management changes",
            "sector weakness"
        ]
    
    # Pick 1-2 reasons
    selected_reasons = random.sample(reasons, k=min(2, len(reasons)))
    
    return {
        "symbol": stock_symbol,
        "name": stock_name,
        "sentiment": sentiment,
        "score": score,
        "news_count": news_count, 
        "trending": trending,
        "key_reasons": selected_reasons
    }

def generate_market_sentiment() -> Dict[str, Dict[str, Any]]:
    """Generate dynamic market sentiment data"""
    sentiments = {}
    sectors = list(STOCK_INFO.keys())
    
    for sector in sectors:
        # Generate sentiment values
        sentiment_value = random.choice(["positive", "neutral", "negative"])
        trending = "up" if sentiment_value == "positive" else "down" if sentiment_value == "negative" else "stable"
        
        # Score range based on sentiment
        score = 0
        if sentiment_value == "positive":
            score = round(random.uniform(0.65, 0.95), 2)
        elif sentiment_value == "neutral":
            score = round(random.uniform(0.45, 0.65), 2)
        else:
            score = round(random.uniform(0.15, 0.45), 2)
            
        # Generate some stocks in this sector
        stocks_in_sector = list(STOCK_INFO[sector].keys())
        top_performers = random.sample(stocks_in_sector, k=min(2, len(stocks_in_sector)))
            
        sentiments[sector] = {
            "sentiment": sentiment_value,
            "score": score,
            "trending": trending,
            "top_performers": top_performers
        }
    
    return sentiments

def identify_intent(message: str) -> Dict[str, Any]:
    """Enhanced intent identification with extracted entities and context"""
    message_lower = message.lower()
    
    intent_data = {
        "primary_intent": "general_query",
        "secondary_intent": None,
        "entities": {},
        "sentiment": "neutral",
        "is_question": False
    }
    
    # Check if it's a question
    if re.search(r'\?$|^(what|how|why|when|where|who|can|could|would|will|should|is|are|do|does)', message_lower):
        intent_data["is_question"] = True
    
    # Extract sentiment in the query itself
    if re.search(r'\b(happy|excited|pleased|good|great)\b', message_lower):
        intent_data["sentiment"] = "positive"
    elif re.search(r'\b(sad|unhappy|disappointed|frustrated|bad|awful)\b', message_lower):
        intent_data["sentiment"] = "negative"
    
    # Basic conversation patterns
    if re.search(r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)( there)?[.!]?$', message_lower):
        intent_data["primary_intent"] = "greeting"
        return intent_data
    
    if re.search(r"^how are you|how(?:'s)? it going|how have you been|what(?:'s)? up$", message_lower):
        intent_data["primary_intent"] = "how_are_you"
        return intent_data
    
    if re.search(r'^(bye|goodbye|farewell|see you|talk to you later)[.!]?$', message_lower):
        intent_data["primary_intent"] = "goodbye"
        return intent_data
    
    if re.search(r'^(thanks|thank you|appreciate it|thx)[.!]?$', message_lower):
        intent_data["primary_intent"] = "thanks"
        return intent_data
    
    if re.search(r'\bjoke\b|\bfunny\b|\bmake me laugh\b', message_lower):
        intent_data["primary_intent"] = "joke"
        return intent_data
    
    # Sentiment analysis specific patterns
    if re.search(r'analyze (this|the|my|following) (statement|sentence|text|news)', message_lower):
        intent_data["primary_intent"] = "analyze_sentiment"
        
        # Extract the statement to analyze
        match = re.search(r'analyze (this|the|my|following).*?[:\-] *(.*)', message_lower)
        if match and match.group(2):
            intent_data["entities"]["statement"] = match.group(2)
        return intent_data
    
    # Market/Stock Sentiment patterns
    if re.search(r'\b(sentiment|feeling|opinion|mood)\b', message_lower):
        if re.search(r'\b(market|markets|sector|sectors|industry|industries)\b', message_lower):
            intent_data["primary_intent"] = "market_sentiment"
            
            # Extract specific sectors if mentioned
            for sector in STOCK_INFO.keys():
                if sector.lower() in message_lower:
                    if "sectors" not in intent_data["entities"]:
                        intent_data["entities"]["sectors"] = []
                    intent_data["entities"]["sectors"].append(sector)
            return intent_data
            
        elif re.search(r'\b(stock|ticker|company|symbol)\b', message_lower):
            intent_data["primary_intent"] = "stock_sentiment"
            
            # Check if specific stocks are mentioned
            for sector, stocks in STOCK_INFO.items():
                for stock in stocks:
                    if stock.lower() in message_lower:
                        if "stocks" not in intent_data["entities"]:
                            intent_data["entities"]["stocks"] = []
                        intent_data["entities"]["stocks"].append(stock)
            return intent_data
    
    # Financial product information patterns
    if re.search(r'\b(fd|fixed deposit|deposits|deposit rates|insurance|policy|policies|plan|protection|mutual fund|etf|ulip)\b', message_lower):
        intent_data["primary_intent"] = "product_information"
        
        # Extract specific product types
        product_keywords = {
            "fixed_deposit": ["fd", "fixed deposit", "deposits", "deposit rates"],
            "insurance": ["insurance", "policy", "protection", "coverage"],
            "mutual_fund": ["mutual fund", "mf", "fund"],
            "etf": ["etf", "exchange traded fund", "exchange-traded fund"],
            "ulip": ["ulip", "unit linked", "unit-linked"]
        }
        
        for product, keywords in product_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intent_data["entities"]["product_type"] = product
                break
        
        # Check if it's a recommendation request
        if re.search(r'\b(recommend|suggest|best|top|good|should I|which one|better|compare)\b', message_lower):
            intent_data["secondary_intent"] = "recommendation"
        
        return intent_data
    
    # Investment recommendation patterns
    if re.search(r'\b(recommend|suggest|buy|invest|good stock|pick|advice|strategy|approach)\b', message_lower):
        intent_data["primary_intent"] = "investment_recommendation"
        
        # Extract investment type
        investment_types = {
            "stock": ["stock", "share", "equity"],
            "mutual_fund": ["mutual fund", "fund"],
            "etf": ["etf", "exchange traded"],
            "bond": ["bond", "fixed income"],
            "real_estate": ["real estate", "property", "reit"],
            "crypto": ["crypto", "bitcoin", "ethereum", "digital currency"]
        }
        
        for inv_type, keywords in investment_types.items():
            if any(keyword in message_lower for keyword in keywords):
                intent_data["entities"]["investment_type"] = inv_type
                break
        
        # Look for risk preference
        risk_levels = {
            "conservative": ["safe", "low risk", "conservative", "secure"],
            "moderate": ["balanced", "moderate", "medium risk"],
            "aggressive": ["aggressive", "high risk", "growth"]
        }
        
        for risk, keywords in risk_levels.items():
            if any(keyword in message_lower for keyword in keywords):
                intent_data["entities"]["risk_preference"] = risk
                break
                
        return intent_data
    
    # Educational content patterns
    if re.search(r'(how to|get into|start|begin|learn about|explain|what is|what are)\b', message_lower):
        intent_data["primary_intent"] = "educational"
        
        # Check for specific financial concepts
        for concept in FINANCIAL_CONCEPTS:
            concept_term = concept.replace('_', ' ')
            if concept_term in message_lower:
                intent_data["entities"]["concept"] = concept
                return intent_data
        
        # Check for educational topics
        edu_topics = {
            "investing_basics": ["investing basics", "start investing", "begin investing"],
            "stock_market": ["stock market", "how stocks work", "buying stocks"],
            "retirement": ["retirement", "retirement planning", "retirement account"],
            "personal_finance": ["personal finance", "budgeting", "saving money"]
        }
        
        for topic, keywords in edu_topics.items():
            if any(keyword in message_lower for keyword in keywords):
                intent_data["entities"]["topic"] = topic
                break
                
        return intent_data
    
    # If no specific intent is identified, treat as a general query
    return intent_data

def generate_response(intent_data: Dict[str, Any], message: str, memory: ConversationMemory) -> str:
    """Generate a dynamic, context-aware response based on identified intent and conversation memory"""
    primary_intent = intent_data["primary_intent"]
    
    # Handle basic conversation intents
    if primary_intent == "greeting":
        # Check if this is the first interaction
        if len(memory.messages) <= 1:
            return random.choice([
                "Hi there! I'm your friendly financial assistant. How can I help you today?",
                "Hello! I'm here to help with any financial questions or topics you'd like to discuss. What's on your mind?",
                "Hey! I'm your AI financial assistant. Whether you're interested in investing, saving, or learning about financial concepts, I'm here to help. What would you like to talk about?"
            ])
        else:
            return random.choice([
                "Hello again! What financial topic would you like to discuss now?",
                "Hi there! Ready to continue our financial conversation. What's on your mind?",
                "Hey! Great to chat again. What financial questions can I help with today?"
            ])
    
    if primary_intent == "how_are_you":
        return random.choice([
            "I'm doing great! Ready to talk about markets, investments, or any financial topics you're interested in. What's on your mind?",
            "I'm excellent, thanks for asking! Always ready to help with financial questions or discussions. What would you like to explore today?"
            "I'm well, thank you! The world of finance is always changing, and I'm here to help you navigate it. What financial topic would you like to discuss today?"
        ])
    
    if primary_intent == "goodbye":
        return random.choice([
            "Goodbye! Remember, the best investment you can make is in yourself. Feel free to come back anytime with your financial questions.",
            "See you later! I'm here whenever you need guidance on financial matters. Have a great day!",
            "Take care! Remember that financial knowledge is a journey, not a destination. I'll be here when you want to continue that journey."
        ])
    
    if primary_intent == "thanks":
        return random.choice([
            "You're welcome! I'm happy to help with any other financial questions you might have.",
            "Anytime! Financial literacy is empowering, and I'm glad to be part of your journey.",
            "My pleasure! If you have more questions in the future, don't hesitate to ask."
        ])
    
    if primary_intent == "joke":
        finance_jokes = [
            "Why don't economists like to go to the beach? Because the tide raises their liquidity concerns.",
            "How many economists does it take to change a light bulb? None. If the light bulb needed changing, the market would have done it by now.",
            "What do you call a financial instrument that's way too complicated? Probably your bank's newest product.",
            "I told my wife she was overreacting when she caught me looking at stock charts at 3am. She said I was being defensive. I said no, I was being a contrarian investor.",
            "Why are Bitcoin investors always calm? Because they've HODL'd onto their feelings.",
            "What's an actuary's favorite candy? Mortality mints.",
            "What did the stock broker say to his friend on the ski slope? That dividend is going downhill fast!",
            "What's a banker's favorite James Bond movie? 'The Spy Who Collateralized Me'.",
            "What's a venture capitalist's favorite song? 'Don't Stop Believing... in Unicorns'."
        ]
        return random.choice(finance_jokes)
    
    # Handle sentiment analysis
    if primary_intent == "analyze_sentiment":
        statement = intent_data["entities"].get("statement", message)
        return analyze_sentiment(statement)
    
    # Handle market sentiment queries
    if primary_intent == "market_sentiment":
        market_data = generate_market_sentiment()
        
        # Check if specific sectors were mentioned
        specific_sectors = intent_data["entities"].get("sectors", [])
        
        if specific_sectors:
            # Provide focused information about mentioned sectors
            sector_insights = []
            for sector in specific_sectors:
                if sector in market_data:
                    sector_info = market_data[sector]
                    sentiment_desc = f"{sector.title()} sector shows {sector_info['sentiment']} sentiment (score: {sector_info['score']})"
                    trend_desc = f"and is trending {sector_info['trending']}"
                    performers = f"Top performers include {', '.join(sector_info['top_performers'])}"
                    sector_insights.append(f"{sentiment_desc} {trend_desc}. {performers}.")
            
            if sector_insights:
                response = f"Here's the latest sentiment analysis for your requested sectors:\n\n{' '.join(sector_insights)}\n\nThis analysis is based on recent news articles, social media sentiment, and trading patterns. Would you like more specific information about any of these sectors or their top-performing stocks?"
            else:
                response = "I don't have specific sentiment data for those sectors. I can provide information about technology, healthcare, finance, energy, and consumer sectors. Which would you like to learn about?"
        else:
            # Provide general market sentiment overview
            positive_sectors = [s for s, data in market_data.items() if data["sentiment"] == "positive"]
            negative_sectors = [s for s, data in market_data.items() if data["sentiment"] == "negative"]
            neutral_sectors = [s for s, data in market_data.items() if data["sentiment"] == "neutral"]
            
            overall_sentiment = "positive" if len(positive_sectors) > len(negative_sectors) else "mixed" if len(positive_sectors) == len(negative_sectors) else "cautious"
            
            response = f"The overall market sentiment is currently {overall_sentiment}. "
            
            if positive_sectors:
                response += f"The {', '.join([s.title() for s in positive_sectors])} {len(positive_sectors) > 1 and 'sectors are' or 'sector is'} showing positive sentiment. "
            
            if negative_sectors:
                response += f"The {', '.join([s.title() for s in negative_sectors])} {len(negative_sectors) > 1 and 'sectors are' or 'sector is'} facing challenges with negative sentiment. "
            
            if neutral_sectors:
                response += f"The {', '.join([s.title() for s in neutral_sectors])} {len(neutral_sectors) > 1 and 'sectors are' or 'sector is'} showing neutral sentiment. "
            
            response += "\nWould you like more specific information about any particular sector or stock?"
        
        return response
    
    # Handle stock sentiment queries
    if primary_intent == "stock_sentiment":
        specific_stocks = intent_data["entities"].get("stocks", [])
        
        if specific_stocks:
            # Provide sentiment for specific stocks
            stock_insights = []
            for stock in specific_stocks:
                sentiment_data = generate_dynamic_stock_sentiment(stock)
                
                insight = f"{sentiment_data['name']} ({sentiment_data['symbol']}) shows {sentiment_data['sentiment']} sentiment "
                insight += f"with a score of {sentiment_data['score']:.2f} based on {sentiment_data['news_count']} recent news articles. "
                insight += f"The stock is trending {sentiment_data['trending']}"
                
                if sentiment_data['key_reasons']:
                    insight += f", influenced by {' and '.join(sentiment_data['key_reasons'])}"
                insight += "."
                
                stock_insights.append(insight)
            
            response = "\n\n".join(stock_insights)
            response += "\n\nWould you like more details about any of these stocks or information about other stocks?"
        else:
            # Suggest some stocks to analyze
            suggested_stocks = []
            for sector in STOCK_INFO:
                stocks = list(STOCK_INFO[sector].keys())
                if stocks:
                    suggested_stocks.append(random.choice(stocks))
                if len(suggested_stocks) >= 3:
                    break
            
            response = "I'd be happy to analyze stock sentiment for you. Which stocks are you interested in? "
            response += f"Some popular stocks to analyze include {', '.join(suggested_stocks)}. Just let me know which one(s) you'd like sentiment information for."
        
        return response
    
    # Handle product information queries
    if primary_intent == "product_information":
        product_type = intent_data["entities"].get("product_type")
        is_recommendation = intent_data["secondary_intent"] == "recommendation"
        
        if product_type == "fixed_deposit":
            if is_recommendation:
                fd_options = [
                    {
                        "tenure": "Short-term (6-12 months)",
                        "typical_rate": "4.5-5.5%",
                        "benefits": "Liquidity, guaranteed returns",
                        "ideal_for": "Emergency funds, short-term goals"
                    },
                    {
                        "tenure": "Medium-term (1-3 years)",
                        "typical_rate": "5.5-6.5%",
                        "benefits": "Better interest rates than short-term",
                        "ideal_for": "Planned expenses in 1-3 years"
                    },
                    {
                        "tenure": "Long-term (3-5+ years)",
                        "typical_rate": "6.5-7.5%",
                        "benefits": "Higher interest, possible tax benefits",
                        "ideal_for": "Long-term wealth building, retirement planning"
                    }
                ]
                
                # Choose most suitable option based on conversation context
                recommended_option = random.choice(fd_options)
                
                response = f"Based on general market conditions, {recommended_option['tenure']} fixed deposits might be worth considering. "
                response += f"They typically offer rates around {recommended_option['typical_rate']} and are particularly good for {recommended_option['ideal_for']}. "
                response += f"Key benefits include {recommended_option['benefits']}.\n\n"
                response += "Remember that actual rates vary by bank and economic conditions. What's your timeline for this investment?"
            else:
                response = "Fixed Deposits (FDs) are secure investments offered by banks where you deposit money for a fixed period at a guaranteed interest rate. "
                response += "They're low-risk and provide predictable returns, making them popular for conservative investors. "
                response += "FDs come in various tenures from a few months to several years, with longer terms generally offering higher interest rates. "
                response += "Most banks allow premature withdrawals with a small penalty. Are you considering investing in FDs or would you like to know about specific FD options?"
        elif product_type == "insurance":
            if is_recommendation:
                insurance_options = [
                    {
                        "type": "Term Insurance",
                        "features": "Pure life coverage, no maturity benefits",
                        "ideal_for": "Primary income earners with dependents",
                        "benefits": "Maximum coverage at minimum premium"
                    },
                    {
                        "type": "Health Insurance",
                        "features": "Coverage for medical expenses",
                        "ideal_for": "Everyone, regardless of age",
                        "benefits": "Financial protection against healthcare costs"
                    },
                    {
                        "type": "ULIP",
                        "features": "Insurance + Investment",
                        "ideal_for": "Those seeking both protection and investment",
                        "benefits": "Tax benefits, market-linked returns"
                    }
                ]
                
                recommended_option = random.choice(insurance_options)
                
                response = f"Many people in similar situations consider {recommended_option['type']} options. "
                response += f"These provide {recommended_option['features']} and are ideal for {recommended_option['ideal_for']}. "
                response += f"Key benefits include {recommended_option['benefits']}.\n\n"
                response += "Insurance needs are highly personal and depend on your specific situation. Would you like to know more about different insurance types or discuss specific protection needs?"
            else:
                response = "Insurance policies provide financial protection against various risks. Common types include term insurance (pure protection), "
                response += "health insurance (medical coverage), ULIPs (insurance + investment), endowment plans (insurance + savings), "
                response += "and general insurance for assets like homes and vehicles.\n\n"
                response += "Each type serves different needs and has unique features. What specific aspect of insurance would you like to explore further?"
        elif product_type in ["mutual_fund", "etf", "ulip"]:
            product_info = FINANCIAL_PRODUCTS.get(product_type if product_type != "mutual_fund" else "mutual_funds", {})
            
            if product_info:
                response = f"{product_info['description']}. "
                response += f"Key benefits include {', '.join(product_info['benefits'][:3])}. "
                response += f"Important considerations include {', '.join(product_info['considerations'][:2])}. "
                response += f"This product is typically suitable for {product_info['ideal_for']}."
                
                if is_recommendation:
                    response += "\n\nWould you like me to suggest some specific strategies for investing in this product based on your goals?"
            else:
                response = f"I'd be happy to provide information about {product_type.replace('_', ' ').upper()}s. "
                response += "Could you tell me more specifically what you'd like to know about them? For example, their benefits, risks, or how they work?"
        else:
            # General financial product information
            response = "I can provide information on various financial products including fixed deposits, insurance policies, mutual funds, ETFs, and ULIPs. "
            response += "Each serves different financial needs and goals. Which specific product would you like to learn more about?"
        
        return response
    
    # Handle investment recommendation queries
    if primary_intent == "investment_recommendation":
        investment_type = intent_data["entities"].get("investment_type", "general")
        risk_preference = intent_data["entities"].get("risk_preference", "moderate")
        
        if investment_type == "stock":
            # Sample stock recommendation based on risk preference
            stock_recommendations = {
                "conservative": ["PG", "JNJ", "KO"],
                "moderate": ["MSFT", "AAPL", "JPM"],
                "aggressive": ["NVDA", "AMZN", "GOOGL"]
            }
            
            recommended_stocks = stock_recommendations.get(risk_preference, stock_recommendations["moderate"])
            stock_names = []
            for stock in recommended_stocks:
                for sector, stocks in STOCK_INFO.items():
                    if stock in stocks:
                        stock_names.append(f"{stock} ({stocks[stock]['name']})")
                        break
            
            response = "While I can't provide personalized investment advice, investors with a "
            response += f"{risk_preference} risk profile often consider stocks like {', '.join(stock_names)}. "
            response += "These suggestions are based on general market information, not personalized advice.\n\n"
            response += "Always research thoroughly and consider consulting with a financial advisor before investing. "
            response += "Would you like to know more about any of these companies or learn about investment strategies for stocks?"
        elif investment_type in ["mutual_fund", "etf"]:
            fund_types = {
                "conservative": ["Bond funds", "Dividend funds", "Value funds"],
                "moderate": ["Balanced funds", "Index funds", "Blue-chip funds"],
                "aggressive": ["Growth funds", "Sector-specific funds", "Small-cap funds"]
            }
            
            recommended_funds = fund_types.get(risk_preference, fund_types["moderate"])
            
            response = f"For {risk_preference} investors interested in {investment_type.replace('_', ' ')}s, "
            response += f"these types are commonly considered: {', '.join(recommended_funds)}. "
            response += f"Each type has different risk-return characteristics that align with a {risk_preference} approach.\n\n"
            response += "Would you like more specific information about any of these fund types and their typical performance characteristics?"
        else:
            # General investment recommendation
            strategies = {
                "conservative": {
                    "allocation": "60-70% bonds, 30-40% stocks",
                    "focus": "Income generation and capital preservation",
                    "products": "Bond funds, dividend stocks, CDs, fixed deposits"
                },
                "moderate": {
                    "allocation": "40-60% bonds, 40-60% stocks",
                    "focus": "Balance between growth and income",
                    "products": "Index funds, blue-chip stocks, balanced mutual funds"
                },
                "aggressive": {
                    "allocation": "20-30% bonds, 70-80% stocks",
                    "focus": "Long-term growth and capital appreciation",
                    "products": "Growth stocks, sector-specific ETFs, emerging markets"
                }
            }
            
            strategy = strategies.get(risk_preference, strategies["moderate"])
            
            response = f"For investors with a {risk_preference} risk profile, a common approach includes:\n\n"
            response += f"- Asset allocation: Approximately {strategy['allocation']}\n"
            response += f"- Focus: {strategy['focus']}\n"
            response += f"- Financial products to consider: {strategy['products']}\n\n"
            response += "Remember that investment decisions should be based on your specific financial goals, time horizon, and personal circumstances. "
            response += "What's your primary investment goal and timeline?"
        
        return response
    
    # Handle educational queries
    if primary_intent == "educational":
        concept = intent_data["entities"].get("concept")
        topic = intent_data["entities"].get("topic")
        
        if concept and concept in FINANCIAL_CONCEPTS:
            concept_info = FINANCIAL_CONCEPTS[concept]
            response = f"{concept_info['detailed']}\n\n"
            response += "Would you like to know more about how this concept applies to specific financial situations or learn about related concepts?"
        elif topic and topic in FINANCIAL_EDUCATION:
            topic_info = FINANCIAL_EDUCATION[topic]
            response = f"{topic_info['title']}: {topic_info['content']}\n\n"
            response += "Would you like to explore any specific aspect of this topic in more detail?"
        else:
            # General educational response
            response = "I'm happy to help with financial education! I can explain concepts like compound interest, diversification, or P/E ratios. "
            response += "I can also provide information about investing basics, the stock market, personal finance, or retirement planning. "
            response += "What specific financial topic or concept would you like to learn about?"
        
        return response
    
    # Handle general queries with improved conversation flow
    # Extract key financial terms and concepts
    financial_terms = [
        "stocks", "bonds", "invest", "market", "finance", "money", "saving", 
        "retirement", "budget", "debt", "credit", "loan", "mortgage", "bank", 
        "interest", "dividend", "portfolio", "fund"
    ]
    
    message_lower = message.lower()
    found_terms = [term for term in financial_terms if term in message_lower]
    
    # Get conversation context
    conversation_summary = memory.get_conversation_summary()
    recent_topics = conversation_summary.get("topics_discussed", [])
    
    # Generate contextual response
    if found_terms:
        # Financial topic identified
        primary_term = found_terms[0]
        
        # Check if this is continuing a previous topic
        continuing_topic = primary_term in recent_topics
        
        if continuing_topic:
            # Continuing previous discussion
            responses = [
                f"To continue our discussion about {primary_term}, what specific aspect interests you most?",
                f"I'd be happy to explore {primary_term} further. Is there a particular element you'd like to focus on?",
                f"Let's dive deeper into {primary_term}. What questions do you have about this topic?"
            ]
        else:
            # New financial topic
            responses = [
                f"That's an interesting question about {primary_term}. To provide the most helpful information, could you share what you're looking to achieve with {primary_term}?",
                f"When it comes to {primary_term}, there are several approaches to consider. What's your main goal regarding this topic?",
                f"I'd be happy to discuss {primary_term}. To better assist you, could you share your experience level with this topic?"
            ]
        
        # Get resource recommendations based on query
        recommendations = get_resource_recommendations(message, conversation_summary.get("user_profile", {}))
        if recommendations and random.random() < 0.3:  # 30% chance to include a recommendation
            recommendation = recommendations[0]
            responses = [r + f" By the way, many people interested in {primary_term} also find '{recommendation['title']}' helpful to understand." for r in responses]
        
        return random.choice(responses)
    
    # Check if it's a question without financial terms
    if intent_data["is_question"]:
        return "That's an interesting question! While I specialize in financial topics, I'd be happy to chat about this. To help focus our conversation, would you like to know how this relates to personal finance or investments?"
    
    # For very short messages that don't fit other categories
    if len(message.split()) <= 3:
        return "I see! I'm here to chat about financial topics like investing, saving, budgeting, or market trends. What aspect of personal finance or investing would you like to explore today?"
    
    # Default response for anything else
    return "Thanks for sharing that. I'm primarily focused on financial topics, so I'd be happy to discuss anything related to personal finance, investing, or markets. Is there a specific financial topic you'd like to explore today?"

def analyze_sentiment(statement: str) -> str:
    """Analyze sentiment of financial text"""
    if FINBERT_ENABLED:
        try:
            payload = {
                "inputs": statement,
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Process FinBERT API response
            if isinstance(result, list) and len(result) > 0:
                sentiment_data = result[0]
                if isinstance(sentiment_data, list) and len(sentiment_data) > 0:
                    top_sentiment = max(sentiment_data, key=lambda x: x['score'])
                    sentiment = top_sentiment['label']
                    score = top_sentiment['score']
                    
                    sentiment_descriptions = {
                        "positive": "optimistic, which suggests potential upside",
                        "neutral": "balanced, without strong positive or negative indicators",
                        "negative": "cautious or concerned, which suggests potential challenges"
                    }
                    
                    return f"""I analyzed the financial sentiment of: "{statement}"

Financial sentiment: **{sentiment.title()}** ({score:.1%} confidence)

This statement appears {sentiment_descriptions.get(sentiment.lower(), "neutral")} from a financial perspective. Financial markets and investors would likely interpret this as {sentiment.lower()}.

Would you like me to explain what aspects of the statement might contribute to this sentiment analysis?"""
                
            # Fallback to simulated response
            return generate_simulated_sentiment(statement)
            
        except Exception as e:
            # Fallback to simulated response
            return generate_simulated_sentiment(statement)
    else:
        # Use simulated response when FinBERT is not enabled
        return generate_simulated_sentiment(statement)

def generate_simulated_sentiment(statement: str) -> str:
    """Generate a simulated sentiment analysis for financial text"""
    # Simple keyword-based sentiment analysis
    positive_words = ["growth", "profit", "increase", "gain", "positive", "up", "bullish", "opportunity",
                     "succeed", "success", "strong", "strengthen", "improved", "improving", "outperform"]
    negative_words = ["decline", "decrease", "loss", "debt", "risk", "bearish", "down", "fail", "weak",
                     "negative", "problem", "issue", "challenge", "underperform", "concern"]
    
    statement_lower = statement.lower()
    
    # Count sentiment words
    positive_count = sum(1 for word in positive_words if word in statement_lower)
    negative_count = sum(1 for word in negative_words if word in statement_lower)
    
    # Determine overall sentiment
    if positive_count > negative_count:
        sentiment = "positive"
        score = min(0.5 + (positive_count - negative_count) * 0.1, 0.95)
    elif negative_count > positive_count:
        sentiment = "negative"
        score = min(0.5 + (negative_count - positive_count) * 0.1, 0.95)
    else:
        sentiment = "neutral"
        score = 0.5 + random.uniform(-0.1, 0.1)
    
    sentiment_descriptions = {
        "positive": "optimistic, which suggests potential upside",
        "neutral": "balanced, without strong positive or negative indicators",
        "negative": "cautious or concerned, which suggests potential challenges"
    }
    
    # Identify key phrases that influenced the sentiment
    words = statement.split()
    key_phrases = []
    
    for i, word in enumerate(words):
        word_lower = word.lower().strip(".,!?;:")
        if word_lower in positive_words and sentiment == "positive":
            start = max(0, i-2)
            end = min(len(words), i+3)
            phrase = " ".join(words[start:end])
            key_phrases.append(phrase)
        elif word_lower in negative_words and sentiment == "negative":
            start = max(0, i-2)
            end = min(len(words), i+3)
            phrase = " ".join(words[start:end])
            key_phrases.append(phrase)
    
    # Limit to 2 key phrases
    key_phrases = key_phrases[:2]
    
    response = f"""I analyzed the financial sentiment of: "{statement}"

Financial sentiment: **{sentiment.title()}** ({score:.1%} confidence)

This statement appears {sentiment_descriptions[sentiment]} from a financial perspective. Financial markets and investors would likely interpret this as {sentiment}.
"""
    
    if key_phrases:
        response += f"\nKey phrases that influenced this analysis:\n- {'\n- '.join(key_phrases)}"
    
    response += "\n\nWould you like me to explain what other aspects of the statement might contribute to this sentiment analysis?"
    
    return response

def chatbot(message: str, chat_history: List[Tuple[str, str]]) -> str:
    """Main chatbot function with conversation memory and improved context handling"""
    # Initialize or retrieve conversation memory
    if not hasattr(chatbot, "memory"):
        chatbot.memory = ConversationMemory()
    
    # Add user message to memory
    chatbot.memory.add_message("user", message)
    
    # Identify intent and generate response
    intent_data = identify_intent(message)
    response = generate_response(intent_data, message, chatbot.memory)
    
    # Add assistant response to memory
    chatbot.memory.add_message("assistant", response)
    
    return response

# Create a more user-friendly Gradio interface
with gr.Blocks(theme="soft") as chat_ui:
    gr.Markdown("""# 💰 Financial Assistant
    Your intelligent financial companion for stocks, investments, planning, and education.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot_interface = gr.Chatbot(
                label="Chat with your Financial Assistant",
                height=450
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me about stocks, investments, financial concepts, or planning advice...",
                    label="Your Message",
                    scale=8
                )
                submit = gr.Button("Send", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Conversation")
        
        with gr.Column(scale=1):
            gr.Markdown("""### Quick Topics
            Click any topic to start a conversation about it.
            """)
            
            topic_buttons = [
                gr.Button("Stock Market Basics"),
                gr.Button("Investment Strategies"),
                gr.Button("Retirement Planning"),
                gr.Button("Market Sentiment"),
                gr.Button("Financial Products")
            ]
    
    gr.Markdown("""### Example Questions
    - "What's the current market sentiment?"
    - "Tell me about AAPL stock sentiment"
    - "How should I start investing in stocks?"
    - "Explain compound interest to me"
    - "What investment strategies are good for beginners?"
    - "What's the difference between ETFs and mutual funds?"
    """)
    
    # Set up event handlers
    msg_handler = msg.submit(
        fn=lambda message, history: (None, history + [[message, chatbot(message, history)]]),
        inputs=[msg, chatbot_interface],
        outputs=[msg, chatbot_interface]
    )
    
    submit.click(
        fn=lambda message, history: (None, history + [[message, chatbot(message, history)]]),
        inputs=[msg, chatbot_interface],
        outputs=[msg, chatbot_interface]
    )
    
    clear.click(lambda: None, None, chatbot_interface, queue=False)
    
    # Set up topic button handlers
    topic_questions = [
        "Can you explain stock market basics for beginners?",
        "What are some common investment strategies and which one might be right for me?",
        "How should I approach retirement planning?",
        "What's the current market sentiment across different sectors?",
        "Can you compare different financial products like mutual funds, ETFs, and fixed deposits?"
    ]
    
    for i, button in enumerate(topic_buttons):
        def make_click_handler(index):
            def handler(history):
                question = topic_questions[index]
                response = chatbot(question, history)
                return None, history + [[question, response]]
            return handler

        button.click(
            fn=make_click_handler(i),
            inputs=[chatbot_interface],
            outputs=[msg, chatbot_interface]
        )

# Launch the app
if __name__ == "__main__":
    chat_ui.launch(share=True)