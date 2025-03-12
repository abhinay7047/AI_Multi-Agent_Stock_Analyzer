from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from duckduckgo_search import DDGS
from typing import Any

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7,
    api_key=openai_api_key
)

def analyze_stock_data(symbol: str) -> str:
    """Analyzes fundamental stock data including price, market cap, P/E ratio, and other key metrics"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Determine currency based on exchange
        currency = "₹" if ".NS" in symbol else "$"
        
        # Format market cap based on size
        market_cap = info.get('marketCap', 0)
        if market_cap >= 1e12:
            market_cap_str = f"{market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            market_cap_str = f"{market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            market_cap_str = f"{market_cap/1e6:.2f}M"
        else:
            market_cap_str = f"{market_cap:,.2f}"
        
        analysis = {
            "Company Name": info.get('longName', 'N/A'),
            "Current Price": f"{currency}{info.get('currentPrice', 'N/A'):,.2f}",
            "Market Cap": f"{currency}{market_cap_str}",
            "P/E Ratio": f"{info.get('forwardPE', 'N/A'):.2f}",
            "52 Week Range": f"{currency}{info.get('fiftyTwoWeekLow', 'N/A'):,.2f} - {currency}{info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
            "Revenue Growth": f"{info.get('revenueGrowth', 'N/A')*100:.2f}%" if info.get('revenueGrowth') else 'N/A',
            "Profit Margins": f"{info.get('profitMargins', 'N/A')*100:.2f}%" if info.get('profitMargins') else 'N/A',
            "Analyst Rating": info.get('recommendationKey', 'N/A').upper(),
            "Currency": currency
        }
        
        return str(analysis)
    except Exception as e:
        return f"Error analyzing stock data: {str(e)}"

def search_stock_news(symbol: str) -> str:
    """Searches for recent news about a stock or company"""
    try:
        with DDGS() as ddgs:
            company_name = yf.Ticker(symbol).info.get('longName', symbol)
            # Search for different types of news
            search_terms = [
                f"{company_name} stock news last 30 days",
                f"{company_name} earnings report",
                f"{company_name} company announcements",
                f"{company_name} market analysis",
                
            ]
            
            all_results = []
            for term in search_terms:
                try:
                    results = list(ddgs.text(term, max_results=3))
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    print(f"Error searching for term '{term}': {str(e)}")
                    continue
            
            if not all_results:
                return "No recent news found for this stock."
            
            # Remove duplicates based on title
            seen_titles = set()
            news_summary = []
            for result in all_results:
                if result.get('title') and result['title'] not in seen_titles:
                    seen_titles.add(result['title'])
                    news_summary.append({
                        "title": result.get('title', 'No Title'),
                        "snippet": result.get('body', 'No content available'),
                        "source": result.get('source', 'Unknown Source'),
                        "date": result.get('date', 'Unknown Date'),
                        "url": result.get('link', '#')
                    })
            
            if not news_summary:
                return "No valid news articles found."
            
            # Sort by date (most recent first)
            news_summary.sort(key=lambda x: x['date'], reverse=True)
            
            # Format the output
            formatted_news = "Recent News and Updates:\n\n"
            for news in news_summary[:5]:  # Show top 5 most recent news
                formatted_news += f"📰 {news['title']}\n"
                formatted_news += f"📅 {news['date']}\n"
                formatted_news += f"🔍 {news['snippet']}\n"
                formatted_news += f"📌 Source: {news['source']}\n"
                formatted_news += f"🔗 Link: {news['url']}\n"
                formatted_news += "-" * 80 + "\n\n"
            
            return formatted_news
    except Exception as e:
        return f"Error fetching news: {str(e)}"

def analyze_technical_indicators(symbol: str, period: str = '1y') -> str:
    """Analyzes technical indicators including moving averages, RSI, and Bollinger Bands"""
    try:
        stock = yf.Ticker(symbol)
        # Fix period format
        valid_periods = {
            '1m': '1mo',
            '3m': '3mo',
            '6m': '6mo',
            '1y': '1y',
            '2y': '2y',
            '5y': '5y',
            '10y': '10y',
            'ytd': 'ytd',
            'max': 'max'
        }
        period = valid_periods.get(period, '1y')
        hist = stock.history(period=period)
        
        if hist.empty:
            return "No historical data available for the specified period"
            
        # Determine currency based on exchange
        currency = "₹" if ".NS" in symbol else "$"
        
        sma_20 = SMAIndicator(close=hist['Close'], window=20)
        sma_50 = SMAIndicator(close=hist['Close'], window=50)
        rsi = RSIIndicator(close=hist['Close'])
        bb = BollingerBands(close=hist['Close'])
        
        # Get the last values using iloc to avoid deprecation warnings
        current_price = hist['Close'].iloc[-1]
        sma_20_val = sma_20.sma_indicator().iloc[-1]
        sma_50_val = sma_50.sma_indicator().iloc[-1]
        rsi_val = rsi.rsi().iloc[-1]
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        
        analysis = {
            "Current Price": f"{currency}{current_price:,.2f}",
            "SMA 20": f"{currency}{sma_20_val:,.2f}",
            "SMA 50": f"{currency}{sma_50_val:,.2f}",
            "RSI": f"{rsi_val:.2f}",
            "Bollinger Upper": f"{currency}{bb_upper:,.2f}",
            "Bollinger Lower": f"{currency}{bb_lower:,.2f}",
            "Currency": currency
        }
        
        return str(analysis)
    except Exception as e:
        return f"Error calculating technical indicators: {str(e)}"


class StockAnalyzer:
    def __init__(self):
        # Create tools
        self.stock_data_tool = Tool(
            name="Stock Fundamental Analysis",
            func=analyze_stock_data,
            description="Analyzes fundamental stock data including price, market cap, P/E ratio, and other key metrics"
        )

        self.web_news_tool = Tool(
            name="Web News Search",
            func=search_stock_news,
            description="Searches for news from various web sources using DuckDuckGo"
        )

        self.technical_analysis_tool = Tool(
            name="Technical Analysis",
            func=analyze_technical_indicators,
            description="Analyzes technical indicators including moving averages, RSI, and Bollinger Bands"
        )

        # Create specialized agents
        self.financial_analyst = Agent(
            role='Financial Analyst',
            goal='Analyze stock financial data and provide comprehensive insights',
            backstory="""You are an experienced financial analyst with expertise in 
            technical and fundamental analysis. You analyze market data to provide 
            actionable insights and predictions.""",
            tools=[self.stock_data_tool],
            llm=llm,
            verbose=True
        )

        self.research_analyst = Agent(
            role='Research Analyst',
            goal='Gather and analyze market news and sentiment from multiple sources',
            backstory="""You are a thorough research analyst who gathers and analyzes 
            market news, sentiment, and trends from and web sources 
            to provide comprehensive context for investment decisions.""",
            tools=[self.web_news_tool],
            llm=llm,
            verbose=True
        )

        self.technical_analyst = Agent(
            role='Technical Analyst',
            goal='Analyze technical indicators and chart patterns',
            backstory="""You are a technical analysis expert who analyzes price patterns,
            indicators, and charts to identify trading opportunities and risks.""",
            tools=[self.technical_analysis_tool],
            llm=llm,
            verbose=True
        )

    def analyze_stock(self, symbol: str, analysis_types: list = None) -> str:
        """Main method to analyze a stock"""
        if analysis_types is None:
            analysis_types = ["Fundamental Analysis", "Technical Analysis", "News Analysis"]
        
        tasks = []
        agents = []
        
        try:
            # Add business information to the analysis
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                return f"Error: Could not fetch information for {symbol}"
            
            business_info = f"""
### Company Business Information
**Company Name:** {info.get('longName', 'N/A')}
**Sector:** {info.get('sector', 'N/A')}
**Industry:** {info.get('industry', 'N/A')}
**Country:** {info.get('country', 'N/A')}

**Business Overview:**
{info.get('longBusinessSummary', 'No business summary available.')}
"""
            
            # Create tasks based on selected analysis types
            if "Fundamental Analysis" in analysis_types:
                fundamental_analysis = Task(
                    description=f"""Analyze the fundamental data for {symbol} and provide insights on:
                    1. Company financial health
                    2. Valuation metrics
                    3. Growth prospects
                    4. Competitive position
                    5. Ownership pattern""",
                    agent=self.financial_analyst,
                    expected_output="""A detailed analysis of the company's fundamental data including:
                    - Financial health assessment
                    - Valuation analysis
                    - Growth potential evaluation
                    - Competitive position analysis
                    - Ownership pattern analysis"""
                )
                tasks.append(fundamental_analysis)
                agents.append(self.financial_analyst)

            if "News Analysis" in analysis_types:
                news_analysis = Task(
                    description=f"""Research and analyze recent news and market sentiment for {symbol}:
                    1. Key news events and their impact
                    2. Earnings reports and financial updates
                    3. Company announcements and developments
                    4. Market sentiment and analyst opinions
                    5. Industry trends and competitive landscape
                    6. Potential catalysts for future growth
                    7. Risk factors and challenges""",
                    agent=self.research_analyst,
                    expected_output="""A comprehensive summary of recent news and market sentiment including:
                    - Key news events and their impact 
                    - Earnings and financial performance
                    - Company developments and announcements
                    - Market sentiment analysis
                    - Industry trend analysis
                    - Potential catalysts and risks"""
                )
                tasks.append(news_analysis)
                agents.append(self.research_analyst)

            if "Technical Analysis" in analysis_types:
                technical_analysis = Task(
                    description=f"""Analyze technical indicators and chart patterns for {symbol}:
                    1. Price trends
                    2. Support and resistance levels
                    3. Technical indicators
                    4. Trading signals""",
                    agent=self.technical_analyst,
                    expected_output="""A detailed technical analysis including:
                    - Price trend analysis
                    - Support and resistance levels
                    - Technical indicator interpretations
                    - Trading signals and recommendations""",
                    context={"period": "6mo"}  # Set default period to 6mo
                )
                tasks.append(technical_analysis)
                agents.append(self.technical_analyst)

            if not tasks:
                return "Please select at least one analysis type."

            # Create and run the crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential
            )

            result = crew.kickoff()
            # Convert CrewOutput to string and handle potential None values
            result_str = str(result) if result is not None else "No analysis results available."
            return business_info + "\n\n" + result_str
        except Exception as e:
            return f"Error analyzing {symbol}: {str(e)}"

# Example usage
if __name__ == "__main__":
    analyzer = StockAnalyzer()
    result = analyzer.analyze_stock("AAPL")
    print(result) 