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
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from duckduckgo_search import DDGS
from typing import Any
from scipy import stats

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

def analyze_advanced_technical_indicators(symbol: str, period: str = '1y') -> str:
    """Analyzes advanced technical indicators including MACD, Stochastic, and ATR"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return "No historical data available for the specified period"
            
        # Calculate MACD
        macd = MACD(close=hist['Close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        macd_hist = macd.macd_diff()
        
        # Calculate Stochastic Oscillator
        stoch = StochasticOscillator(high=hist['High'], low=hist['Low'], close=hist['Close'])
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        
        # Calculate ATR
        atr = AverageTrueRange(high=hist['High'], low=hist['Low'], close=hist['Close'])
        atr_value = atr.average_true_range()
        
        # Calculate Fibonacci Retracement levels
        high = hist['High'].max()
        low = hist['Low'].min()
        diff = high - low
        fib_levels = {
            '0%': low,
            '23.6%': low + 0.236 * diff,
            '38.2%': low + 0.382 * diff,
            '50%': low + 0.5 * diff,
            '61.8%': low + 0.618 * diff,
            '100%': high
        }
        
        analysis = {
            "MACD": {
                "MACD Line": f"{macd_line.iloc[-1]:.2f}",
                "Signal Line": f"{signal_line.iloc[-1]:.2f}",
                "MACD Histogram": f"{macd_hist.iloc[-1]:.2f}"
            },
            "Stochastic Oscillator": {
                "K Line": f"{stoch_k.iloc[-1]:.2f}",
                "D Line": f"{stoch_d.iloc[-1]:.2f}"
            },
            "ATR": f"{atr_value.iloc[-1]:.2f}",
            "Fibonacci Retracement Levels": fib_levels
        }
        
        return str(analysis)
    except Exception as e:
        return f"Error calculating advanced technical indicators: {str(e)}"

def calculate_risk_metrics(symbol: str, period: str = '1y') -> str:
    """Calculates various risk metrics including VaR, Sharpe ratio, and Beta"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return "No historical data available for the specified period"
            
        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # Calculate Beta (using appropriate market index based on exchange)
        try:
            # Determine market index based on exchange
            if ".NS" in symbol:
                market = yf.Ticker("^NSEI")  # NIFTY 50 for Indian stocks
            else:
                market = yf.Ticker("^GSPC")  # S&P 500 for US stocks
            
            market_hist = market.history(period=period)
            if not market_hist.empty:
                market_returns = market_hist['Close'].pct_change().dropna()
                
                # Align the dates between stock and market returns
                aligned_returns = pd.DataFrame({
                    'stock': returns,
                    'market': market_returns
                }).dropna()
                
                if len(aligned_returns) > 1:
                    beta = stats.linregress(aligned_returns['market'], aligned_returns['stock'])[0]
                else:
                    beta = 0
            else:
                beta = 0
        except Exception as e:
            print(f"Error calculating beta: {str(e)}")
            beta = 0
        
        # Calculate Maximum Drawdown
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
        except Exception as e:
            print(f"Error calculating max drawdown: {str(e)}")
            max_drawdown = 0
        
        analysis = {
            "Value at Risk (95%)": f"{var_95:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Beta": f"{beta:.2f}",
            "Maximum Drawdown": f"{max_drawdown:.2%}"
        }
        
        return str(analysis)
    except Exception as e:
        return f"Error calculating risk metrics: {str(e)}"

def calculate_financial_ratios(symbol: str) -> str:
    """Calculates key financial ratios including D/E, current ratio, ROE, ROA, and operating margin"""
    try:
        stock = yf.Ticker(symbol)
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        
        if balance_sheet.empty or income_stmt.empty:
            return "Unable to calculate financial ratios: Missing financial data"
        
        # Get the most recent values
        latest_bs = balance_sheet.iloc[:, 0]
        latest_is = income_stmt.iloc[:, 0]
        
        # Calculate ratios
        ratios = {}
        
        # Debt-to-Equity Ratio
        try:
            total_debt = latest_bs.get('Total Debt', 0)
            total_equity = latest_bs.get('Total Stockholder Equity', 0)
            if total_equity != 0:
                ratios['Debt-to-Equity'] = f"{total_debt/total_equity:.2f}"
            else:
                ratios['Debt-to-Equity'] = 'N/A'
        except:
            ratios['Debt-to-Equity'] = 'N/A'
        
        # Current Ratio
        try:
            current_assets = latest_bs.get('Total Current Assets', 0)
            current_liabilities = latest_bs.get('Total Current Liabilities', 0)
            if current_liabilities != 0:
                ratios['Current Ratio'] = f"{current_assets/current_liabilities:.2f}"
            else:
                ratios['Current Ratio'] = 'N/A'
        except:
            ratios['Current Ratio'] = 'N/A'
        
        # Return on Equity (ROE)
        try:
            net_income = latest_is.get('Net Income', 0)
            total_equity = latest_bs.get('Total Stockholder Equity', 0)
            if total_equity != 0:
                ratios['ROE'] = f"{(net_income/total_equity)*100:.2f}%"
            else:
                ratios['ROE'] = 'N/A'
        except:
            ratios['ROE'] = 'N/A'
        
        # Return on Assets (ROA)
        try:
            total_assets = latest_bs.get('Total Assets', 0)
            if total_assets != 0:
                ratios['ROA'] = f"{(net_income/total_assets)*100:.2f}%"
            else:
                ratios['ROA'] = 'N/A'
        except:
            ratios['ROA'] = 'N/A'
        
        # Operating Margin
        try:
            operating_income = latest_is.get('Operating Income', 0)
            total_revenue = latest_is.get('Total Revenue', 0)
            if total_revenue != 0:
                ratios['Operating Margin'] = f"{(operating_income/total_revenue)*100:.2f}%"
            else:
                ratios['Operating Margin'] = 'N/A'
        except:
            ratios['Operating Margin'] = 'N/A'
        
        # Add industry-specific ratios based on sector
        sector = stock.info.get('sector', '').lower()
        
        # Technology sector ratios
        if 'technology' in sector:
            try:
                # R&D to Revenue
                r_and_d = latest_is.get('Research And Development', 0)
                if total_revenue != 0:
                    ratios['R&D to Revenue'] = f"{(r_and_d/total_revenue)*100:.2f}%"
            except:
                pass
            
            try:
                # Free Cash Flow Margin
                fcf = latest_is.get('Free Cash Flow', 0)
                if total_revenue != 0:
                    ratios['FCF Margin'] = f"{(fcf/total_revenue)*100:.2f}%"
            except:
                pass
        
        # Healthcare sector ratios
        elif 'healthcare' in sector:
            try:
                # Gross Margin
                gross_profit = latest_is.get('Gross Profit', 0)
                if total_revenue != 0:
                    ratios['Gross Margin'] = f"{(gross_profit/total_revenue)*100:.2f}%"
            except:
                pass
        
        # Energy sector ratios
        elif 'energy' in sector:
            try:
                # EBITDA Margin
                ebitda = latest_is.get('EBITDA', 0)
                if total_revenue != 0:
                    ratios['EBITDA Margin'] = f"{(ebitda/total_revenue)*100:.2f}%"
            except:
                pass
        
        # Financial sector ratios (including banks)
        elif 'financial' in sector:
            try:
                # Net Interest Margin
                net_interest_income = latest_is.get('Net Interest Income', 0)
                average_earning_assets = latest_bs.get('Total Assets', 0)
                if average_earning_assets != 0:
                    ratios['Net Interest Margin'] = f"{(net_interest_income/average_earning_assets)*100:.2f}%"
            except:
                pass
            
            try:
                # Capital Adequacy Ratio
                total_capital = latest_bs.get('Total Stockholder Equity', 0)
                risk_weighted_assets = latest_bs.get('Total Assets', 0)
                if risk_weighted_assets != 0:
                    ratios['Capital Adequacy Ratio'] = f"{(total_capital/risk_weighted_assets)*100:.2f}%"
            except:
                pass
        
        # Retail sector ratios
        elif 'retail' in sector:
            try:
                # Inventory Turnover
                inventory = latest_bs.get('Inventory', 0)
                cogs = latest_is.get('Cost Of Revenue', 0)
                if inventory != 0:
                    ratios['Inventory Turnover'] = f"{cogs/inventory:.2f}"
            except:
                pass
        
        # Add common ratios for all sectors
        try:
            # Asset Turnover
            if total_assets != 0:
                ratios['Asset Turnover'] = f"{total_revenue/total_assets:.2f}"
        except:
            pass
        
        try:
            # Quick Ratio
            quick_assets = current_assets - latest_bs.get('Inventory', 0)
            if current_liabilities != 0:
                ratios['Quick Ratio'] = f"{quick_assets/current_liabilities:.2f}"
        except:
            pass
        
        return str(ratios)
    except Exception as e:
        return f"Error calculating financial ratios: {str(e)}"

def analyze_sector_and_peers(symbol: str) -> str:
    """Analyzes sector performance and compares with peer companies"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info:
            return "Unable to fetch company information"
        
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        # Get sector performance
        sector_performance = {}
        try:
            # Determine appropriate sector ETF based on sector and exchange
            if ".NS" in symbol:
                # For Indian stocks, use appropriate sector index
                if 'financial' in sector.lower():
                    sector_etf = yf.Ticker("^NSEBANK")  # NIFTY BANK for financials
                elif 'technology' in sector.lower():
                    sector_etf = yf.Ticker("^CNXIT")  # NIFTY IT for technology
                elif 'healthcare' in sector.lower():
                    sector_etf = yf.Ticker("^CNXPHARMA")  # NIFTY PHARMA for healthcare
                else:
                    sector_etf = yf.Ticker("^NSEI")  # NIFTY 50 as default
            else:
                # For US stocks, use appropriate sector ETF
                if 'technology' in sector.lower():
                    sector_etf = yf.Ticker("XLK")  # Technology Select Sector SPDR
                elif 'healthcare' in sector.lower():
                    sector_etf = yf.Ticker("XLV")  # Healthcare Select Sector SPDR
                elif 'financial' in sector.lower():
                    sector_etf = yf.Ticker("XLF")  # Financial Select Sector SPDR
                elif 'energy' in sector.lower():
                    sector_etf = yf.Ticker("XLE")  # Energy Select Sector SPDR
                elif 'consumer' in sector.lower():
                    sector_etf = yf.Ticker("XLP")  # Consumer Staples Select Sector SPDR
                else:
                    sector_etf = yf.Ticker("^GSPC")  # S&P 500 as default
            
            sector_performance['Sector ETF Performance'] = f"{sector_etf.info.get('regularMarketChangePercent', 0):.2f}%"
            sector_performance['Sector ETF Price'] = f"${sector_etf.info.get('regularMarketPrice', 0):,.2f}"
            sector_performance['Sector ETF Volume'] = f"{sector_etf.info.get('regularMarketVolume', 0):,}"
        except Exception as e:
            print(f"Error getting sector performance: {str(e)}")
            sector_performance['Sector ETF Performance'] = 'N/A'
        
        # Get peer comparison
        peers = info.get('recommendedSymbols', [])
        peer_comparison = {}
        
        # Add default peers based on sector if no recommended symbols
        if not peers:
            if ".NS" in symbol:
                if 'financial' in sector.lower():
                    peers = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
                elif 'technology' in sector.lower():
                    peers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"]
                elif 'healthcare' in sector.lower():
                    peers = ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "BIOCON.NS"]
            else:
                if 'technology' in sector.lower():
                    peers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                elif 'healthcare' in sector.lower():
                    peers = ["JNJ", "PFE", "UNH", "MRK", "ABBV"]
                elif 'financial' in sector.lower():
                    peers = ["JPM", "BAC", "WFC", "GS", "MS"]
                elif 'energy' in sector.lower():
                    peers = ["XOM", "CVX", "COP", "SLB", "EOG"]
        
        for peer in peers[:5]:  # Compare with top 5 peers
            try:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                # Determine currency based on exchange
                currency = "₹" if ".NS" in peer else "$"
                
                # Get sector-specific metrics
                metrics = {
                    'Price': f"{currency}{peer_info.get('currentPrice', 'N/A'):,.2f}",
                    'Market Cap': f"{currency}{peer_info.get('marketCap', 0)/1e9:.2f}B",
                    'P/E': f"{peer_info.get('forwardPE', 'N/A'):.2f}",
                    'Change': f"{peer_info.get('regularMarketChangePercent', 0):.2f}%"
                }
                
                # Add sector-specific metrics
                if 'technology' in sector.lower():
                    metrics['Revenue Growth'] = f"{peer_info.get('revenueGrowth', 'N/A')*100:.2f}%" if peer_info.get('revenueGrowth') else 'N/A'
                    metrics['R&D to Revenue'] = f"{peer_info.get('researchAndDevelopmentToRevenue', 'N/A')*100:.2f}%" if peer_info.get('researchAndDevelopmentToRevenue') else 'N/A'
                elif 'healthcare' in sector.lower():
                    metrics['Gross Margin'] = f"{peer_info.get('grossMargins', 'N/A')*100:.2f}%" if peer_info.get('grossMargins') else 'N/A'
                    metrics['Operating Margin'] = f"{peer_info.get('operatingMargins', 'N/A')*100:.2f}%" if peer_info.get('operatingMargins') else 'N/A'
                elif 'financial' in sector.lower():
                    metrics['ROE'] = f"{peer_info.get('returnOnEquity', 'N/A')*100:.2f}%" if peer_info.get('returnOnEquity') else 'N/A'
                    metrics['ROA'] = f"{peer_info.get('returnOnAssets', 'N/A')*100:.2f}%" if peer_info.get('returnOnAssets') else 'N/A'
                elif 'energy' in sector.lower():
                    metrics['EBITDA Margin'] = f"{peer_info.get('ebitdaMargins', 'N/A')*100:.2f}%" if peer_info.get('ebitdaMargins') else 'N/A'
                    metrics['Operating Margin'] = f"{peer_info.get('operatingMargins', 'N/A')*100:.2f}%" if peer_info.get('operatingMargins') else 'N/A'
                
                peer_comparison[peer] = metrics
            except Exception as e:
                print(f"Error analyzing peer {peer}: {str(e)}")
                continue
        
        analysis = {
            'Sector': sector,
            'Industry': industry,
            'Sector Performance': sector_performance,
            'Peer Comparison': peer_comparison
        }
        
        return str(analysis)
    except Exception as e:
        return f"Error analyzing sector and peers: {str(e)}"

def analyze_dividend_and_growth(symbol: str) -> str:
    """Analyzes dividend metrics and earnings growth rate"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        earnings = stock.earnings
        
        if earnings.empty:
            return "Unable to analyze dividend and growth: Missing earnings data"
        
        # Determine currency based on exchange
        currency = "₹" if ".NS" in symbol else "$"
        
        # Get sector for sector-specific analysis
        sector = info.get('sector', '').lower()
        
        # Dividend Analysis
        dividend_metrics = {
            'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
            'Dividend Rate': f"{currency}{info.get('dividendRate', 0):,.2f}" if info.get('dividendRate') else 'N/A',
            'Payout Ratio': f"{info.get('payoutRatio', 0)*100:.2f}%" if info.get('payoutRatio') else 'N/A',
            '5-Year Average Dividend Yield': f"{info.get('fiveYearAvgDividendYield', 0)*100:.2f}%" if info.get('fiveYearAvgDividendYield') else 'N/A'
        }
        
        # Earnings Growth Analysis
        earnings_growth = {}
        try:
            # Calculate year-over-year growth rates
            earnings['YoY Growth'] = earnings['Earnings'].pct_change(periods=4) * 100
            earnings['Revenue Growth'] = earnings['Revenue'].pct_change(periods=4) * 100
            
            # Get latest growth rates
            earnings_growth['Latest Earnings Growth'] = f"{earnings['YoY Growth'].iloc[-1]:.2f}%"
            earnings_growth['Latest Revenue Growth'] = f"{earnings['Revenue Growth'].iloc[-1]:.2f}%"
            
            # Calculate 5-year CAGR
            if len(earnings) >= 20:  # 5 years of quarterly data
                earnings_cagr = ((earnings['Earnings'].iloc[-1] / earnings['Earnings'].iloc[-20]) ** (1/5) - 1) * 100
                revenue_cagr = ((earnings['Revenue'].iloc[-1] / earnings['Revenue'].iloc[-20]) ** (1/5) - 1) * 100
                earnings_growth['5-Year Earnings CAGR'] = f"{earnings_cagr:.2f}%"
                earnings_growth['5-Year Revenue CAGR'] = f"{revenue_cagr:.2f}%"
        except:
            earnings_growth['Latest Earnings Growth'] = 'N/A'
            earnings_growth['Latest Revenue Growth'] = 'N/A'
            earnings_growth['5-Year Earnings CAGR'] = 'N/A'
            earnings_growth['5-Year Revenue CAGR'] = 'N/A'
        
        # Sector-specific metrics
        sector_metrics = {}
        try:
            if 'technology' in sector:
                # Technology sector metrics
                sector_metrics['R&D to Revenue'] = f"{info.get('researchAndDevelopmentToRevenue', 0)*100:.2f}%" if info.get('researchAndDevelopmentToRevenue') else 'N/A'
                sector_metrics['Free Cash Flow Margin'] = f"{info.get('freeCashflowToRevenue', 0)*100:.2f}%" if info.get('freeCashflowToRevenue') else 'N/A'
                sector_metrics['Operating Cash Flow'] = f"{currency}{info.get('operatingCashflow', 0):,.2f}" if info.get('operatingCashflow') else 'N/A'
            
            elif 'healthcare' in sector:
                # Healthcare sector metrics
                sector_metrics['Gross Margin'] = f"{info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else 'N/A'
                sector_metrics['Operating Margin'] = f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A'
                sector_metrics['EBITDA Margin'] = f"{info.get('ebitdaMargins', 0)*100:.2f}%" if info.get('ebitdaMargins') else 'N/A'
            
            elif 'energy' in sector:
                # Energy sector metrics
                sector_metrics['EBITDA Margin'] = f"{info.get('ebitdaMargins', 0)*100:.2f}%" if info.get('ebitdaMargins') else 'N/A'
                sector_metrics['Operating Margin'] = f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A'
                sector_metrics['Profit Margin'] = f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A'
            
            elif 'financial' in sector:
                # Financial sector metrics
                sector_metrics['Net Interest Margin'] = f"{info.get('netInterestMargin', 0)*100:.2f}%" if info.get('netInterestMargin') else 'N/A'
                sector_metrics['Capital Adequacy Ratio'] = f"{info.get('capitalAdequacyRatio', 0)*100:.2f}%" if info.get('capitalAdequacyRatio') else 'N/A'
                sector_metrics['Cost to Income Ratio'] = f"{info.get('costToIncomeRatio', 0)*100:.2f}%" if info.get('costToIncomeRatio') else 'N/A'
            
            elif 'retail' in sector:
                # Retail sector metrics
                sector_metrics['Gross Margin'] = f"{info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else 'N/A'
                sector_metrics['Operating Margin'] = f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A'
                sector_metrics['Inventory Turnover'] = f"{info.get('inventoryTurnover', 0):.2f}" if info.get('inventoryTurnover') else 'N/A'
            
            elif 'industrial' in sector:
                # Industrial sector metrics
                sector_metrics['Operating Margin'] = f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A'
                sector_metrics['EBITDA Margin'] = f"{info.get('ebitdaMargins', 0)*100:.2f}%" if info.get('ebitdaMargins') else 'N/A'
                sector_metrics['Asset Turnover'] = f"{info.get('assetTurnover', 0):.2f}" if info.get('assetTurnover') else 'N/A'
            
            elif 'consumer' in sector:
                # Consumer sector metrics
                sector_metrics['Gross Margin'] = f"{info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else 'N/A'
                sector_metrics['Operating Margin'] = f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A'
                sector_metrics['Inventory Turnover'] = f"{info.get('inventoryTurnover', 0):.2f}" if info.get('inventoryTurnover') else 'N/A'
        except:
            pass
        
        analysis = {
            'Dividend Metrics': dividend_metrics,
            'Growth Metrics': earnings_growth,
            'Sector-Specific Metrics': sector_metrics
        }
        
        return str(analysis)
    except Exception as e:
        return f"Error analyzing dividend and growth: {str(e)}"

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

        self.financial_ratios_tool = Tool(
            name="Financial Ratios Analysis",
            func=calculate_financial_ratios,
            description="Calculates key financial ratios including D/E, current ratio, ROE, ROA, and operating margin"
        )

        self.sector_analysis_tool = Tool(
            name="Sector and Peer Analysis",
            func=analyze_sector_and_peers,
            description="Analyzes sector performance and compares with peer companies"
        )

        self.dividend_growth_tool = Tool(
            name="Dividend and Growth Analysis",
            func=analyze_dividend_and_growth,
            description="Analyzes dividend metrics and earnings growth rate"
        )

        # Create specialized agents
        self.financial_analyst = Agent(
            role='Financial Analyst',
            goal='Provide comprehensive financial analysis and investment insights using numbers and data to show your analysis',
            backstory="""You are an experienced financial analyst with expertise in:
            1. Fundamental analysis and valuation
            2. Financial statement analysis
            3. Industry and peer comparison
            4. Growth and profitability metrics
            5. Risk assessment and management
            6. Market positioning and competitive analysis
            7. Dividend and capital allocation analysis
            8. Economic impact assessment""",
            tools=[self.stock_data_tool, self.financial_ratios_tool, self.sector_analysis_tool, self.dividend_growth_tool],
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
            goal='Analyze technical patterns and provide trading insights',
            backstory="""You are a technical analysis expert who specializes in:
            1. Price action and chart patterns
            2. Technical indicators and oscillators
            3. Volume analysis and market depth
            4. Support and resistance levels
            5. Trend analysis and momentum
            6. Market structure and cycles
            7. Volatility analysis
            8. Trading signals and entry/exit points""",
            tools=[self.technical_analysis_tool],
            llm=llm,
            verbose=True
        )

        self.risk_analyst = Agent(
            role='Risk Analyst',
            goal='Assess and quantify investment risks',
            backstory="""You are a risk management specialist who focuses on:
            1. Portfolio risk assessment
            2. Market risk analysis
            3. Volatility metrics
            4. Correlation analysis
            5. Value at Risk (VaR) calculation
            6. Stress testing
            7. Risk-adjusted returns
            8. Position sizing recommendations""",
            tools=[self.technical_analysis_tool, self.financial_ratios_tool],
            llm=llm,
            verbose=True
        )

        self.portfolio_analyst = Agent(
            role='Portfolio Analyst',
            goal='Optimize portfolio allocation and performance',
            backstory="""You are a portfolio management expert who specializes in:
            1. Asset allocation optimization
            2. Portfolio rebalancing
            3. Risk management
            4. Performance attribution
            5. Diversification strategies
            6. Sector rotation
            7. Portfolio construction
            8. Investment style analysis""",
            tools=[self.stock_data_tool, self.technical_analysis_tool, self.sector_analysis_tool, self.financial_ratios_tool],
            llm=llm,
            verbose=True
        )

    def analyze_stock(self, symbol: str, analysis_types: list = None) -> str:
        """Main method to analyze a stock"""
        if analysis_types is None:
            analysis_types = ["Fundamental Analysis", "Technical Analysis", "Risk Analysis", "Portfolio Analysis"]
        
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
                    1. Company financial health and performance
                    2. Valuation metrics and ratios
                    3. Growth prospects and potential
                    4. Competitive position and market share
                    5. Industry trends and dynamics
                    6. Management effectiveness
                    7. Dividend policy and capital allocation
                    8. Economic impact and external factors
                    9. Financial ratios (D/E, Current Ratio, ROE, ROA, Operating Margin)
                    10. Sector and peer comparison
                    11. Dividend analysis and yield metrics
                    12. Earnings growth rate analysis

                    IMPORTANT: You MUST use all available tools to provide comprehensive analysis:
                    1. Use stock_data_tool for basic company information and metrics
                    2. Use financial_ratios_tool to analyze key financial ratios
                    3. Use sector_analysis_tool to compare with peers and sector performance
                    4. Use dividend_growth_tool to analyze dividend metrics and growth rates

                    For each section, provide specific numbers and data points from the tools.
                    If any data is unavailable, explicitly state that it could not be retrieved.""",
                    agent=self.financial_analyst,
                    expected_output="""A comprehensive fundamental analysis including:
                    - Financial health assessment with specific metrics
                    - Valuation analysis with P/E and other ratios
                    - Growth potential evaluation with growth rates
                    - Competitive position analysis with peer comparisons
                    - Industry trend analysis with sector performance
                    - Management assessment with efficiency metrics
                    - Dividend analysis with yield and payout ratios
                    - Economic impact assessment
                    - Financial ratios analysis (D/E, Current Ratio, ROE, ROA, Operating Margin)
                    - Sector and peer comparison with specific metrics
                    - Growth rate analysis with historical trends."""
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
                    7. Risk factors and challenges

                    IMPORTANT: You MUST use the web_news_tool to:
                    1. Search for recent news articles
                    2. Analyze earnings reports
                    3. Review company announcements
                    4. Gather market analysis
                    
                    For each section, provide specific details from the news articles including dates, sources, and key points.
                    If any data is unavailable, explicitly state that it could not be retrieved.""",
                    agent=self.research_analyst,
                    expected_output="""A comprehensive summary of recent news and market sentiment including:
                    - Key news events with dates and sources
                    - Earnings reports with specific numbers
                    - Company announcements with details
                    - Market sentiment analysis with supporting data
                    - Industry trend analysis with specific examples
                    - Potential catalysts with supporting evidence
                    - Risk factors with specific examples"""
                )
                tasks.append(news_analysis)
                agents.append(self.research_analyst)

            if "Technical Analysis" in analysis_types:
                technical_analysis = Task(
                    description=f"""Analyze technical indicators and chart patterns for {symbol}:
                    1. Price trends and patterns
                    2. Support and resistance levels
                    3. Technical indicators and oscillators
                    4. Volume analysis and market depth
                    5. Market structure and cycles
                    6. Volatility analysis
                    7. Trading signals
                    8. Entry and exit points

                    IMPORTANT: You MUST use the technical_analysis_tool to:
                    1. Calculate and analyze moving averages (SMA 20, SMA 50)
                    2. Evaluate RSI levels and trends
                    3. Analyze Bollinger Bands
                    4. Provide specific price levels and indicators
                    
                    For each section, provide specific numbers and levels from the technical indicators.
                    If any data is unavailable, explicitly state that it could not be retrieved.""",
                    agent=self.technical_analyst,
                    expected_output="""A detailed technical analysis including:
                    - Price trend analysis with specific levels
                    - Support and resistance levels with prices
                    - Technical indicator values and interpretations
                    - Volume analysis with specific numbers
                    - Market structure analysis with key levels
                    - Volatility assessment with specific metrics
                    - Trading signals with entry/exit prices
                    - Entry/exit recommendations with price targets""",
                    context={"period": "6mo"}
                )
                tasks.append(technical_analysis)
                agents.append(self.technical_analyst)

            if "Risk Analysis" in analysis_types:
                risk_analysis = Task(
                    description=f"""Assess investment risks for {symbol}:
                    1. Market risk assessment
                    2. Volatility metrics
                    3. Correlation analysis
                    4. Value at Risk (VaR)
                    5. Stress testing scenarios
                    6. Risk-adjusted returns
                    7. Position sizing recommendations
                    8. Risk mitigation strategies

                    IMPORTANT: You MUST use:
                    1. technical_analysis_tool for volatility metrics and market risk
                    2. financial_ratios_tool for financial risk assessment
                    
                    For each section, provide specific numbers and metrics from the tools.
                    If any data is unavailable, explicitly state that it could not be retrieved.""",
                    agent=self.risk_analyst,
                    expected_output="""A comprehensive risk analysis including:
                    - Market risk assessment with specific metrics
                    - Volatility analysis with exact numbers
                    - Correlation metrics with specific values
                    - VaR calculations with confidence levels
                    - Stress test results with specific scenarios
                    - Risk-adjusted performance metrics
                    - Position sizing recommendations with specific numbers
                    - Risk mitigation strategies with concrete steps"""
                )
                tasks.append(risk_analysis)
                agents.append(self.risk_analyst)

            if "Portfolio Analysis" in analysis_types:
                portfolio_analysis = Task(
                    description=f"""Analyze portfolio implications for {symbol}:
                    1. Asset allocation optimization
                    2. Portfolio rebalancing needs
                    3. Risk management considerations
                    4. Performance attribution
                    5. Diversification benefits
                    6. Sector rotation impact
                    7. Portfolio construction advice
                    8. Investment style analysis

                    IMPORTANT: You MUST use:
                    1. stock_data_tool for company fundamentals
                    2. technical_analysis_tool for price trends
                    3. sector_analysis_tool for sector positioning
                    4. financial_ratios_tool for financial metrics
                    
                    For each section, provide specific numbers and recommendations from the tools.
                    If any data is unavailable, explicitly state that it could not be retrieved.""",
                    agent=self.portfolio_analyst,
                    expected_output="""A detailed portfolio analysis including:
                    - Asset allocation recommendations with specific percentages
                    - Rebalancing suggestions with target weights
                    - Risk management insights with specific metrics
                    - Performance attribution with exact numbers
                    - Diversification analysis with correlation metrics
                    - Sector rotation advice with specific timing
                    - Portfolio construction guidance with exact allocations
                    - Investment style assessment with supporting data"""
                )
                tasks.append(portfolio_analysis)
                agents.append(self.portfolio_analyst)

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