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
import streamlit as st

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
        # Create a unique key for this stock's risk metrics
        risk_metrics_key = f"risk_metrics_{symbol}_{period}"
        
        # Check if we already have calculated risk metrics in session state for this specific stock
        if hasattr(st.session_state, risk_metrics_key) and st.session_state[risk_metrics_key] is not None:
            print(f"Using cached risk metrics for {symbol} from session state")
            return str(st.session_state[risk_metrics_key])
            
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return "No historical data available for the specified period"
            
        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # Calculate Sortino Ratio (downside risk only)
        negative_returns = returns[returns < 0]
        sortino_ratio = 0
        if len(negative_returns) > 0 and negative_returns.std() != 0:
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / negative_returns.std()
        
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
            print(f"Error calculating beta for {symbol}: {str(e)}")
            beta = 0
        
        # Calculate Maximum Drawdown
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
        except Exception as e:
            print(f"Error calculating max drawdown for {symbol}: {str(e)}")
            max_drawdown = 0
            
        # Stress Testing Scenarios
        try:
            # Market Crash Scenario (simulate 2008-like crash)
            market_crash_impact = returns.mean() - 2 * returns.std() * 5  # 5-sigma event
            
            # Interest Rate Hike Scenario
            if beta > 0:
                interest_rate_impact = -0.05 * beta  # 5% market drop due to rate hike
            else:
                interest_rate_impact = -0.02  # Default impact
                
            # Sector-Specific Shock
            sector_shock = returns.mean() - 1.5 * returns.std()
            
            # Calculate Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95 * 1.5
        except Exception as e:
            print(f"Error in stress testing for {symbol}: {str(e)}")
            market_crash_impact = -0.15  # Default value
            interest_rate_impact = -0.05  # Default value
            sector_shock = -0.10  # Default value
            cvar_95 = var_95 * 1.5  # Default value
        
        # Risk-Adjusted Returns
        try:
            # Treynor Ratio
            treynor_ratio = 0
            if beta != 0:
                treynor_ratio = (returns.mean() * 252 - risk_free_rate) / beta
                
            # Information Ratio (assuming market as benchmark)
            if 'market' in locals() and len(aligned_returns) > 1:
                tracking_error = (aligned_returns['stock'] - aligned_returns['market']).std() * np.sqrt(252)
                information_ratio = 0
                if tracking_error != 0:
                    information_ratio = (aligned_returns['stock'].mean() - aligned_returns['market'].mean()) * 252 / tracking_error
            else:
                information_ratio = 0
                
            # Calmar Ratio
            calmar_ratio = 0
            if max_drawdown != 0:
                calmar_ratio = (returns.mean() * 252) / abs(max_drawdown)
        except Exception as e:
            print(f"Error calculating risk-adjusted returns for {symbol}: {str(e)}")
            treynor_ratio = 0
            information_ratio = 0
            calmar_ratio = 0
        
        analysis = {
            "Value at Risk (95%)": f"{var_95:.2%}",
            "Value at Risk (99%)": f"{var_99:.2%}",
            "Conditional VaR (95%)": f"{cvar_95:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Beta": f"{beta:.2f}",
            "Maximum Drawdown": f"{max_drawdown:.2%}",
            "Market Crash Impact": f"{market_crash_impact:.2%}",
            "Interest Rate Hike Impact": f"{interest_rate_impact:.2%}",
            "Sector Shock Impact": f"{sector_shock:.2%}",
            "Treynor Ratio": f"{treynor_ratio:.4f}",
            "Information Ratio": f"{information_ratio:.2f}",
            "Calmar Ratio": f"{calmar_ratio:.2f}"
        }
        
        # Store in session state with unique key for this stock
        if hasattr(st, 'session_state'):
            st.session_state[risk_metrics_key] = analysis
            print(f"Stored risk metrics for {symbol} in session state: {analysis}")
        
        return str(analysis)
    except Exception as e:
        print(f"Error calculating risk metrics for {symbol}: {str(e)}")
        return f"Error calculating risk metrics: {str(e)}"

def calculate_financial_ratios(symbol: str) -> str:
    """Calculates key financial ratios for a given stock symbol"""
    try:
        # Check if we already have calculated ratios in session state
        if hasattr(st.session_state, 'financial_ratios') and st.session_state.financial_ratios is not None:
            print(f"Using cached financial ratios from session state: {st.session_state.financial_ratios}")
            return str(st.session_state.financial_ratios)
        
        print(f"Calculating financial ratios for {symbol}...")
        stock = yf.Ticker(symbol)
        
        # Initialize ratios dictionary
        ratios = {}
        
        # Try to get ratios from info first (most reliable source)
        try:
            info = stock.info
            if info:
                # Get company name for better identification
                company_name = info.get('longName', '')
                print(f"Analyzing {company_name} ({symbol})")
                
                # Try to get metrics directly from info
                if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                    ratios['ROE'] = f"{info['returnOnEquity']*100:.2f}%"
                
                if 'returnOnAssets' in info and info['returnOnAssets'] is not None:
                    ratios['ROA'] = f"{info['returnOnAssets']*100:.2f}%"
                
                if 'operatingMargins' in info and info['operatingMargins'] is not None:
                    ratios['Operating Margin'] = f"{info['operatingMargins']*100:.2f}%"
                
                if 'currentRatio' in info and info['currentRatio'] is not None:
                    ratios['Current Ratio'] = f"{info['currentRatio']:.2f}"
                
                # Calculate Debt-to-Equity if we have the components
                if 'totalDebt' in info and 'totalShareholderEquity' in info:
                    if info['totalShareholderEquity'] > 0:
                        de_ratio = info['totalDebt'] / info['totalShareholderEquity']
                        ratios['Debt-to-Equity'] = f"{de_ratio:.2f}"
                
                # Add additional ratios if available
                if 'priceToBook' in info and info['priceToBook'] is not None:
                    ratios['Price/Book'] = f"{info['priceToBook']:.2f}"
                
                if 'forwardPE' in info and info['forwardPE'] is not None:
                    ratios['P/E Ratio'] = f"{info['forwardPE']:.2f}"
                elif 'trailingPE' in info and info['trailingPE'] is not None:
                    ratios['P/E Ratio'] = f"{info['trailingPE']:.2f}"
                
                if 'enterpriseToEbitda' in info and info['enterpriseToEbitda'] is not None:
                    ratios['EV/EBITDA'] = f"{info['enterpriseToEbitda']:.2f}"
                
                if 'profitMargins' in info and info['profitMargins'] is not None:
                    ratios['Net Profit Margin'] = f"{info['profitMargins']*100:.2f}%"
                
                # Add ROCE if available
                
                print(f"Got ratios from info: {ratios}")
        except Exception as e:
            print(f"Error getting ratios from info: {str(e)}")
        
        # If we don't have all the ratios from info, try financial statements
        if len(ratios) < 5:  # If we're missing some key ratios
            try:
                # Try to get annual data first (more reliable for ratios)
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                
                if balance_sheet.empty or income_stmt.empty:
                    # If annual data is empty, try quarterly data
                    print("Annual financial statements empty, trying quarterly data...")
                    balance_sheet = stock.quarterly_balance_sheet
                    income_stmt = stock.quarterly_income_stmt
                
                if not balance_sheet.empty and not income_stmt.empty:
                    # Get the latest financial data
                    latest_bs = balance_sheet.iloc[:, 0].to_dict()
                    latest_is = income_stmt.iloc[:, 0].to_dict()
                    
                    print(f"Latest balance sheet date: {balance_sheet.columns[0]}")
                    print(f"Latest income statement date: {income_stmt.columns[0]}")
                    
                    # Print available keys for debugging
                    print(f"Balance sheet keys: {list(latest_bs.keys())}")
                    print(f"Income statement keys: {list(latest_is.keys())}")
                    
                    # Total Debt calculation with multiple attempts
                    total_debt = latest_bs.get('Total Debt', 
                                latest_bs.get('Long Term Debt', 
                                latest_bs.get('LongTermDebt', 
                                latest_bs.get('Total Liabilities Net Minority Interest', 
                                latest_bs.get('TotalDebt', 0)))))
                    
                    # Total Equity calculation with multiple attempts
                    total_equity = latest_bs.get('Total Equity Gross Minority Interest', 
                                  latest_bs.get('Stockholders Equity', 
                                  latest_bs.get('StockholdersEquity', 
                                  latest_bs.get('Total Equity', 
                                  latest_bs.get('TotalEquityGrossMinorityInterest', 
                                  latest_bs.get('TotalStockholdersEquity', 0))))))
                    
                    # Current Assets calculation with multiple attempts
                    current_assets = latest_bs.get('Current Assets', 
                                   latest_bs.get('Total Current Assets', 
                                   latest_bs.get('CurrentAssets', 
                                   latest_bs.get('TotalCurrentAssets', 
                                   latest_bs.get('Cash And Cash Equivalents', 
                                   latest_bs.get('CashAndCashEquivalents', 0))))))
                    
                    # Current Liabilities calculation with multiple attempts
                    current_liabilities = latest_bs.get('Current Liabilities', 
                                        latest_bs.get('Total Current Liabilities', 
                                        latest_bs.get('CurrentLiabilities', 
                                        latest_bs.get('TotalCurrentLiabilities', 
                                        latest_bs.get('Accounts Payable', 
                                        latest_bs.get('AccountsPayable', 0))))))
                    
                    # Net Income calculation with multiple attempts
                    net_income = latest_is.get('Net Income', 
                                latest_is.get('Net Income Common Stockholders', 
                                latest_is.get('NetIncome', 
                                latest_is.get('NetIncomeCommonStockholders', 
                                latest_is.get('Net Income from Continuing Operations', 
                                latest_is.get('NetIncomeFromContinuingOperations', 0))))))
                    
                    # Total Assets calculation with multiple attempts
                    total_assets = latest_bs.get('Total Assets', 
                                  latest_bs.get('TotalAssets', 0))
                    
                    # Operating Income calculation with multiple attempts
                    operating_income = latest_is.get('Operating Income', 
                                     latest_is.get('EBIT', 
                                     latest_is.get('OperatingIncome', 
                                     latest_is.get('Operating Income or Loss', 
                                     latest_is.get('OperatingIncomeOrLoss', 0)))))
                    
                    # Total Revenue calculation with multiple attempts
                    total_revenue = latest_is.get('Total Revenue',
                                   latest_is.get('Revenue',
                                   latest_is.get('TotalRevenue',
                                   latest_is.get('Gross Profit',
                                   latest_is.get('GrossProfit', 0)))))
                    
                    
                    # Calculate and format ratios if not already in the dictionary
                    
                    # Debt-to-Equity Ratio
                    if 'Debt-to-Equity' not in ratios and total_equity != 0:
                        ratios['Debt-to-Equity'] = f"{(total_debt / total_equity):.2f}"
                    
                    # Current Ratio
                    if 'Current Ratio' not in ratios and current_liabilities != 0:
                        ratios['Current Ratio'] = f"{(current_assets / current_liabilities):.2f}"
                    
                    # ROE
                    if 'ROE' not in ratios and total_equity != 0:
                        ratios['ROE'] = f"{(net_income / total_equity * 100):.2f}%"
                    
                    # ROA
                    if 'ROA' not in ratios and total_assets != 0:
                        ratios['ROA'] = f"{(net_income / total_assets * 100):.2f}%"
                    
                    # Operating Margin
                    if 'Operating Margin' not in ratios and total_revenue != 0:
                        ratios['Operating Margin'] = f"{(operating_income / total_revenue * 100):.2f}%"
                    
                    # Asset Turnover
                    if 'Asset Turnover' not in ratios and total_assets != 0:
                        ratios['Asset Turnover'] = f"{(total_revenue / total_assets):.2f}"
                    
                    # Net Profit Margin
                    if 'Net Profit Margin' not in ratios and total_revenue != 0:
                        ratios['Net Profit Margin'] = f"{(net_income / total_revenue * 100):.2f}%"
                    
                    print(f"Added ratios from financial statements: {ratios}")
            except Exception as e:
                print(f"Error calculating ratios from financial statements: {str(e)}")
        
        # Try to get more accurate data from alternative sources if still missing key ratios
        if 'Debt-to-Equity' not in ratios or 'ROE' not in ratios or 'ROA' not in ratios or 'Operating Margin' not in ratios or 'ROCE' not in ratios or 'Dividend Yield' not in ratios or 'Net Profit Margin' not in ratios or 'P/E Ratio' not in ratios or 'Price/Book' not in ratios:
            try:
                # Try to get more accurate data from stock.info
                info = stock.info
                
                # Fallback to info for missing metrics
                if 'Debt-to-Equity' not in ratios and 'debtToEquity' in info and info['debtToEquity'] is not None:
                    ratios['Debt-to-Equity'] = f"{info['debtToEquity']/100:.2f}"
                
                if 'ROE' not in ratios and 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                    ratios['ROE'] = f"{info['returnOnEquity']*100:.2f}%"
                
                if 'ROA' not in ratios and 'returnOnAssets' in info and info['returnOnAssets'] is not None:
                    ratios['ROA'] = f"{info['returnOnAssets']*100:.2f}%"
                
                if 'Operating Margin' not in ratios and 'operatingMargins' in info and info['operatingMargins'] is not None:
                    ratios['Operating Margin'] = f"{info['operatingMargins']*100:.2f}%"
                
                if 'Current Ratio' not in ratios and 'currentRatio' in info and info['currentRatio'] is not None:
                    ratios['Current Ratio'] = f"{info['currentRatio']:.2f}"
                
                # Try to calculate ROCE if missing
                if 'ROCE' not in ratios:
                    if 'returnOnCapital' in info and info['returnOnCapital'] is not None:
                        ratios['ROCE'] = f"{info['returnOnCapital']*100:.2f}%"
                    elif 'ebit' in info and 'totalAssets' in info and 'totalCurrentLiabilities' in info:
                        capital_employed = info['totalAssets'] - info['totalCurrentLiabilities']
                        if capital_employed > 0:
                            roce = (info['ebit'] / capital_employed) * 100
                            ratios['ROCE'] = f"{roce:.2f}%"
                
                # Try to get Dividend Yield if missing
                if 'Dividend Yield' not in ratios and 'dividendYield' in info and info['dividendYield'] is not None:
                    # Ensure dividend yield is reasonable (less than 15%)
                    if 0 <= info['dividendYield'] <= 0.15:
                        ratios['Dividend Yield'] = f"{info['dividendYield']*100:.2f}%"
                    else:
                        print(f"Unreasonable dividend yield: {info['dividendYield']*100:.2f}%, ignoring")
                
                # Try to calculate missing ratios from other available data
                if 'Debt-to-Equity' not in ratios and 'totalDebt' in info and 'totalShareholderEquity' in info:
                    if info['totalShareholderEquity'] > 0:
                        de_ratio = info['totalDebt'] / info['totalShareholderEquity']
                        ratios['Debt-to-Equity'] = f"{de_ratio:.2f}"
                
                # For Indian stocks, try to use more accurate data
                if '.NS' in symbol or '.BO' in symbol:
                    # For Indian stocks, sometimes the debt-to-equity is reported differently
                    if 'Debt-to-Equity' not in ratios and 'longTermDebt' in info and 'totalShareholderEquity' in info:
                        if info['totalShareholderEquity'] > 0:
                            de_ratio = info['longTermDebt'] / info['totalShareholderEquity']
                            ratios['Debt-to-Equity'] = f"{de_ratio:.2f}"
            except Exception as e:
                print(f"Error getting additional ratios from info: {str(e)}")
        
        # Ensure we have at least some values for the key ratios
        for key in ['Debt-to-Equity', 'Current Ratio', 'ROE', 'ROA', 'Operating Margin', 'ROCE', 'Dividend Yield', 'Net Profit Margin', 'P/E Ratio', 'Price/Book']:
            if key not in ratios:
                ratios[key] = 'N/A'
        
        # Validate the ratios to ensure they're reasonable
        try:
            # Debt-to-Equity validation
            if 'Debt-to-Equity' in ratios and ratios['Debt-to-Equity'] != 'N/A':
                de_value = float(ratios['Debt-to-Equity'].replace('%', ''))
                if de_value < 0 or de_value > 10:  # Unreasonable D/E ratio
                    print(f"Unreasonable Debt-to-Equity ratio: {de_value}, setting to N/A")
                    ratios['Debt-to-Equity'] = 'N/A'
            
            # Current Ratio validation
            if 'Current Ratio' in ratios and ratios['Current Ratio'] != 'N/A':
                cr_value = float(ratios['Current Ratio'].replace('%', ''))
                if cr_value < 0 or cr_value > 100:  # Unreasonable Current Ratio
                    print(f"Unreasonable Current Ratio: {cr_value}, setting to N/A")
                    ratios['Current Ratio'] = 'N/A'
            
            # ROE validation
            if 'ROE' in ratios and ratios['ROE'] != 'N/A':
                roe_value = float(ratios['ROE'].replace('%', ''))
                if roe_value < -100 or roe_value > 100:  # Unreasonable ROE
                    print(f"Unreasonable ROE: {roe_value}, setting to N/A")
                    ratios['ROE'] = 'N/A'
            
            # ROA validation
            if 'ROA' in ratios and ratios['ROA'] != 'N/A':
                roa_value = float(ratios['ROA'].replace('%', ''))
                if roa_value < -50 or roa_value > 50:  # Unreasonable ROA
                    print(f"Unreasonable ROA: {roa_value}, setting to N/A")
                    ratios['ROA'] = 'N/A'
            
            # Operating Margin validation
            if 'Operating Margin' in ratios and ratios['Operating Margin'] != 'N/A':
                om_value = float(ratios['Operating Margin'].replace('%', ''))
                if om_value < -100 or om_value > 100:  # Unreasonable Operating Margin
                    print(f"Unreasonable Operating Margin: {om_value}, setting to N/A")
                    ratios['Operating Margin'] = 'N/A'
            
            # ROCE validation
            if 'ROCE' in ratios and ratios['ROCE'] != 'N/A':
                roce_value = float(ratios['ROCE'].replace('%', ''))
                if roce_value < -100 or roce_value > 100:  # Unreasonable ROCE
                    print(f"Unreasonable ROCE: {roce_value}, setting to N/A")
                    ratios['ROCE'] = 'N/A'
            
            # Dividend Yield validation
            if 'Dividend Yield' in ratios and ratios['Dividend Yield'] != 'N/A':
                dy_value = float(ratios['Dividend Yield'].replace('%', ''))
                if dy_value < 0 or dy_value > 15:  # Unreasonable Dividend Yield (over 15%)
                    print(f"Unreasonable Dividend Yield: {dy_value}, setting to N/A")
                    ratios['Dividend Yield'] = 'N/A'
            
            # Net Profit Margin validation
            if 'Net Profit Margin' in ratios and ratios['Net Profit Margin'] != 'N/A':
                npm_value = float(ratios['Net Profit Margin'].replace('%', ''))
                if npm_value < -100 or npm_value > 100:  # Unreasonable Net Profit Margin
                    print(f"Unreasonable Net Profit Margin: {npm_value}, setting to N/A")
                    ratios['Net Profit Margin'] = 'N/A'
        except Exception as e:
            print(f"Error validating ratios: {str(e)}")
        
        # Store in session state for future use
        if hasattr(st, 'session_state'):
            st.session_state.financial_ratios = ratios
            print(f"Stored financial ratios in session state: {ratios}")
        
        print(f"Final calculated ratios: {ratios}")
        return str(ratios)
    except Exception as e:
        print(f"Error calculating financial ratios: {str(e)}")
        # Return a dictionary with N/A values for all ratios
        default_ratios = {
            'Debt-to-Equity': 'N/A',
            'Current Ratio': 'N/A',
            'ROE': 'N/A',
            'ROA': 'N/A',
            'Operating Margin': 'N/A',
            'ROCE': 'N/A',
            'Dividend Yield': 'N/A',
            'Net Profit Margin': 'N/A',
            'P/E Ratio': 'N/A',
            'Price/Book': 'N/A'
        }
        return str(default_ratios)

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
                    metrics['Net Interest Margin'] = f"{peer_info.get('netInterestMargin', 'N/A')*100:.2f}%" if peer_info.get('netInterestMargin') else 'N/A'
                    metrics['Capital Adequacy Ratio'] = f"{peer_info.get('capitalAdequacyRatio', 'N/A')*100:.2f}%" if peer_info.get('capitalAdequacyRatio') else 'N/A'
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
        
        # Get historical data for growth calculations
        hist_data = stock.history(period="5y")
        
        # Determine currency based on exchange
        currency = "₹" if ".NS" in symbol else "$"
        
        # Get sector for sector-specific analysis
        sector = info.get('sector', '').lower()
        
        # Dividend Analysis with multiple attempts for each metric
        dividend_metrics = {}
        
        # Dividend Yield
        dividend_yield = (
            info.get('dividendYield',
            info.get('yield',
            info.get('trailingAnnualDividendYield', None))))
        dividend_metrics['Dividend Yield'] = f"{dividend_yield*100:.2f}%" if dividend_yield else 'N/A'
        
        # Dividend Rate
        dividend_rate = (
            info.get('dividendRate',
            info.get('trailingAnnualDividendRate',
            info.get('lastDividendValue', None))))
        dividend_metrics['Dividend Rate'] = f"{currency}{dividend_rate:,.2f}" if dividend_rate else 'N/A'
        
        # Payout Ratio
        payout_ratio = (
            info.get('payoutRatio',
            info.get('dividendPayout',
            info.get('dividendPayoutRatio', None))))
        dividend_metrics['Payout Ratio'] = f"{payout_ratio*100:.2f}%" if payout_ratio else 'N/A'
        
        # 5-Year Average Dividend Yield
        five_year_avg_yield = (
            info.get('fiveYearAvgDividendYield',
            info.get('5yearAverageDividendYield',
            info.get('averageDividendYield', None))))
        dividend_metrics['5-Year Average Dividend Yield'] = f"{five_year_avg_yield:.2f}%" if five_year_avg_yield else 'N/A'
        
        # Earnings Growth Analysis
        earnings_growth = {}
        try:
            # Get quarterly financials
            financials = stock.quarterly_financials
            if not financials.empty:
                # Calculate year-over-year growth rates
                if 'Net Income' in financials.index:
                    earnings = financials.loc['Net Income']
                    if len(earnings) >= 4:
                        yoy_growth = ((earnings.iloc[0] / earnings.iloc[4]) - 1) * 100
                    else:
                        earnings_growth['Latest Earnings Growth'] = 'N/A (insufficient data)'
                
                # Revenue Growth
                if 'Total Revenue' in financials.index:
                    revenue = financials.loc['Total Revenue']
                    if len(revenue) >= 4:
                        rev_growth = ((revenue.iloc[0] / revenue.iloc[4]) - 1) * 100
                    else:
                        earnings_growth['Latest Revenue Growth'] = 'N/A (insufficient data)'
                
                # Calculate 5-year CAGR if enough data
                if len(hist_data) >= 1250:  # Approximately 5 years of trading days
                    # Price CAGR
                    start_price = hist_data['Close'].iloc[0]
                    end_price = hist_data['Close'].iloc[-1]
                    price_cagr = ((end_price / start_price) ** (1/5) - 1) * 100
                    earnings_growth['5-Year Price CAGR'] = f"{price_cagr:.2f}%"
                    
                    if len(financials.index) >= 20:  # 5 years of quarterly data
                        if 'Net Income' in financials.index:
                            start_earnings = financials.loc['Net Income'].iloc[-1]
                            end_earnings = financials.loc['Net Income'].iloc[0]
                            if start_earnings > 0 and end_earnings > 0:
                                earnings_cagr = ((end_earnings / start_earnings) ** (1/5) - 1) * 100
                                earnings_growth['5-Year Earnings CAGR'] = f"{earnings_cagr:.2f}%"
        except Exception as e:
            print(f"Error calculating growth metrics: {str(e)}")
            earnings_growth['Latest Earnings Growth'] = 'N/A'
            earnings_growth['Latest Revenue Growth'] = 'N/A'
            earnings_growth['5-Year Price CAGR'] = 'N/A'
            earnings_growth['5-Year Earnings CAGR'] = 'N/A'
        
        # Add additional growth metrics
        try:
            # Revenue Growth from info
            rev_growth = info.get('revenueGrowth', None)
            if rev_growth is not None:
                earnings_growth['Revenue Growth (TTM)'] = f"{rev_growth*100:.2f}%"
            
            # Earnings Growth from info
            earn_growth = info.get('earningsGrowth', None)
            if earn_growth is not None:
                earnings_growth['Earnings Growth (TTM)'] = f"{earn_growth*100:.2f}%"
            
            # EPS Growth
            eps_growth = info.get('earningsQuarterlyGrowth', None)
            if eps_growth is not None:
                earnings_growth['EPS Growth (QoQ)'] = f"{eps_growth*100:.2f}%"
        except Exception as e:
            print(f"Error calculating additional growth metrics: {str(e)}")
        
        analysis = {
            'Dividend Metrics': dividend_metrics,
            'Growth Metrics': earnings_growth
        }
        
        return str(analysis)
    except Exception as e:
        print(f"Error in analyze_dividend_and_growth: {str(e)}")
        return "Error analyzing dividend and growth metrics"

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
            8. Economic impact assessment
            
            You provide detailed financial analysis with specific numbers and metrics.
            When analyzing a stock, you always include key financial ratios such as Debt-to-Equity,
            Current Ratio, ROE, ROA, and Operating Margin in your analysis, with clear explanations
            of what these metrics indicate about the company's financial health.""",
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
            8. Trading signals and entry/exit points
            
            You provide detailed technical analysis with specific numbers and metrics.
            When analyzing a stock, you always include key technical indicators such as
            moving averages (SMA 20, SMA 50), RSI, Bollinger Bands, MACD, and Stochastic
            Oscillator in your analysis, with clear explanations of what these indicators
            suggest about potential price movements.""",
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
            8. Position sizing recommendations
            
            You provide detailed risk analysis with specific numbers and metrics.
            When analyzing a stock, you always include Value at Risk (VaR), stress testing scenarios,
            and risk-adjusted returns in your analysis.""",
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

                    IMPORTANT: You MUST use all available tools to provide comprehensive analysis.
                    
                    CRITICAL: Always include the following in your analysis:
                    - Debt-to-Equity Ratio: Explain what this indicates about financial leverage
                    - Current Ratio: Explain what this indicates about short-term liquidity
                    - ROE (Return on Equity): Explain what this indicates about profitability
                    - ROA (Return on Assets): Explain what this indicates about asset efficiency
                    - Operating Margin: Explain what this indicates about operational efficiency
                    
                    For each section, provide specific numbers and data points from the tools.
                    If any data is unavailable, explicitly state that it could not be retrieved and explain
                    what this might indicate about the company.""",
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
                    
                    CRITICAL: Always include the following in your analysis:
                    - Current price in relation to SMA 20 and SMA 50 (above/below)
                    - RSI value and whether it indicates overbought (>70) or oversold (<30) conditions
                    - Bollinger Bands position (upper, middle, lower) and what it suggests
                    - Key support and resistance levels with specific price points
                    - Clear trading signals (buy, sell, hold) based on technical indicators
                    
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
                    - Entry/exit recommendations with price targets"""
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
                    
                    CRITICAL: Always include the following in your analysis:
                    - Value at Risk (VaR) at 95% and 99% confidence levels
                    - Conditional VaR (Expected Shortfall)
                    - Stress test scenarios (market crash, interest rate hike, sector shock)
                    - Risk-adjusted returns (Sharpe, Sortino, Treynor, Information, and Calmar ratios)
                    
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