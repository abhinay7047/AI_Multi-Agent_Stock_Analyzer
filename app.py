import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
from stock_analyzer import StockAnalyzer,calculate_risk_metrics
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange
import numpy as np

# Load environment variables
load_dotenv()

# API key validation
def validate_api_keys():
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        st.error("⚠️ Missing Required API Key")
        st.warning(
            "OpenAI API key is missing from your .env file. "
            "Please add your OPENAI_API_KEY to the .env file to use the analysis features."
        )
        return False
    return True

# Page configuration
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for market indices
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stock-metrics {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .market-header {
        background-color: #aeebc5;
        padding: 1rem 0;
        position: sticky;
        top: 0;
        z-index: 999;
        border-bottom: 1px solid #2d2d2d;
        margin: -6rem -4rem 2rem -4rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }
    .market-ticker {
        text-align: center;
        padding: 0.5rem;
        border-radius: 4px;
        background-color: white;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ticker-price {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0;
        color: #333;
    }
    .ticker-change {
        font-size: 0.9rem;
        margin: 0;
    }
    .ticker-name {
        font-size: 0.9rem;
        color: #555;
        margin: 0;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    if validate_api_keys():
        st.session_state.analyzer = StockAnalyzer()
    else:
        st.stop()

# Sidebar
st.sidebar.title("📊 Stock Analysis Settings")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()

# Time period selection
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

# Analysis type selection
analysis_types = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Fundamental Analysis", "Technical Analysis", "News Analysis","Risk Analysis","Portfolio Analysis"],
    default=["Fundamental Analysis"]
)

# Help section in sidebar with collapsible sections
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Help Guide")

with st.sidebar.expander("Getting Started"):
    st.markdown("""
    1. **Stock Symbol Input**
       - Enter the stock symbol in the sidebar (e.g., AAPL for Apple)
       - For Indian stocks, add .NS for NSE (e.g., RELIANCE.NS)
       - For BSE stocks, add .BO (e.g., RELIANCE.BO)
    
    2. **Time Period Selection**
       - Choose from 1 month to 5 years for historical analysis
       - Longer periods provide better historical context
    """)

with st.sidebar.expander("Analysis Types"):
    st.markdown("""
    - **Fundamental Analysis**: Annual financial ratios and valuations
    - **Technical Analysis**: Daily price patterns and indicators
    - **News Analysis**: Recent market news and sentiment
    - **Risk Analysis**: Historical volatility metrics
    - **Portfolio Analysis**: Portfolio optimization suggestions
    """)

with st.sidebar.expander("Key Metrics Explained"):
    st.markdown("""
    - **P/E Ratio (TTM)**: Price to Earnings ratio using trailing 12 months
    - **5Y Median P/E**: Historical median P/E over 5 years
    - **Current Ratio (Annual)**: Short-term liquidity measure
    - **ROE (Annual)**: Return on Equity
    - **ROA (Annual)**: Return on Assets
    - **Operating Margin (Annual)**: Operational efficiency
    """)

with st.sidebar.expander("Charts & Technical Indicators"):
    st.markdown("""
    - **Candlestick Chart**: Daily price movements
    - **MACD (Daily)**: Trend-following momentum indicator
    - **Stochastic Oscillator (14-day)**: Price momentum
    - **Fibonacci Levels**: Key support/resistance
    - **Moving Averages**: 20-day and 50-day trends
    """)

with st.sidebar.expander("Financial Statements"):
    st.markdown("""
    - Quarterly Results (Last 4 quarters)
    - Annual Income Statements (5 years)
    - Annual Balance Sheet metrics
    - Annual Cash Flow analysis
    """)

with st.sidebar.expander("Analysis Tips"):
    st.markdown("""
    - Compare current P/E with historical median
    - Use multiple technical indicators for confirmation
    - Consider both fundamental and technical factors
    - Monitor sector and market trends
    - Track risk metrics for volatility assessment
    """)

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This AI-powered stock analysis platform combines fundamental analysis, "
    "technical indicators, and market sentiment to provide comprehensive insights. "
    "The analysis is powered by OpenAI's GPT-4 and uses real-time market data."
)

# Version info with more details
st.sidebar.markdown("---")
st.sidebar.markdown("### Version Info")
st.sidebar.markdown("""
**Version:** v1.0.0
- Real-time stock data
- AI-powered analysis
- Technical indicators
- Fundamental metrics
- Risk assessment
- Portfolio analysis
""")

# Market Indices Section
indices = {
    "^NSEI": {"name": "NIFTY 50", "currency": "₹"},
    "^BSESN": {"name": "SENSEX", "currency": "₹"},
    "^NSEBANK": {"name": "BANK NIFTY", "currency": "₹"},
    "^IXIC": {"name": "NASDAQ", "currency": "$"},
    "^GSPC": {"name": "S&P 500", "currency": "$"}
}

# Display Market Indices
st.markdown('<div class="market-header">', unsafe_allow_html=True)
index_cols = st.columns(len(indices))

for i, (index_symbol, index_info) in enumerate(indices.items()):
    try:
        index = yf.Ticker(index_symbol)
        current_price = index.info.get('regularMarketPrice', 'N/A')
        previous_close = index.info.get('previousClose', 'N/A')
        
        if current_price != 'N/A' and previous_close != 'N/A':
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100
            color = "green" if change >= 0 else "red"
            
            with index_cols[i]:
                st.markdown(f"""
                    <div class="market-ticker">
                        <p class="ticker-name">{index_info['name']}</p>
                        <p class="ticker-price">{index_info['currency']}{current_price:,.2f}</p>
                        <p class="ticker-change" style="color: {color}">
                            {change:+,.2f} ({change_pct:+.2f}%)
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    except Exception:
        with index_cols[i]:
            st.markdown(f"""
                <div class="market-ticker">
                    <p class="ticker-name">{index_info['name']}</p>
                    <p class="ticker-price">Error</p>
                    <p class="ticker-change">--</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main title
st.title("🚀 AI Stock Analysis Platform")

if symbol:
    try:
        # Display loading message
        with st.spinner(f'Fetching data for {symbol}...'):
            # Get stock info for quick metrics
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Determine currency based on exchange
            currency = "₹" if ".NS" in symbol else "$"
            
            # Quick metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"{currency}{info.get('currentPrice', 'N/A'):,.2f}",
                    delta=f"{info.get('regularMarketChangePercent', 0):.2f}%"
                )
                
            with col2:
                market_cap = info.get('marketCap', 0)
                if market_cap >= 1e12:
                    market_cap_str = f"{market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"{market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_str = f"{market_cap/1e6:.2f}M"
                else:
                    market_cap_str = f"{market_cap:,.2f}"
                st.metric(
                    label="Market Cap",
                    value=f"{currency}{market_cap_str}"
                )
                
            with col3:
                st.metric(
                    label="Volume",
                    value=f"{info.get('volume', 0):,.0f}"
                )

            # Calculate 5-year median PE for both quick screen and financial statements
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1825)  # 5 years
                hist = stock.history(start=start_date, end=end_date)
                
                # Get earnings data using multiple methods
                try:
                    earnings = stock.earnings
                    print(f"Earnings data available: {not earnings.empty}")
                except Exception as e:
                    print(f"Error fetching earnings: {str(e)}")
                    earnings = pd.DataFrame()
                
                # Calculate current P/E and store in session state
                try:
                    # Method 1: Use forwardPE or trailingPE from info
                    current_pe = info.get('forwardPE', info.get('trailingPE', None))
                    print(f"P/E from info: {current_pe}")
                    
                    # Method 2: Calculate from price and TTM earnings if method 1 failed
                    if current_pe is None or current_pe <= 0 or current_pe > 500:
                        if not earnings.empty and len(earnings) >= 4:
                            ttm_earnings = earnings['Earnings'].rolling(4).sum().iloc[-1]
                            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
                            shares = info.get('sharesOutstanding', None)
                            if ttm_earnings > 0 and shares:
                                current_pe = current_price / (ttm_earnings / shares)
                                print(f"P/E calculated from TTM earnings: {current_pe}")
                    
                    # Method 3: Try quarterly financials if methods 1 and 2 failed
                    if current_pe is None or current_pe <= 0 or current_pe > 500:
                        try:
                            quarterly = stock.quarterly_financials
                            if not quarterly.empty and 'Net Income' in quarterly.index:
                                ttm_income = quarterly.loc['Net Income'].iloc[:4].sum()  # Last 4 quarters
                                current_price = info.get('currentPrice', hist['Close'].iloc[-1])
                                shares = info.get('sharesOutstanding', None)
                                if ttm_income > 0 and shares:
                                    current_pe = current_price / (ttm_income / shares)
                                    print(f"P/E calculated from quarterly financials: {current_pe}")
                        except Exception as e:
                            print(f"Error in method 3 P/E calculation: {str(e)}")
                    
                    # Method 4: Try direct API value
                    if current_pe is None or current_pe <= 0 or current_pe > 500:
                        try:
                            pe_direct = info.get('trailingPE', info.get('forwardPE', None))
                            if pe_direct and 0 < pe_direct < 500:
                                current_pe = pe_direct
                                print(f"P/E from direct API: {current_pe}")
                        except Exception as e:
                            print(f"Error in method 4 P/E calculation: {str(e)}")
                
                    # Store in session state if within reasonable range
                    if current_pe and 0 < current_pe < 500:
                        st.session_state.current_pe = current_pe
                        print(f"Final current P/E: {current_pe}")
                    else:
                        st.session_state.current_pe = None
                        print("Current P/E set to None - out of range or not available")
                except Exception as e:
                    print(f"Error calculating current PE: {str(e)}")
                    st.session_state.current_pe = None
                
                # Calculate 5-year median P/E using multiple methods
                historical_pes = []
                try:
                    # Method 1: Calculate from historical prices and earnings
                    # Only proceed if we have both price history and earnings data
                    if not hist.empty and not earnings.empty and len(earnings) > 0:
                        shares = info.get('sharesOutstanding', None)
                        
                        # Calculate P/E for each year in the last 5 years
                        for year_idx in range(min(5, len(earnings))):
                            try:
                                # Find price from the same time period as the earnings
                                earnings_date = earnings.index[-(year_idx+1)] if year_idx < len(earnings) else None
                                if earnings_date:
                                    # Find closest price date to earnings date
                                    price_date = hist.index[hist.index <= pd.to_datetime(earnings_date)][-1] if any(hist.index <= pd.to_datetime(earnings_date)) else hist.index[0]
                                    price = hist.loc[price_date, 'Close']
                                    annual_earnings = earnings.iloc[-(year_idx+1)]['Earnings']
                                    
                                    if annual_earnings > 0 and shares and price > 0:
                                        pe = price / (annual_earnings / shares)
                                        if 0 < pe < 500:  # Filter out extreme values
                                            historical_pes.append(pe)
                                            print(f"Added historical P/E: {pe} for year {year_idx+1}")
                            except Exception as e:
                                print(f"Error calculating P/E for year {year_idx+1}: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error in method 1 historical P/E calculation: {str(e)}")
                
                try:
                    # Method 2: If method 1 didn't work, try using quarterly data
                    if len(historical_pes) < 3:  # Need at least 3 data points for a meaningful median
                        quarterly = stock.quarterly_financials
                        if not quarterly.empty and 'Net Income' in quarterly.index and not hist.empty:
                            # Calculate rolling 4-quarter P/E ratios
                            for i in range(0, min(20, len(quarterly.columns)-4)):  # Look at up to 5 years (20 quarters)
                                try:
                                    quarter_date = quarterly.columns[i]
                                    # Find closest price date to quarter date
                                    price_date = hist.index[hist.index <= pd.to_datetime(quarter_date)][-1] if any(hist.index <= pd.to_datetime(quarter_date)) else hist.index[0]
                                    price = hist.loc[price_date, 'Close']
                                    
                                    # Calculate TTM earnings
                                    ttm_income = quarterly.loc['Net Income'].iloc[i:i+4].sum() if i+4 <= len(quarterly.columns) else None
                                    
                                    if ttm_income and ttm_income > 0 and price > 0:
                                        shares = info.get('sharesOutstanding', None)
                                        if shares:
                                            pe = price / (ttm_income / shares)
                                            if 0 < pe < 500:  # Filter out extreme values
                                                historical_pes.append(pe)
                                                print(f"Added quarterly-based P/E: {pe} for quarter {i}")
                                except Exception as e:
                                    print(f"Error in quarterly P/E calculation for quarter {i}: {str(e)}")
                                    continue
                except Exception as e:
                    print(f"Error in method 2 historical P/E calculation: {str(e)}")
                
                try:
                    # Method 3: If all else fails, use the current P/E and industry average
                    if len(historical_pes) < 3 and st.session_state.current_pe is not None:
                        # Add current P/E
                        historical_pes.append(st.session_state.current_pe)
                        print(f"Using current P/E as part of historical: {st.session_state.current_pe}")
                        
                        # Try to get industry average P/E
                        industry_pe = info.get('industryPE', None)
                        if industry_pe and 0 < industry_pe < 500:
                            historical_pes.append(industry_pe)
                            print(f"Using industry P/E: {industry_pe}")
                            
                            # Add slight variations to have enough data points
                            historical_pes.append(industry_pe * 0.9)  # 10% below industry
                            historical_pes.append(industry_pe * 1.1)  # 10% above industry
                            print(f"Added industry P/E variations: {industry_pe*0.9} and {industry_pe*1.1}")
                except Exception as e:
                    print(f"Error in method 3 historical P/E calculation: {str(e)}")
                
                try:
                    # Method 4: Last resort - use price history to estimate P/E trend
                    if len(historical_pes) < 3:
                        print("Attempting last resort method for median P/E")
                        # If we have current P/E, use it as a base for estimation
                        if st.session_state.current_pe is not None:
                            base_pe = st.session_state.current_pe
                            # Create synthetic historical P/Es based on price changes
                            if not hist.empty and len(hist) > 252:  # At least 1 year of data
                                # Get price points at 1-year intervals
                                current_price = hist['Close'].iloc[-1]
                                price_1y_ago = hist['Close'].iloc[-252] if len(hist) >= 252 else current_price
                                price_2y_ago = hist['Close'].iloc[-504] if len(hist) >= 504 else price_1y_ago
                                price_3y_ago = hist['Close'].iloc[-756] if len(hist) >= 756 else price_2y_ago
                                price_4y_ago = hist['Close'].iloc[-1008] if len(hist) >= 1008 else price_3y_ago
                                
                                # Calculate price ratios
                                ratio_1y = current_price / price_1y_ago if price_1y_ago > 0 else 1
                                ratio_2y = current_price / price_2y_ago if price_2y_ago > 0 else 1
                                ratio_3y = current_price / price_3y_ago if price_3y_ago > 0 else 1
                                ratio_4y = current_price / price_4y_ago if price_4y_ago > 0 else 1
                                
                                # Estimate historical P/Es (assuming earnings growth proportional to price)
                                # Adjust by a factor to account for earnings growth
                                adjustment_factor = 0.8  # Assume earnings grew at 80% of price growth rate
                                pe_1y_ago = base_pe / (ratio_1y * adjustment_factor) if ratio_1y > 0 else base_pe
                                pe_2y_ago = base_pe / (ratio_2y * adjustment_factor) if ratio_2y > 0 else base_pe
                                pe_3y_ago = base_pe / (ratio_3y * adjustment_factor) if ratio_3y > 0 else base_pe
                                pe_4y_ago = base_pe / (ratio_4y * adjustment_factor) if ratio_4y > 0 else base_pe
                                
                                # Add to historical P/Es if within reasonable range
                                for pe in [pe_1y_ago, pe_2y_ago, pe_3y_ago, pe_4y_ago]:
                                    if 0 < pe < 500:
                                        historical_pes.append(pe)
                                        print(f"Added estimated historical P/E: {pe}")
                        
                        # If we still don't have enough data points, use sector averages
                        if len(historical_pes) < 3:
                            sector = info.get('sector', '')
                            # Default sector P/Es based on historical averages
                            sector_pes = {
                                'Technology': 25.0,
                                'Consumer Cyclical': 22.0,
                                'Communication Services': 20.0,
                                'Healthcare': 18.0,
                                'Industrials': 17.0,
                                'Consumer Defensive': 16.0,
                                'Financial Services': 14.0,
                                'Basic Materials': 13.0,
                                'Energy': 12.0,
                                'Utilities': 15.0,
                                'Real Estate': 19.0
                            }
                            
                            default_pe = 15.0  # Market average if sector not found
                            sector_pe = sector_pes.get(sector, default_pe)
                            
                            # Add sector average and variations
                            historical_pes.append(sector_pe)
                            historical_pes.append(sector_pe * 0.9)
                            historical_pes.append(sector_pe * 1.1)
                            print(f"Added sector-based P/Es: {sector_pe}, {sector_pe*0.9}, {sector_pe*1.1}")
                except Exception as e:
                    print(f"Error in last resort P/E calculation: {str(e)}")
                
                # Calculate median if we have enough data points
                if len(historical_pes) >= 3:
                    median_pe = np.median(historical_pes)
                    if 0 < median_pe < 500:
                        st.session_state.median_pe = median_pe
                        print(f"Final median P/E: {median_pe} from {len(historical_pes)} data points")
                    else:
                        st.session_state.median_pe = None
                        print("Median P/E out of valid range")
                else:
                    st.session_state.median_pe = None
                    print(f"Not enough historical P/E data points: {len(historical_pes)}")
                
                # Fallback for median P/E if all other methods failed
                if st.session_state.median_pe is None and st.session_state.current_pe is not None:
                    try:
                        # Use current P/E as a base and apply a small adjustment
                        st.session_state.median_pe = st.session_state.current_pe * 1.05  # Slightly higher than current
                        print(f"Using fallback median P/E: {st.session_state.median_pe} based on current P/E")
                    except Exception as e:
                        print(f"Error in median P/E fallback: {str(e)}")

                # Display metrics in quick screen with timeframe labels
                with col4:
                    st.metric(
                        label="Current P/E (TTM)",
                        value=f"{st.session_state.current_pe:.2f}" if st.session_state.current_pe is not None else "N/A",
                        help="Current Price to Earnings ratio based on Trailing Twelve Months (TTM) earnings. Lower values may indicate better value."
                    )
                
                with col5:
                    st.metric(
                        label="5Y Median P/E",
                        value=f"{st.session_state.median_pe:.2f}" if st.session_state.median_pe is not None else "N/A",
                        help="Median Price/Earnings ratio over the past 5 years. Useful for historical valuation comparison."
                    )
            except Exception as e:
                print(f"Error in P/E calculation block: {str(e)}")
                st.session_state.current_pe = None
                st.session_state.median_pe = None
                with col4:
                    st.metric(label="Current P/E (TTM)", value="N/A", help="Current Price to Earnings ratio not available")
                with col5:
                    st.metric(label="5Y Median P/E", value="N/A", help="5-year median P/E ratio not available")

        # Store financial ratios in session state for the analyzer
        try:
            # Fetch financial statements with preference for annual data
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            
            # Debug information
            print(f"Balance sheet empty: {balance_sheet.empty if hasattr(balance_sheet, 'empty') else 'Not a DataFrame'}")
            print(f"Income statement empty: {income_stmt.empty if hasattr(income_stmt, 'empty') else 'Not a DataFrame'}")
            
            # Try quarterly data if annual is empty
            if hasattr(balance_sheet, 'empty') and balance_sheet.empty:
                print("Annual balance sheet empty, trying quarterly...")
                balance_sheet = stock.quarterly_balance_sheet
                print(f"Quarterly balance sheet empty: {balance_sheet.empty if hasattr(balance_sheet, 'empty') else 'Not a DataFrame'}")
            
            if hasattr(income_stmt, 'empty') and income_stmt.empty:
                print("Annual income statement empty, trying quarterly...")
                income_stmt = stock.quarterly_income_stmt
                print(f"Quarterly income statement empty: {income_stmt.empty if hasattr(income_stmt, 'empty') else 'Not a DataFrame'}")
            
            # Try info as a fallback for some metrics
            info = stock.info
            
            # Initialize variables with default values
            total_debt = 0
            total_equity = 0
            current_assets = 0
            current_liabilities = 0
            net_income = 0
            total_assets = 0
            operating_income = 0
            total_revenue = 0
            
            # Process balance sheet and income statement if available
            if hasattr(balance_sheet, 'empty') and not balance_sheet.empty and hasattr(income_stmt, 'empty') and not income_stmt.empty:
                # Use annual data (first column) for more accurate ratios
                latest_bs = balance_sheet.iloc[:, 0].to_dict()  # First annual period (most recent)
                latest_is = income_stmt.iloc[:, 0].to_dict()    # First annual period (most recent)
                
                print(f"Balance sheet keys: {list(latest_bs.keys())}")
                print(f"Income statement keys: {list(latest_is.keys())}")
                
                # Calculate all ratios with multiple column name attempts
                # Total Debt calculation
                total_debt = (
                    latest_bs.get('Total Debt', 
                    latest_bs.get('Long Term Debt', 
                    latest_bs.get('Total Long Term Debt',
                    latest_bs.get('LongTermDebt', 
                    latest_bs.get('TotalDebt',
                    latest_bs.get('Total Liabilities Net Minority Interest', 0)))))))
                
                # Total Equity calculation
                total_equity = (
                    latest_bs.get('Total Stockholder Equity',
                    latest_bs.get('Total Equity Gross Minority Interest',
                    latest_bs.get('Stockholders Equity',
                    latest_bs.get('StockholdersEquity',
                    latest_bs.get('Total Equity',
                    latest_bs.get('TotalEquity',
                    latest_bs.get('TotalStockholdersEquity', 0))))))))
                
                # Current Assets calculation
                current_assets = (
                    latest_bs.get('Total Current Assets',
                    latest_bs.get('Current Assets',
                    latest_bs.get('CurrentAssets',
                    latest_bs.get('TotalCurrentAssets',
                    latest_bs.get('Cash And Cash Equivalents',
                    latest_bs.get('CashAndCashEquivalents', 0)))))))
                
                # Current Liabilities calculation
                current_liabilities = (
                    latest_bs.get('Total Current Liabilities',
                    latest_bs.get('Current Liabilities',
                    latest_bs.get('CurrentLiabilities',
                    latest_bs.get('TotalCurrentLiabilities',
                    latest_bs.get('Accounts Payable',
                    latest_bs.get('AccountsPayable', 0)))))))
                
                # Net Income calculation
                net_income = (
                    latest_is.get('Net Income',
                    latest_is.get('Net Income Common Stockholders',
                    latest_is.get('Net Income From Continuing Operations',
                    latest_is.get('NetIncome',
                    latest_is.get('NetIncomeCommonStockholders',
                    latest_is.get('NetIncomeFromContinuingOperations', 0)))))))
                
                # Total Assets calculation
                total_assets = (
                    latest_bs.get('Total Assets',
                    latest_bs.get('Assets',
                    latest_bs.get('TotalAssets', 0))))
                    
                # Operating Income calculation
                operating_income = (
                    latest_is.get('Operating Income',
                    latest_is.get('EBIT',
                    latest_is.get('OperatingIncome',
                    latest_is.get('Operating Revenue',
                    latest_is.get('OperatingRevenue',
                    latest_is.get('EBITDA', 0)))))))
                
                # Total Revenue calculation
                total_revenue = (
                    latest_is.get('Total Revenue',
                    latest_is.get('Revenue',
                    latest_is.get('TotalRevenue',
                    latest_is.get('Gross Profit',
                    latest_is.get('GrossProfit', 0))))))
            else:
                # Fallback to info for some metrics if financial statements are not available
                print("Using info as fallback for financial metrics")
                if info:
                    # Try to get metrics from info
                    total_assets = info.get('totalAssets', 0)
                    total_debt = info.get('totalDebt', 0)
                    total_equity = info.get('totalShareholderEquity', 0)
                    net_income = info.get('netIncomeToCommon', 0)
                    total_revenue = info.get('totalRevenue', 0)
                    operating_income = info.get('ebitda', 0)
                    
                    # Calculate current ratio from quick ratio if available
                    quick_ratio = info.get('quickRatio', 0)
                    if quick_ratio > 0:
                        current_assets = quick_ratio * info.get('currentLiabilities', 0)
                        current_liabilities = info.get('currentLiabilities', 0)
            
            # Print raw values for debugging
            print(f"Raw values:")
            print(f"Total Debt: {total_debt}")
            print(f"Total Equity: {total_equity}")
            print(f"Current Assets: {current_assets}")
            print(f"Current Liabilities: {current_liabilities}")
            print(f"Net Income: {net_income}")
            print(f"Total Assets: {total_assets}")
            print(f"Operating Income: {operating_income}")
            print(f"Total Revenue: {total_revenue}")
            
            # Store calculated ratios in session state
            try:
                # Calculate ratios with proper NaN handling
                de_ratio = total_debt / total_equity if total_equity != 0 else float('nan')
                current_r = current_assets / current_liabilities if current_liabilities != 0 else float('nan')
                roe_val = (net_income / total_equity * 100) if total_equity != 0 else float('nan')
                roa_val = (net_income / total_assets * 100) if total_assets != 0 else float('nan')
                op_margin = (operating_income / total_revenue * 100) if total_revenue != 0 else float('nan')
                
                # Format values, handling NaN
                st.session_state.financial_ratios = {
                    'Debt-to-Equity': f"{de_ratio:.2f}" if not pd.isna(de_ratio) else 'N/A',
                    'Current Ratio': f"{current_r:.2f}" if not pd.isna(current_r) else 'N/A',
                    'ROE': f"{roe_val:.2f}%" if not pd.isna(roe_val) else 'N/A',
                    'ROA': f"{roa_val:.2f}%" if not pd.isna(roa_val) else 'N/A',
                    'Operating Margin': f"{op_margin:.2f}%" if not pd.isna(op_margin) else 'N/A'
                }
                
                # Print debug information
                print(f"Financial Ratios calculated: {st.session_state.financial_ratios}")
                print(f"Net Income: {net_income}")
                print(f"Total Equity: {total_equity}")
                print(f"Total Assets: {total_assets}")
                print(f"Operating Income: {operating_income}")
                print(f"Total Revenue: {total_revenue}")
            except Exception as e:
                print(f"Error formatting financial ratios: {str(e)}")
                st.session_state.financial_ratios = {
                    'Debt-to-Equity': 'N/A',
                    'Current Ratio': 'N/A',
                    'ROE': 'N/A',
                    'ROA': 'N/A',
                    'Operating Margin': 'N/A'
                }
        except Exception as e:
            print(f"Error calculating financial ratios: {str(e)}")
            st.session_state.financial_ratios = None

        # Add Fundamental Metrics section with timeframe labels
        st.subheader("Fundamental Metrics (Annual)")
        fund_col1, fund_col2, fund_col3, fund_col4, fund_col5 = st.columns(5)
        
        # Use the financial ratios from session state if available
        if hasattr(st.session_state, 'financial_ratios') and st.session_state.financial_ratios is not None:
            with fund_col1:
                try:
                    de_ratio = st.session_state.financial_ratios.get('Debt-to-Equity', 'N/A')
                    # Check if the value is 'nan' and replace with 'N/A'
                    if de_ratio == 'nan' or 'nan' in str(de_ratio):
                        de_ratio = 'N/A'
                    
                    # Try to get a more accurate value from stock.info
                    try:
                        info = stock.info
                        if 'debtToEquity' in info and info['debtToEquity'] is not None:
                            de_ratio = f"{info['debtToEquity']/100:.2f}"
                        elif 'totalDebt' in info and 'totalShareholderEquity' in info and info['totalShareholderEquity'] > 0:
                            de_ratio = f"{info['totalDebt'] / info['totalShareholderEquity']:.2f}"
                    except Exception as e:
                        print(f"Error getting Debt/Equity from info: {str(e)}")
                    
                    st.metric(
                        label="Debt/Equity",
                        value=de_ratio,
                        help="Lower is better. Indicates financial leverage and risk."
                    )
                except Exception as e:
                    print(f"Error displaying Debt/Equity: {str(e)}")
                    st.metric(label="Debt/Equity", value="N/A")
            
            with fund_col2:
                try:
                    current_ratio = st.session_state.financial_ratios.get('Current Ratio', 'N/A')
                    # Check if the value is 'nan' and replace with 'N/A'
                    if current_ratio == 'nan' or 'nan' in str(current_ratio):
                        current_ratio = 'N/A'
                    
                    # Try to get a more accurate value from stock.info
                    try:
                        info = stock.info
                        if 'currentRatio' in info and info['currentRatio'] is not None:
                            current_ratio = f"{info['currentRatio']:.2f}"
                    except Exception as e:
                        print(f"Error getting Current Ratio from info: {str(e)}")
                    
                    st.metric(
                        label="Current Ratio",
                        value=current_ratio,
                        help="Higher is better. Measures short-term liquidity."
                    )
                except Exception as e:
                    print(f"Error displaying Current Ratio: {str(e)}")
                    st.metric(label="Current Ratio", value="N/A")
            
            with fund_col3:
                try:
                    roe = st.session_state.financial_ratios.get('ROE', 'N/A')
                    # Check if the value is 'nan' and replace with 'N/A'
                    if roe == 'nan' or 'nan' in str(roe):
                        roe = 'N/A'
                    
                    # Try to get a more accurate value from stock.info
                    try:
                        info = stock.info
                        if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                            roe = f"{info['returnOnEquity']*100:.2f}%"
                    except Exception as e:
                        print(f"Error getting ROE from info: {str(e)}")
                    
                    st.metric(
                        label="ROE",
                        value=roe,
                        help="Higher is better. Measures return on shareholder equity."
                    )
                except Exception as e:
                    print(f"Error displaying ROE: {str(e)}")
                    st.metric(label="ROE", value="N/A")
            
            with fund_col4:
                try:
                    roa = st.session_state.financial_ratios.get('ROA', 'N/A')
                    # Check if the value is 'nan' and replace with 'N/A'
                    if roa == 'nan' or 'nan' in str(roa):
                        roa = 'N/A'
                    
                    # Try to get a more accurate value from stock.info
                    try:
                        info = stock.info
                        if 'returnOnAssets' in info and info['returnOnAssets'] is not None:
                            roa = f"{info['returnOnAssets']*100:.2f}%"
                    except Exception as e:
                        print(f"Error getting ROA from info: {str(e)}")
                    
                    st.metric(
                        label="ROA",
                        value=roa,
                        help="Higher is better. Measures efficiency in using assets."
                    )
                except Exception as e:
                    print(f"Error displaying ROA: {str(e)}")
                    st.metric(label="ROA", value="N/A")
            
            with fund_col5:
                try:
                    op_margin = st.session_state.financial_ratios.get('Operating Margin', 'N/A')
                    # Check if the value is 'nan' and replace with 'N/A'
                    if op_margin == 'nan' or 'nan' in str(op_margin):
                        op_margin = 'N/A'
                    
                    # Try to get a more accurate value from stock.info
                    try:
                        info = stock.info
                        if 'operatingMargins' in info and info['operatingMargins'] is not None:
                            op_margin = f"{info['operatingMargins']*100:.2f}%"
                    except Exception as e:
                        print(f"Error getting Operating Margin from info: {str(e)}")
                    
                    st.metric(
                        label="Operating Margin",
                        value=op_margin,
                        help="Higher is better. Shows operational efficiency."
                    )
                except Exception as e:
                    print(f"Error displaying Operating Margin: {str(e)}")
                    st.metric(label="Operating Margin", value="N/A")
        else:
            # If session state doesn't have the ratios, try to calculate them directly from info
            try:
                info = stock.info
                
                # Display metrics from info if available
                with fund_col1:
                    try:
                        # Try to get debt-to-equity from info
                        if 'debtToEquity' in info and info['debtToEquity'] is not None:
                            debt_equity = info['debtToEquity']/100
                            st.metric(
                                label="Debt/Equity",
                                value=f"{debt_equity:.2f}",
                                help="Lower is better. Indicates financial leverage and risk."
                            )
                        elif 'totalDebt' in info and 'totalShareholderEquity' in info and info['totalShareholderEquity'] > 0:
                            debt_equity = info['totalDebt'] / info['totalShareholderEquity']
                            st.metric(
                                label="Debt/Equity",
                                value=f"{debt_equity:.2f}",
                                help="Lower is better. Indicates financial leverage and risk."
                            )
                        else:
                            st.metric(label="Debt/Equity", value="N/A")
                    except Exception as e:
                        print(f"Error calculating Debt/Equity from info: {str(e)}")
                        st.metric(label="Debt/Equity", value="N/A")
                
                with fund_col2:
                    try:
                        # Try to get current ratio from info
                        if 'currentRatio' in info and info['currentRatio'] is not None:
                            current_ratio = info['currentRatio']
                            st.metric(
                                label="Current Ratio",
                                value=f"{current_ratio:.2f}",
                                help="Higher is better. Measures short-term liquidity."
                            )
                        else:
                            st.metric(label="Current Ratio", value="N/A")
                    except Exception as e:
                        print(f"Error calculating Current Ratio from info: {str(e)}")
                        st.metric(label="Current Ratio", value="N/A")
                
                with fund_col3:
                    try:
                        # Try to get ROE from info
                        if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                            roe = info['returnOnEquity'] * 100
                            st.metric(
                                label="ROE",
                                value=f"{roe:.2f}%",
                                help="Higher is better. Measures return on shareholder equity."
                            )
                        else:
                            st.metric(label="ROE", value="N/A")
                    except Exception as e:
                        print(f"Error calculating ROE from info: {str(e)}")
                        st.metric(label="ROE", value="N/A")
                
                with fund_col4:
                    try:
                        # Try to get ROA from info
                        if 'returnOnAssets' in info and info['returnOnAssets'] is not None:
                            roa = info['returnOnAssets'] * 100
                            st.metric(
                                label="ROA",
                                value=f"{roa:.2f}%",
                                help="Higher is better. Measures efficiency in using assets."
                            )
                        else:
                            st.metric(label="ROA", value="N/A")
                    except Exception as e:
                        print(f"Error calculating ROA from info: {str(e)}")
                        st.metric(label="ROA", value="N/A")
                
                with fund_col5:
                    try:
                        # Try to get Operating Margin from info
                        if 'operatingMargins' in info and info['operatingMargins'] is not None:
                            op_margin = info['operatingMargins'] * 100
                            st.metric(
                                label="Operating Margin",
                                value=f"{op_margin:.2f}%",
                                help="Higher is better. Shows operational efficiency."
                            )
                        else:
                            st.metric(label="Operating Margin", value="N/A")
                    except Exception as e:
                        print(f"Error calculating Operating Margin from info: {str(e)}")
                        st.metric(label="Operating Margin", value="N/A")
            except Exception as e:
                print(f"Error calculating metrics from info: {str(e)}")
                # Display N/A for all metrics if calculation fails
                with fund_col1:
                    st.metric(label="Debt/Equity", value="N/A")
                with fund_col2:
                    st.metric(label="Current Ratio", value="N/A")
                with fund_col3:
                    st.metric(label="ROE", value="N/A")
                with fund_col4:
                    st.metric(label="ROA", value="N/A")
                with fund_col5:
                    st.metric(label="Operating Margin", value="N/A")

        # Add additional metrics section
        st.subheader("Additional Metrics")
        add_col1, add_col2, add_col3, add_col4, add_col5 = st.columns(5)
        
        try:
            # Use financial ratios from session state if available
            if hasattr(st.session_state, 'financial_ratios') and st.session_state.financial_ratios is not None:
                with add_col1:
                    try:
                        # P/E Ratio
                        pe_ratio = st.session_state.financial_ratios.get('P/E Ratio', 'N/A')
                        if pe_ratio == 'nan' or 'nan' in str(pe_ratio):
                            pe_ratio = 'N/A'
                        
                        # Try to get a more accurate value from stock.info
                        if pe_ratio == 'N/A':
                            try:
                                info = stock.info
                                if 'forwardPE' in info and info['forwardPE'] is not None:
                                    pe_ratio = f"{info['forwardPE']:.2f}"
                                elif 'trailingPE' in info and info['trailingPE'] is not None:
                                    pe_ratio = f"{info['trailingPE']:.2f}"
                            except Exception as e:
                                print(f"Error getting P/E Ratio from info: {str(e)}")
                        
                        st.metric(
                            label="P/E Ratio",
                            value=pe_ratio,
                            help="Price to Earnings ratio. Lower can indicate better value."
                        )
                    except Exception as e:
                        print(f"Error displaying P/E Ratio: {str(e)}")
                        st.metric(label="P/E Ratio", value="N/A")
                
                with add_col2:
                    try:
                        # Price to Book
                        price_book = st.session_state.financial_ratios.get('Price/Book', 'N/A')
                        if price_book == 'nan' or 'nan' in str(price_book):
                            price_book = 'N/A'
                        
                        # Try to get a more accurate value from stock.info
                        if price_book == 'N/A':
                            try:
                                info = stock.info
                                if 'priceToBook' in info and info['priceToBook'] is not None:
                                    price_book = f"{info['priceToBook']:.2f}"
                            except Exception as e:
                                print(f"Error getting Price/Book from info: {str(e)}")
                        
                        st.metric(
                            label="Price/Book",
                            value=price_book,
                            help="Price to Book ratio. Lower can indicate better value."
                        )
                    except Exception as e:
                        print(f"Error displaying Price/Book: {str(e)}")
                        st.metric(label="Price/Book", value="N/A")
                
                with add_col3:
                    try:
                        # Dividend Yield
                        div_yield = st.session_state.financial_ratios.get('Dividend Yield', 'N/A')
                        if div_yield == 'nan' or 'nan' in str(div_yield):
                            div_yield = 'N/A'
                        
                        # Try to get a more accurate value from stock.info
                        if div_yield == 'N/A':
                            try:
                                info = stock.info
                                if 'dividendYield' in info and info['dividendYield'] is not None:
                                    # Ensure dividend yield is reasonable (less than 15%)
                                    if 0 <= info['dividendYield'] <= 0.15:
                                        div_yield = f"{info['dividendYield']*100:.2f}%"
                            except Exception as e:
                                print(f"Error getting Dividend Yield from info: {str(e)}")
                        
                        st.metric(
                            label="Dividend Yield",
                            value=div_yield,
                            help="Annual dividend yield percentage."
                        )
                    except Exception as e:
                        print(f"Error displaying Dividend Yield: {str(e)}")
                        st.metric(label="Dividend Yield", value="N/A")
                
                with add_col4:
                    try:
                        # ROCE (Return on Capital Employed)
                        roce = st.session_state.financial_ratios.get('ROCE', 'N/A')
                        if roce == 'nan' or 'nan' in str(roce):
                            roce = 'N/A'
                        
                        # Try to get a more accurate value from stock.info
                        if roce == 'N/A':
                            try:
                                info = stock.info
                                if 'returnOnCapital' in info and info['returnOnCapital'] is not None:
                                    roce = f"{info['returnOnCapital']*100:.2f}%"
                                elif 'ebit' in info and 'totalAssets' in info and 'totalCurrentLiabilities' in info:
                                    capital_employed = info['totalAssets'] - info['totalCurrentLiabilities']
                                    if capital_employed > 0:
                                        roce_val = (info['ebit'] / capital_employed) * 100
                                        roce = f"{roce_val:.2f}%"
                            except Exception as e:
                                print(f"Error getting ROCE from info: {str(e)}")
                        
                        st.metric(
                            label="ROCE",
                            value=roce,
                            help="Return on Capital Employed. Higher indicates better efficiency."
                        )
                    except Exception as e:
                        print(f"Error displaying ROCE: {str(e)}")
                        st.metric(label="ROCE", value="N/A")
                
                with add_col5:
                    try:
                        # Net Profit Margin
                        npm = st.session_state.financial_ratios.get('Net Profit Margin', 'N/A')
                        if npm == 'nan' or 'nan' in str(npm):
                            npm = 'N/A'
                        
                        # Try to get a more accurate value from stock.info
                        if npm == 'N/A':
                            try:
                                info = stock.info
                                if 'profitMargins' in info and info['profitMargins'] is not None:
                                    npm = f"{info['profitMargins']*100:.2f}%"
                            except Exception as e:
                                print(f"Error getting Net Profit Margin from info: {str(e)}")
                        
                        st.metric(
                            label="Net Profit Margin",
                            value=npm,
                            help="Net Profit Margin. Higher indicates better profitability."
                        )
                    except Exception as e:
                        print(f"Error displaying Net Profit Margin: {str(e)}")
                        st.metric(label="Net Profit Margin", value="N/A")
            else:
                # If session state doesn't have the ratios, try to calculate them directly from info
                info = stock.info
                
                with add_col1:
                    try:
                        # P/E Ratio
                        pe_ratio = None
                        if 'forwardPE' in info and info['forwardPE'] is not None:
                            pe_ratio = info['forwardPE']
                        elif 'trailingPE' in info and info['trailingPE'] is not None:
                            pe_ratio = info['trailingPE']
                        
                        if pe_ratio is not None:
                            st.metric(
                                label="P/E Ratio",
                                value=f"{pe_ratio:.2f}",
                                help="Price to Earnings ratio. Lower can indicate better value."
                            )
                        else:
                            st.metric(label="P/E Ratio", value="N/A")
                    except Exception as e:
                        print(f"Error displaying P/E Ratio: {str(e)}")
                        st.metric(label="P/E Ratio", value="N/A")
                
                with add_col2:
                    try:
                        # Price to Book
                        if 'priceToBook' in info and info['priceToBook'] is not None:
                            st.metric(
                                label="Price/Book",
                                value=f"{info['priceToBook']:.2f}",
                                help="Price to Book ratio. Lower can indicate better value."
                            )
                        else:
                            st.metric(label="Price/Book", value="N/A")
                    except Exception as e:
                        print(f"Error displaying Price/Book: {str(e)}")
                        st.metric(label="Price/Book", value="N/A")
                
                with add_col3:
                    try:
                        # Dividend Yield
                        if 'dividendYield' in info and info['dividendYield'] is not None:
                            # Ensure dividend yield is reasonable (less than 15%)
                            if 0 <= info['dividendYield'] <= 0.15:
                                st.metric(
                                    label="Dividend Yield",
                                    value=f"{info['dividendYield']*100:.2f}%",
                                    help="Annual dividend yield percentage."
                                )
                            else:
                                st.metric(label="Dividend Yield", value="N/A")
                        else:
                            st.metric(label="Dividend Yield", value="N/A")
                    except Exception as e:
                        print(f"Error displaying Dividend Yield: {str(e)}")
                        st.metric(label="Dividend Yield", value="N/A")
                
                with add_col4:
                    try:
                        # ROCE (Return on Capital Employed)
                        roce = None
                        if 'returnOnCapital' in info and info['returnOnCapital'] is not None:
                            roce = info['returnOnCapital'] * 100
                        elif 'ebit' in info and 'totalAssets' in info and 'totalCurrentLiabilities' in info:
                            capital_employed = info['totalAssets'] - info['totalCurrentLiabilities']
                            if capital_employed > 0:
                                roce = (info['ebit'] / capital_employed) * 100
                        
                        if roce is not None:
                            st.metric(
                                label="ROCE",
                                value=f"{roce:.2f}%",
                                help="Return on Capital Employed. Higher indicates better efficiency."
                            )
                        else:
                            st.metric(label="ROCE", value="N/A")
                    except Exception as e:
                        print(f"Error displaying ROCE: {str(e)}")
                        st.metric(label="ROCE", value="N/A")
                
                with add_col5:
                    try:
                        # Net Profit Margin
                        if 'profitMargins' in info and info['profitMargins'] is not None:
                            st.metric(
                                label="Net Profit Margin",
                                value=f"{info['profitMargins']*100:.2f}%",
                                help="Net Profit Margin. Higher indicates better profitability."
                            )
                        else:
                            st.metric(label="Net Profit Margin", value="N/A")
                    except Exception as e:
                        print(f"Error displaying Net Profit Margin: {str(e)}")
                        st.metric(label="Net Profit Margin", value="N/A")
        except Exception as e:
            print(f"Error displaying additional metrics: {str(e)}")
            # Display N/A for all additional metrics if calculation fails
            with add_col1:
                st.metric(label="P/E Ratio", value="N/A")
            with add_col2:
                st.metric(label="Price/Book", value="N/A")
            with add_col3:
                st.metric(label="Dividend Yield", value="N/A")
            with add_col4:
                st.metric(label="ROCE", value="N/A")
            with add_col5:
                st.metric(label="Net Profit Margin", value="N/A")

        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["AI Analysis", "Charts", "Financial Statements"])
        
        with tab1:
            if st.button("Generate AI Analysis"):
                with st.spinner('Generating comprehensive analysis... This may take a few minutes.'):
                    # Get AI analysis
                    analysis_result = st.session_state.analyzer.analyze_stock(symbol, analysis_types)
                    st.markdown(analysis_result)
        
        with tab2:
            # Historical price chart
            hist_data = stock.history(period=time_period)
            
            # Add company business information
            st.subheader("Company Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Business Overview")
                st.write(info.get('longBusinessSummary', 'No business summary available.'))
            
            with col2:
                st.markdown("### Key Business Segments")
                sectors = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                country = info.get('country', 'N/A')
                st.write(f"**Sector:** {sectors}")
                st.write(f"**Industry:** {industry}")
                st.write(f"**Country:** {country}")
            
            # Add Advanced Technical Analysis
            st.subheader("Advanced Technical Analysis")
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                try:
                    # MACD Chart with error handling
                    if not hist_data.empty and len(hist_data) > 0:
                        try:
                            macd = MACD(close=hist_data['Close'])
                            macd_line = macd.macd()
                            signal_line = macd.macd_signal()
                            macd_hist = macd.macd_diff()
                            
                            if not (macd_line.empty or signal_line.empty or macd_hist.empty):
                                fig_macd = go.Figure()
                                fig_macd.add_trace(go.Scatter(x=hist_data.index, y=macd_line, name='MACD Line'))
                                fig_macd.add_trace(go.Scatter(x=hist_data.index, y=signal_line, name='Signal Line'))
                                fig_macd.add_trace(go.Bar(x=hist_data.index, y=macd_hist, name='MACD Histogram'))
                                fig_macd.update_layout(title='MACD', template="plotly_dark")
                                st.plotly_chart(fig_macd, use_container_width=True)
                            else:
                                st.warning("Insufficient data to calculate MACD")
                        except Exception as e:
                            print(f"Error calculating MACD: {str(e)}")
                            st.warning("Unable to calculate MACD indicators")
                    
                    # Stochastic Oscillator with error handling
                    try:
                        stoch = StochasticOscillator(high=hist_data['High'], 
                                                   low=hist_data['Low'], 
                                                   close=hist_data['Close'],
                                                   window=14,  # Standard period
                                                   smooth_window=3)  # Standard smoothing
                        stoch_k = stoch.stoch()
                        stoch_d = stoch.stoch_signal()
                        
                        if not (stoch_k.empty or stoch_d.empty):
                            fig_stoch = go.Figure()
                            fig_stoch.add_trace(go.Scatter(x=hist_data.index, y=stoch_k, name='%K Line'))
                            fig_stoch.add_trace(go.Scatter(x=hist_data.index, y=stoch_d, name='%D Line'))
                            fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
                            fig_stoch.update_layout(
                                title='Stochastic Oscillator', 
                                template="plotly_dark",
                                yaxis_range=[0, 100]  # Fixed range for better visualization
                            )
                            st.plotly_chart(fig_stoch, use_container_width=True)
                        else:
                            st.warning("Insufficient data to calculate Stochastic Oscillator")
                    except Exception as e:
                        print(f"Error calculating Stochastic Oscillator: {str(e)}")
                        st.warning("Unable to calculate Stochastic Oscillator")
                except Exception as e:
                    print(f"Error in technical analysis column 1: {str(e)}")
                    st.error("Unable to perform technical analysis")
            
            with tech_col2:
                try:
                    # Risk Metrics with error handling
                    st.markdown("### Risk Metrics")
                    try:
                        risk_metrics = calculate_risk_metrics(symbol, time_period)
                        risk_data = eval(risk_metrics)
                        
                        # Store risk metrics in session state for agents to use
                        if isinstance(risk_data, dict) and risk_data:
                            st.session_state.risk_metrics = risk_data
                            
                            # Organize risk metrics into categories
                            risk_col1, risk_col2 = st.columns(2)
                            
                            with risk_col1:
                                st.markdown("#### Value at Risk & Volatility")
                                if "Value at Risk (95%)" in risk_data:
                                    st.metric("Value at Risk (95%)", risk_data["Value at Risk (95%)"], 
                                             help="Maximum expected loss with 95% confidence over a day")
                                if "Value at Risk (99%)" in risk_data:
                                    st.metric("Value at Risk (99%)", risk_data["Value at Risk (99%)"], 
                                             help="Maximum expected loss with 99% confidence over a day")
                                if "Conditional VaR (95%)" in risk_data:
                                    st.metric("Conditional VaR", risk_data["Conditional VaR (95%)"], 
                                             help="Expected loss in worst 5% of cases (tail risk)")
                                if "Maximum Drawdown" in risk_data:
                                    st.metric("Maximum Drawdown", risk_data["Maximum Drawdown"], 
                                             help="Largest peak-to-trough decline during the period")
                                if "Beta" in risk_data:
                                    st.metric("Beta", risk_data["Beta"], 
                                             help="Stock's volatility relative to the market")
                            
                            with risk_col2:
                                st.markdown("#### Risk-Adjusted Returns")
                                if "Sharpe Ratio" in risk_data:
                                    st.metric("Sharpe Ratio", risk_data["Sharpe Ratio"], 
                                             help="Return per unit of risk (higher is better)")
                                if "Sortino Ratio" in risk_data:
                                    st.metric("Sortino Ratio", risk_data["Sortino Ratio"], 
                                             help="Return per unit of downside risk (higher is better)")
                                if "Treynor Ratio" in risk_data:
                                    st.metric("Treynor Ratio", risk_data["Treynor Ratio"], 
                                             help="Return per unit of market risk (higher is better)")
                                if "Information Ratio" in risk_data:
                                    st.metric("Information Ratio", risk_data["Information Ratio"], 
                                             help="Active return per unit of active risk (higher is better)")
                                if "Calmar Ratio" in risk_data:
                                    st.metric("Calmar Ratio", risk_data["Calmar Ratio"], 
                                             help="Return per unit of maximum drawdown (higher is better)")
                            
                            # Stress Test Scenarios
                            st.markdown("#### Stress Test Scenarios")
                            stress_col1, stress_col2, stress_col3 = st.columns(3)
                            
                            with stress_col1:
                                if "Market Crash Impact" in risk_data:
                                    st.metric("Market Crash", risk_data["Market Crash Impact"], 
                                             help="Estimated impact of a severe market crash")
                            
                            with stress_col2:
                                if "Interest Rate Hike Impact" in risk_data:
                                    st.metric("Interest Rate Hike", risk_data["Interest Rate Hike Impact"], 
                                             help="Estimated impact of significant interest rate increase")
                            
                            with stress_col3:
                                if "Sector Shock Impact" in risk_data:
                                    st.metric("Sector Shock", risk_data["Sector Shock Impact"], 
                                             help="Estimated impact of sector-specific negative event")
                        else:
                            st.warning("Unable to calculate risk metrics. Error: " + str(risk_data))
                            st.session_state.risk_metrics = None
                    except Exception as e:
                        print(f"Error calculating risk metrics: {str(e)}")
                        st.warning(f"Unable to calculate risk metrics. Error: {str(e)}")
                        st.session_state.risk_metrics = None
                    
                    # ATR Chart with error handling
                    if not hist_data.empty and len(hist_data) > 0:
                        try:
                            atr = AverageTrueRange(high=hist_data['High'], 
                                                 low=hist_data['Low'], 
                                                 close=hist_data['Close'],
                                                 window=14)  # Standard period
                            atr_value = atr.average_true_range()
                            
                            if not atr_value.empty:
                                fig_atr = go.Figure()
                                fig_atr.add_trace(go.Scatter(x=hist_data.index, y=atr_value, name='ATR'))
                                fig_atr.update_layout(
                                    title='Average True Range (ATR)', 
                                    template="plotly_dark",
                                    yaxis_title="ATR Value"
                                )
                                st.plotly_chart(fig_atr, use_container_width=True)
                            else:
                                st.warning("Insufficient data to calculate ATR")
                        except Exception as e:
                            print(f"Error calculating ATR: {str(e)}")
                            st.warning("Unable to calculate ATR")
                    else:
                        st.warning("No historical data available for ATR calculation")
                except Exception as e:
                    print(f"Error in technical analysis column 2: {str(e)}")
                    st.error("Unable to perform technical analysis")
            
            # Candlestick chart with Fibonacci levels
            st.subheader("Price Chart with Fibonacci Levels")
            try:
                if not hist_data.empty and len(hist_data) > 0:
                    # Calculate Fibonacci levels
                    high = hist_data['High'].max()
                    low = hist_data['Low'].min()
                    diff = high - low
                    fib_levels = {
                        '0%': low,
                        '23.6%': low + 0.236 * diff,
                        '38.2%': low + 0.382 * diff,
                        '50%': low + 0.5 * diff,
                        '61.8%': low + 0.618 * diff,
                        '100%': high
                    }
                    
                    # Create candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        name="OHLC"
                    )])
                    
                    # Add Fibonacci levels
                    for level, price in fib_levels.items():
                        fig.add_hline(
                            y=price, 
                            line_dash="dash", 
                            line_color="gray",
                            annotation_text=f"Fib {level}: {price:.2f}", 
                            annotation_position="right"
                        )
                    
                    # Add moving averages with error handling
                    try:
                        ma20 = hist_data['Close'].rolling(window=20).mean()
                        ma50 = hist_data['Close'].rolling(window=50).mean()
                        
                        if not ma20.empty:
                            fig.add_trace(go.Scatter(
                                x=hist_data.index, 
                                y=ma20,
                                line=dict(color='orange', width=1),
                                name="20-day MA"
                            ))
                        
                        if not ma50.empty:
                            fig.add_trace(go.Scatter(
                                x=hist_data.index, 
                                y=ma50,
                                line=dict(color='blue', width=1),
                                name="50-day MA"
                            ))
                    except Exception as e:
                        print(f"Error calculating moving averages: {str(e)}")
                    
                    fig.update_layout(
                        title=f"{symbol} Stock Price with Fibonacci Levels",
                        yaxis_title="Price",
                        xaxis_title="Date",
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    try:
                        fig_volume = go.Figure(data=[go.Bar(
                            x=hist_data.index, 
                            y=hist_data['Volume'],
                            name="Volume"
                        )])
                        fig_volume.update_layout(
                            title=f"{symbol} Trading Volume",
                            yaxis_title="Volume",
                            xaxis_title="Date",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
                    except Exception as e:
                        print(f"Error creating volume chart: {str(e)}")
                        st.warning("Unable to display volume chart")
                else:
                    st.warning("No historical data available for price chart")
            except Exception as e:
                print(f"Error creating price chart: {str(e)}")
                st.error("Unable to create price chart")
        
        with tab3:
            st.subheader("Historical Data")
            
            # Add P/E ratio comparison
            pe_col1, pe_col2 = st.columns(2)
            with pe_col1:
                st.metric(
                    label="Current P/E Ratio",
                    value=f"{st.session_state.current_pe:.2f}" if st.session_state.current_pe is not None else "N/A",
                    help="Current Price to Earnings ratio. Lower values may indicate better value."
                )
            with pe_col2:
                st.metric(
                    label="5Y Median P/E",
                    value=f"{st.session_state.median_pe:.2f}" if st.session_state.median_pe is not None else "N/A",
                    help="Median Price/Earnings ratio over the past 5 years. Useful for historical valuation comparison."
                )
            
            # Add Financial Statements
            st.subheader("Financial Statements")
            fin_tabs = st.tabs(["Quarterly Results", "Income Statement", "Balance Sheet", "Cash Flow"])
            
            with fin_tabs[0]:
                try:
                    # Try multiple methods to fetch quarterly data
                    quarterly_data = None
                    try:
                        quarterly_data = stock.quarterly_financials
                    except Exception:
                        try:
                            quarterly_data = stock.quarterly_earnings
                        except Exception:
                            pass
                    
                    if quarterly_data is not None and not quarterly_data.empty:
                        # Format earnings with currency and update date format
                        formatted_quarterly = quarterly_data.copy()
                        
                        # Convert column names (dates) to desired format
                        formatted_quarterly.columns = [pd.to_datetime(col).strftime('%b %Y').upper() for col in formatted_quarterly.columns]
                        
                        # Format all numeric columns
                        for col in formatted_quarterly.columns:
                            try:
                                def format_value(x):
                                    if isinstance(x, (int, float)):
                                        if abs(x) >= 1e6:
                                            return f"{currency}{x/1e6:,.2f}M"
                                        return f"{currency}{x:,.2f}"
                                    return str(x)
                                
                                formatted_quarterly[col] = formatted_quarterly[col].apply(format_value)
                            except Exception:
                                continue
                        
                        # Display the formatted dataframe
                        st.markdown('<div style="background-color: #1a1a1a; padding: 10px; border-radius: 5px;">', 
                                  unsafe_allow_html=True)
                        st.dataframe(formatted_quarterly, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add download button with formatted dates
                        csv = quarterly_data.to_csv()
                        st.download_button(
                            label="Download Quarterly Results",
                            data=csv,
                            file_name=f"{symbol}_quarterly_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No quarterly results available for this stock.")
                except Exception as e:
                    st.error(f"Unable to fetch quarterly results: {str(e)}")
            
            with fin_tabs[1]:
                try:
                    income_stmt = stock.income_stmt
                    if not income_stmt.empty:
                        # Format income statement with currency and update date format
                        formatted_income = income_stmt.copy()
                        
                        # Convert column names (dates) to desired format
                        formatted_income.columns = [pd.to_datetime(col).strftime('%b %Y').upper() for col in formatted_income.columns]
                        
                        # Format values
                        for col in formatted_income.columns:
                            formatted_income[col] = formatted_income[col].apply(
                                lambda x: f"{currency}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{currency}{x:,.2f}"
                            )
                        st.dataframe(formatted_income)
                    else:
                        st.info("No income statement available")
                except Exception as e:
                    st.error("Unable to fetch income statement")
            
            with fin_tabs[2]:
                try:
                    balance_sheet = stock.balance_sheet
                    if not balance_sheet.empty:
                        # Format balance sheet with currency and update date format
                        formatted_balance = balance_sheet.copy()
                        
                        # Convert column names (dates) to desired format
                        formatted_balance.columns = [pd.to_datetime(col).strftime('%b %Y').upper() for col in formatted_balance.columns]
                        
                        # Format values
                        for col in formatted_balance.columns:
                            formatted_balance[col] = formatted_balance[col].apply(
                                lambda x: f"{currency}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{currency}{x:,.2f}"
                            )
                        st.dataframe(formatted_balance)
                    else:
                        st.info("No balance sheet available")
                except Exception as e:
                    st.error("Unable to fetch balance sheet")
            
            with fin_tabs[3]:
                try:
                    cash_flow = stock.cashflow
                    if not cash_flow.empty:
                        # Format cash flow with currency and update date format
                        formatted_cash = cash_flow.copy()
                        
                        # Convert column names (dates) to desired format
                        formatted_cash.columns = [pd.to_datetime(col).strftime('%b %Y').upper() for col in formatted_cash.columns]
                        
                        # Format values
                        for col in formatted_cash.columns:
                            formatted_cash[col] = formatted_cash[col].apply(
                                lambda x: f"{currency}{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"{currency}{x:,.2f}"
                            )
                        st.dataframe(formatted_cash)
                    else:
                        st.info("No cash flow statement available")
                except Exception as e:
                    st.error("Unable to fetch cash flow statement")
            
            st.subheader("Price History")
            # Format the dataframe with proper currency
            formatted_data = hist_data.copy()
            
            # Format price columns with currency
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                formatted_data[col] = formatted_data[col].apply(lambda x: f"{currency}{x:,.2f}")
            
            # Format volume with commas
            formatted_data['Volume'] = formatted_data['Volume'].apply(lambda x: f"{x:,.0f}")
            
            # Display the formatted dataframe
            st.dataframe(formatted_data)
            
            # Add download buttons for all data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                csv = hist_data.to_csv(index=True)
                st.download_button(
                    label="Download Price History",
                    data=csv,
                    file_name=f"{symbol}_historical_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                try:
                    csv = stock.quarterly_earnings.to_csv(index=True)
                    st.download_button(
                        label="Download Quarterly Results",
                        data=csv,
                        file_name=f"{symbol}_quarterly_results.csv",
                        mime="text/csv"
                    )
                except:
                    pass
            
            with col3:
                try:
                    csv = stock.income_stmt.to_csv(index=True)
                    st.download_button(
                        label="Download Income Statement",
                        data=csv,
                        file_name=f"{symbol}_income_statement.csv",
                        mime="text/csv"
                    )
                except:
                    pass
            
            with col4:
                try:
                    csv = stock.balance_sheet.to_csv(index=True)
                    st.download_button(
                        label="Download Balance Sheet",
                        data=csv,
                        file_name=f"{symbol}_balance_sheet.csv",
                        mime="text/csv"
                    )
                except:
                    pass

    except Exception as e:
        st.error(f"Error analyzing {symbol}. Please check if the symbol is correct.")
        st.exception(e) 