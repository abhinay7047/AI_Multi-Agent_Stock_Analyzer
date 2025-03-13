import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os
import requests
from stock_analyzer import StockAnalyzer,calculate_risk_metrics
import json
from bs4 import BeautifulSoup
from typing import Optional, Dict
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from ta.volatility import AverageTrueRange

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
    default=["Fundamental Analysis", "Technical Analysis", "News Analysis","Risk Analysis","Portfolio Analysis"]
)

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
            col1, col2, col3, col4 = st.columns(4)
            
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
                
            with col4:
                st.metric(
                    label="P/E Ratio",
                    value=f"{info.get('forwardPE', 'N/A'):.2f}"
                )

            # Add Fundamental Metrics section
            st.subheader("Fundamental Metrics")
            fund_col1, fund_col2, fund_col3, fund_col4, fund_col5 = st.columns(5)
            
            # Get financial data once to avoid multiple API calls
            try:
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                
                with fund_col1:
                    try:
                        # Calculate Debt/Equity ratio
                        if not balance_sheet.empty:
                            # Use exact column names from the balance sheet
                            debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else None
                            equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] if 'Total Equity Gross Minority Interest' in balance_sheet.index else None
                            
                            if debt is not None and equity is not None and equity != 0:
                                debt_equity = debt / equity
                                st.metric(
                                    label="Debt/Equity",
                                    value=f"{debt_equity:.2f}",
                                    help="Lower is better. Indicates financial leverage and risk."
                                )
                            else:
                                st.metric(label="Debt/Equity", value="N/A")
                        else:
                            st.metric(label="Debt/Equity", value="N/A")
                    except Exception as e:
                        st.metric(label="Debt/Equity", value="N/A")
                
                with fund_col2:
                    try:
                        # Calculate Current Ratio
                        if not balance_sheet.empty:
                            # Use exact column names from the balance sheet
                            current_assets = balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0] if 'Cash Cash Equivalents And Short Term Investments' in balance_sheet.index else None
                            current_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                            
                            if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
                                current_ratio = current_assets / current_liabilities
                                st.metric(
                                    label="Current Ratio",
                                    value=f"{current_ratio:.2f}",
                                    help="Higher is better. Measures short-term liquidity."
                                )
                            else:
                                st.metric(label="Current Ratio", value="N/A")
                        else:
                            st.metric(label="Current Ratio", value="N/A")
                    except Exception as e:
                        st.metric(label="Current Ratio", value="N/A")
                
                with fund_col3:
                    try:
                        # Calculate ROE
                        if not income_stmt.empty and not balance_sheet.empty:
                            # Use exact column names from the financial statements
                            net_income = income_stmt.loc['Net Income Common Stockholders'].iloc[0] if 'Net Income Common Stockholders' in income_stmt.index else None
                            equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] if 'Total Equity Gross Minority Interest' in balance_sheet.index else None
                            
                            if net_income is not None and equity is not None and equity != 0:
                                roe = (net_income / equity) * 100
                                st.metric(
                                    label="ROE",
                                    value=f"{roe:.2f}%",
                                    help="Higher is better. Measures return on shareholder equity."
                                )
                            else:
                                st.metric(label="ROE", value="N/A")
                        else:
                            st.metric(label="ROE", value="N/A")
                    except Exception as e:
                        st.metric(label="ROE", value="N/A")
                
                with fund_col4:
                    try:
                        # Calculate ROA
                        if not income_stmt.empty and not balance_sheet.empty:
                            # Use exact column names from the financial statements
                            net_income = income_stmt.loc['Net Income Common Stockholders'].iloc[0] if 'Net Income Common Stockholders' in income_stmt.index else None
                            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
                            
                            if net_income is not None and total_assets is not None and total_assets != 0:
                                roa = (net_income / total_assets) * 100
                                st.metric(
                                    label="ROA",
                                    value=f"{roa:.2f}%",
                                    help="Higher is better. Measures efficiency in using assets."
                                )
                            else:
                                st.metric(label="ROA", value="N/A")
                        else:
                            st.metric(label="ROA", value="N/A")
                    except Exception as e:
                        st.metric(label="ROA", value="N/A")
                
                with fund_col5:
                    try:
                        # Calculate Operating Margin
                        if not income_stmt.empty:
                            # Use exact column names from the income statement
                            operating_income = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else None
                            total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else None
                            
                            if operating_income is not None and total_revenue is not None and total_revenue != 0:
                                operating_margin = (operating_income / total_revenue) * 100
                                st.metric(
                                    label="Operating Margin",
                                    value=f"{operating_margin:.2f}%",
                                    help="Higher is better. Shows operational efficiency."
                                )
                            else:
                                st.metric(label="Operating Margin", value="N/A")
                        else:
                            st.metric(label="Operating Margin", value="N/A")
                    except Exception as e:
                        st.metric(label="Operating Margin", value="N/A")

            except Exception as e:
                st.error(f"Error fetching financial data: {str(e)}")
                st.metric(label="Debt/Equity", value="N/A")
                st.metric(label="Current Ratio", value="N/A")
                st.metric(label="ROE", value="N/A")
                st.metric(label="ROA", value="N/A")
                st.metric(label="Operating Margin", value="N/A")

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
                    # MACD Chart
                    macd = MACD(close=hist_data['Close'])
                    macd_line = macd.macd()
                    signal_line = macd.macd_signal()
                    macd_hist = macd.macd_diff()
                    
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=hist_data.index, y=macd_line, name='MACD Line'))
                    fig_macd.add_trace(go.Scatter(x=hist_data.index, y=signal_line, name='Signal Line'))
                    fig_macd.add_trace(go.Bar(x=hist_data.index, y=macd_hist, name='MACD Histogram'))
                    fig_macd.update_layout(title='MACD', template="plotly_dark")
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # Stochastic Oscillator
                    stoch = StochasticOscillator(high=hist_data['High'], low=hist_data['Low'], close=hist_data['Close'])
                    stoch_k = stoch.stoch()
                    stoch_d = stoch.stoch_signal()
                    
                    fig_stoch = go.Figure()
                    fig_stoch.add_trace(go.Scatter(x=hist_data.index, y=stoch_k, name='K Line'))
                    fig_stoch.add_trace(go.Scatter(x=hist_data.index, y=stoch_d, name='D Line'))
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
                    fig_stoch.update_layout(title='Stochastic Oscillator', template="plotly_dark")
                    st.plotly_chart(fig_stoch, use_container_width=True)
                
                with tech_col2:
                    # Risk Metrics
                    st.markdown("### Risk Metrics")
                    risk_metrics = calculate_risk_metrics(symbol, time_period)
                    risk_data = eval(risk_metrics)
                    
                    for metric, value in risk_data.items():
                        st.metric(metric, value)
                    
                    # ATR Chart
                    atr = AverageTrueRange(high=hist_data['High'], low=hist_data['Low'], close=hist_data['Close'])
                    atr_value = atr.average_true_range()
                    
                    fig_atr = go.Figure()
                    fig_atr.add_trace(go.Scatter(x=hist_data.index, y=atr_value, name='ATR'))
                    fig_atr.update_layout(title='Average True Range', template="plotly_dark")
                    st.plotly_chart(fig_atr, use_container_width=True)
                
                # Candlestick chart with Fibonacci levels
                st.subheader("Price Chart with Fibonacci Levels")
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
                
                fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                            open=hist_data['Open'],
                            high=hist_data['High'],
                            low=hist_data['Low'],
                            close=hist_data['Close'],
                            name="OHLC")])
                
                # Add Fibonacci levels
                for level, price in fib_levels.items():
                    fig.add_hline(y=price, line_dash="dash", line_color="gray",
                                annotation_text=level, annotation_position="right")
                
                # Add moving averages
                fig.add_trace(go.Scatter(x=hist_data.index, 
                                       y=hist_data['Close'].rolling(window=20).mean(),
                                       line=dict(color='orange', width=1),
                                       name="20-day MA"))
                fig.add_trace(go.Scatter(x=hist_data.index, 
                                       y=hist_data['Close'].rolling(window=50).mean(),
                                       line=dict(color='blue', width=1),
                                       name="50-day MA"))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price with Fibonacci Levels",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure(data=[go.Bar(x=hist_data.index, 
                                                   y=hist_data['Volume'],
                                                   name="Volume")])
                fig_volume.update_layout(
                    title=f"{symbol} Trading Volume",
                    yaxis_title="Volume",
                    xaxis_title="Date",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_volume, use_container_width=True)
            
            with tab3:
                st.subheader("Historical Data")
                
                # Add P/E ratio comparison
                pe_col1, pe_col2 = st.columns(2)
                with pe_col1:
                    current_pe = info.get('forwardPE', info.get('trailingPE', 'N/A'))
                    st.metric(
                        label="Current P/E Ratio",
                        value=f"{current_pe:.2f}" if isinstance(current_pe, (int, float)) else 'N/A'
                    )
                with pe_col2:
                    try:
                        # Try multiple methods to get median PE
                        median_pe = None
                        try:
                            income_stmt = stock.income_stmt
                            if not income_stmt.empty:
                                # Look for Net Income in income statement
                                if 'Net Income' in income_stmt.index:
                                    earnings = income_stmt.loc['Net Income']
                                    if len(earnings) > 0:
                                        avg_earnings = earnings.mean()
                                        current_price = info.get('regularMarketPrice', 0)
                                        if avg_earnings != 0:
                                            median_pe = current_price / (avg_earnings / info.get('sharesOutstanding', 1))
                                # If Net Income not found, try Total Revenue
                                elif 'Total Revenue' in income_stmt.index:
                                    earnings = income_stmt.loc['Total Revenue']
                                    if len(earnings) > 0:
                                        avg_earnings = earnings.mean()
                                        current_price = info.get('regularMarketPrice', 0)
                                        if avg_earnings != 0:
                                            median_pe = current_price / (avg_earnings / info.get('sharesOutstanding', 1))
                        except Exception as e:
                            print(f"Error calculating median PE from income statement: {str(e)}")
                            pass
                        
                        if median_pe is None:
                            median_pe = info.get('fiveYearAvgDividendYield', info.get('trailingPE', 'N/A'))
                        
                        st.metric(
                            label="Median P/E Ratio (5Y)",
                            value=f"{median_pe:.2f}" if isinstance(median_pe, (int, float)) else 'N/A'
                        )
                    except Exception as e:
                        st.metric(
                            label="Median P/E Ratio (5Y)",
                            value='N/A'
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

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This AI-powered stock analysis platform combines fundamental analysis, "
    "technical indicators, and market sentiment to provide comprehensive insights. "
    "The analysis is powered by OpenAI's GPT-4 and uses real-time market data."
)

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("### Version Info")
st.sidebar.text("v1.0.0") 