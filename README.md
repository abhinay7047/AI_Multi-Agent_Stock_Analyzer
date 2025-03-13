# AI Stock Analysis Platform

A comprehensive stock analysis platform that combines fundamental analysis, technical indicators, and market sentiment to provide detailed insights into stock performance. The platform uses OpenAI's GPT-4 for intelligent analysis and real-time market data from Yahoo Finance.

## Features

### 1. Market Overview
- Real-time market indices display (NIFTY 50, SENSEX, BANK NIFTY, NASDAQ, S&P 500)
- Live price updates with percentage changes
- Currency-specific formatting (₹ for Indian markets, $ for US markets)

### 2. Quick Metrics
- Current Price with change percentage
- Market Capitalization
- Trading Volume
- P/E Ratio

### 3. Fundamental Analysis
- Debt/Equity Ratio
- Current Ratio
- Return on Equity (ROE)
- Return on Assets (ROA)
- Operating Margin
- Comprehensive financial statements (Income Statement, Balance Sheet, Cash Flow)

### 4. Technical Analysis
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Average True Range (ATR)
- Moving Averages (20-day and 50-day)
- Fibonacci Levels
- Candlestick Charts
- Volume Analysis

### 5. Risk Analysis
- Volatility Metrics
- Risk-adjusted Returns
- Market Risk Indicators

### 6. AI-Powered Analysis
- Comprehensive stock analysis using GPT-4
- Sector-specific insights
- Market sentiment analysis
- Growth and dividend analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-analysis-platform
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Enter a stock symbol in the sidebar (e.g., "AAPL" for Apple Inc.)

3. Select the desired analysis types:
   - Fundamental Analysis
   - Technical Analysis
   - News Analysis
   - Risk Analysis
   - Portfolio Analysis

4. Choose the time period for analysis:
   - 1 month
   - 3 months
   - 6 months
   - 1 year
   - 2 years
   - 5 years

## Project Structure

- `app.py`: Main Streamlit application
- `stock_analyzer.py`: Core analysis functionality
- `.env`: Environment variables (API keys)
- `requirements.txt`: Project dependencies

## Dependencies

- streamlit
- yfinance
- pandas
- plotly
- ta (Technical Analysis library)
- python-dotenv
- requests
- beautifulsoup4
- openai

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Disclaimer

This tool is for educational and research purposes only. Always do your own research and consider consulting with financial advisors before making investment decisions.

## Version

Current Version: 1.0.0 