# Advanced Financial Analysis System

A comprehensive financial analysis platform powered by AI that combines multiple data sources and analysis tools to provide in-depth market insights.

## Features

- Real-time stock price analysis
- Technical and fundamental analysis
- Interactive charts and visualizations
- Risk assessment
- Comparative peer analysis

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Enter a stock symbol in the sidebar
2. Select the desired timeframe for analysis
3. Choose the types of analysis you want to perform
4. Click "Generate AI Analysis" to get results

## Features Details

### Technical Analysis
- Price trends and patterns
- Moving averages
- Volume analysis
- Technical indicators

### Fundamental Analysis
- Financial ratios
- Company fundamentals
- Market position
- Peer comparison

### Risk Assessment
- Volatility metrics
- Risk indicators
- Market correlation

## Dependencies

- Streamlit
- YFinance
- Plotly
- Pandas
- NumPy
- ta (Technical Analysis Library)

## Notes

- Ensure all API keys are valid and have sufficient credits
- Some analyses may take a few minutes to generate
- Keep your dependencies up to date
- Monitor API rate limits 