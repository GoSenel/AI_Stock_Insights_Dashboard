# AI Stock Insights Dashboard

A beginner-friendly stock market dashboard built with **Streamlit**, **Yahoo Finance (yfinance)**, **Plotly**, and **Machine Learning**.

This application is designed as:

- A portfolio project
- An educational learning platform
- A stock analysis dashboard
- A financial data visualization tool

The goal is to help users understand stocks, technical indicators, company fundamentals, and forecasting concepts in a simple and visual way. Check it Here 

https://aistockinsightsdashboard-macejdfgtwv2vgjlzchkik.streamlit.app/

---

# Features

## Stock Search

Search any publicly traded company by ticker symbol.

Examples:

- AAPL (Apple)
- MSFT (Microsoft)
- NVDA (NVIDIA)
- AMD
- AMZN
- META
- TSLA
- GOOGL

The application validates data and displays clear messages when information is unavailable.

---

## Company Overview

View important company information including:

- Company Name
- Ticker Symbol
- Sector
- Industry
- Market Capitalization
- P/E Ratio
- Beta
- Dividend Information
- Company Website
- Business Description
- Latest Share Price

The dashboard presents this information in a beginner-friendly format with explanations.

---

## Interactive Price Charts

Powered by Plotly.

Features include:

- Candlestick Charts
- Zooming
- Panning
- Hover Information
- Responsive Layout
- Shared Volume Analysis

Users can explore historical market behavior visually.

---

## Technical Analysis

### Moving Averages

Available overlays:

- MA20
- MA50
- MA200

Helpful for visualizing market trends.

---

### RSI (Relative Strength Index)

Includes:

- RSI Chart
- Current RSI Value
- Educational Interpretation

Learn:

- Momentum
- Overbought Conditions
- Oversold Conditions

---

### MACD

Includes:

- MACD Line
- Signal Line
- Momentum Analysis

Learn:

- Trend Direction
- Momentum Shifts
- Crossovers

---

### Bollinger Bands

The dashboard calculates and visualizes Bollinger Bands concepts.

Learn:

- Volatility
- Price Expansion
- Price Compression
- Market Uncertainty

---

## AI Forecasting Lab

Educational machine learning demonstrations.

### Models Included

#### Linear Regression

A simple trend-fitting model.

Useful for:

- Understanding price trends
- Learning regression concepts

---

#### Random Forest Regressor

Ensemble-based machine learning model.

Useful for:

- Capturing more complex patterns
- Demonstrating modern predictive techniques

---

### Forecast Horizons

Users can generate forecasts for:

- 7 Days
- 14 Days
- 30 Days

---

### Model Evaluation Metrics

For each model, the dashboard calculates:

- RMSE
- MAE
- R² Score

Helping users understand prediction quality.

---

## News Integration

Displays recent company news from Yahoo Finance.

News cards may include:

- Headline
- Source
- Publication Date
- Summary
- Article Link
- Thumbnail Image (when available)

---

## Watchlist

Create a personal watchlist.

Features:

- Add Tickers
- Remove Tickers
- Persistent Local Storage
- Quick Company Reference

Watchlist data is stored in:

```text
watchlist.json
```

---

## Education Center

Designed for complete beginners.

Topics include:

### What is a Stock?

Learn how company ownership works.

### Why Stock Prices Change

Understand supply, demand, expectations, and market psychology.

### What is a Stock Market?

Simple explanation of how exchanges operate.

### Market Capitalization

Learn what company value means.

### Dividends

Understand shareholder distributions.

### Candlestick Charts

Learn:

- Open
- High
- Low
- Close

---

### Moving Averages

Learn:

- Trend Direction
- Support
- Resistance

---

### RSI

Learn:

- Momentum
- Overbought Markets
- Oversold Markets

---

### MACD

Learn:

- Trend Changes
- Crossovers
- Momentum

---

### Bollinger Bands

Learn:

- Volatility
- Market Expansion
- Market Compression

---

### Forecasting Concepts

Understand:

- Model Uncertainty
- Why Predictions Fail
- Historical Data Limitations

---

### Diversification

Learn:

- Risk Management
- Portfolio Construction

---

## Data Export

Download historical stock prices as CSV.

Export includes:

- Open
- High
- Low
- Close
- Volume
- Technical Indicators

Forecast outputs can also be exported for further analysis.

---

# Technologies Used

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance
- Scikit-learn

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-stock-insights-dashboard.git
```

Enter the project directory:

```bash
cd ai-stock-insights-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

# Screenshots

Add screenshots here.

Suggested sections:

### Dashboard Overview

images/dashboard.png

### Technical Analysis

images/technical.png

### Forecasting Lab

images/forecast.png

### Education Center

images/education.png

---

# Deployment

## Streamlit Community Cloud

1. Push project to GitHub
2. Sign in to Streamlit Cloud
3. Create New App
4. Select repository
5. Select:

```text
app.py
```

6. Deploy

A public shareable URL will be generated.

---

## Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

# Educational Disclaimer

This application is intended solely for educational and informational purposes.

The forecasting models included are demonstrations of machine learning techniques and should not be considered accurate predictions of future market performance.

Financial markets involve risk and uncertainty.

Nothing in this application constitutes:

- Financial advice
- Investment advice
- Trading advice
- Portfolio recommendations

Always perform your own research and consult qualified financial professionals before making investment decisions.

---

# Data Source

Market data and company information are provided through:

- Yahoo Finance
- yfinance

Data may be delayed, incomplete, unavailable, or subject to change.

---

# License

MIT License

Copyright (c) 2026 Göknil Şenel
