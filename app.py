import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Page setup and visual style
# ------------------------------------------------------------
st.set_page_config(
    page_title="Stock Insights | Learn as You Explore",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { 
        background-color: #f5f5dc; /* Cream white */
        color: #333333; /* Dark text color */
      }
      [data-testid="stSidebar"] { 
        background-color: #f5f5dc; 
      }
      .intro-card {
        background-color: #ffffff; /* White */
        border: 1px solid #cccccc; /* Light grey border */
        border-radius: 14px; 
        padding: 18px 20px; 
        margin-bottom: 14px;
      }
      .explain {
        color: #555555; 
        font-size: 0.92rem; 
        line-height: 1.55;
      }
      .news-card {
        background-color: #ffffff; /* White */
        border: 1px solid #cccccc; /* Light grey border */
        border-radius: 14px; 
        padding: 16px; 
        margin-top: 8px;
      }
      div[data-testid="stMetric"] {
        background-color: #ffffff; /* White */
        border: 1px solid #cccccc; /* Light grey border */
        padding: 14px; 
        border-radius: 12px;
      }
      h1, h2, h3 { 
        color: #333333; /* Dark text color */
      }
      a { 
        color: #1e90ff !important; /* Bright blue */
      }
    </style>
    """,
    unsafe_allow_html=True
)
    


WATCHLIST = Path("watchlist.json")


# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_history(ticker: str, period: str):
    return yf.Ticker(ticker).history(period=period, auto_adjust=False)


@st.cache_data(ttl=3600)
def get_company_info(ticker: str):
    return yf.Ticker(ticker).info or {}


@st.cache_data(ttl=1800)
def get_company_news(ticker: str):
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


def get_news_fields(item):
    """Yahoo Finance has returned more than one news-item format."""
    content = item.get("content", item) if isinstance(item, dict) else {}
    title = content.get("title") or item.get("title") or "Company news"
    summary = content.get("summary") or item.get("summary") or ""
    link = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}
    if isinstance(link, dict):
        link = link.get("url", "")
    if not link:
        link = item.get("link", "")
    provider = content.get("provider") or item.get("publisher") or {}
    if isinstance(provider, dict):
        provider = provider.get("displayName", "News source")
    pub_date = content.get("pubDate") or item.get("providerPublishTime")
    if isinstance(pub_date, (int, float)):
        pub_date = pd.to_datetime(pub_date, unit="s", utc=True).strftime("%d %b %Y")
    elif pub_date:
        pub_date = str(pub_date)[:10]

    thumbnail = content.get("thumbnail") or item.get("thumbnail") or {}
    image_url = ""
    if isinstance(thumbnail, dict):
        resolutions = thumbnail.get("resolutions", [])
        if resolutions:
            image_url = resolutions[0].get("url", "")
        image_url = image_url or thumbnail.get("url", "")
    return title, summary, link, provider, pub_date, image_url


def calculate_rsi(series, window=14):
    change = series.diff()
    gains = change.clip(lower=0).rolling(window).mean()
    losses = (-change.clip(upper=0)).rolling(window).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def add_indicators(data):
    data = data.copy()
    for window in (20, 50, 200):
        data[f"MA{window}"] = data["Close"].rolling(window).mean()

    middle = data["Close"].rolling(20).mean()
    deviation = data["Close"].rolling(20).std()
    data["BBM"] = middle
    data["BBU"] = middle + 2 * deviation
    data["BBL"] = middle - 2 * deviation
    data["RSI"] = calculate_rsi(data["Close"])

    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = short_ema - long_ema
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data


def load_watchlist():
    try:
        return json.loads(WATCHLIST.read_text()) if WATCHLIST.exists() else []
    except (json.JSONDecodeError, OSError):
        return []


def save_watchlist(items):
    WATCHLIST.write_text(json.dumps(sorted(set(items))))


def money(value, currency="USD"):
    if value is None or not np.isfinite(value):
        return "Not available"
    return f"{currency} {value:,.2f}"


def compact_number(value):
    if value is None or not np.isfinite(value):
        return "Not available"
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.2f}{suffix}"
    return f"{value:,.0f}"


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.title("Stock explorer")
    st.caption("Choose a company to learn how its stock works.")
    ticker = st.text_input("Insert Here one Company ticker to Start, e.g AAPL for Apple inc. or NVDA for Nvidia Co. and press enter", "AAPL").strip().upper()
    period_label = st.selectbox(
        "How much history should we show?",
        ["1 month", "3 months", "6 months", "1 year", "2 years", "5 years"],
        index=3,
    )
    period_map = {
        "1 month": "1mo", "3 months": "3mo", "6 months": "6mo",
        "1 year": "1y", "2 years": "2y", "5 years": "5y",
    }
    show_ma20 = st.checkbox("Show short-term trend (20 days)", True)
    show_ma50 = st.checkbox("Show medium-term trend (50 days)", False)
    show_ma200 = st.checkbox("Show long-term trend (200 days)", False)
    forecast_days = st.selectbox("Forecast horizon (trading days)", [7, 14, 30], index=1)
    st.divider()
    st.caption("A ticker is a short code used to identify a listed company. For example, AAPL is Apple.")


# ------------------------------------------------------------
# Load selected company
# ------------------------------------------------------------
if not ticker:
    st.info("Enter a company ticker in the sidebar to begin.")
    st.stop()

try:
    info = get_company_info(ticker)
    history = get_history(ticker, period_map[period_label])
    if history.empty:
        st.error("No price data was found. Check the ticker and try again.")
        st.stop()
except Exception as error:
    st.error(f"Could not load company data: {error}")
    st.stop()

data = add_indicators(history)
company_name = info.get("longName") or info.get("shortName") or ticker
currency = info.get("currency") or "USD"
logo_url = info.get("logo_url") or info.get("logoUrl")
website = info.get("website", "")
business_description = info.get("longBusinessSummary") or (
    "A company description is not available from the data provider."
)


# ------------------------------------------------------------
# Company identity and beginner introduction
# ------------------------------------------------------------
logo_col, name_col = st.columns([1, 7], vertical_alignment="center")
with logo_col:
    if logo_url:
        st.image(logo_url, width=76)
    else:
        st.markdown(f"## {ticker}")
with name_col:
    st.title(company_name)
    st.caption(f"{ticker}  ·  {info.get('sector', 'Sector not available')}  ·  {info.get('industry', 'Industry not available')}")

st.markdown(
    f"""
    <div class="intro-card">
      <h3>What does this company do?</h3>
      <div class="explain">{business_description}</div>
      <p class="explain"><b>In simple terms:</b> A stock is a small ownership share in a company.
      When you buy {ticker}, you are buying a tiny part of this business.
      Its market price changes as buyers and sellers react to information and expectations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, charts_tab, forecast_tab, watchlist_tab, learn_tab = st.tabs(
    ["Company overview", "Price & charts", "Forecast lab", "My watchlist", "Learn the basics"]
)


# ------------------------------------------------------------
# Company overview
# ------------------------------------------------------------
with overview_tab:
    st.header("The company at a glance")
    st.caption("These figures are a snapshot. They change over time and are not a recommendation to buy or sell.")

    latest_price = float(data["Close"].dropna().iloc[-1])
    previous_price = float(data["Close"].dropna().iloc[-2]) if len(data) > 1 else latest_price
    daily_change = latest_price - previous_price
    daily_change_pct = (daily_change / previous_price * 100) if previous_price else 0

    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE")
    beta = info.get("beta")
    dividend_yield = info.get("dividendYield")

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Latest share price", money(latest_price, currency),
        f"{daily_change_pct:+.2f}% vs previous close",
        help="The latest available closing price for one share. The percentage shows the change from the previous trading day's close.",
    )
    metric_cols[1].metric(
        "Company size", compact_number(market_cap),
        help="Market value: share price multiplied by all shares. It is one way to estimate how large the company is.",
    )
    metric_cols[2].metric(
        "P/E ratio", f"{pe_ratio:.1f}" if pe_ratio else "Not available",
        help="Price-to-earnings ratio. Roughly, how many dollars investors pay for each dollar the company earns in a year.",
    )
    metric_cols[3].metric(
        "Beta", f"{beta:.2f}" if beta is not None else "Not available",
        help="A rough measure of how much the share price has moved compared with the wider market. Around 1 means similar historical movement; above 1 means larger movement.",
    )

    with st.expander("Understand these four numbers"):
        st.markdown("""
        - **Share price:** the price of one share. A lower share price does not automatically mean a company is cheaper.
        - **Company size (market value):** the estimated total market value of all its shares. Bigger does not automatically mean safer.
        - **P/E ratio:** compares the share price with company earnings. A high value can mean investors expect growth, or that the share is expensive. It depends on the business and industry.
        - **Beta:** describes past price movement relative to the market. It does not predict what will happen next.
        """)

    st.subheader("One piece of company news")
    news_items = get_company_news(ticker)
    if news_items:
        title, summary, link, provider, published, image_url = get_news_fields(news_items[0])
        news_left, news_right = st.columns([1, 1.7])
        with news_left:
            if image_url:
                st.image(image_url, use_container_width=True)
            else:
                st.info("The news provider did not supply an image for this story.")
        with news_right:
            st.markdown(f"### {title}")
            if published:
                st.caption(f"{provider} · {published}")
            else:
                st.caption(provider)
            st.write(summary if summary else "A short summary was not supplied by the news provider. Open the original story for details.")
            if link:
                st.markdown(f"[Read the full story]({link})")
        st.caption("This is one recent story returned by the news provider. A headline is not the full story; read the source before drawing conclusions.")
    else:
        st.info("No recent news story is available for this ticker right now.")

    st.subheader("What is this company worth?")
    st.write(
        f"**Market capitalisation:** {compact_number(market_cap)} {currency if market_cap else ''}. "
        "This is the stock market's estimated value of all the company's shares combined. "
        "It is not the same as the cash the company has in its bank account."
    )
    if dividend_yield is not None:
        st.write(
            f"**Dividend yield:** {dividend_yield * 100:.2f}%. "
            "A dividend is money a company may pay to shareholders. Yield compares the yearly dividend with the share price. "
            "Dividends can change or stop."
        )

    if website:
        st.markdown(f"Company website: [{website}]({website})")


# ------------------------------------------------------------
# Price and charts
# ------------------------------------------------------------
with charts_tab:
    st.header("How has the share price moved?")
    st.write(
        "This chart shows the price people paid for one share over time. "
        "The candles summarise each trading period: where the price started, ended, and how high or low it moved."
    )

    chart = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.72, 0.28],
    )
    chart.add_trace(
        go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Share price",
        ),
        row=1, col=1,
    )
    if show_ma20:
        chart.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="20-day average"), row=1, col=1)
    if show_ma50:
        chart.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="50-day average"), row=1, col=1)
    if show_ma200:
        chart.add_trace(go.Scatter(x=data.index, y=data["MA200"], name="200-day average"), row=1, col=1)
    chart.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Shares traded"), row=2, col=1)
    chart.update_layout(
        template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02), margin=dict(l=10, r=10, t=50, b=10),
    )
    chart.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    chart.update_yaxes(title_text="Trading volume", row=2, col=1)
    st.plotly_chart(chart, use_container_width=True)

    with st.expander("How to read this chart"):
        st.markdown("""
        - **Each candle:** shows the opening, highest, lowest, and closing price for that time period.
        - **Green candle:** the price ended higher than it started. **Red candle:** it ended lower.
        - **Trading volume:** how many shares changed hands. High volume means more shares were traded; it does not tell us by itself whether the price should rise or fall.
        - **Moving average:** the average closing price over a chosen number of trading days. It smooths daily ups and downs so the general direction is easier to see.
        """)

    st.subheader("Two indicators, explained simply")
    left, right = st.columns(2)
    with left:
        st.markdown("**RSI — recent buying and selling strength**")
        st.line_chart(data["RSI"].dropna())
        rsi_value = data["RSI"].dropna().iloc[-1] if not data["RSI"].dropna().empty else np.nan
        if np.isfinite(rsi_value):
            st.write(f"Latest RSI: **{rsi_value:.1f} out of 100**")
        st.caption("RSI is a momentum indicator. Above 70 is often called 'overbought' and below 30 'oversold'. These are warning zones, not automatic buy or sell signals.")
    with right:
        st.markdown("**MACD — a way to compare price trends**")
        st.line_chart(data[["MACD", "Signal"]].dropna())
        st.caption("MACD compares two smoothed price trends. When its lines cross, some traders watch for a possible change in momentum. Crossings can also produce false signals.")

    st.download_button(
        "Download the data (CSV)",
        data.to_csv().encode("utf-8"),
        file_name=f"{ticker.lower()}_historical_data.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# Forecasting
# ------------------------------------------------------------
with forecast_tab:
    st.header("Forecast lab: what might happen next?")
    st.warning(
        "A model is making an educated calculation from past prices, not seeing the future. "
        "Forecasts can be very wrong. This section is for learning, not investment decisions."
    )
    st.write(
        "The models below look for patterns in historical closing prices and extend those patterns forward. "
        "They do not know future news, company results, economic changes, or unexpected events."
    )

    clean = data[["Close"]].dropna().reset_index(drop=True)
    if len(clean) < 40:
        st.info("There is not enough historical data for a useful demonstration. Choose a longer history period.")
    else:
        clean["day_number"] = np.arange(len(clean))
        features = clean[["day_number"]]
        target = clean["Close"]
        split_at = max(int(len(clean) * 0.8), 1)
        x_train, x_test = features.iloc[:split_at], features.iloc[split_at:]
        y_train, y_test = target.iloc[:split_at], target.iloc[split_at:]

        models = {
            "Straight-line model": LinearRegression(),
            "Random Forest model": RandomForestRegressor(n_estimators=200, random_state=42),
        }
        results = []
        predictions = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            test_prediction = model.predict(x_test)
            results.append({
                "Model": model_name,
                "Typical test error (RMSE)": mean_squared_error(y_test, test_prediction) ** 0.5,
                "Average absolute error (MAE)": mean_absolute_error(y_test, test_prediction),
                "Fit score (R²)": r2_score(y_test, test_prediction) if len(y_test) > 1 else np.nan,
            })
            future_x = pd.DataFrame({"day_number": np.arange(len(clean), len(clean) + forecast_days)})
            predictions[model_name] = model.predict(future_x)

        results_df = pd.DataFrame(results)
        st.subheader("How well did the models do on past data?")
        st.dataframe(
            results_df.style.format({
                "Typical test error (RMSE)": "{:.2f}",
                "Average absolute error (MAE)": "{:.2f}",
                "Fit score (R²)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("What do these model scores mean?"):
            st.markdown("""
            - **Typical test error (RMSE):** roughly how far predictions were from actual prices in the test period. Smaller is generally better.
            - **Average absolute error (MAE):** the average size of the model's mistake, in the share's currency. Smaller is generally better.
            - **Fit score (R²):** compares the model with a simple baseline. A value closer to 1 can indicate a better fit on this test set; a negative value means it performed worse than predicting the test-period average.
            - **Important:** a model that performed best on past data is not necessarily the model that will perform best in the future.
            """)

        future_dates = pd.bdate_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
        )
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], name="Past closing price",
            line=dict(width=2),
        ))
        for model_name, values in predictions.items():
            forecast_fig.add_trace(go.Scatter(
                x=future_dates, y=values, name=f"{model_name} estimate",
                line=dict(dash="dash"),
            ))
        forecast_fig.update_layout(
            template="plotly_dark", height=470,
            title="Past prices and model estimates",
            xaxis_title="Date", yaxis_title=f"Price ({currency})",
        )
        st.plotly_chart(forecast_fig, use_container_width=True)
        st.caption("Dashed lines are model estimates, not promised prices or a range of guaranteed outcomes.")

        forecast_table = pd.DataFrame({"Date": future_dates})
        for model_name, values in predictions.items():
            forecast_table[model_name] = values
        st.download_button(
            "Download forecast (CSV)",
            forecast_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker.lower()}_educational_forecast.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# Watchlist
# ------------------------------------------------------------
with watchlist_tab:
    st.header("My watchlist")
    st.write("A watchlist is simply a list of companies you want to keep an eye on. Adding a company does not buy its shares.")
    watchlist = load_watchlist()
    new_ticker = st.text_input("Add a ticker to your list", key="watchlist_new").strip().upper()
    if st.button("Add company") and new_ticker:
        if new_ticker not in watchlist:
            watchlist.append(new_ticker)
            save_watchlist(watchlist)
        st.rerun()

    if not watchlist:
        st.info("Your watchlist is empty. Add a ticker above.")
    else:
        for item in watchlist:
            col_name, col_remove = st.columns([4, 1])
            with col_name:
                st.write(f"**{item}**")
                try:
                    item_info = get_company_info(item)
                    item_price = item_info.get("currentPrice")
                    item_currency = item_info.get("currency") or "USD"
                    st.caption(f"{item_info.get('longName', item)} · Latest reported price: {money(item_price, item_currency)}")
                except Exception:
                    st.caption("Company details are currently unavailable.")
            with col_remove:
                if st.button("Remove", key=f"remove_{item}"):
                    watchlist.remove(item)
                    save_watchlist(watchlist)
                    st.rerun()


# ------------------------------------------------------------
# Beginner education
# ------------------------------------------------------------
with learn_tab:
    st.header("Financial markets — the beginner's guide")
    st.write("Start here if terms such as stock, market value, or RSI are new to you.")

    lessons = [
        ("What is a stock?",
         "A stock is a small ownership share in a company. If you own a share, you own a very small part of that business. The share price can rise or fall."),
        ("Why do stock prices change?",
         "Prices move because buyers and sellers disagree about what a share is worth. Company profits, expectations, interest rates, the economy, and news can all influence those decisions."),
        ("What is a stock market?",
         "A stock market is a system where investors buy and sell shares. A stock exchange, such as the New York Stock Exchange, is one place where those trades happen."),
        ("What is market capitalisation?",
         "Market capitalisation, or market cap, is the share price multiplied by the total number of shares. It estimates the market value of the whole company."),
        ("What is a dividend?",
         "A dividend is money a company may distribute to its shareholders. Not every company pays dividends, and payments can change or stop."),
        ("What is a candlestick?",
         "A candlestick is a compact picture of trading during a period. It shows the opening price, closing price, highest price, and lowest price."),
        ("What is a moving average?",
         "A moving average is the average closing price over a chosen number of trading days. It smooths short-term movements to make a trend easier to see."),
        ("What is RSI?",
         "The Relative Strength Index (RSI) is a score from 0 to 100 that summarises recent price momentum. Some traders watch readings above 70 or below 30, but these are not instructions to trade."),
        ("What is MACD?",
         "Moving Average Convergence Divergence (MACD) compares two smoothed price trends. Traders use it to study momentum and possible trend changes. It can give misleading signals."),
        ("What is a Bollinger Band?",
         "Bollinger Bands draw a middle moving average and two outer lines based on recent price variation. Wider bands mean prices have been more variable; narrow bands mean they have been quieter."),
        ("What is a forecast model?",
         "A forecast model uses data and a mathematical method to estimate possible future values. It can make mistakes, especially when the future differs from the past."),
        ("What is diversification?",
         "Diversification means spreading investments across different companies or types of assets instead of relying on just one. It can reduce some risks, but it cannot remove all risk."),
        ("What does risk mean?",
         "Risk is the possibility that an investment loses value or behaves differently from what you expected. Never invest money you cannot afford to lose."),
    ]
    for question, answer in lessons:
        with st.expander(question):
            st.write(answer)

    st.info(
        "Remember: a company can be excellent while its stock is overpriced; "
        "a stock can rise or fall for reasons unrelated to the company's long-term quality. "
        "This dashboard is an educational tool, not personalised financial advice."
    )

st.divider()
st.caption(
    "Data supplied by Yahoo Finance via yfinance. Market data may be delayed, incomplete, or revised. "
    "This application is for educational purposes only and is not financial advice."
)
