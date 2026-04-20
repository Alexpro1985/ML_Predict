import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from prophet import Prophet


# ---------- LOAD ----------
def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    if df is None or df.empty:
        return None

    df = df.reset_index()[["Date", "Close"]].dropna()
    df.columns = ["ds", "y"]

    return df


# ---------- PROPHET ----------
def run_prophet(df, future_days=365):
    df = df.copy()
    df["y"] = df["y"].astype(float)

    model = Prophet(interval_width=0.95)  # 95% confidence interval
    model.fit(df)

    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)

    last = forecast.iloc[-1]

    pred = float(last["yhat"])
    lower = float(last["yhat_lower"])
    upper = float(last["yhat_upper"])

    # ---------- confidence heuristic ----------
    # чем уже диапазон — тем выше "уверенность"
    range_width = upper - lower
    confidence = max(0, 1 - (range_width / abs(pred))) * 100

    return pred, confidence


# ---------- MAIN ----------
tickers = ["AAPL", "BTC-USD", "EURUSD=X"]
start_date = "2020-01-01"
end_date = "2027-01-01"

for ticker in tqdm(tickers, desc="Processing assets"):
    df = download_data(ticker, start_date, end_date)

    if df is None:
        continue

    print("\n" + "=" * 60)
    print(f"Ticker: {ticker}")

    try:
        pred, conf = run_prophet(df)

        print(f"Forecast: {pred:.4f}")
        print(f"Confidence: {conf:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
