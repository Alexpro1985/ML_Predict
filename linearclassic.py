import datetime as dt
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def download_ticker_data(ticker_code, start_date, end_date):
    try:
        data = yf.download(ticker_code, start=start_date, end=end_date)
        return data
    except Exception:
        return None


def prepare_features(data):
    data = data.reset_index()
    data["Date_ordinal"] = data["Date"].map(dt.datetime.toordinal)

    X = data["Date_ordinal"].values.reshape(-1, 1)
    y = data["Close"].values.reshape(-1, 1)

    return X, y


def predict_ticker_price(ticker_data, future_date):
    X, y = prepare_features(ticker_data)

    model = LinearRegression()
    model.fit(X, y)

    future_x = dt.datetime.strptime(future_date, "%Y-%m-%d").toordinal()
    predicted_price = model.predict(np.array([[future_x]]))
    score = model.score(X, y)

    return predicted_price, score


tickers = ["AAPL", "EURUSD=X", "GOLD", "BTC-USD"]
start_date = "2020-01-01"
end_date = "2024-01-01"
future_date = "2027-01-01"


for ticker in tqdm(tickers, desc="Processing tickers"):
    data = download_ticker_data(ticker.strip(), start_date, end_date)

    if data is None or data.empty:
        continue

    price, score = predict_ticker_price(data, future_date)

    price = float(price.item())
    score = float(score)

    print(
        f"{ticker} -> {future_date}: "
        f"{price:.5f}, confidence: {score * 100:.2f}%"
    )
