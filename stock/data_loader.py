"""
data_loader.py
Module for downloading, cleaning, and preparing time-series stock data for LSTM.

Designed for beginner-friendly understanding:
- Uses Yahoo Finance (yfinance) to fetch historical stock prices.
- Supports Indian stocks (Tata Motors, Bharti Airtel, State Bank of India).
- Prevents data leakage by fitting MinMaxScaler strictly on training data.
- Generates 60-day sequence lookback windows for LSTM training and evaluation.
"""

import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# Company configurations with ticker fallbacks
COMPANIES = {
    "TATAMOTORS": {
        "name": "Tata Motors",
        "primary_ticker": "TMPV.NS",      # Tata Motors Passenger Vehicles (post-demerger primary active ticker)
        "fallback_tickers": ["TATAMOTORS.NS", "TMCV.NS", "TATAMOTORS.BO"],
        "industry": "Automobile / EV",
        "description": "Leading Indian multinational automotive manufacturing company."
    },
    "BHARTIARTL": {
        "name": "Bharti Airtel",
        "primary_ticker": "BHARTIARTL.NS",
        "fallback_tickers": ["BHARTIARTL.BO"],
        "industry": "Telecommunications",
        "description": "Major Indian multinational telecommunications services provider."
    },
    "SBIN": {
        "name": "State Bank of India (SBI)",
        "primary_ticker": "SBIN.NS",
        "fallback_tickers": ["SBIN.BO"],
        "industry": "Banking / Public Financial Services",
        "description": "India's largest public sector banking and financial services statutory body."
    }
}


def get_available_companies():
    """Return list of supported companies with ticker information."""
    return [
        {
            "id": key,
            "name": val["name"],
            "ticker": val["primary_ticker"],
            "industry": val["industry"],
            "description": val["description"]
        }
        for key, val in COMPANIES.items()
    ]


def fetch_stock_raw_data(ticker_symbol, period="2y"):
    """
    Download historical stock data from Yahoo Finance with fallback handling.

    Parameters:
    - ticker_symbol (str): The Yahoo Finance ticker symbol (e.g. 'TMPV.NS', 'BHARTIARTL.NS').
    - period (str): Historical period (e.g. '1mo', '6mo', '1y', '2y', '5y').

    Returns:
    - (pd.DataFrame, str): The cleaned DataFrame and the active ticker resolved.
    """
    tickers_to_try = [ticker_symbol]

    # Check if there are known fallbacks for this ticker
    for key, info in COMPANIES.items():
        if ticker_symbol in [info["primary_ticker"], key] + info.get("fallback_tickers", []):
            tickers_to_try = [info["primary_ticker"]] + info.get("fallback_tickers", [])
            break

    df = pd.DataFrame()
    resolved_ticker = ticker_symbol

    for sym in tickers_to_try:
        try:
            ticker_obj = yf.Ticker(sym)
            temp_df = ticker_obj.history(period=period)
            if temp_df is not None and not temp_df.empty and len(temp_df) > 65:
                df = temp_df
                resolved_ticker = sym
                break
        except Exception as e:
            print(f"[DataLoader] Warning: Could not fetch {sym}: {e}")
            continue

    if df.empty or len(df) <= 65:
        # Fallback to longer period if short period had insufficient data
        for sym in tickers_to_try:
            try:
                temp_df = yf.Ticker(sym).history(period="2y")
                if temp_df is not None and not temp_df.empty and len(temp_df) > 65:
                    df = temp_df
                    resolved_ticker = sym
                    break
            except Exception:
                continue

    if df.empty:
        raise ValueError(f"Unable to download historical data for {ticker_symbol}. Please check internet connection or ticker.")

    # Clean data: reset index, sort chronologically, remove nulls
    df = df.reset_index()
    
    # Standardize column names
    if "Date" not in df.columns and "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    
    # Strip timezones from Date for clean JSON serialization
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])

    return df, resolved_ticker


def get_stock_summary(df, company_id=None, resolved_ticker=""):
    """
    Generate current market summary metrics from the DataFrame.
    """
    if df.empty:
        return {}

    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else latest_row

    latest_close = float(latest_row["Close"])
    prev_close = float(prev_row["Close"])
    change = latest_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0

    high_52w = float(df["High"].max())
    low_52w = float(df["Low"].min())
    day_high = float(latest_row["High"])
    day_low = float(latest_row["Low"])
    volume = int(latest_row.get("Volume", 0))

    company_name = resolved_ticker
    if company_id and company_id in COMPANIES:
        company_name = COMPANIES[company_id]["name"]

    return {
        "company_name": company_name,
        "ticker": resolved_ticker,
        "latest_close": round(latest_close, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "day_high": round(day_high, 2),
        "day_low": round(day_low, 2),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "volume": volume,
        "latest_date": latest_row["Date"].strftime("%Y-%m-%d"),
        "currency": "INR"
    }


def prepare_lstm_data(df, sequence_length=60, train_ratio=0.8):
    """
    Prepares chronological train and test sequences for the LSTM model.

    To strictly prevent Data Leakage:
    1. We split the data chronologically (first train_ratio for training, remaining for testing).
    2. We fit the MinMaxScaler ONLY on the training portion.
    3. We then transform the test portion using the scaler fitted on training data.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.
    - sequence_length (int): Lookback window in trading days (default: 60).
    - train_ratio (float): Chronological training split proportion (default: 0.8).

    Returns:
    - dict: Contains scaled X_train, y_train, X_test, y_test, scaler, actual test prices, and dates.
    """
    if len(df) < sequence_length + 20:
        raise ValueError(f"Insufficient historical data: {len(df)} days found, need at least {sequence_length + 20}.")

    # Extract Close prices as 2D array for scaler
    close_prices = df["Close"].values.reshape(-1, 1)
    dates = df["Date"].values

    total_len = len(close_prices)
    train_size = int(total_len * train_ratio)

    # Chronological Split: strictly past data for training, future data for testing
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size - sequence_length:]  # Include lookback for the first test point

    # FIT SCALER ONLY ON TRAIN DATA (CRITICAL TO AVOID DATA LEAKAGE)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    # 1. Create Training Sequences
    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_train)):
        X_train.append(scaled_train[i - sequence_length:i, 0])
        y_train.append(scaled_train[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshape X for LSTM: [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 2. Create Testing Sequences
    X_test, y_test = [], []
    test_dates = []
    actual_test_prices = []

    for i in range(sequence_length, len(scaled_test)):
        X_test.append(scaled_test[i - sequence_length:i, 0])
        y_test.append(scaled_test[i, 0])
        # The corresponding date index in original df is train_size + (i - sequence_length)
        idx = train_size + (i - sequence_length)
        test_dates.append(pd.to_datetime(dates[idx]).strftime("%Y-%m-%d"))
        actual_test_prices.append(float(close_prices[idx][0]))

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "test_dates": test_dates,
        "actual_test_prices": actual_test_prices,
        "full_scaled_close": scaler.transform(close_prices),
        "full_dates": [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates],
        "full_close": [float(c[0]) for c in close_prices],
        "sequence_length": sequence_length
    }
