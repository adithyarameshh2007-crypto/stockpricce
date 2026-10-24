"""
app.py
Flask Web Application for Stock Price Prediction System using LSTM.

Indian Companies Supported:
- Tata Motors (NSE: TMPV.NS / TATAMOTORS)
- Bharti Airtel (NSE: BHARTIARTL.NS)
- State Bank of India (NSE: SBIN.NS)

Features:
- RESTful API endpoints for historical data, LSTM training, evaluation, and future predictions.
- Robust error handling for network dropouts and delisted/modified tickers.
- Clean JSON responses for interactive Chart.js frontend.
"""

import os
import datetime
import traceback
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

import data_loader
import model as lstm_engine

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index():
    """Render main application dashboard."""
    return render_template("index.html")


@app.route("/api/companies", methods=["GET"])
def api_get_companies():
    """Return list of supported Indian companies."""
    try:
        companies = data_loader.get_available_companies()
        return jsonify({"success": True, "companies": companies})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stock-data", methods=["GET"])
def api_get_stock_data():
    """
    Fetch historical stock data for the selected company and period.
    Query parameters:
    - company: Company key (e.g. 'TATAMOTORS', 'BHARTIARTL', 'SBIN')
    - period: '1mo', '6mo', '1y', '2y' (default: '1y')
    """
    company_id = request.args.get("company", "TATAMOTORS").strip().upper()
    period = request.args.get("period", "1y").strip().lower()

    valid_periods = ["1mo", "6mo", "1y", "2y", "5y"]
    if period not in valid_periods:
        period = "1y"

    try:
        df, resolved_ticker = data_loader.fetch_stock_raw_data(company_id, period=period)
        summary = data_loader.get_stock_summary(df, company_id=company_id, resolved_ticker=resolved_ticker)

        # Format historical records for Chart.js
        history = []
        for _, row in df.iterrows():
            history.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "volume": int(row.get("Volume", 0))
            })

        return jsonify({
            "success": True,
            "company_id": company_id,
            "resolved_ticker": resolved_ticker,
            "period": period,
            "summary": summary,
            "history": history
        })

    except Exception as e:
        print(f"[App Error] Stock Data: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Failed to load stock data: {str(e)}"
        }), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Run or load LSTM model, evaluate on test set, and forecast future prices.
    Request body (JSON):
    {
        "company": "TATAMOTORS",
        "period": "2y",
        "force_retrain": false,
        "forecast_days": 7
    }
    """
    data = request.get_json() or {}
    company_id = data.get("company", "TATAMOTORS").strip().upper()
    period = data.get("period", "2y").strip().lower()
    force_retrain = bool(data.get("force_retrain", False))
    forecast_days = int(data.get("forecast_days", 7))

    # Clamp forecast days between 1 and 14 days
    forecast_days = max(1, min(forecast_days, 14))

    try:
        # 1. Download clean historical data (at least 2y recommended for optimal sequence depth)
        df, resolved_ticker = data_loader.fetch_stock_raw_data(company_id, period=period)
        summary = data_loader.get_stock_summary(df, company_id=company_id, resolved_ticker=resolved_ticker)

        # 2. Chronological sequence preparation (strictly prevents data leakage)
        prepared = data_loader.prepare_lstm_data(df, sequence_length=60, train_ratio=0.8)

        # 3. Train or load pre-trained LSTM model
        model, newly_trained = lstm_engine.train_or_load_model(
            ticker=resolved_ticker,
            X_train=prepared["X_train"],
            y_train=prepared["y_train"],
            force_retrain=force_retrain,
            epochs=10
        )

        # 4. Evaluate on unseen chronological test set
        metrics, test_predictions = lstm_engine.evaluate_model(
            model=model,
            X_test=prepared["X_test"],
            y_test=prepared["y_test"],
            scaler=prepared["scaler"],
            actual_prices=prepared["actual_test_prices"],
            test_dates=prepared["test_dates"]
        )

        # 5. Forecast future trading days
        latest_date_str = prepared["full_dates"][-1]
        future_forecast = lstm_engine.predict_future(
            model=model,
            full_scaled_close=prepared["full_scaled_close"],
            scaler=prepared["scaler"],
            latest_date_str=latest_date_str,
            n_days=forecast_days,
            rmse_error=metrics["rmse"]
        )

        # Calculate expected change for next trading day
        next_day = future_forecast[0] if future_forecast else {}
        next_price = next_day.get("predicted_price", summary.get("latest_close", 0))
        curr_price = summary.get("latest_close", 0)
        exp_change = next_price - curr_price
        exp_change_pct = (exp_change / curr_price) * 100 if curr_price != 0 else 0

        forecast_summary = {
            "next_trading_day": next_day.get("date"),
            "next_trading_day_name": next_day.get("day_name"),
            "predicted_price": next_price,
            "expected_change": round(exp_change, 2),
            "expected_change_pct": round(exp_change_pct, 2),
            "trend": "bullish" if exp_change >= 0 else "bearish"
        }

        # Build full continuous historical series for seamless chart display
        full_history = [
            {"date": d, "close": c}
            for d, c in zip(prepared["full_dates"], prepared["full_close"])
        ]

        return jsonify({
            "success": True,
            "company_id": company_id,
            "resolved_ticker": resolved_ticker,
            "newly_trained": newly_trained,
            "summary": summary,
            "metrics": metrics,
            "forecast_summary": forecast_summary,
            "test_predictions": test_predictions,
            "future_forecast": future_forecast,
            "full_history": full_history
        })

    except Exception as e:
        print(f"[App Error] Predict: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Model prediction error: {str(e)}"
        }), 500


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Force re-train the LSTM model for a company."""
    data = request.get_json() or {}
    company_id = data.get("company", "TATAMOTORS").strip().upper()
    try:
        df, resolved_ticker = data_loader.fetch_stock_raw_data(company_id, period="2y")
        prepared = data_loader.prepare_lstm_data(df, sequence_length=60, train_ratio=0.8)
        model, _ = lstm_engine.train_or_load_model(
            ticker=resolved_ticker,
            X_train=prepared["X_train"],
            y_train=prepared["y_train"],
            force_retrain=True,
            epochs=12
        )
        metrics, _ = lstm_engine.evaluate_model(
            model=model,
            X_test=prepared["X_test"],
            y_test=prepared["y_test"],
            scaler=prepared["scaler"],
            actual_prices=prepared["actual_test_prices"],
            test_dates=prepared["test_dates"]
        )
        return jsonify({
            "success": True,
            "message": f"Successfully retrained model for {resolved_ticker}",
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Create saved_models folder if not present
    os.makedirs(os.path.join(os.path.dirname(__file__), "saved_models"), exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[Server] Running on http://localhost:{port} (host={host})")
    app.run(host=host, port=port, debug=False)


