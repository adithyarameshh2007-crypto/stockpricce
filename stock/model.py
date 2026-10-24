"""
model.py
Module for building, training, evaluating, and predicting with the LSTM Neural Network.

Features:
- Standard 2-layer LSTM architecture with Dropout regularization.
- Inverse-transforms predictions to original price scale (₹ INR).
- Calculates MAE, RMSE, and MAPE evaluation metrics on unseen test data.
- Multi-step iterative future forecasting for upcoming trading days.
- Model caching to provide instantaneous responses after initial training.
"""

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress verbose TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_lstm_model(sequence_length=60, feature_count=1):
    """
    Constructs a beginner-friendly, high-performance Sequential LSTM model.

    Architecture:
    1. Input Layer: (sequence_length=60, feature_count=1)
    2. LSTM Layer 1: 50 memory cells with return_sequences=True (passes sequence to next layer)
    3. Dropout Layer 1: 20% dropout to prevent overfitting
    4. LSTM Layer 2: 50 memory cells with return_sequences=False (condenses features)
    5. Dropout Layer 2: 20% dropout
    6. Dense Layer: 25 neurons with ReLU activation
    7. Output Layer: 1 neuron producing the predicted closing price for the next day
    """
    model = Sequential([
        Input(shape=(sequence_length, feature_count)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25, activation="relu"),
        Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error"
    )
    return model


def get_model_file_path(ticker):
    """Clean ticker name and return saved model path."""
    clean_ticker = ticker.replace(".", "_").replace(":", "_").replace("^", "")
    return os.path.join(MODELS_DIR, f"{clean_ticker}_lstm.keras")


def train_or_load_model(ticker, X_train, y_train, force_retrain=False, epochs=12, batch_size=32):
    """
    Loads an existing trained model for the ticker or trains a new one.

    Parameters:
    - ticker (str): Stock ticker (e.g. 'TMPV.NS').
    - X_train (np.ndarray): 3D input sequence array.
    - y_train (np.ndarray): 1D target closing prices.
    - force_retrain (bool): If True, re-trains even if saved model exists.
    - epochs (int): Training iterations.
    - batch_size (int): Batch size per gradient update.

    Returns:
    - (model, bool): The trained/loaded Keras model and whether it was newly trained.
    """
    model_path = get_model_file_path(ticker)

    if not force_retrain and os.path.exists(model_path):
        try:
            print(f"[Model] Loading pre-trained model for {ticker} from {model_path}...")
            model = load_model(model_path)
            return model, False
        except Exception as e:
            print(f"[Model] Warning: Could not load saved model ({e}), retraining...")

    print(f"[Model] Training new LSTM model for {ticker} on {len(X_train)} samples...")
    model = build_lstm_model(sequence_length=X_train.shape[1], feature_count=X_train.shape[2])

    early_stopping = EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    try:
        model.save(model_path)
        print(f"[Model] Model successfully saved to {model_path}.")
    except Exception as e:
        print(f"[Model] Warning: Failed to save model: {e}")

    return model, True


def evaluate_model(model, X_test, y_test, scaler, actual_prices, test_dates):
    """
    Evaluates the LSTM model on chronological test data.

    Returns:
    - metrics: MAE, RMSE, MAPE, accuracy percentage.
    - test_predictions: List of dicts with date, actual price, and predicted price.
    """
    # Predict scaled values
    scaled_predictions = model.predict(X_test, verbose=0)

    # Inverse transform to original ₹ INR price scale
    predicted_prices = scaler.inverse_transform(scaled_predictions).flatten()
    actual_prices = np.array(actual_prices)

    # Calculate Evaluation Metrics
    mae = float(mean_absolute_error(actual_prices, predicted_prices))
    rmse = float(np.sqrt(mean_squared_error(actual_prices, predicted_prices)))
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = float(np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100)
    
    # Directional Accuracy (% of days where predicted direction matched actual)
    if len(actual_prices) > 1:
        actual_diff = np.diff(actual_prices)
        pred_diff = np.diff(predicted_prices)
        correct_direction = np.sum((actual_diff * pred_diff) > 0)
        directional_accuracy = round(float((correct_direction / len(actual_diff)) * 100), 1)
    else:
        directional_accuracy = 100.0

    test_results = []
    for d, act, pred in zip(test_dates, actual_prices, predicted_prices):
        test_results.append({
            "date": d,
            "actual": round(float(act), 2),
            "predicted": round(float(pred), 2)
        })

    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "directional_accuracy": directional_accuracy,
        "sample_count": len(test_results)
    }

    return metrics, test_results


def predict_future(model, full_scaled_close, scaler, latest_date_str, n_days=7, rmse_error=0.0):
    """
    Generates multi-step future price forecasts using recursive prediction.

    Parameters:
    - model: Trained LSTM model.
    - full_scaled_close: Scaled close prices array.
    - scaler: Fitted MinMaxScaler.
    - latest_date_str (str): Latest trading date in YYYY-MM-DD.
    - n_days (int): Number of future trading days to predict.
    - rmse_error (float): Model's RMSE to calculate confidence interval bands.

    Returns:
    - list of dicts: [ { "date", "day_name", "predicted_price", "lower_bound", "upper_bound" }, ... ]
    """
    sequence_length = 60
    if len(full_scaled_close) < sequence_length:
        raise ValueError("Not enough data to create 60-day lookback window for future prediction.")

    # Start with the last 60 known days
    current_sequence = full_scaled_close[-sequence_length:].reshape(1, sequence_length, 1).copy()

    latest_date = datetime.datetime.strptime(latest_date_str, "%Y-%m-%d").date()
    future_predictions = []
    curr_date = latest_date

    for day_idx in range(1, n_days + 1):
        # Move to next trading day (skip Saturday = 5, Sunday = 6)
        curr_date += datetime.timedelta(days=1)
        while curr_date.weekday() >= 5:
            curr_date += datetime.timedelta(days=1)

        # Predict next 1 day
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        predicted_scaled_val = next_pred_scaled[0, 0]

        # Inverse transform to ₹ INR
        predicted_price = float(scaler.inverse_transform([[predicted_scaled_val]])[0, 0])

        # Estimate confidence interval using cumulative uncertainty
        uncertainty = rmse_error * (1 + 0.1 * day_idx) if rmse_error > 0 else predicted_price * 0.02
        lower_bound = max(0.0, predicted_price - uncertainty)
        upper_bound = predicted_price + uncertainty

        future_predictions.append({
            "step": day_idx,
            "date": curr_date.strftime("%Y-%m-%d"),
            "day_name": curr_date.strftime("%A"),
            "predicted_price": round(predicted_price, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        })

        # Slide sequence window: remove oldest, append newest predicted value
        new_point = np.array([[[predicted_scaled_val]]])
        current_sequence = np.append(current_sequence[:, 1:, :], new_point, axis=1)

    return future_predictions
