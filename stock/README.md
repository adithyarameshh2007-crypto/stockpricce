# 📈 Stock Price Prediction System Using LSTM (Deep Learning)

An end-to-end, full-stack machine learning web application that downloads historical stock price data, trains an LSTM (Long Short-Term Memory) recurrent neural network, evaluates it on unseen test data, and forecasts future stock prices for top Indian equities.

---

## 🏢 Supported Indian Equities

| Company | Primary NSE Ticker | Fallback / Notes |
| :--- | :--- | :--- |
| **Tata Motors** | `TMPV.NS` | Handles `TATAMOTORS.NS` & `TMCV.NS` automatically |
| **Bharti Airtel** | `BHARTIARTL.NS` | `BHARTIARTL.BO` |
| **State Bank of India (SBI)** | `SBIN.NS` | `SBIN.BO` |

---

## 🎨 Theme & User Interface

- **Theme Palette**: Fresh Light Mint-Green (`#f4fbf8`, `#ecfdf5`) & Crisp White (`#ffffff`) with Vivid Teal accents (`#0d9488`, `#14b8a6`).
- **Charts**: Interactive [Chart.js](https://www.chartjs.org/) with:
  1. **Forecast & History**: Continuous actual prices linked seamlessly to a 7-day dashed future projection with an uncertainty confidence band ($\pm\text{RMSE}$).
  2. **Model Validation**: Side-by-side comparison of **Actual Prices vs. LSTM Predictions** on unseen test data.
  3. **Historical View**: Daily close price trends across 1M, 6M, 1Y, and 2Y timeframes.
- **7-Day Forward Projections**: Clean cards displaying estimated prices for upcoming trading days (weekends skipped).
- **Evaluation Metrics**: Real-time RMSE, MAE, MAPE (%), and Directional Trend Accuracy.
- **Educational Guide**: Step-by-step breakdown of how LSTM works for engineering students and beginners.
- **Official Disclaimer**: Clear disclaimer clarifying estimates are educational, not financial advice.

---

## 🧠 LSTM Deep Learning Architecture & Principles

### 1. Chronological Split (No Shuffling)
Time-series data has temporal order. Shuffling data causes the model to "cheat" by looking at future prices to guess past prices.
- **Split Ratio**: First **80%** chronological days for Training; latest **20%** strictly for Testing.

### 2. Preventing Data Leakage
The `MinMaxScaler` is fitted **ONLY** on the training set:
$$\text{Scaled Value} = \frac{X - X_{\text{train\_min}}}{X_{\text{train\_max}} - X_{\text{train\_min}}}$$
The test data and future forecast inputs are transformed using the fitted training scaler parameters.

### 3. 60-Day Lookback Sequence Window
For every day $t$, the input features $X_t$ are the preceding 60 trading days:
$$X_t = [P_{t-60}, P_{t-59}, \dots, P_{t-1}], \quad y_t = P_t$$

### 4. Neural Network Layers
- **Input Layer**: Shape `(60, 1)`
- **LSTM Layer 1**: 50 memory cells (`return_sequences=True`)
- **Dropout Layer 1**: 20% dropout to prevent overfitting
- **LSTM Layer 2**: 50 memory cells (`return_sequences=False`)
- **Dropout Layer 2**: 20% dropout
- **Dense Hidden Layer**: 25 units with ReLU activation
- **Output Layer**: 1 unit producing the predicted close price

---

## 🚀 Setup & Execution Guide (Windows & VS Code)

### ⚠️ Python & TensorFlow Compatibility Note
- **Python 3.14**: Official TensorFlow does not yet have binary wheels for Python 3.14.
- **Python 3.11**: Fully stable and compatible with TensorFlow 2.17.1, NumPy 1.26.4, and Pandas 2.3.3.
- The virtual environment (`.venv`) is already created using **Python 3.11**.

### Step 1: Open Terminal in VS Code
Open your project folder in VS Code (`Ctrl + ~` to toggle the terminal). Ensure you are in PowerShell or Command Prompt.

### Step 2: Activate the Virtual Environment
In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If you get a PowerShell ExecutionPolicy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once).*

In Command Prompt (`cmd`):
```cmd
.\.venv\Scripts\activate.bat
```

### Step 3: Verify Installed Dependencies (Already Completed)
```powershell
pip list
```

### Step 4: Run the Flask Web Server
```powershell
python app.py
```

### Step 5: Open in Your Browser
Visit:
```
http://127.0.0.1:5000
```
Open in Microsoft Edge, Google Chrome, Brave, or Firefox.

---

## 📁 Project Structure

```
stock/
├── app.py                  # Flask web backend & REST API endpoints
├── data_loader.py          # yfinance data downloader, cleaner & sequence builder
├── model.py                # LSTM architecture, training, evaluation & recursive forecasting
├── requirements.txt        # Verified package dependencies
├── saved_models/           # Pre-trained .keras model files for instant inference
│   ├── TMPV_NS_lstm.keras       # Tata Motors pre-trained weights
│   ├── BHARTIARTL_NS_lstm.keras # Bharti Airtel pre-trained weights
│   └── SBIN_NS_lstm.keras       # SBI pre-trained weights
├── templates/
│   └── index.html          # Semantic HTML5 dashboard
└── static/
    ├── style.css           # Mint-green & teal design system
    └── script.js           # Chart.js controller & API client
```

---

## 💡 Quick Tanglish Explanation (Second-Year Engineering Guide)

- **LSTM na enna?**: Simple RNN-la "Vanishing Gradient" problem irukum (pazhaya data marandhu poidum). LSTM-la *Forget Gate, Input Gate, Output Gate* irukardhala 60 trading days oda pattern-ah nalla remember pannikitu adutha naal price-ah predict pannum.
- **Data Leakage thadukradhu eppadi?**: Train data-voda Min and Max use panni mattum dhaan scaler fit pannuvom. Test data-va scaler-la fit panna koodathu, transform mattum dhaan pannanum.
- **Why No Shuffle?**: Stock market time-series sequence data. Monday, Tuesday, Wednesday sequence-ah irundha dhaan Thursday enna aagum nu predict panna mudiyum. Shuffling time order-ah break pannidum.
- **Models Caching**: First time train aana odaney model `.keras` file-ah save aaidum. Next time app open pannumbodhu model instant-ah load aagi 1 second-la prediction kaatum!

---

## 🛡️ Disclaimer
Stock price predictions are computational estimates generated by an experimental Long Short-Term Memory (LSTM) deep learning model for educational and research purposes. They do not constitute financial advice or investment recommendations.
