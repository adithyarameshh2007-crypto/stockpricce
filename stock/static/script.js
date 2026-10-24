/**
 * static/script.js
 * Frontend controller for Stock Price Prediction System using LSTM
 * Handles Chart.js visualization, API communication, and interactive dashboard state.
 */

// Global Dashboard State
const state = {
    company: "TATAMOTORS",
    period: "1y",
    activeTab: "forecast",
    predictionData: null,
    historyData: null,
    chartInstance: null,
    isLoading: false
};

// DOM Element References
const dom = {
    companySelect: document.getElementById("companySelect"),
    periodPills: document.getElementById("periodPills"),
    predictBtn: document.getElementById("predictBtn"),
    retrainBtn: document.getElementById("retrainBtn"),
    guideBtn: document.getElementById("guideBtn"),
    chartTabs: document.getElementById("chartTabs"),
    chartHeading: document.getElementById("chartHeading"),
    chartSubheading: document.getElementById("chartSubheading"),
    chartLoader: document.getElementById("chartLoader"),
    loaderStatus: document.getElementById("loaderStatus"),
    stockChartCanvas: document.getElementById("stockChart"),
    customLegend: document.getElementById("customLegend"),
    
    // Metric displays
    stockNameDisplay: document.getElementById("stockNameDisplay"),
    tickerBadge: document.getElementById("tickerBadge"),
    currentPriceDisplay: document.getElementById("currentPriceDisplay"),
    priceChangePill: document.getElementById("priceChangePill"),
    priceChangeDisplay: document.getElementById("priceChangeDisplay"),
    range52wDisplay: document.getElementById("range52wDisplay"),
    
    predictedPriceDisplay: document.getElementById("predictedPriceDisplay"),
    predictedTrendPill: document.getElementById("predictedTrendPill"),
    predictedChangeDisplay: document.getElementById("predictedChangeDisplay"),
    targetDateDisplay: document.getElementById("targetDateDisplay"),
    
    rmseDisplay: document.getElementById("rmseDisplay"),
    maeDisplay: document.getElementById("maeDisplay"),
    mapeDisplay: document.getElementById("mapeDisplay"),
    accuracyDisplay: document.getElementById("accuracyDisplay"),
    accuracyQuality: document.getElementById("accuracyQuality"),
    
    forecastCardsContainer: document.getElementById("forecastCardsContainer"),
    toast: document.getElementById("toast")
};

// Initialize Application on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    // Initial fetch and prediction
    loadDashboardData();
});

/**
 * Attach UI event listeners.
 */
function setupEventListeners() {
    // Company selection changed
    dom.companySelect.addEventListener("change", (e) => {
        state.company = e.target.value;
        loadDashboardData();
    });

    // Timeframe filter pills
    dom.periodPills.addEventListener("click", (e) => {
        const pill = e.target.closest(".pill");
        if (!pill || pill.classList.contains("active")) return;

        dom.periodPills.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        state.period = pill.dataset.period;
        
        loadHistoricalDataOnly();
    });

    // Run prediction button
    dom.predictBtn.addEventListener("click", () => {
        runPrediction(false);
    });

    // Retrain model button
    dom.retrainBtn.addEventListener("click", () => {
        runRetrain();
    });

    // Guide scroll button
    dom.guideBtn.addEventListener("click", () => {
        const guideEl = document.querySelector(".guide-section");
        if (guideEl) {
            guideEl.scrollIntoView({ behavior: "smooth" });
        }
    });

    // Chart view tabs
    dom.chartTabs.addEventListener("click", (e) => {
        const tabBtn = e.target.closest(".chart-tab");
        if (!tabBtn || tabBtn.classList.contains("active")) return;

        dom.chartTabs.querySelectorAll(".chart-tab").forEach(t => t.classList.remove("active"));
        tabBtn.classList.add("active");
        state.activeTab = tabBtn.dataset.tab;

        renderActiveChart();
    });
}

/**
 * Show / Hide loading spinner with dynamic message.
 */
function setLoading(loading, message = "Loading...") {
    state.isLoading = loading;
    if (loading) {
        dom.loaderStatus.textContent = message;
        dom.chartLoader.classList.remove("hidden");
        dom.predictBtn.disabled = true;
        dom.retrainBtn.disabled = true;
    } else {
        dom.chartLoader.classList.add("hidden");
        dom.predictBtn.disabled = false;
        dom.retrainBtn.disabled = false;
    }
}

/**
 * Display toast notification.
 */
function showToast(message, type = "info") {
    dom.toast.textContent = message;
    dom.toast.className = `toast ${type}`;
    setTimeout(() => {
        dom.toast.className = "toast hidden";
    }, 4500);
}

/**
 * Format currency in Indian Rupees format (₹).
 */
function formatRupee(value) {
    if (value === undefined || value === null || isNaN(value)) return "--";
    return Number(value).toLocaleString("en-IN", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

/**
 * Master function: fetches both historical data and runs LSTM prediction.
 */
async function loadDashboardData() {
    setLoading(true, "Fetching stock data and running LSTM model...");
    try {
        await Promise.all([
            fetchHistoricalData(state.company, state.period),
            runPrediction(false, false) // avoid dual setLoading calls
        ]);
        renderActiveChart();
    } catch (err) {
        console.error("Dashboard Load Error:", err);
        showToast("Error loading stock data: " + err.message, "error");
    } finally {
        setLoading(false);
    }
}

/**
 * Fetch raw historical stock data from backend.
 */
async function fetchHistoricalData(company, period) {
    const res = await fetch(`/api/stock-data?company=${encodeURIComponent(company)}&period=${encodeURIComponent(period)}`);
    const data = await res.json();
    if (!data.success) {
        throw new Error(data.error || "Failed to fetch historical data");
    }
    state.historyData = data;
    updateMarketSummaryUI(data.summary);
    return data;
}

/**
 * Quick refresh for changing timeframe pills.
 */
async function loadHistoricalDataOnly() {
    setLoading(true, `Loading ${state.period.toUpperCase()} market data...`);
    try {
        await fetchHistoricalData(state.company, state.period);
        renderActiveChart();
    } catch (err) {
        showToast(err.message, "error");
    } finally {
        setLoading(false);
    }
}

/**
 * Run LSTM inference and future predictions.
 */
async function runPrediction(forceRetrain = false, manageLoading = true) {
    if (manageLoading) {
        const msg = forceRetrain 
            ? "Training deep learning LSTM model on historical sequences..." 
            : "Executing LSTM sequence inference & multi-step forecasting...";
        setLoading(true, msg);
    }

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                company: state.company,
                period: "2y",
                force_retrain: forceRetrain,
                forecast_days: 7
            })
        });

        const data = await res.json();
        if (!data.success) {
            throw new Error(data.error || "Prediction request failed");
        }

        state.predictionData = data;
        updatePredictionMetricsUI(data);
        renderForecastCards(data.future_forecast);
        
        if (manageLoading) {
            renderActiveChart();
            showToast("LSTM prediction generated successfully!", "success");
        }

    } catch (err) {
        console.error("Prediction Error:", err);
        showToast(err.message, "error");
    } finally {
        if (manageLoading) setLoading(false);
    }
}

/**
 * Trigger explicit retraining of the neural network.
 */
async function runRetrain() {
    setLoading(true, "Re-optimizing neural network weights with 12 epochs...");
    try {
        const res = await fetch("/api/retrain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ company: state.company })
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || "Retraining failed");

        showToast("Model retrained successfully!", "success");
        // Reload full prediction with fresh weights
        await runPrediction(false, false);
        renderActiveChart();
    } catch (err) {
        showToast("Retraining error: " + err.message, "error");
    } finally {
        setLoading(false);
    }
}

/**
 * Update current stock market metrics on UI.
 */
function updateMarketSummaryUI(summary) {
    if (!summary) return;

    dom.stockNameDisplay.textContent = summary.company_name || state.company;
    dom.tickerBadge.textContent = summary.ticker || "--";
    dom.currentPriceDisplay.textContent = formatRupee(summary.latest_close);

    const isPositive = summary.change >= 0;
    const sign = isPositive ? "+" : "";
    dom.priceChangeDisplay.textContent = `${sign}₹${formatRupee(summary.change)} (${sign}${summary.change_pct}%)`;
    dom.priceChangePill.className = `price-change-pill ${isPositive ? "positive" : "negative"}`;

    dom.range52wDisplay.textContent = `₹${formatRupee(summary.low_52w)} - ₹${formatRupee(summary.high_52w)}`;
}

/**
 * Update LSTM model evaluation metrics and next-day forecast UI.
 */
function updatePredictionMetricsUI(data) {
    const summary = data.forecast_summary || {};
    const metrics = data.metrics || {};

    // Next day forecast card
    dom.predictedPriceDisplay.textContent = formatRupee(summary.predicted_price);
    
    const isBullish = summary.trend === "bullish";
    const sign = summary.expected_change >= 0 ? "+" : "";
    dom.predictedChangeDisplay.textContent = `${sign}₹${formatRupee(summary.expected_change)} (${sign}${summary.expected_change_pct}%)`;
    dom.predictedTrendPill.className = `trend-pill ${isBullish ? "positive" : "negative"}`;
    
    dom.targetDateDisplay.textContent = `${summary.next_trading_day_name || ""} (${summary.next_trading_day || "--"})`;

    // Evaluation metrics
    dom.rmseDisplay.textContent = `₹${formatRupee(metrics.rmse)}`;
    dom.maeDisplay.textContent = `₹${formatRupee(metrics.mae)}`;
    dom.mapeDisplay.textContent = `${metrics.mape || "--"}%`;
    
    const dirAcc = metrics.directional_accuracy || 0;
    dom.accuracyDisplay.textContent = `${dirAcc}%`;
    if (dirAcc >= 60) {
        dom.accuracyQuality.textContent = "High Quality";
        dom.accuracyQuality.style.background = "var(--bullish-bg)";
        dom.accuracyQuality.style.color = "var(--bullish)";
    } else {
        dom.accuracyQuality.textContent = "Moderate Fit";
        dom.accuracyQuality.style.background = "var(--bg-mint-soft)";
        dom.accuracyQuality.style.color = "var(--primary)";
    }
}

/**
 * Render 7-Day Day-by-Day Forecast Cards.
 */
function renderForecastCards(forecastList) {
    if (!forecastList || forecastList.length === 0) {
        dom.forecastCardsContainer.innerHTML = `<div class="empty-state">No future forecasts available.</div>`;
        return;
    }

    let html = "";
    forecastList.forEach((day, idx) => {
        const isFirst = idx === 0;
        html += `
            <div class="forecast-day-card ${isFirst ? 'highlight' : ''}">
                <div class="day-step">Day ${day.step}</div>
                <div class="day-name">${day.day_name}</div>
                <div class="day-date">${day.date}</div>
                <div class="day-price">₹${formatRupee(day.predicted_price)}</div>
                <div class="day-band">Est: ₹${formatRupee(day.lower_bound)} - ₹${formatRupee(day.upper_bound)}</div>
            </div>
        `;
    });
    dom.forecastCardsContainer.innerHTML = html;
}

/**
 * Render Chart based on the active tab mode.
 */
function renderActiveChart() {
    if (state.activeTab === "forecast") {
        renderForecastChart();
    } else if (state.activeTab === "validation") {
        renderValidationChart();
    } else {
        renderHistoricalChart();
    }
}

/**
 * Tab 1: Continuous Historical + 7-Day Future Forecast Chart.
 */
function renderForecastChart() {
    if (!state.predictionData) return;

    dom.chartHeading.textContent = "Stock Price Trajectory & LSTM Future Forecast";
    dom.chartSubheading.textContent = "Showing recent continuous historical prices seamlessly connected to the 7-day future estimate.";

    const fullHistory = state.predictionData.full_history || [];
    const futureForecast = state.predictionData.future_forecast || [];

    // Take the last 90 days of history for clean, legible visual scaling
    const recentHistory = fullHistory.slice(-90);

    const labels = [];
    const historicalSeries = [];
    const forecastSeries = [];
    const lowerBandSeries = [];
    const upperBandSeries = [];

    // Push historical points
    recentHistory.forEach(item => {
        labels.push(item.date);
        historicalSeries.push(item.close);
        forecastSeries.push(null);
        lowerBandSeries.push(null);
        upperBandSeries.push(null);
    });

    // Bridge the last historical point with the first forecast point
    const lastHistoryItem = recentHistory[recentHistory.length - 1];
    if (lastHistoryItem && futureForecast.length > 0) {
        forecastSeries[forecastSeries.length - 1] = lastHistoryItem.close;
        lowerBandSeries[lowerBandSeries.length - 1] = lastHistoryItem.close;
        upperBandSeries[upperBandSeries.length - 1] = lastHistoryItem.close;
    }

    // Push future forecast points
    futureForecast.forEach(item => {
        labels.push(item.date);
        historicalSeries.push(null);
        forecastSeries.push(item.predicted_price);
        lowerBandSeries.push(item.lower_bound);
        upperBandSeries.push(item.upper_bound);
    });

    destroyChart();

    const ctx = dom.stockChartCanvas.getContext("2d");

    // Teal Gradient for historical fill
    const histGradient = ctx.createLinearGradient(0, 0, 0, 400);
    histGradient.addColorStop(0, "rgba(13, 148, 136, 0.25)");
    histGradient.addColorStop(1, "rgba(13, 148, 136, 0.0)");

    // Mint Gradient for future projection
    const futureGradient = ctx.createLinearGradient(0, 0, 0, 400);
    futureGradient.addColorStop(0, "rgba(45, 212, 191, 0.2)");
    futureGradient.addColorStop(1, "rgba(45, 212, 191, 0.0)");

    state.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Historical Close (₹)",
                    data: historicalSeries,
                    borderColor: "#0d9488",
                    backgroundColor: histGradient,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: "#0d9488",
                    fill: true,
                    tension: 0.15
                },
                {
                    label: "LSTM 7-Day Forecast (₹)",
                    data: forecastSeries,
                    borderColor: "#14b8a6",
                    backgroundColor: futureGradient,
                    borderWidth: 2.8,
                    borderDash: [6, 4],
                    pointRadius: 4,
                    pointBackgroundColor: "#ffffff",
                    pointBorderColor: "#14b8a6",
                    pointBorderWidth: 2,
                    pointHoverRadius: 7,
                    fill: false,
                    tension: 0.2
                },
                {
                    label: "Confidence Upper Bound (₹)",
                    data: upperBandSeries,
                    borderColor: "rgba(20, 184, 166, 0.25)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: "+1",
                    backgroundColor: "rgba(20, 184, 166, 0.1)",
                    tension: 0.2
                },
                {
                    label: "Confidence Lower Bound (₹)",
                    data: lowerBandSeries,
                    borderColor: "rgba(20, 184, 166, 0.25)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.2
                }
            ]
        },
        options: getCommonChartOptions()
    });

    updateCustomLegend([
        { color: "#0d9488", label: "Actual Historical Close Price" },
        { color: "#14b8a6", label: "LSTM Recursive Future Prediction" },
        { color: "rgba(20, 184, 166, 0.35)", label: "Estimated Confidence Band (±RMSE)" }
    ]);
}

/**
 * Tab 2: Test Set Actual vs Predicted Comparison Chart.
 */
function renderValidationChart() {
    if (!state.predictionData) return;

    dom.chartHeading.textContent = "Model Validation: Actual vs LSTM Predicted (Unseen Test Data)";
    dom.chartSubheading.textContent = "Evaluating how accurately the trained LSTM tracks real price movements on the chronological 20% test set.";

    const testList = state.predictionData.test_predictions || [];
    const labels = testList.map(t => t.date);
    const actuals = testList.map(t => t.actual);
    const predictions = testList.map(t => t.predicted);

    destroyChart();

    const ctx = dom.stockChartCanvas.getContext("2d");

    state.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Actual Test Price (₹)",
                    data: actuals,
                    borderColor: "#334155",
                    backgroundColor: "rgba(51, 65, 85, 0.05)",
                    borderWidth: 2.2,
                    pointRadius: 2,
                    pointHoverRadius: 6,
                    tension: 0.1
                },
                {
                    label: "LSTM Predicted Price (₹)",
                    data: predictions,
                    borderColor: "#0d9488",
                    backgroundColor: "rgba(13, 148, 136, 0.1)",
                    borderWidth: 2.4,
                    borderDash: [4, 4],
                    pointRadius: 2,
                    pointHoverRadius: 6,
                    pointBackgroundColor: "#ffffff",
                    pointBorderColor: "#0d9488",
                    tension: 0.1
                }
            ]
        },
        options: getCommonChartOptions()
    });

    updateCustomLegend([
        { color: "#334155", label: "Actual Unseen Test Price" },
        { color: "#0d9488", label: "LSTM Model Predicted Price" }
    ]);
}

/**
 * Tab 3: Historical Prices Only Chart.
 */
function renderHistoricalChart() {
    if (!state.historyData) return;

    dom.chartHeading.textContent = `Historical Closing Prices (${state.period.toUpperCase()})`;
    dom.chartSubheading.textContent = `High-resolution OHLCV closing price movement for ${state.historyData.resolved_ticker}.`;

    const history = state.historyData.history || [];
    const labels = history.map(h => h.date);
    const closes = history.map(h => h.close);

    destroyChart();

    const ctx = dom.stockChartCanvas.getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, "rgba(13, 148, 136, 0.28)");
    gradient.addColorStop(1, "rgba(13, 148, 136, 0.0)");

    state.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Closing Price (₹)",
                    data: closes,
                    borderColor: "#0d9488",
                    backgroundColor: gradient,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.15
                }
            ]
        },
        options: getCommonChartOptions()
    });

    updateCustomLegend([
        { color: "#0d9488", label: `${state.historyData.resolved_ticker} Daily Close Price` }
    ]);
}

/**
 * Shared Chart.js configuration options.
 */
function getCommonChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: "index",
            intersect: false
        },
        plugins: {
            legend: {
                display: false // We use our own clean custom legend below
            },
            tooltip: {
                backgroundColor: "#ffffff",
                titleColor: "#0f172a",
                bodyColor: "#334155",
                borderColor: "rgba(13, 148, 136, 0.25)",
                borderWidth: 1.5,
                padding: 12,
                boxPadding: 6,
                usePointStyle: true,
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || "";
                        if (label) label += ": ";
                        if (context.parsed.y !== null) {
                            label += "₹" + formatRupee(context.parsed.y);
                        }
                        return label;
                    }
                }
            }
        },
        scales: {
            x: {
                grid: {
                    display: false,
                    drawBorder: false
                },
                ticks: {
                    color: "#94a3b8",
                    font: { family: "Inter", size: 11 },
                    maxTicksLimit: 10
                }
            },
            y: {
                grid: {
                    color: "rgba(226, 232, 240, 0.7)",
                    drawBorder: false
                },
                ticks: {
                    color: "#94a3b8",
                    font: { family: "Inter", size: 11 },
                    callback: function(val) {
                        return "₹" + formatRupee(val);
                    }
                }
            }
        }
    };
}

/**
 * Safely destroy active Chart.js instance.
 */
function destroyChart() {
    if (state.chartInstance) {
        state.chartInstance.destroy();
        state.chartInstance = null;
    }
}

/**
 * Render custom color-coded legend badges below chart.
 */
function updateCustomLegend(items) {
    let html = "";
    items.forEach(item => {
        html += `
            <div class="legend-item">
                <span class="legend-dot" style="background-color: ${item.color};"></span>
                <span>${item.label}</span>
            </div>
        `;
    });
    dom.customLegend.innerHTML = html;
}
