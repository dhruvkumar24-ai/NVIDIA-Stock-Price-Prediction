# 📈 NVIDIA Stock Price Prediction
### LSTM · ARIMA · SARIMA · Time Series Analysis · Python · Self-Project

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA%2FSARIMA-blue?style=flat-square)
![Status](https://img.shields.io/badge/Best_Model-LSTM_(RMSE%3D11.4)-brightgreen?style=flat-square)
![Data](https://img.shields.io/badge/Data-15_Years_(2010--2025)-orange?style=flat-square)

---

## 🧩 Problem Statement

NVIDIA's stock has been one of the most volatile and high-growth equities in recent history — driven by AI adoption, GPU demand, and market cycles. This project builds and compares **statistical and deep learning forecasting models** on 15 years of NVDA daily closing prices.

> *"Given 15 years of NVIDIA trading data, which model best captures the non-linear dynamics of a high-growth tech stock — ARIMA, SARIMA, or LSTM?"*

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| **Ticker** | NVDA (NVIDIA Corporation) |
| **Source** | Yahoo Finance (`yfinance`) |
| **Period** | January 2010 – August 2025 |
| **Records** | ~3,900–4,000 trading days |
| **Target** | Adjusted Close Price (USD) |
| **Features** | Open, High, Low, Close, Adj Close, Volume |

---

## ⚙️ Pipeline Overview

```
Raw NVDA Data (yfinance)
        │
        ▼
EDA & Visualization
  ├── Price trends, volatility regimes
  ├── Drawdown analysis
  └── Day-of-week seasonality effects
        │
        ▼
Stationarity Testing
  ├── ADF Test (Augmented Dickey-Fuller)
  ├── ACF & PACF plots
  └── Differencing for stationarity
        │
        ▼
Statistical Models
  ├── ARIMA (auto-order via pmdarima)
  └── SARIMA (with seasonal components)
        │
        ▼
Deep Learning
  ├── LSTM (sequence_length=180, 2 stacked layers)
  ├── MinMaxScaler normalization
  └── EarlyStopping + ModelCheckpoint
        │
        ▼
Model Comparison → RMSE, MAE, MAPE
        │
        ▼
LSTM wins: RMSE = 11.4 vs SARIMA: RMSE = 21.3
```

---

## 📊 Results

| Model | RMSE | Notes |
|---|---|---|
| **LSTM** | **11.4** ✅ | Best model — captures non-linear trends |
| SARIMA | 21.3 | Best statistical model |
| ARIMA | Higher | Baseline statistical model |

**LSTM Architecture:**
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(180, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

---

## 🔑 Key Insights

- **ADF Test** confirmed non-stationarity in raw prices — first-order differencing applied
- **Sequence length = 180 days** (6 months) gave best LSTM performance
- **LSTM outperformed all statistical models** — NVDA's price dynamics are highly non-linear
- **EarlyStopping** prevented overfitting — best weights restored automatically
- SARIMA captured seasonality better than ARIMA but still lagged LSTM significantly

---

## 📁 Repository Structure

```
📦 NVIDIA-Stock-Price-Prediction/
├── NVIDIA_Stock_Price_Prediction_LSTM.ipynb   # Full pipeline notebook
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

```bash
pip install yfinance pandas numpy matplotlib seaborn statsmodels pmdarima tensorflow scikit-learn
```

Open `NVIDIA_Stock_Price_Prediction_LSTM.ipynb` and run sequentially — data is fetched live via `yfinance`.

---

## 🧠 Concepts Covered

`Time Series Analysis` · `Stationarity Testing (ADF)` · `ACF/PACF` · `ARIMA` · `SARIMA` · `LSTM` · `Sequence Modeling` · `MinMax Scaling` · `EarlyStopping` · `Stock Price Forecasting`

---

## 👤 Author

**Dhruv Kumar Sahu**
M.Tech, Industrial & Management Engineering — IIT Kanpur
GATE 2024 AIR 33 | [LinkedIn](https://www.linkedin.com/in/dhruv-kumar-sahu-157ab9193/) · [GitHub](https://github.com/dhruvkumar24-ai)
