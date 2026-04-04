# 📈 SensexAI: Market Prediction & Analysis Framework

> AI-powered stock market prediction system for BSE Sensex combining statistical models, deep learning, anomaly detection, and explainable insights via a RAG-based intelligence system.

---

##  Overview

**SensexAI** is an end-to-end financial time series forecasting and analysis framework built on the BSE Sensex (^BSESN) dataset (2016–2025).

The project provides a comprehensive comparison between classical statistical models and deep learning approaches, while extending beyond prediction through anomaly detection, sentiment analysis, and an AI-powered chatbot capable of both explaining past events and generating forward-looking insights.

---

##  Objectives

- Compare classical models vs deep learning for stock prediction  
- Analyze effectiveness of AI in short-term financial forecasting  
- Detect anomalous market conditions (crashes, volatility spikes)  
- Evaluate the impact of news sentiment  
- Build an explainable AI system for market understanding  

---

##  Dataset

| Property | Value |
|----------|------|
| Index | BSE Sensex (^BSESN) |
| Source | Yahoo Finance (`yfinance`) |
| Period | 2016 – 2025 |
| Trading Days | 2,462 |
| Features | 24 |

###  Train/Test Split
- Train: 80% (2016–2024)  
- Test: 20% (2024–2026)

---

##  Feature Engineering

- RSI (14-day)  
- MACD (12, 26, 9)  
- Bollinger Bands  
- SMA (20, 50), EMA (20)  
- ATR (14)  
- On-Balance Volume (OBV)  
- Realized Volatility (20-day)  
- Lag Features (1, 3, 5, 10 days)  

---

##  Models Implemented

###  Classical Models
- ARIMA  
- SARIMA  
- ARIMA-GARCH  

###  Deep Learning Models
- LSTM  
- GRU  
- Bidirectional LSTM (BiLSTM)  

---

## Performance Comparison

| Model          | RMSE   | R²     | Insight |
|----------------|--------|--------|--------|
| ARIMA-GARCH    | 630.72 | 0.9746 | Best overall |
| ARIMA          | 632.87 | 0.9744 | Strong baseline |
| SARIMA         | 734.98 | 0.9655 | Captures seasonality |
| GRU            | 1309.53| 0.8904 | Best DL model |
| LSTM           | 1897.19| 0.7699 | Overfitting observed |
| BiLSTM         | 1915.43| 0.7655 | No improvement |

---

##  Key Insights

- Classical models outperform deep learning for one-step-ahead prediction  
- Stock prices follow a near-random walk behavior  
- GRU outperforms LSTM due to simpler architecture  
- Sentiment analysis did not improve performance (temporal mismatch)  
- ARIMA-GARCH provides both prediction and volatility estimation  

---

##  Anomaly Detection

### Methods Used:
- LSTM Autoencoder (sequence-based anomalies)  
- Isolation Forest (feature-space anomalies)  

### Results:
- 72 consensus anomalies detected  
- Key periods identified:
  - COVID-19 crash (2020)  
  - Rate hike cycle (2022)  
  - Election volatility (2024)  

---

##  Sentiment Analysis

- Dataset: 1.58M news headlines  
- Tool: VADER sentiment analyzer  

### Findings:
- Majority of news is negative (~75%)  
- Weak correlation with market returns  
- No performance gain due to dataset misalignment  

---

##  RAG Chatbot (Explainable & Forward-Looking AI)

An AI-powered chatbot that provides both **historical explanations** and **forward-looking market insights** using a Retrieval-Augmented Generation (RAG) pipeline.

###  Components
- Embeddings: `all-MiniLM-L6-v2`  
- LLM: DeepSeek Chat API  
- Frontend: Streamlit  
- Data: News headlines + market data  

---

###  Capabilities

####  Historical Explanation
- Explains past market events using real data  
- Grounds responses in retrieved news and price context  

**Examples:**
- “Why did Sensex crash in March 2020?”  
- “What caused the 2022 volatility?”  

---

####  Forward-Looking Insights
- Generates **context-aware projections** based on:
  - Historical trends  
  - Similar past events  
  - Market conditions  

**Examples:**
- “What might happen if inflation rises again?”  
- “How could elections impact the market?”  
- “What is the expected short-term trend?”  



---

###  Purpose

To bridge the gap between:
-  Quantitative prediction  
-  Qualitative reasoning  

by delivering an **interactive financial intelligence system**.

---

## 🧱 Tech Stack

| Category | Tools |
|----------|------|
| Data | pandas, numpy, yfinance |
| Features | pandas_ta |
| Models | pmdarima, arch |
| Deep Learning | TensorFlow, Keras |
| ML | scikit-learn |
| Sentiment | vaderSentiment |
| Embeddings | sentence-transformers |
| LLM | DeepSeek API |
| Visualization | matplotlib, seaborn |
| App | Streamlit |

---

## 📁 Project Structure
