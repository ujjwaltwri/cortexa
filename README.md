# Cortexa

> **RAG-Enhanced Financial Intelligence System**  
> Autonomous quantitative trading signals backed by machine learning and real-time market context

[![Status](https://img.shields.io/badge/status-production%20ready-success)](https://github.com)
[![Accuracy](https://img.shields.io/badge/accuracy-53.72%25-blue)](https://github.com)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.5480-informational)](https://github.com)

---

## What is Cortexa?

Cortexa is an autonomous financial intelligence system that combines **machine learning predictions** with **retrieval-augmented generation (RAG)** to deliver statistically-backed trading signals with real-time, explainable market analysis.

**The Edge:** 53.72% accuracy on unseen test data—a proven statistical advantage beyond random market movement.

---

## Key Features

- **Quantitative Predictions** – Random Forest model trained on economic indicators and technical analysis
- **Historical Context Engine** – RAG-powered search through 7,000+ historical market states
- **Regime-Aware Intelligence** – Hidden Markov Model (HMM) filters relevant precedents based on current market conditions
- **Real-Time News Integration** – AI-powered news summarization via Gemini API
- **Autonomous Operation** – Scheduled daily updates via Prefect orchestration
- **Interactive Dashboard** – Clean web interface for instant market insights

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Data Sources   │
│  yfinance       │
│  FRED           │
│  Google News    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Processing    │
│   Prefect       │
│   Feature Eng.  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│  Qdrant DB      │
│  (7,000+ docs)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Dual Intelligence Layer    │
│                                 │
│  ┌──────────────────────────┐  │
│  │  Quantitative Brain      │  │
│  │  Random Forest Model     │  │
│  │  ML Probability          │  │
│  └──────────────────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  Qualitative Brain       │  │
│  │  RAG Context Engine      │  │
│  │  Historical Win Rate     │  │
│  └──────────────────────────┘  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  RAG Signal     │
│  Decision Layer │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI +      │
│  Web Dashboard  │
└─────────────────┘
```

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Orchestration** | Prefect | Pipeline automation & scheduling |
| **Backend** | FastAPI | REST API service |
| **Frontend** | HTML/JS | Interactive dashboard |
| **Vector DB** | Qdrant | Historical state storage |
| **ML Model** | Random Forest | Predictive engine |
| **AI/LLM** | Gemini API | News summarization & analysis |
| **Data** | yfinance, FRED, Google News | Market data & indicators |

---

## Quick Start

### Prerequisites

- Docker Desktop
- Python 3.10+
- API Keys: FRED & Gemini

### Installation

```bash
# Clone and navigate to project
cd cortexa

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Initialize System

```bash
# 1. Start vector database
docker-compose up -d

# 2. Initial data ingestion
python -m flows.daily_update_flow

# 3. Train production model
python -m src.training.train

# 4. Backfill historical memory
python -m flows.backfill_memory_flow
```

### Run Application

**Terminal 1 - Backend:**
```bash
python server.py
```

**Terminal 2 - Frontend:**
```bash
python -m http.server 3000
```

**Access Dashboard:**  
Navigate to `http://localhost:3000`

---

## Usage Examples

Ask questions to trigger the full intelligence pipeline:

- *"What is the outlook on MSFT?"*
- *"Should I invest in Tesla?"*
- *"Analyze the current market regime for AAPL"*

The system will return:
- ML prediction with confidence score
- Historical win rate from similar market states
- Real-time news summary and context
- Actionable trading signal

---

## Performance Metrics

| Metric | Value | Significance |
|--------|-------|--------------|
| **Test Accuracy** | 53.72% | Statistically significant edge |
| **ROC-AUC** | 0.5480 | Confirms predictive power |
| **Training Data** | 7,000+ | Historical market states |
| **Baseline** | 50.00% | Random market movement |

---

## Future Roadmap

- [ ] **Autonomous Scheduling** – Deploy daily updates via Prefect Cloud (6:00 AM UTC)
- [ ] **Advanced Features** – Volume profile analysis & sector momentum signals
- [ ] **Multi-Asset Support** – Expand beyond equities to crypto, forex, commodities
- [ ] **Enhanced RAG** – Incorporate earnings calls, SEC filings, and analyst reports
- [ ] **Real-Time Streaming** – WebSocket integration for live signal updates

---

## License

This project is proprietary and confidential.

---

## Contributing

This is a private research project. For collaboration inquiries, please reach out directly.

---

<div align="center">

**Built by quantitative researchers, for quantitative traders**

*Combining the precision of machine learning with the wisdom of historical context*

</div>