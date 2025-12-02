Cortexa: RAG-Enhanced Financial Intelligence System
Project Status: Production-Ready (Engineering Complete)

An autonomous quantitative system developed to generate statistically-backed trading signals and provide real-time, explainable market analysis through a combined Machine Learning (ML) and Retrieval-Augmented Generation (RAG) framework.

Project Summary and Performance Metrics

Cortexa successfully established a predictive edge in financial forecasting, moving beyond the 50% threshold typically associated with random market movement.

Final Achieved Edge: 53.72% Accuracy on the unseen test dataset (corresponding to an ROC-AUC of 0.5480). This confirms a statistically significant and tradable signal.

Core Value Proposition: The system provides a combined quantitative signal that is backed by proven statistical performance and supported by real-time news context.

Architectural Overview: The Hybrid Advisor Model

Cortexa utilizes a dual-component architecture to combine predictive modeling with qualitative context, ensuring decisions are both data-driven and explainable.

Data Flow and Architecture

The system follows a modular architecture where data flows from external sources (yfinance, FRED, Google News) into a central processing and storage layer (Qdrant) before being queried by the client applications (FastAPI/Dashboard).

The Quantitative Brain (The Predictor): A Random Forest model trained on Economic Indicators and advanced technicals. This calculates the statistical probability of a future price move.

The Qualitative Brain (The Context Engine): A RAG Engine that uses a vector database (Qdrant) to search through 7,000+ historical market states and news articles, filtered by the current Hidden Markov Model (HMM) Regime to find relevant historical precedents.

The RAG Signal Engine (Decision Layer): This final layer synthesizes the ML probability with the historical win rate derived from the RAG search, producing the final, actionable signal.

Technology Stack

Category	Component	Function
Orchestration	Prefect	Manages the scheduled execution of daily data updates, feature engineering, and model training pipelines.
API / Frontend	FastAPI / Custom HTML/JS	Provides the core backend service for model inference and handles communication with the web dashboard.
Vector Database	Qdrant	Persistent storage for all 7,000+ historical market states and news articles.
Machine Learning	Random Forest	The validated predictive model demonstrating a clear statistical advantage.
AI/LLM	Gemini API	Facilitates real-time news summarization and qualitative interpretation of retrieved context.
Data Sources	yfinance, FRED, Google News	Provides raw market data, macroeconomic indicators, and news feeds.
Setup and Installation Guide

Prerequisites

Docker Desktop: Essential for running the Qdrant Vector Database.

Python 3.10+: Recommended environment version.

API Keys: Valid keys for FRED and Gemini.

1. Repository and Environment Setup

Bash
# Navigate to the project directory
cd cortexa

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all project dependencies
pip install -r requirements.txt
2. Database Initialization and Start

The Qdrant Vector Database must be started before proceeding with data flows.

Bash
docker-compose up -d
3. System Training and Memory Backfill (One-Time Procedure)

Execute the primary data flows to establish the foundational data and train the predictive model.

Bash
# A. Initial Data Ingestion and Feature Creation
python -m flows.daily_update_flow 

# B. Train the Production Model
python -m src.training.train 

# C. Backfill RAG Memory (Populate Qdrant with history)
python -m flows.backfill_memory_flow 
Operating the Application

1. Launch the Backend API (Terminal 1)

This initiates the core service responsible for all model and database interactions.

Bash
python server.py
2. Launch the Frontend Interface (Terminal 2)

This starts a simple web server to host the analytical dashboard.

Bash
python -m http.server 3000
3. Accessing the Dashboard

Navigate to the following URL in your web browser: http://localhost:3000

Ask a question that triggers the logic:

"What is the outlook on MSFT?" (Triggers ML Prediction, RAG Signal, News Summary)

"Should I invest in Tesla?" (Triggers ML Prediction, RAG Signal, News Summary)

Future Development and Operationalization

Autonomous Operation: Configure the Prefect Cloud scheduler to run the daily_update_flow.py job daily (e.g., 6:00 AM UTC), making the system truly autonomous.

Feature Expansion: Integrate more complex predictive features, such as Volume Profile analysis or sector-specific momentum signals, to further increase the model's edge.