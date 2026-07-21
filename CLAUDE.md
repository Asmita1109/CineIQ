# CineIQ

## Project
A self-improving movie intelligence platform that predicts content demand, personalizes recommendations, and optimizes engagement over time using MovieLens 25M.

## Core Question
How do we move from static recommendations to a self-improving content platform that predicts demand, personalizes delivery, and optimizes engagement over time?

## Components
1. Trend Forecasting -- LightGBM, predict which genres/movies gain traction over time, train locally, artifacts to S3
2. Recommendation Engine -- Neural Collaborative Filtering, train on SageMaker, serve via SageMaker endpoint
3. RL Optimization -- Contextual Bandit, learns best recommendation strategy per user segment, train locally, serve via Lambda
4. LLM Layer -- Claude API generates personalized explanation for each recommendation

## Stack
- Python, Pandas, NumPy, Scikit-learn, LightGBM, PyTorch
- AWS: S3, SageMaker, Lambda, API Gateway, DynamoDB, CloudWatch
- Anthropic API
- Streamlit (dashboard)
- MLflow (experiment tracking)

## Folder Structure
cineiq/
├── data/
├── notebooks/
├── src/
│   ├── forecasting/
│   ├── recommender/
│   ├── rl/
│   └── llm/
├── api/
├── dashboard/
├── models/
├── tests/
└── CLAUDE.md

## Dataset
MovieLens 25M -- ratings.csv, movies.csv, tags.csv
