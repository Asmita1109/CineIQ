# CineIQ — A Self-Improving Movie Intelligence Platform

Predicts what movies and genres will trend, personalizes recommendations per user, and continuously improves engagement through reinforcement learning and LLM-generated explanations — all built on the MovieLens dataset.

> **Core question:** How do we move from static recommendations to a self-improving content platform that predicts demand, personalizes delivery, and optimizes engagement over time?

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://cineiq-bh73b6kuriapczrbswjsuq.streamlit.app)

---

## Architecture

```mermaid
flowchart TD
    A[("MovieLens Data<br/>Ratings · Tags · Genome Scores")] --> B["Data Pipeline<br/>Ingest → Clean → Feature Engineering"]

    B --> C["Trend Forecasting<br/>LightGBM"]
    B --> D["Recommendation Engine<br/>NCF + BPR"]
    B --> E["RL Optimization<br/>LinUCB Bandit"]

    C --> F["Merge Signals"]
    D --> F
    E --> F

    F --> G["LLM Explanation Layer<br/>Claude API"]
    G --> H(["User"])

    H -. "reward / engagement signal" .-> E

    subgraph AWS["AWS Infrastructure"]
        S3[("S3<br/>Data & Model Artifacts")]
        SM["SageMaker<br/>NCF Training"]
        LAM["Lambda<br/>RL Serving"]
        APIGW["API Gateway"]
    end

    B -.-> S3
    D -.-> SM
    SM -.-> S3
    E -.-> LAM
    LAM -.-> APIGW
    APIGW -.-> H
```

---

## Overview

CineIQ is an end-to-end movie intelligence platform built on the MovieLens dataset that treats recommendation as a continuously-learning system rather than a static model. It forecasts which genres are gaining traction before they peak, ranks movies for each user with a neural collaborative filtering model trained via pairwise ranking, and uses a contextual bandit to learn which of those ranked candidates actually earns engagement — closing the loop with real reward signal instead of a fixed heuristic. A Claude-powered explanation layer then turns each recommendation's underlying signals (taste profile, tag similarity, genre momentum) into a short, human-readable reason. The goal is to demonstrate the shift from "here's a ranked list" to "here's a system that gets better at ranking as it observes what actually works."

---

## Components

### 1. Trend Forecasting
LightGBM model predicting next-week genre demand from historical rating velocity (lag + rolling-average features). **82% RMSE improvement** over a naive last-week baseline on the held-out test set.

```mermaid
flowchart LR
    A["Weekly Genre<br/>Rating History"] --> B["LightGBM<br/>Lag + Rolling Features"]
    B --> C["Predicted Next-Week<br/>Genre Demand"]
```

### 2. Recommendation Engine
Neural Collaborative Filtering trained with Bayesian Personalized Ranking (BPR) pairwise loss on AWS SageMaker. **NDCG@10 = 0.80** on the held-out test set (negative-sampled ranking evaluation).

```mermaid
flowchart LR
    A["User + Movie Embeddings<br/>+ Genome Tags"] --> B["NCF<br/>BPR Pairwise Ranking"]
    B --> C["Top-N Ranked<br/>Movie Candidates"]
```

### 3. RL Optimization
A LinUCB contextual bandit that learns, per user segment, which of the recommender's top candidates to actually surface — improving **+16.8% over a most-popular baseline** on training data (generalization to unseen users narrows this gap, see [Key Results](#key-results)).

```mermaid
flowchart LR
    A["NCF Candidates +<br/>User Context"] --> B["LinUCB<br/>Contextual Bandit"]
    B --> C["Selected<br/>Recommendation"]
    C -. "reward" .-> B
```

### 4. LLM Explanation
Claude API generates a short, personalized explanation for each recommendation, grounded in the user's taste profile, the movie's tag/genre profile, and the current genre trend signal.

```mermaid
flowchart LR
    A["Recommendation +<br/>Taste Profile + Trend"] --> B["Claude API"]
    B --> C["Personalized<br/>Explanation Text"]
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?logo=lightgbm&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazonaws&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Anthropic Claude](https://img.shields.io/badge/Claude%20API-Anthropic-D97757?logo=anthropic&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)

---

## Dataset

Built on **MovieLens (ml-latest)**, augmented with TMDB metadata for recent releases:

| Stat | Value |
|---|---|
| Users | 307,412 |
| Movies | 43,855 |
| Ratings / interactions | 33.7M |
| Genome tags | 1,128 |
| Time span | Jan 1995 – Jul 2023 |

---

## Key Results

| Component | Model | Metric | Result |
|---|---|---|---|
| Trend Forecasting | LightGBM | Test RMSE improvement vs. naive baseline | **+82.0%** |
| Trend Forecasting | LightGBM | Test MAE improvement vs. naive baseline | +79.8% |
| Recommendation Engine | NCF (BPR) | Test NDCG@10 | **0.80** |
| Recommendation Engine | NCF (BPR) | Test Precision@10 / Recall@10 | 0.44 / 0.88 |
| RL Optimization | LinUCB | Improvement over most-popular baseline (training) | **+16.8%** |
| RL Optimization | LinUCB | Improvement over most-popular baseline (held-out test) | +2.2% |
| LLM Explanation | Claude API | — | Real-time, grounded, personalized explanations |

The gap between the RL agent's training-time and test-time improvement over the popularity baseline is a real and expected generalization effect — see [Limitations](#limitations).

---

## Project Structure

```
CineIQ/
├── dashboard/              # Streamlit app (live demo)
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── raw/                 # Raw MovieLens + TMDB data
│   ├── processed/            # Cleaned, deduplicated tables
│   └── features/             # Model-ready feature tables (train/val/test splits)
├── models/                  # Trained model artifacts + evaluation results
│   └── figures/
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── pipeline/                 # Ingestion, cleaning, feature engineering scripts
├── src/
│   ├── forecasting/          # LightGBM trend model
│   ├── recommender/          # NCF (BPR) model, SageMaker training/launch
│   ├── rl/                   # LinUCB contextual bandit
│   └── llm/                  # Claude explanation layer
├── results/                  # Aggregated model results
├── api/                      # (scaffolded) serving layer
├── tests/                    # (scaffolded)
├── CLAUDE.md
└── requirements.txt
```

---

## How to Run

### Try it live
The fastest way to explore CineIQ is the hosted demo — no setup required:
**[cineiq-bh73b6kuriapczrbswjsuq.streamlit.app](https://cineiq-bh73b6kuriapczrbswjsuq.streamlit.app)**

### Run locally

```bash
git clone https://github.com/Asmita1109/CineIQ.git
cd CineIQ
pip install -r dashboard/requirements.txt
```

Create a `.env` file at the project root with:
```
ANTHROPIC_API_KEY=your_key_here
```

The dashboard downloads its required data/model artifacts (~900MB) from S3 on first run if they're not already present locally. For that, set AWS credentials in `.streamlit/secrets.toml`:
```toml
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_DEFAULT_REGION = "us-east-1"
```

Then launch:
```bash
streamlit run dashboard/app.py
```

### Reproducing the full pipeline
The `pipeline/`, `src/forecasting/`, `src/recommender/`, and `src/rl/` directories contain the scripts used to build every artifact the dashboard consumes, from raw MovieLens ingestion through SageMaker training to RL agent training. See `CLAUDE.md` for the full component breakdown.

---

## Limitations

- **Dataset ends July 2023** — recent viewing patterns and new releases aren't captured.
- **Explicit ratings are rare in production** — most real platforms rely on implicit feedback (clicks, watch time, skips), not 5-star ratings.
- **The RL agent is trained on offline historical data** — online learning with real-time feedback would likely improve results further than what offline replay can show.
- **Casual users are better served by the popularity baseline** due to cold start — the bandit has too little interaction history to personalize effectively for low-activity users.
- **The LLM explanation layer adds latency** — a production deployment at scale would need caching (already partially addressed) and/or async generation.
- **Model accuracy is limited by CPU training and dataset size** — GPU training and the full MovieLens 25M+ dataset would likely improve all three model components.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
