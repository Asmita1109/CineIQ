"""CineIQ Streamlit dashboard.

Two sections:
  1. An interactive Plotly genre trend chart (forecasting_features.csv):
     annual rating counts per genre, 1995-2023.
  2. A per-user recommendation panel: profile lookup, top-5 unrated movies
     scored by the trained BPR NCF model, and a Claude-generated
     explanation for the top recommendation.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# dashboard/ is a sibling of src/ -- add the specific package dirs we need
# so bare imports (matching those modules' own same-directory convention)
# resolve correctly regardless of where `streamlit run` is invoked from.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recommender"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "llm"))
from model import NCF, _build_user_rated_sets  # noqa: E402
from evaluate import build_genome_lookup  # noqa: E402
from explainer import RecommendationExplainer  # noqa: E402

CORE_QUESTION = (
    "How do we move from static recommendations to a self-improving content "
    "platform that predicts demand, personalizes delivery, and optimizes "
    "engagement over time?"
)

TOP_N_GENRES = 5
TOP_N_RECS = 5

st.set_page_config(page_title="CineIQ", page_icon="\U0001f3ac", layout="wide")


# ------------------------------------------------------------------
# Cached loaders -- each of these is expensive enough (full-catalog model
# scoring, an 18M-row genome table, a 33M-row rating history) that it must
# run once per app process, not once per interaction/rerun.
# ------------------------------------------------------------------
@st.cache_data
def load_forecasting_features():
    return pd.read_csv(FEATURES_DIR / "forecasting_features.csv", parse_dates=["week_start"])


@st.cache_data
def load_user_features():
    return pd.read_parquet(FEATURES_DIR / "user_features.parquet")


@st.cache_data
def load_movie_catalog():
    movie_features = pd.read_parquet(FEATURES_DIR / "movie_features.parquet", columns=["movieId", "genres"])
    movies = pd.read_csv(PROCESSED_DIR / "movies_clean.csv", usecols=["movieId", "title"])
    catalog = movie_features.merge(movies, on="movieId", how="left")
    return catalog.set_index("movieId")


@st.cache_resource
def load_bpr_model():
    ckpt = torch.load(MODELS_DIR / "recommender_model_bpr.pt", map_location="cpu", weights_only=False)
    model = NCF(
        num_users=ckpt["num_users"],
        num_movies=ckpt["num_movies"],
        embedding_dim=ckpt["embedding_dim"],
        genome_dim=ckpt["genome_dim"],
        hidden_layers=ckpt["hidden_layers"],
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    movie_features = pd.read_parquet(FEATURES_DIR / "movie_features.parquet")
    genome_lookup = build_genome_lookup(movie_features, ckpt["movie_id_map"], ckpt["genome_dim"])
    return model, ckpt, genome_lookup


@st.cache_resource
def load_rated_movie_sets(_user_id_map, _movie_id_map):
    # Leading underscore tells st.cache_resource not to hash these (large,
    # already-fixed once the BPR checkpoint is loaded) -- only the function
    # identity matters for cache validity here, it only ever runs once.
    return _build_user_rated_sets(FEATURES_DIR / "rl_features.parquet", _user_id_map, _movie_id_map)


@st.cache_resource
def load_explainer():
    return RecommendationExplainer()


# ------------------------------------------------------------------
# Recommender: score top-N unrated movies for a user
# ------------------------------------------------------------------
@torch.no_grad()
def score_top_movies(user_id, model, ckpt, genome_lookup, rated_sets, movie_catalog, top_n=TOP_N_RECS):
    user_id_map = ckpt["user_id_map"]
    movie_id_map = ckpt["movie_id_map"]
    if user_id not in user_id_map:
        return None

    user_idx = user_id_map[user_id]
    rated = rated_sets.get(user_idx, frozenset())

    n = len(movie_id_map)
    user_idx_t = torch.full((n,), user_idx, dtype=torch.long)
    movie_idx_t = torch.arange(n, dtype=torch.long)
    genome_t = torch.from_numpy(genome_lookup)
    scores = model(user_idx_t, movie_idx_t, genome_t).numpy()

    idx_to_movie_id = np.array([mid for mid, _ in sorted(movie_id_map.items(), key=lambda kv: kv[1])])
    unrated_mask = np.array([i not in rated for i in range(n)])
    candidate_idx = np.where(unrated_mask)[0]
    top_local = candidate_idx[np.argsort(-scores[candidate_idx])[:top_n]]

    rows = []
    for rank, idx in enumerate(top_local, start=1):
        movie_id = int(idx_to_movie_id[idx])
        info = movie_catalog.loc[movie_id] if movie_id in movie_catalog.index else None
        title = info["title"] if info is not None and pd.notna(info["title"]) else f"movie {movie_id}"
        genres = info["genres"] if info is not None and pd.notna(info["genres"]) else ""
        rows.append(
            {
                "rank": rank,
                "movieId": movie_id,
                "title": title,
                "genres": genres,
                "relevance_score": round(float(scores[idx]), 3),
            }
        )
    return pd.DataFrame(rows)


def format_recs_for_display(recs):
    """Presentation-only reshaping of score_top_movies()'s output -- the
    raw recs DataFrame (with movieId and full-precision score) is still
    used internally (e.g. passed to the explainer), this is just what's
    shown in the table."""
    display = recs.drop(columns=["movieId"]).copy()
    display["genres"] = display["genres"].apply(lambda g: " • ".join(g.split("|")) if g else "")
    display["relevance_score"] = display["relevance_score"].round(2)
    display = display.rename(
        columns={"rank": "#", "title": "Movie", "genres": "Genres", "relevance_score": "Score"}
    )
    return display[["#", "Movie", "Genres", "Score"]]


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
st.title("CineIQ")
st.markdown(
    f"<p style='font-size:18px;font-style:italic;color:#37474F;margin-bottom:20px;'>{CORE_QUESTION}</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Section 1: genre trend chart -- annual rating counts, 1995-2023, Plotly
# ------------------------------------------------------------------
st.markdown("<h3>\U0001f525 Genre Trends</h3>", unsafe_allow_html=True, anchors=False)
st.caption("28 years of genre popularity trends")

forecasting_df = load_forecasting_features()

top_genres = (
    forecasting_df.groupby("genre")["rating_count"].sum().nlargest(TOP_N_GENRES).index.tolist()
)
GENRE_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8338ec"]

year_df = forecasting_df[forecasting_df["genre"].isin(top_genres)].copy()
year_df["year"] = year_df["week_start"].dt.year
year_agg = year_df.groupby(["year", "genre"], as_index=False)["rating_count"].sum()

fig = go.Figure()
for genre, color in zip(top_genres, GENRE_COLORS):
    genre_data = year_agg[year_agg["genre"] == genre].sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=genre_data["year"],
            y=genre_data["rating_count"],
            name=genre,
            mode="lines+markers",
            line=dict(shape="spline", smoothing=0.6, width=3, color=color),
            marker=dict(size=6, color=color),
            hovertemplate=f"Year: %{{x}}<br>Genre: {genre}<br>Rating Count: %{{y:,}}<extra></extra>",
        )
    )

fig.update_xaxes(title="Year", dtick=1, fixedrange=True)
fig.update_yaxes(title="Annual Rating Count", fixedrange=True)
fig.update_layout(
    height=460,
    dragmode=False,
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
)

# Legend-click-to-toggle-trace and hover tooltips are native Plotly
# behavior, no extra code needed. fixedrange=True on both axes disables
# scroll/drag zoom and pan; dragmode=False disables the default
# click-drag box-zoom; displayModeBar hides the zoom/pan toolbar --
# together this leaves just hover + legend clicks, nothing else.
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})


# ------------------------------------------------------------------
# Section 2: user recommendations
# ------------------------------------------------------------------
st.header("Get Personalized Recommendations")

user_features = load_user_features()
movie_catalog = load_movie_catalog()
bpr_model, bpr_ckpt, genome_lookup = load_bpr_model()
rated_sets = load_rated_movie_sets(bpr_ckpt["user_id_map"], bpr_ckpt["movie_id_map"])

MIN_USER_ID = int(user_features["userId"].min())
MAX_USER_ID = int(user_features["userId"].max())

default_power_user_id = int(
    user_features.loc[user_features["user_segment"] == "power", "userId"].sort_values().iloc[0]
)

if "user_id_input" not in st.session_state:
    st.session_state.user_id_input = default_power_user_id

st.caption(f"Valid user IDs range from {MIN_USER_ID:,} to {MAX_USER_ID:,}")

user_id = st.number_input("User ID", min_value=1, step=1, key="user_id_input")

if "rec_result" not in st.session_state:
    st.session_state.rec_result = None

clicked_get_recommendations = st.button("Get Recommendations")
# First-ever page load: rec_result is still None and nothing was clicked --
# auto-fetch for the default power user (user_id already defaults to it).
first_load = st.session_state.rec_result is None and not clicked_get_recommendations

if clicked_get_recommendations or first_load:
    profile_row = user_features[user_features["userId"] == user_id]
    if profile_row.empty:
        st.session_state.rec_result = {
            "error": f"User ID {user_id} not found. Try a different ID between {MIN_USER_ID:,} and {MAX_USER_ID:,}."
        }
    else:
        with st.spinner("Scoring movies..."):
            profile = profile_row.iloc[0].to_dict()
            recs = score_top_movies(user_id, bpr_model, bpr_ckpt, genome_lookup, rated_sets, movie_catalog)

        if recs is None or recs.empty:
            st.session_state.rec_result = {
                "error": f"User {user_id} isn't known to the trained BPR model (no candidates could be scored)."
            }
        else:
            top = recs.iloc[0]
            explanation, explanation_error = None, None
            with st.spinner("Generating explanation..."):
                try:
                    explainer = load_explainer()
                    explanation = explainer.explain(int(user_id), int(top["movieId"]), float(top["relevance_score"]))
                except Exception as e:
                    explanation_error = str(e)

            st.session_state.rec_result = {
                "profile": profile,
                "recs": recs,
                "top_title": top["title"],
                "explanation": explanation,
                "explanation_error": explanation_error,
            }

result = st.session_state.rec_result
if result is not None:
    if "error" in result:
        st.warning(result["error"])
    else:
        profile = result["profile"]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Segment", profile["user_segment"])
        p2.metric("Favorite genre", profile["favorite_genre"])
        p3.metric("Avg rating", f"{profile['avg_rating']:.2f}")
        p4.metric("Total ratings", f"{profile['total_ratings']:,}")

        st.subheader("Top 5 Recommendations")
        st.dataframe(format_recs_for_display(result["recs"]), width="stretch", hide_index=True)

        st.subheader(f"Why we recommend “{result['top_title']}”")
        if result["explanation"]:
            st.success(result["explanation"])
        else:
            st.warning(f"Couldn't generate an explanation: {result['explanation_error']}")

# ------------------------------------------------------------------
# Section 3: footer
# ------------------------------------------------------------------
st.markdown(
    "<span style='color:#666666;font-size:16px;'>Built with PyTorch · LightGBM · AWS SageMaker · "
    "RL Agent (LinUCB) · Claude API · [GitHub](https://github.com/Asmita1109/CineIQ)</span>",
    unsafe_allow_html=True,
)
