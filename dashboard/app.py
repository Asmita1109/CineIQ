"""CineIQ Streamlit dashboard.

Two sections:
  1. An interactive Plotly genre trend chart (forecasting_features.csv):
     annual rating counts per genre, 1995-2023.
  2. A per-user recommendation panel: profile lookup, top-5 unrated movies
     scored by the trained BPR NCF model, and a Claude-generated
     explanation for the top recommendation.

Run with: streamlit run dashboard/app.py
"""

import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# TEMPORARY: force stdout to flush after every line. A prior deploy
# crashed with NO error message shown on-page at all -- if that's an
# external kill (OOM, container resource limit) rather than a Python
# exception, our own try/except can't catch it (the interpreter itself
# gets terminated), and print() calls are only useful for pinpointing
# *where* it died if they've actually reached the log before the kill.
# Default stdout buffering when not attached to a terminal (as on Cloud)
# can hold lines in memory rather than writing them immediately, which
# would silently lose exactly the diagnostic info we need most.
sys.stdout.reconfigure(line_buffering=True)


def log_mem(tag):
    """Logs current process RSS memory. Uses the `resource` module (Linux
    only, which is what Streamlit Cloud runs) to actually measure this --
    on Windows (local dev) it just no-ops rather than guessing."""
    try:
        import resource

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[mem] {tag}: RSS = {rss_mb:,.0f} MB")
    except ImportError:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# TEMPORARY diagnostics for the Streamlit Cloud "crashes loading
# recommendations, works locally" investigation -- remove once resolved.
print(f"[paths] __file__={__file__}")
print(f"[paths] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[paths] FEATURES_DIR={FEATURES_DIR}")
print(f"[paths] PROCESSED_DIR={PROCESSED_DIR}")
print(f"[paths] MODELS_DIR={MODELS_DIR}")
log_mem("app start")

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

# ------------------------------------------------------------------
# First-run setup: download required data/model files from S3 if they
# aren't already present locally. Lets the app run on a fresh deploy
# (e.g. Streamlit Community Cloud) where only the code is checked out, not
# the ~900MB of data/model artifacts a full local dev checkout would have.
# ------------------------------------------------------------------
S3_BUCKET = "cineiq-ml-bucket"
S3_REGION_DEFAULT = "us-east-1"

# (local path, S3 key) for files fetched as-is.
S3_REQUIRED_FILES = [
    (FEATURES_DIR / "forecasting_features.csv", "features/forecasting_features.csv"),
    (MODELS_DIR / "forecasting_model.pkl", "models/forecasting_model.pkl"),
    (FEATURES_DIR / "user_features.parquet", "features/user_features.parquet"),
    (FEATURES_DIR / "movie_features.parquet", "features/movie_features.parquet"),
    (FEATURES_DIR / "rl_features.parquet", "features/rl_features.parquet"),
    (PROCESSED_DIR / "movies_clean.csv", "processed/movies_clean.csv"),
    (PROCESSED_DIR / "genome_tags_clean.csv", "processed/genome_tags_clean.csv"),
    (PROCESSED_DIR / "genome_scores_clean.csv", "processed/genome_scores_clean.csv"),
]
# The BPR model isn't stored directly -- it's inside the SageMaker training
# job's output tarball, so it needs downloading and extracting separately.
BPR_MODEL_TARGET = MODELS_DIR / "recommender_model_bpr.pt"
BPR_MODEL_S3_KEY = "models/recommender/cineiq-recommender-20260726-053202/output/model.tar.gz"


def get_s3_client():
    # st.secrets raises two *different* exception types depending on how
    # it's missing: StreamlitSecretNotFoundError when no secrets are
    # configured at all (e.g. .streamlit/secrets.toml doesn't exist and
    # nothing's set in the Cloud app's Secrets panel), KeyError when
    # secrets exist but a specific key is absent/misspelled. Verified both
    # directly -- catching only KeyError (the original bug) let the first
    # case escape as an unhandled traceback instead of the friendly error.
    try:
        return boto3.client(
            "s3",
            region_name=st.secrets.get("AWS_DEFAULT_REGION", S3_REGION_DEFAULT),
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        )
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        st.error(
            "Required files are missing and no AWS credentials were found in Streamlit "
            "secrets (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION). "
            "On Streamlit Community Cloud, add them under your app's Settings -> Secrets. "
            "For local dev, add them to .streamlit/secrets.toml."
        )
        st.stop()


def _download_bpr_model(s3):
    if BPR_MODEL_TARGET.exists():
        return
    print(f"[setup] Downloading + extracting BPR model from s3://{S3_BUCKET}/{BPR_MODEL_S3_KEY} ...")
    BPR_MODEL_TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = Path(tmp_dir) / "model.tar.gz"
        s3.download_file(S3_BUCKET, BPR_MODEL_S3_KEY, str(tar_path))
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp_dir)
        shutil.move(str(Path(tmp_dir) / "recommender_model.pt"), str(BPR_MODEL_TARGET))
    print(f"[setup] BPR model ready at {BPR_MODEL_TARGET} ({BPR_MODEL_TARGET.stat().st_size:,} bytes)")


def setup_required_files():
    """No-ops (and never touches S3 or secrets) if everything's already
    present -- always true for local dev once the training pipeline has
    generated these files.

    Deliberately NOT @st.cache_resource-wrapped: the function's own
    Path.exists() checks are already near-free (9 stat calls), so caching
    saved nothing meaningful, while it added a real failure mode -- if
    something inside ever raised in a way the cache layer didn't expect
    (e.g. st.stop()'s internal control-flow exception), a broken/partial
    run could plausibly get treated as a completed one and never retried
    on the next script run. Calling this as a plain function every rerun
    is simpler to reason about and costs nothing once files exist.
    """
    missing = [p for p, _ in S3_REQUIRED_FILES if not p.exists()] + (
        [] if BPR_MODEL_TARGET.exists() else [BPR_MODEL_TARGET]
    )
    print(f"[setup] setup_required_files() running. Missing: {[str(p) for p in missing] or 'none'}")
    if not missing:
        return

    with st.spinner("Setting up CineIQ..."):
        st.write(f"Downloading {len(missing)} missing file(s) from S3 -- this only happens once...")
        s3 = get_s3_client()
        try:
            for local_path, s3_key in S3_REQUIRED_FILES:
                if not local_path.exists():
                    print(f"[setup] Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(S3_BUCKET, s3_key, str(local_path))
                    print(f"[setup] Done: {local_path} ({local_path.stat().st_size:,} bytes)")
            _download_bpr_model(s3)
        except Exception as e:
            print(f"[setup] S3 download FAILED: {type(e).__name__}: {e}")
            st.error(f"Failed to download required files from S3: {type(e).__name__}: {e}")
            st.stop()
    st.write("Setup complete.")
    print("[setup] setup_required_files() complete -- all required files present.")
    log_mem("after setup_required_files")


st.set_page_config(page_title="CineIQ", page_icon="\U0001f3ac", layout="wide")
setup_required_files()


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
    ckpt_path = MODELS_DIR / "recommender_model_bpr.pt"
    print(f"[recs] load_bpr_model: torch.load({ckpt_path}) exists={ckpt_path.exists()} ...")
    log_mem("before torch.load BPR checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    log_mem("after torch.load BPR checkpoint")
    print("[recs] load_bpr_model: checkpoint loaded, building NCF model ...")
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
    print("[recs] load_bpr_model: done.")
    log_mem("after load_bpr_model")
    return model, ckpt, genome_lookup


@st.cache_resource
def load_rated_movie_sets(_user_id_map, _movie_id_map):
    # Leading underscore tells st.cache_resource not to hash these (large,
    # already-fixed once the BPR checkpoint is loaded) -- only the function
    # identity matters for cache validity here, it only ever runs once.
    rl_path = FEATURES_DIR / "rl_features.parquet"
    print(f"[recs] load_rated_movie_sets: reading {rl_path} exists={rl_path.exists()} ...")
    log_mem("before _build_user_rated_sets (33.7M-row rl_features.parquet)")
    result = _build_user_rated_sets(rl_path, _user_id_map, _movie_id_map)
    log_mem("after _build_user_rated_sets")
    print("[recs] load_rated_movie_sets: done.")
    return result


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

log_mem("before load_forecasting_features")
forecasting_df = load_forecasting_features()
log_mem("after load_forecasting_features")

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

# TEMPORARY debug output for the Streamlit Cloud "crashes loading
# recommendations, works locally" investigation -- remove once resolved.
st.write("DEBUG: Starting recommendations")
st.write(f"DEBUG: MODELS_DIR = {MODELS_DIR}")
st.write(f"DEBUG: MODELS_DIR exists = {MODELS_DIR.exists()}")
st.write(f"DEBUG: recommender_model_bpr.pt exists = {(MODELS_DIR / 'recommender_model_bpr.pt').exists()}")
st.write(f"DEBUG: sys.path = {sys.path}")

with st.expander("Diagnostics (paths + file status)", expanded=True):
    st.write(f"PROJECT_ROOT: `{PROJECT_ROOT}`")
    st.write(f"MODELS_DIR: `{MODELS_DIR}` (exists: {MODELS_DIR.exists()})")
    st.write(f"FEATURES_DIR: `{FEATURES_DIR}` (exists: {FEATURES_DIR.exists()})")
    st.write(f"PROCESSED_DIR: `{PROCESSED_DIR}` (exists: {PROCESSED_DIR.exists()})")
    bpr_path = MODELS_DIR / "recommender_model_bpr.pt"
    bpr_size = f"{bpr_path.stat().st_size:,} bytes" if bpr_path.exists() else "MISSING"
    st.write(f"BPR checkpoint: `{bpr_path}` ({bpr_size})")
    rl_path = FEATURES_DIR / "rl_features.parquet"
    rl_size = f"{rl_path.stat().st_size:,} bytes" if rl_path.exists() else "MISSING"
    st.write(f"rl_features.parquet: `{rl_path}` ({rl_size})")
    genome_path = PROCESSED_DIR / "genome_scores_clean.csv"
    genome_size = f"{genome_path.stat().st_size:,} bytes" if genome_path.exists() else "MISSING"
    st.write(f"genome_scores_clean.csv: `{genome_path}` ({genome_size})")

# TEMPORARY: wrap the whole loading sequence so a crash here shows the full
# traceback on-page instead of Cloud just dying silently / showing its
# generic error screen. st.stop() after displaying it, since letting
# execution fall through would immediately NameError on the unassigned
# variables below.
try:
    print("[recs] Loading user_features.parquet ...")
    user_features = load_user_features()
    log_mem("after load_user_features")
    print("[recs] Loading movie_catalog (movie_features.parquet + movies_clean.csv) ...")
    movie_catalog = load_movie_catalog()
    log_mem("after load_movie_catalog")
    print("[recs] Loading BPR model ...")
    bpr_model, bpr_ckpt, genome_lookup = load_bpr_model()
    log_mem("after load_bpr_model (cached call)")
    print("[recs] Loading rated_movie_sets (rl_features.parquet) ...")
    rated_sets = load_rated_movie_sets(bpr_ckpt["user_id_map"], bpr_ckpt["movie_id_map"])
    log_mem("after load_rated_movie_sets (cached call)")
    print("[recs] All recommendation dependencies loaded successfully.")
except Exception as e:
    import traceback

    print(f"[recs] Loading dependencies FAILED: {type(e).__name__}: {e}")
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
    st.stop()

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
            try:
                print(f"[recs] score_top_movies for user {user_id} ...")
                log_mem("before score_top_movies")
                recs = score_top_movies(user_id, bpr_model, bpr_ckpt, genome_lookup, rated_sets, movie_catalog)
                log_mem("after score_top_movies")
                print("[recs] score_top_movies done.")
            except Exception as e:
                import traceback

                print(f"[recs] score_top_movies FAILED: {type(e).__name__}: {e}")
                st.error(f"Error scoring movies: {e}")
                st.code(traceback.format_exc())
                st.stop()

        if recs is None or recs.empty:
            st.session_state.rec_result = {
                "error": f"User {user_id} isn't known to the trained BPR model (no candidates could be scored)."
            }
        else:
            top = recs.iloc[0]
            explanation, explanation_error = None, None
            with st.spinner("Generating explanation..."):
                try:
                    print("[recs] Loading explainer (genome_scores_clean.csv, ~333MB) ...")
                    log_mem("before load_explainer")
                    explainer = load_explainer()
                    log_mem("after load_explainer")
                    print("[recs] Explainer loaded, calling Claude ...")
                    explanation = explainer.explain(int(user_id), int(top["movieId"]), float(top["relevance_score"]))
                    print("[recs] Explanation generated successfully.")
                except Exception as e:
                    print(f"[recs] Explanation FAILED: {type(e).__name__}: {e}")
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
