"""Train the LinUCB contextual bandit on rl_train.parquet, sequentially in
temporal order, and compare it against two non-personalized baselines:
always recommend the single most popular movie, or recommend a uniformly
random movie.

Each round: build the round's raw state and BPR-scored action space from
the environment, have the agent pick a candidate via UCB, score that pick
against historical data, and update the agent's per-action linear model
with the observed reward. Both baselines are scored the same way every
round (same reward source), so the comparison is apples-to-apples.

Checkpointing: this is a ~27M-row loop that can run for hours, and has been
observed to get killed by something outside this script's control partway
through (varying survival time across identical-seed runs -- not a fixed
timeout or a clean memory ceiling). To survive that, progress is
checkpointed to models/rl_checkpoint.pkl every CHECKPOINT_INTERVAL
interactions (agent state, how many interactions have been processed, and
running reward sums/counts for all three strategies -- not the raw
per-interaction reward lists, which is what previously grew to ~3GB of
boxed Python ints across 3 lists of 27M entries each). On startup, if that
checkpoint exists, training resumes from it instead of starting over.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from agent import LinUCBAgent
from environment import RecommendationEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

ALPHA = 1.0
RANDOM_STATE = 42
EPISODE_SIZE = 50_000  # interactions per "episode" for reward-curve logging/plotting
LOG_EVERY_EPISODES = 10
CHECKPOINT_INTERVAL = 1_000_000  # interactions between checkpoint saves

CHECKPOINT_PATH = MODELS_DIR / "rl_checkpoint.pkl"
AGENT_PATH = MODELS_DIR / "rl_agent.pkl"

PALETTE = "mako"
STRATEGIES = ("agent", "popular", "random")


def save_checkpoint(path, agent, n_processed, cum_reward, n_reward, episode_avgs, rng):
    """Atomic write (temp file + replace) so a process killed mid-save can't
    leave a corrupted checkpoint behind."""
    tmp_path = path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "agent": agent,
                "n_processed": n_processed,
                "cum_reward": cum_reward,
                "n_reward": n_reward,
                "episode_avgs": episode_avgs,
                "rng_state": rng.bit_generator.state,
                "alpha": ALPHA,
                "episode_size": EPISODE_SIZE,
            },
            f,
        )
    tmp_path.replace(path)
    print(f"  Checkpoint saved -> {path} (interaction {n_processed:,})")


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_reward_curves(episode_avgs, episode_size):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette(PALETTE, 3)
    labels = {"agent": "LinUCB agent", "popular": "Most-popular baseline", "random": "Random baseline"}
    for strategy, color in zip(STRATEGIES, colors):
        ax.plot(episode_avgs[strategy], label=labels[strategy], color=color)
    ax.set_xlabel(f"Episode (each = {episode_size:,} interactions)")
    ax.set_ylabel("Average reward")
    ax.set_title("Contextual Bandit vs Baselines -- Average Reward per Episode")
    ax.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "rl_reward_curve.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def main(max_interactions=None):
    print("Loading environment (rl_train.parquet) ...")
    env = RecommendationEnv(
        interactions_path=FEATURES_DIR / "rl_train.parquet",
        max_interactions=max_interactions,
    )

    n_processed = 0
    cum_reward = {s: 0 for s in STRATEGIES}
    n_reward = {s: 0 for s in STRATEGIES}
    episode_avgs = {s: [] for s in STRATEGIES}
    rng = np.random.default_rng(RANDOM_STATE)

    if CHECKPOINT_PATH.exists():
        print(f"\nFound checkpoint at {CHECKPOINT_PATH} -- resuming training from it.")
        ckpt = load_checkpoint(CHECKPOINT_PATH)
        agent = ckpt["agent"]
        n_processed = ckpt["n_processed"]
        cum_reward = ckpt["cum_reward"]
        n_reward = ckpt["n_reward"]
        episode_avgs = ckpt["episode_avgs"]
        rng.bit_generator.state = ckpt["rng_state"]
        print(f"  Resuming from interaction {n_processed:,} (of {len(env.data):,})")
    else:
        print("No checkpoint found -- starting from scratch.")
        print("Initializing LinUCB agent ...")
        agent = LinUCBAgent(movie_features_path=FEATURES_DIR / "movie_features.parquet", alpha=ALPHA)

    popular_movie_id = env.most_popular_movie()

    print(f"\n{'=' * 70}\nTRAINING\n{'=' * 70}")
    print(f"Interactions: {len(env.data):,}   context_dim: {agent.context_dim}   alpha: {ALPHA}")

    n_skipped = 0
    log_every = EPISODE_SIZE * LOG_EVERY_EPISODES
    # Running sums for the current, not-yet-complete episode -- flushed into
    # episode_avgs[...] once EPISODE_SIZE fresh interactions accumulate.
    episode_sum = {s: 0 for s in STRATEGIES}
    episode_count = 0
    next_checkpoint_at = n_processed - (n_processed % CHECKPOINT_INTERVAL) + CHECKPOINT_INTERVAL

    for i, (user_id, timestamp) in enumerate(env.iter_interactions(start_index=n_processed), start=n_processed):
        action_space = env.get_action_space(user_id)
        if not action_space:
            n_skipped += 1
            n_processed = i + 1
            continue

        state = env.get_state(user_id, timestamp)
        context = agent.build_context(state)
        action = agent.select_action(context, action_space)
        r_agent = env.step(user_id, action)
        agent.update(action, context, r_agent)
        r_popular = env.step(user_id, popular_movie_id)
        r_random = env.step(user_id, env.random_movie(rng))

        cum_reward["agent"] += r_agent
        cum_reward["popular"] += r_popular
        cum_reward["random"] += r_random
        n_reward["agent"] += 1
        n_reward["popular"] += 1
        n_reward["random"] += 1

        episode_sum["agent"] += r_agent
        episode_sum["popular"] += r_popular
        episode_sum["random"] += r_random
        episode_count += 1
        if episode_count == EPISODE_SIZE:
            for s in STRATEGIES:
                episode_avgs[s].append(episode_sum[s] / EPISODE_SIZE)
                episode_sum[s] = 0
            episode_count = 0

        n_processed = i + 1
        if n_processed % log_every == 0:
            recent_avg = np.mean(episode_avgs["agent"][-LOG_EVERY_EPISODES:]) if episode_avgs["agent"] else 0.0
            print(
                f"  interaction {n_processed:,} | episode {n_processed // EPISODE_SIZE} | "
                f"cumulative reward = {cum_reward['agent']:,} | "
                f"avg reward (last {log_every:,}) = {recent_avg:.4f}"
            )

        if n_processed >= next_checkpoint_at:
            save_checkpoint(CHECKPOINT_PATH, agent, n_processed, cum_reward, n_reward, episode_avgs, rng)
            next_checkpoint_at += CHECKPOINT_INTERVAL

    if n_skipped:
        print(f"\nSkipped {n_skipped:,} interactions (user unknown to the BPR model).")

    print(f"\n{'=' * 70}\nFINAL SUMMARY\n{'=' * 70}")
    labels = {"agent": "LinUCB agent", "popular": "Most-popular baseline", "random": "Random baseline"}
    for s in STRATEGIES:
        avg = cum_reward[s] / n_reward[s] if n_reward[s] else 0.0
        print(f"{labels[s]:<24} cumulative reward = {cum_reward[s]:,}   average reward = {avg:.4f}")

    agent_avg = cum_reward["agent"] / n_reward["agent"] if n_reward["agent"] else 0.0
    popular_avg = cum_reward["popular"] / n_reward["popular"] if n_reward["popular"] else 0.0
    random_avg = cum_reward["random"] / n_reward["random"] if n_reward["random"] else 0.0
    imp_popular = (agent_avg - popular_avg) / popular_avg * 100 if popular_avg else float("nan")
    imp_random = (agent_avg - random_avg) / random_avg * 100 if random_avg else float("nan")
    print(f"\nImprovement over most-popular baseline: {imp_popular:+.1f}%")
    print(f"Improvement over random baseline:       {imp_random:+.1f}%")

    plot_reward_curves(episode_avgs, EPISODE_SIZE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENT_PATH, "wb") as f:
        pickle.dump({"agent": agent, "alpha": ALPHA, "episode_size": EPISODE_SIZE}, f)
    print(f"Saved -> {AGENT_PATH}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"Removed in-progress checkpoint {CHECKPOINT_PATH} (training complete).")


if __name__ == "__main__":
    main()
