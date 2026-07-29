"""Evaluate the trained LinUCB bandit agent on rl_test.parquet: a frozen
policy (agent.update() is never called here), scored against the same two
baselines used during training, with a breakdown by user segment."""

import json
import pickle
from pathlib import Path

import numpy as np

from environment import RecommendationEnv, SEGMENTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"

AGENT_PATH = MODELS_DIR / "rl_agent.pkl"
OUTPUT_PATH = MODELS_DIR / "rl_results.json"
RANDOM_STATE = 42


def summarize(rewards):
    rewards = np.asarray(rewards, dtype="float64")
    return {
        "cumulative_reward": float(rewards.sum()),
        "average_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "n_interactions": int(len(rewards)),
    }


def improvement_pct(agent_avg, baseline_avg):
    if baseline_avg == 0:
        return None
    return (agent_avg - baseline_avg) / baseline_avg * 100


def main():
    print(f"Loading trained agent from {AGENT_PATH}")
    with open(AGENT_PATH, "rb") as f:
        bundle = pickle.load(f)
    agent = bundle["agent"]

    print("Loading environment (rl_test.parquet) ...")
    env = RecommendationEnv(interactions_path=FEATURES_DIR / "rl_test.parquet")
    rng = np.random.default_rng(RANDOM_STATE)
    popular_movie_id = env.most_popular_movie()

    print(f"\n{'=' * 70}\nTEST SET EVALUATION (frozen policy -- no further learning)\n{'=' * 70}")

    rewards_by_segment = {seg: {"agent": [], "most_popular": [], "random": []} for seg in SEGMENTS}
    agent_rewards, popular_rewards, random_rewards = [], [], []
    n_skipped = 0

    for user_id, timestamp in env.iter_interactions():
        action_space = env.get_action_space(user_id)
        if not action_space:
            n_skipped += 1
            continue

        state = env.get_state(user_id, timestamp)
        context = agent.build_context(state)
        action = agent.select_action(context, action_space)
        r_agent = env.step(user_id, action)
        r_popular = env.step(user_id, popular_movie_id)
        r_random = env.step(user_id, env.random_movie(rng))

        agent_rewards.append(r_agent)
        popular_rewards.append(r_popular)
        random_rewards.append(r_random)

        segment = state["user_segment"]
        if segment in rewards_by_segment:
            rewards_by_segment[segment]["agent"].append(r_agent)
            rewards_by_segment[segment]["most_popular"].append(r_popular)
            rewards_by_segment[segment]["random"].append(r_random)

    if n_skipped:
        print(f"Skipped {n_skipped:,} interactions (user unknown to the BPR model).")

    overall_agent = summarize(agent_rewards)
    overall_popular = summarize(popular_rewards)
    overall_random = summarize(random_rewards)

    print(
        f"\nLinUCB agent           cumulative = {overall_agent['cumulative_reward']:,.0f}   "
        f"avg = {overall_agent['average_reward']:.4f}"
    )
    print(
        f"Most-popular baseline  cumulative = {overall_popular['cumulative_reward']:,.0f}   "
        f"avg = {overall_popular['average_reward']:.4f}"
    )
    print(
        f"Random baseline        cumulative = {overall_random['cumulative_reward']:,.0f}   "
        f"avg = {overall_random['average_reward']:.4f}"
    )

    imp_popular = improvement_pct(overall_agent["average_reward"], overall_popular["average_reward"])
    imp_random = improvement_pct(overall_agent["average_reward"], overall_random["average_reward"])
    print(
        f"\nImprovement over most-popular baseline: {imp_popular:+.1f}%"
        if imp_popular is not None
        else "\nImprovement over most-popular baseline: n/a"
    )
    print(
        f"Improvement over random baseline:       {imp_random:+.1f}%"
        if imp_random is not None
        else "Improvement over random baseline:       n/a"
    )

    print(f"\n{'=' * 70}\nBY USER SEGMENT\n{'=' * 70}")
    segment_results = {}
    for segment in SEGMENTS:
        seg_agent = summarize(rewards_by_segment[segment]["agent"])
        seg_popular = summarize(rewards_by_segment[segment]["most_popular"])
        seg_random = summarize(rewards_by_segment[segment]["random"])
        seg_imp_popular = improvement_pct(seg_agent["average_reward"], seg_popular["average_reward"])
        seg_imp_random = improvement_pct(seg_agent["average_reward"], seg_random["average_reward"])
        print(
            f"  {segment:<10} agent avg = {seg_agent['average_reward']:.4f}   "
            f"popular avg = {seg_popular['average_reward']:.4f}   "
            f"random avg = {seg_random['average_reward']:.4f}   (n={seg_agent['n_interactions']:,})"
        )
        segment_results[segment] = {
            "agent": seg_agent,
            "most_popular_baseline": seg_popular,
            "random_baseline": seg_random,
            "improvement_over_most_popular_pct": seg_imp_popular,
            "improvement_over_random_pct": seg_imp_random,
        }

    results = {
        "methodology": {
            "action_space": "top 10 candidates scored by the BPR NCF recommender, per user",
            "reward": "1 if historical data shows the user rated the chosen movie >= 4.0, else 0",
            "policy": "frozen (agent.update() not called during evaluation)",
        },
        "n_interactions_evaluated": overall_agent["n_interactions"],
        "n_interactions_skipped": n_skipped,
        "overall": {
            "agent": overall_agent,
            "most_popular_baseline": overall_popular,
            "random_baseline": overall_random,
            "improvement_over_most_popular_pct": imp_popular,
            "improvement_over_random_pct": imp_random,
        },
        "by_user_segment": segment_results,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
