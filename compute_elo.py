#!/usr/bin/env python3
"""
Compute team ELO ratings via maximum likelihood estimation.

Each point is treated as a minimatch under the ELO model:
  P(team A wins a point) = 1 / (1 + 10^((r_B - r_A) / 400))

Ratings are found by minimising the negative log-likelihood across all matches
simultaneously using scipy (convex problem → global optimum guaranteed).

Usage:
  python3 compute_elo.py                  # reads /tmp/ffvb_matches.csv
  python3 compute_elo.py matches.csv      # or a custom file
  python3 scrape_matches.py | python3 compute_elo.py -  # from stdin
"""
import csv
import sys

import numpy as np
from scipy.optimize import minimize

CACHE = "/tmp/ffvb_matches.csv"


def load_matches(source):
    matches = []
    for row in csv.DictReader(source):
        sa, sb = int(row["score_a"]), int(row["score_b"])
        if sa > 0 and sb > 0:
            matches.append((row["team_a"], row["team_b"], sa, sb))
    return matches


def neg_log_likelihood(r, teams_idx, matches):
    loss = 0.0
    for a, b, sa, sb in matches:
        diff = (r[teams_idx[b]] - r[teams_idx[a]]) / 400.0
        p_a = 1.0 / (1.0 + 10.0 ** diff)
        loss -= sa * np.log(p_a) + sb * np.log(1.0 - p_a)
    return loss


def compute_elo(matches):
    teams = sorted({m[0] for m in matches} | {m[1] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    r0 = np.full(n, 1500.0)
    constraint = {"type": "eq", "fun": lambda r: np.mean(r) - 1500.0}

    res = minimize(
        neg_log_likelihood,
        r0,
        args=(idx, matches),
        method="SLSQP",
        constraints=constraint,
        options={"ftol": 1e-12, "maxiter": 10_000},
    )
    print(f"Converged: {res.success} — {res.message}", file=sys.stderr)
    return {t: res.x[idx[t]] for t in teams}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        matches = load_matches(sys.stdin)
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            matches = load_matches(f)
    else:
        with open(CACHE) as f:
            matches = load_matches(f)

    if not matches:
        print("No matches found.", file=sys.stderr)
        sys.exit(1)

    ratings = compute_elo(matches)
    ranked = sorted(ratings.items(), key=lambda x: -x[1])

    writer = csv.writer(sys.stdout)

    print()
    writer.writerow(["rank", "team", "elo"])
    for rank, (team, elo) in enumerate(ranked, 1):
        writer.writerow([rank, team, f"{elo:.1f}"])

    print()
    writer.writerow(["team_a", "team_b", "score_a", "score_b", "perf_a", "perf_b"])
    for a, b, sa, sb in matches:
        if sa == 0 or sb == 0:
            writer.writerow([a, b, sa, sb, "forfeit", "forfeit"])
            continue
        # PR = opponent_elo + 400 * log10(own_points / opponent_points)
        pr_a = ratings[b] + 400 * np.log10(sa / sb)
        pr_b = ratings[a] + 400 * np.log10(sb / sa)
        writer.writerow([a, b, sa, sb, f"{pr_a:.1f}", f"{pr_b:.1f}"])


if __name__ == "__main__":
    main()
