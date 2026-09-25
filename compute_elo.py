#!/usr/bin/env python3
"""
Compute team ELO ratings via maximum likelihood estimation.

Each point is treated as a minimatch under the ELO model:
  P(team A wins a point) = 1 / (1 + 10^((r_B - r_A) / 400))

Ratings are found by minimising the negative log-likelihood across all matches
simultaneously using scipy (convex problem → global optimum guaranteed).

Older matches are down-weighted: each match's log-likelihood is multiplied by
  w = 0.5 ^ (age_days / half_life)
where age is measured from the most recent match. Only relative weights matter,
so the reference date does not change the ratings.

Usage:
  python3 compute_elo.py                  # reads /tmp/ffvb_matches.csv
  python3 compute_elo.py matches.csv      # or a custom file
  python3 compute_elo.py --half-life 0    # no time decay
  python3 scrape_matches.py | python3 compute_elo.py -  # from stdin
"""
import argparse
import csv
import sys
from datetime import date

import numpy as np
from scipy.optimize import minimize

CACHE = "/tmp/ffvb_matches.csv"
DEFAULT_HALF_LIFE = 180


def load_matches(source):
    matches = []
    for row in csv.DictReader(source):
        sa, sb = int(row["score_a"]), int(row["score_b"])
        if sa > 0 and sb > 0:
            day = date.fromisoformat(row["date"])
            matches.append((row["team_a"], row["team_b"], sa, sb, day))
    return matches


def time_weights(matches, half_life):
    if half_life <= 0:
        return [1.0] * len(matches)
    latest = max(m[4] for m in matches)
    return [0.5 ** ((latest - m[4]).days / half_life) for m in matches]


def neg_log_likelihood(r, teams_idx, matches, weights):
    loss = 0.0
    for (a, b, sa, sb, _), w in zip(matches, weights):
        diff = (r[teams_idx[b]] - r[teams_idx[a]]) / 400.0
        p_a = 1.0 / (1.0 + 10.0 ** diff)
        loss -= w * (sa * np.log(p_a) + sb * np.log(1.0 - p_a))
    return loss


def compute_elo(matches, weights):
    teams = sorted({m[0] for m in matches} | {m[1] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    r0 = np.full(n, 1500.0)
    constraint = {"type": "eq", "fun": lambda r: np.mean(r) - 1500.0}

    res = minimize(
        neg_log_likelihood,
        r0,
        args=(idx, matches, weights),
        method="SLSQP",
        constraints=constraint,
        options={"ftol": 1e-12, "maxiter": 10_000},
    )
    print(f"Converged: {res.success} — {res.message}", file=sys.stderr)
    return {t: res.x[idx[t]] for t in teams}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", nargs="?", default=CACHE,
                        help="matches CSV, or - for stdin (default: %(default)s)")
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE,
                        help="days for a match's weight to halve; 0 disables decay "
                             "(default: %(default)s)")
    args = parser.parse_args()

    if args.source == "-":
        matches = load_matches(sys.stdin)
    else:
        with open(args.source) as f:
            matches = load_matches(f)

    if not matches:
        print("No matches found.", file=sys.stderr)
        sys.exit(1)

    weights = time_weights(matches, args.half_life)
    ratings = compute_elo(matches, weights)
    ranked = sorted(ratings.items(), key=lambda x: -x[1])

    writer = csv.writer(sys.stdout)

    print()
    writer.writerow(["rank", "team", "elo"])
    for rank, (team, elo) in enumerate(ranked, 1):
        writer.writerow([rank, team, f"{elo:.1f}"])

    print()
    writer.writerow(["date", "team_a", "team_b", "score_a", "score_b", "weight", "perf_a", "perf_b"])
    for (a, b, sa, sb, day), w in zip(matches, weights):
        if sa == 0 or sb == 0:
            writer.writerow([day, a, b, sa, sb, f"{w:.3f}", "forfeit", "forfeit"])
            continue
        # PR = opponent_elo + 400 * log10(own_points / opponent_points)
        pr_a = ratings[b] + 400 * np.log10(sa / sb)
        pr_b = ratings[a] + 400 * np.log10(sb / sa)
        writer.writerow([day, a, b, sa, sb, f"{w:.3f}", f"{pr_a:.1f}", f"{pr_b:.1f}"])


if __name__ == "__main__":
    main()
