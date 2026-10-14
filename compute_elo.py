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

Ratings are anchored so that their weighted mean is 0, each team weighted by the
sum of its match weights. Teams that played recently define the zero; teams that
only played long ago barely move it.

Usage:
  python3 compute_elo.py                  # reads /tmp/ffvb_matches.csv
  python3 compute_elo.py matches.csv      # or a custom file
  python3 compute_elo.py --half-life 0    # no time decay
  python3 compute_elo.py --monthly        # rating at each month end
  python3 scrape_matches.py | python3 compute_elo.py -  # from stdin
"""
import argparse
import csv
import sys
from datetime import date, timedelta

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


K = np.log(10) / 400  # P(A wins a point) = sigmoid(K * (r_A - r_B))


def neg_log_likelihood(r, ia, ib, sa, sb, w):
    """Weighted negative log-likelihood and its gradient."""
    x = K * (r[ia] - r[ib])
    # log p_a = -log(1 + e^-x), log(1 - p_a) = -log(1 + e^x)
    loss = w @ (sa * np.logaddexp(0, -x) + sb * np.logaddexp(0, x))
    p_a = np.exp(-np.logaddexp(0, -x))
    g = -w * K * (sa - (sa + sb) * p_a)  # d loss / d r_A for each match
    grad = np.bincount(ia, g, len(r)) - np.bincount(ib, g, len(r))
    return loss, grad


def compute_elo(matches, weights):
    teams = sorted({m[0] for m in matches} | {m[1] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    ia = np.array([idx[m[0]] for m in matches])
    ib = np.array([idx[m[1]] for m in matches])
    sa = np.array([m[2] for m in matches], dtype=float)
    sb = np.array([m[3] for m in matches], dtype=float)
    w = np.asarray(weights, dtype=float)

    team_w = np.bincount(ia, w, n) + np.bincount(ib, w, n)
    team_w /= team_w.sum()

    r0 = np.zeros(n)
    constraint = {"type": "eq", "fun": lambda r: team_w @ r, "jac": lambda r: team_w}

    res = minimize(
        neg_log_likelihood,
        r0,
        args=(ia, ib, sa, sb, w),
        jac=True,
        method="SLSQP",
        constraints=constraint,
        options={"ftol": 1e-12, "maxiter": 10_000},
    )
    if not res.success:
        print(f"Not converged: {res.message}", file=sys.stderr)
    return {t: res.x[idx[t]] for t in teams}


def month_ends(matches):
    first = min(m[4] for m in matches).replace(day=1)
    last = max(m[4] for m in matches)
    month = first
    while month <= last:
        nxt = date(month.year + month.month // 12, month.month % 12 + 1, 1)
        yield month, min(nxt - timedelta(days=1), last)
        month = nxt


def is_connected(matches):
    """Whether every team is linked to every other through a chain of matches.

    Otherwise the gap between groups that never met is undetermined, and the
    ratings are not unique.
    """
    neighbours = {}
    for a, b, *_ in matches:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    seen, todo = set(), [next(iter(neighbours))]
    while todo:
        team = todo.pop()
        if team not in seen:
            seen.add(team)
            todo.extend(neighbours[team] - seen)
    return len(seen) == len(neighbours)


def monthly_elo(matches, half_life):
    """Ratings refitted at each month end, using only the matches played by then.

    Months where the teams are not all connected are left empty and skipped
    in the output.
    """
    history = {}
    for month, end in month_ends(matches):
        key = month.strftime("%Y-%m")
        played = [m for m in matches if m[4] <= end]
        if is_connected(played):
            history[key] = compute_elo(played, time_weights(played, half_life))
        else:
            print(f"{key}: teams not all connected by matches, left empty", file=sys.stderr)
            history[key] = {}
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", nargs="?", default=CACHE,
                        help="matches CSV, or - for stdin (default: %(default)s)")
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE,
                        help="days for a match's weight to halve; 0 disables decay "
                             "(default: %(default)s)")
    parser.add_argument("--monthly", action="store_true",
                        help="output one rating column per month, each refitted on "
                             "the matches played by the end of that month")
    args = parser.parse_args()

    if args.source == "-":
        matches = load_matches(sys.stdin)
    else:
        with open(args.source) as f:
            matches = load_matches(f)

    if not matches:
        print("No matches found.", file=sys.stderr)
        sys.exit(1)

    writer = csv.writer(sys.stdout)

    if args.monthly:
        history = monthly_elo(matches, args.half_life)
        months = [m for m in history if history[m]]
        if not months:
            return
        latest = history[months[-1]]
        teams = sorted(latest, key=lambda t: -latest[t])
        writer.writerow(["month", *teams])
        for m in months:
            writer.writerow([m, *(f"{round(history[m][t], 1) + 0.0:.1f}" if t in history[m] else ""
                                  for t in teams)])
        return

    weights = time_weights(matches, args.half_life)
    ratings = compute_elo(matches, weights)
    ranked = sorted(ratings.items(), key=lambda x: -x[1])

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
