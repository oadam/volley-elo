# Volley ELO

Computes ELO ratings for volleyball teams from match results scraped from ffvbbeach.org.

Each point scored is treated as a minimatch. Ratings are found by maximising the
log-likelihood of all observed point totals simultaneously (convex optimisation via scipy).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Usage

### 1. Scrape match results

Fetches results from the FFVB export endpoint and saves them to `/tmp/ffvb_matches.csv`.

```bash
python3 scrape_matches.py
```

### 2. Compute ELO ratings

Reads from `/tmp/ffvb_matches.csv` (so you can iterate without re-hitting the server).

```bash
venv/bin/python compute_elo.py
```

Older matches are down-weighted with an exponential decay: a match's weight halves
every 180 days (counted back from the most recent match). Change it with
`--half-life DAYS`, or disable it with `--half-life 0`.

Outputs two CSV tables to stdout:

- **Rankings** — `rank, team, elo`
- **Per-match performance ratings** — `date, team_a, team_b, score_a, score_b, weight, perf_a, perf_b`

Performance rating for a team in a given match is the ELO they would need to have made
their observed point fraction exactly expected: `PR = opponent_elo + 400 × log10(own_pts / opp_pts)`.
It is zero-sum: one team's outperformance always equals the other's underperformance.

### Run both in one shot

```bash
python3 scrape_matches.py && venv/bin/python compute_elo.py
```
