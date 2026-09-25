# Volley ELO

Computes ELO ratings for volleyball teams from match results scraped from ffvbbeach.org.

Each point scored is treated as a minimatch. Ratings are found by maximising the
log-likelihood of all observed point totals simultaneously (convex optimisation via scipy).
Older matches are down-weighted so that ratings follow each team's current level.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Usage

### 1. Scrape match results

Fetches results from the FFVB export endpoint for every `(season, poule)` pair listed
in `POULES` in `scrape_matches.py`, and saves them to `/tmp/ffvb_matches.csv` with
columns `saison, poule, date, team_a, team_b, score_a, score_b`.

```bash
python3 scrape_matches.py
```

Poules currently scraped (committee `PTPO17`):

| Season    | Poules                                                   |
|-----------|----------------------------------------------------------|
| 2025/2026 | `ALA`, `ALB` (mixed first phase), `AL1`, `AL2`, `AL3`    |
| 2026/2027 | `LA1`, `LA2`, `LA3`, `LA4`                               |

Only played matches (with a points total) are kept. A poule with no results yet is
reported and skipped, so upcoming poules can be listed in advance.

The mixed first-phase poules (`ALA`, `ALB`) matter: they are the only matches between
teams that later end up in different divisions, and without them ratings across
divisions are not comparable.

**Finding poule codes.** Calling the export with an empty `cal_codpoule` returns every
poule of the committee for that season. The `Match` column is the poule code, a leg
letter (`A` aller / `R` retour) and a number, e.g. `AL1R014` → poule `AL1`:

```bash
curl -s -d "cal_saison=2025/2026&cal_codent=PTPO17&cal_codpoule=&typ_edition=E&type=RES" \
  https://www.ffvbbeach.org/ffvbapp/resu/vbspo_calendrier_export.php \
  | iconv -f latin1 -t utf8 | cut -d';' -f3 | sed 's/...$//' | sort | uniq -c
```

### 2. Compute ELO ratings

Reads from `/tmp/ffvb_matches.csv` by default (so you can iterate without re-hitting
the server), or from a given file, or from stdin with `-`.

```bash
venv/bin/python compute_elo.py
```

#### Time weighting

Each match's contribution to the log-likelihood is multiplied by

```
weight = 0.5 ^ (age_days / half_life)
```

where `age_days` is counted back from the most recent match in the data. The default
half-life is 180 days: a match from six months ago counts half as much as the latest
one, a match from a year ago a quarter. This lets ratings track form within a season
and roster changes between seasons, while older matches still anchor teams that have
only played a few games (typically at the start of a season).

Since only relative weights matter, the choice of reference date does not change the
ratings, and the problem stays convex. The weight of each match is shown in the
per-match output.

```bash
venv/bin/python compute_elo.py --half-life 365   # slower decay
venv/bin/python compute_elo.py --half-life 0     # no decay, all matches equal
```

A too-short half-life makes ratings rest on a handful of matches per team and become
noisy; a too-long one makes them slow to react.

#### Anchoring

Only rating differences are determined by the data, so one constraint fixes the
scale: the **weighted mean of all ratings is 0**, each team weighted by the sum of the
weights of the matches it played. A positive rating means above the current average,
a negative one below.

The zero therefore tracks the current field: teams that played recently define it,
and teams that stopped playing long ago barely move it. It still moves when the field
changes (e.g. new teams in a new season), which shifts every rating by the same
amount — compare rankings or rating gaps rather than raw values over time.

The scale is not comparable to chess ELO: since every point is a minimatch, a
100-point gap means winning about 64% of the points (roughly 25-14 per set).

#### Output

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
