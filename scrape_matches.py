import argparse
import re
import csv
import sys
import urllib.parse
import urllib.request
from pathlib import Path

EXPORT_URL = "https://www.ffvbbeach.org/ffvbapp/resu/vbspo_calendrier_export.php"

# A FINISHED poule is fetched once, then reused from the CSV; an ONGOING one is
# refetched on every run.
FINISHED = "FINISHED"
ONGOING = "ONGOING"

POULES = [
    ("2025/2026", "ALA", FINISHED),
    ("2025/2026", "ALB", FINISHED),
    ("2025/2026", "AL1", FINISHED),
    ("2025/2026", "AL2", FINISHED),
    ("2025/2026", "AL3", FINISHED),
    ("2026/2027", "LA1", ONGOING),
    ("2026/2027", "LA2", ONGOING),
    ("2026/2027", "LA3", ONGOING),
    ("2026/2027", "LA4", ONGOING),
]

PAYLOAD = {
    "cal_codent": "PTPO17",
    "cal_coddiv": "",
    "cal_codtour": "",
    "typ_edition": "E",
    "type": "RES",
    "rech_equipe": "",
}


def fetch_export(season, poule):
    params = {**PAYLOAD, "cal_saison": season, "cal_codpoule": poule}
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(EXPORT_URL, data=data)
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("latin-1", errors="replace")
    return list(csv.reader(text.splitlines(), delimiter=";"))


def parse_matches(rows, season, poule):
    # Columns: Entité, Jo, Match, Date, Heure, EQA_no, EQA_nom, EQB_no, EQB_nom,
    #          Set, Score, Total, Salle, Arb1, Arb2
    matches = []
    for row in rows:
        if len(row) < 12:
            continue
        total = str(row[11]).strip()
        m = re.fullmatch(r"(\d+)-(\d+)", total)
        if not m:
            continue
        team_a = str(row[6]).strip()
        team_b = str(row[8]).strip()
        if not team_a or not team_b:
            continue
        matches.append({
            "saison": season,
            "date": str(row[3]).strip(),
            "poule": poule,
            "team_a": team_a,
            "team_b": team_b,
            "score_a": int(m.group(1)),
            "score_b": int(m.group(2)),
        })
    return matches


CACHE = Path(__file__).with_name("matches.csv")
COLUMNS = ["saison", "poule", "date", "team_a", "team_b", "score_a", "score_b"]


def read_csv(path):
    """Existing matches grouped by (season, poule), or {} if the file is absent."""
    if not path.exists():
        return {}
    cached = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cached.setdefault((row["saison"], row["poule"]), []).append(row)
    return cached


def write_csv(matches, dest):
    writer = csv.DictWriter(dest, COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(matches)


def main():
    parser = argparse.ArgumentParser(description="Scrape FFVB match results.")
    parser.add_argument("--refresh", action="store_true",
                        help="refetch finished poules too instead of reusing the CSV")
    args = parser.parse_args()

    cached = {} if args.refresh else read_csv(CACHE)
    matches = []
    for season, poule, status in POULES:
        if status == FINISHED and (season, poule) in cached:
            poule_matches = cached[(season, poule)]
            print(f"{season} {poule}: {len(poule_matches)} matches (cached)", file=sys.stderr)
        else:
            rows = fetch_export(season, poule)
            poule_matches = parse_matches(rows, season, poule)
            if not poule_matches:
                print(f"{season} {poule}: no played matches yet", file=sys.stderr)
                continue
            print(f"{season} {poule}: {len(poule_matches)} matches", file=sys.stderr)
        matches.extend(poule_matches)

    if not matches:
        print("No matches parsed — check column layout.", file=sys.stderr)
        sys.exit(1)

    with open(CACHE, "w", newline="") as f:
        write_csv(matches, f)
    print(f"Saved {len(matches)} matches to {CACHE}", file=sys.stderr)
    write_csv(matches, sys.stdout)


if __name__ == "__main__":
    main()
