import re
import csv
import sys
import urllib.parse
import urllib.request

EXPORT_URL = "https://www.ffvbbeach.org/ffvbapp/resu/vbspo_calendrier_export.php"

PAYLOAD = {
    "cal_saison": "2025/2026",
    "cal_codent": "PTPO17",
    "cal_codpoule": "AL1",
    "cal_coddiv": "",
    "cal_codtour": "",
    "typ_edition": "E",
    "type": "RES",
    "rech_equipe": "",
}


def fetch_export():
    data = urllib.parse.urlencode(PAYLOAD).encode()
    req = urllib.request.Request(EXPORT_URL, data=data)
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("latin-1", errors="replace")
    return list(csv.reader(text.splitlines(), delimiter=";"))


def parse_matches(rows):
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
            "team_a": team_a,
            "team_b": team_b,
            "score_a": int(m.group(1)),
            "score_b": int(m.group(2)),
        })
    return matches


CACHE = "/tmp/ffvb_matches.csv"


def write_csv(matches, dest):
    writer = csv.writer(dest)
    writer.writerow(["team_a", "team_b", "score_a", "score_b"])
    for m in matches:
        writer.writerow([m["team_a"], m["team_b"], m["score_a"], m["score_b"]])


def main():
    rows = fetch_export()
    matches = parse_matches(rows)
    if not matches:
        print("No matches parsed — check column layout.", file=sys.stderr)
        for r in rows[:5]:
            print(r, file=sys.stderr)
        sys.exit(1)

    with open(CACHE, "w", newline="") as f:
        write_csv(matches, f)
    print(f"Saved {len(matches)} matches to {CACHE}", file=sys.stderr)
    write_csv(matches, sys.stdout)


if __name__ == "__main__":
    main()
