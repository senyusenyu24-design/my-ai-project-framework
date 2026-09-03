
"""
Cleans the raw Stockfish-labeled dataset before training:
 - clips extreme evaluations (hung queens, near-mate scores) to +/-1500 cp
 - scales centipawns down to roughly [-15, 15] for stable regression

Usage:
    python3 clean_data.py data/positions.csv data/positions_clean.csv
"""
import csv
import sys

CLIP_CP = 1500.0
SCALE = 100.0

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 clean_data.py <in.csv> <out.csv>", file=sys.stderr)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    eval_idx = header.index("eval_cp")
    for row in rows:
        v = float(row[eval_idx])
        v = max(-CLIP_CP, min(CLIP_CP, v))
        row[eval_idx] = str(v / SCALE)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Cleaned {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()