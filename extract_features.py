
"""
Given a FEN on argv[1], prints the same feature vector used during training
(comma-separated), so it can be piped straight into the C++ inference tool.

Usage:
    python3 extract_features.py "<FEN>"
"""
import sys
import chess
from generate_data import extract_features, FEATURE_NAMES

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_features.py \"<FEN>\"", file=sys.stderr)
        sys.exit(1)
    fen = sys.argv[1]
    board = chess.Board(fen)
    feats = extract_features(board)
    print(",".join(str(v) for v in feats))

if __name__ == "__main__":
    main()