"""
Generates a labeled dataset for chess position evaluation:
 - plays semi-random self-play games with python-chess
 - samples positions from those games
 - extracts a hand-crafted feature vector per position
 - labels each position with a real Stockfish evaluation (centipawns, White's POV)

Output: CSV with feature columns + target eval (scaled to roughly [-10, 10]).
"""
import chess
import subprocess
import random
import csv
import sys

STOCKFISH_PATH = "/usr/games/stockfish"
MOVETIME_MS = 60          # eval time budget per position
N_GAMES = 260              # number of self-play games
MAX_PLIES = 50
POSITIONS_PER_GAME = 8     # sampled positions per game
MIN_PLY_TO_SAMPLE = 4     

FEATURE_NAMES = [
    "material_pawn", "material_knight", "material_bishop", "material_rook", "material_queen",
    "mobility_diff", "white_king_shield", "black_king_shield",
    "doubled_pawns_diff", "isolated_pawns_diff", "passed_pawns_diff",
    "center_control_diff", "total_material", "side_to_move",
    "white_bishop_pair", "black_bishop_pair", "rook_open_file_diff",
    "development_diff", "castling_rights_diff", "king_attackers_diff",
]

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class Engine:
    """Thin persistent UCI wrapper around Stockfish (spawning per-call is too slow)."""
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if token in line:
                return line

    def eval_cp(self, fen, movetime_ms=MOVETIME_MS):
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        last_score = None
        mate_score = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            if "score cp" in line:
                parts = line.split()
                idx = parts.index("cp")
                last_score = int(parts[idx + 1])
                mate_score = None
            elif "score mate" in line:
                parts = line.split()
                idx = parts.index("mate")
                mate_in = int(parts[idx + 1])
                mate_score = 1000 if mate_in > 0 else -1000
            if line.startswith("bestmove"):
                break
      
        score = mate_score if mate_score is not None else last_score
        if score is None:
            return 0.0
        board = chess.Board(fen)
        if board.turn == chess.BLACK:
            score = -score
        return float(score)

    def close(self):
        try:
            self._send("quit")
            self.proc.terminate()
        except Exception:
            pass


def count_attackers_near_king(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    opp = not color
    attackers = 0
    for sq in chess.SQUARES:
        if chess.square_distance(sq, king_sq) <= 1:
            attackers += len(board.attackers(opp, sq))
    return attackers


def pawn_shield(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield = 0
    forward = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = king_file + df
        if 0 <= f <= 7:
            r = king_rank + forward
            if 0 <= r <= 7:
                sq = chess.square(f, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    shield += 1
    return shield


def doubled_isolated_passed(board, color):
    files_with_pawns = [0] * 8
    pawn_squares = board.pieces(chess.PAWN, color)
    for sq in pawn_squares:
        files_with_pawns[chess.square_file(sq)] += 1
    doubled = sum(c - 1 for c in files_with_pawns if c > 1)
    isolated = 0
    for f in range(8):
        if files_with_pawns[f] > 0:
            left = files_with_pawns[f - 1] if f > 0 else 0
            right = files_with_pawns[f + 1] if f < 7 else 0
            if left == 0 and right == 0:
                isolated += files_with_pawns[f]
    opp_pawns = board.pieces(chess.PAWN, not color)
    passed = 0
    for sq in pawn_squares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocked = False
        for opp_sq in opp_pawns:
            of = chess.square_file(opp_sq)
            orr = chess.square_rank(opp_sq)
            if abs(of - f) <= 1:
                if (color == chess.WHITE and orr > r) or (color == chess.BLACK and orr < r):
                    blocked = True
                    break
        if not blocked:
            passed += 1
    return doubled, isolated, passed


def extract_features(board):
    feats = {}
    for pt, name in [(chess.PAWN, "pawn"), (chess.KNIGHT, "knight"),
                      (chess.BISHOP, "bishop"), (chess.ROOK, "rook"), (chess.QUEEN, "queen")]:
        w = len(board.pieces(pt, chess.WHITE))
        b = len(board.pieces(pt, chess.BLACK))
        feats[f"material_{name}"] = (w - b) * PIECE_VALUES[pt]

    board_w = board.copy(); board_w.turn = chess.WHITE
    board_b = board.copy(); board_b.turn = chess.BLACK
    mobility_w = board_w.legal_moves.count()
    mobility_b = board_b.legal_moves.count()
    feats["mobility_diff"] = mobility_w - mobility_b

    feats["white_king_shield"] = pawn_shield(board, chess.WHITE)
    feats["black_king_shield"] = pawn_shield(board, chess.BLACK)

    dw, iw, pw = doubled_isolated_passed(board, chess.WHITE)
    db, ib, pb = doubled_isolated_passed(board, chess.BLACK)
    feats["doubled_pawns_diff"] = db - dw
    feats["isolated_pawns_diff"] = ib - iw
    feats["passed_pawns_diff"] = pw - pb

    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    center_w = sum(len(board.attackers(chess.WHITE, sq)) for sq in center)
    center_b = sum(len(board.attackers(chess.BLACK, sq)) for sq in center)
    feats["center_control_diff"] = center_w - center_b

    total_material = sum(
        len(board.pieces(pt, c)) * PIECE_VALUES[pt]
        for pt in PIECE_VALUES for c in (chess.WHITE, chess.BLACK)
    )
    feats["total_material"] = total_material
    feats["side_to_move"] = 1.0 if board.turn == chess.WHITE else -1.0

    def bishop_pair(color):
        bishops = board.pieces(chess.BISHOP, color)
        return 1.0 if len(bishops) >= 2 else 0.0
    feats["white_bishop_pair"] = bishop_pair(chess.WHITE)
    feats["black_bishop_pair"] = bishop_pair(chess.BLACK)

    def rook_open_files(color):
        cnt = 0
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            pawns_on_file = any(
                chess.square_file(p) == f
                for p in list(board.pieces(chess.PAWN, chess.WHITE)) + list(board.pieces(chess.PAWN, chess.BLACK))
            )
            if not pawns_on_file:
                cnt += 1
        return cnt
    feats["rook_open_file_diff"] = rook_open_files(chess.WHITE) - rook_open_files(chess.BLACK)

    def development(color):
        back_rank = 0 if color == chess.WHITE else 7
        developed = 0
        for pt in (chess.KNIGHT, chess.BISHOP):
            for sq in board.pieces(pt, color):
                if chess.square_rank(sq) != back_rank:
                    developed += 1
        return developed
    feats["development_diff"] = development(chess.WHITE) - development(chess.BLACK)

    cr = 0
    cr += 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    cr += 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    cr -= 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    cr -= 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    feats["castling_rights_diff"] = cr

    feats["king_attackers_diff"] = (
        count_attackers_near_king(board, chess.BLACK) - count_attackers_near_king(board, chess.WHITE)
    )

    return [feats[name] for name in FEATURE_NAMES]


def random_game_positions():
    board = chess.Board()
    n_plies = random.randint(20, MAX_PLIES)
    positions = []
    for ply in range(n_plies):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        
        captures = [m for m in legal if board.is_capture(m)]
        if captures and random.random() < 0.35:
            move = random.choice(captures)
        else:
            move = random.choice(legal)
        board.push(move)
        if ply >= MIN_PLY_TO_SAMPLE:
            positions.append(board.fen())
    return positions


def main():
    engine = Engine(STOCKFISH_PATH)
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/positions.csv"
    rows_written = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["eval_cp"])

        for g in range(N_GAMES):
            fens = random_game_positions()
            if not fens:
                continue
            sample = random.sample(fens, min(POSITIONS_PER_GAME, len(fens)))
            for fen in sample:
                board = chess.Board(fen)
                if board.is_game_over():
                    continue
                feats = extract_features(board)
                eval_cp = engine.eval_cp(fen)
                writer.writerow(feats + [eval_cp])
                rows_written += 1
            if g % 20 == 0:
                print(f"game {g}/{N_GAMES}, rows so far: {rows_written}", file=sys.stderr)
                f.flush()

    engine.close()
    print(f"Done. Wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()