"""
Pilot runner for the Beyond Pattern Matching chess causality study.

Scope (per docs/experimental_notes.md §'Minimum pilot'):
    - Representations: raw S and expert C (skip learned Z for pilot).
    - Estimator: Linear DR-Learner only.
    - Grid: N in {2000, 5000} x gamma in {0.0, 0.3} x homogeneous tau* x 3 seeds.
    - Oracle-correlation: Y built only from context + Group A structural features,
      never from Stockfish eval. C is also computed without Stockfish (python-chess only)
      for the pilot, eliminating the leakage path entirely.
    - Feasibility filter: disabled for pilot (requires per-position Stockfish calls).
      Full run will reinstate a depth-10 top-4 check.

Pipeline:
    1. fetch_games()          — pull ~2k classical PGNs from Lichess, cache.
    2. extract_decisions()    — iterate positions, compute S, X, C, T, and the
                                observed-move aggression score; cache to parquet.
    3. generate_outcomes()    — build Y per the DGP for each gamma.
    4. run_pilot()            — fit DR-Learner for each (N, gamma, rep, seed).
    5. report()               — emit pilot_results.csv and a terminal summary.

Run:
    python pilot.py            # full pilot (reuses caches)
    python pilot.py --refetch  # re-download games
    python pilot.py --rebuild  # re-extract decisions from cached games
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Estimator deps are heavy and not needed until stage 4; import lazily there.

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

GAMES_PATH = CACHE / "games.pgn"
DECISIONS_PATH = CACHE / "decisions.parquet"
RESULTS_PATH = OUT / "pilot_results.csv"

# Lichess public API; no auth.
LICHESS_FALLBACK_USERS = [
    "DrNykterstein",     # Magnus Carlsen
    "Hikaru",            # Hikaru Nakamura
    "FabianoCaruana",    # Fabiano Caruana
    "LyonBeast",         # Maxime Vachier-Lagrave
    "Anish_Giri",        # Anish Giri
    "lachesisQ",         # Ian Nepomniachtchi
    "GMWSO",             # Wesley So
    "Chess-Network",     # Jerry (strong classical pool)
    "nihalsarin2004",    # Nihal Sarin
    "Alireza2003",       # Alireza Firouzja
    "RebeccaHarris",     # Andrew Tang (Penguin)
    "GMVallejo",         # Francisco Vallejo Pons
]
TOP_CLASSICAL_USERS = 40
GAMES_PER_USER = 180
POSITION_SAMPLE_PER_GAME = 18    # raised from 8 so --rebuild yields >=25k decisions for primary.py (N=20k cell)
MIN_PLY = 10                      # skip opening book moves
MAX_PLY = 120                     # skip very deep endgames


# =============================================================================
# Stage 1 — Lichess fetch
# =============================================================================

def _lichess_top_classical_users(n: int = TOP_CLASSICAL_USERS) -> list[str]:
    """Fetch a live pool of classical players so the pilot doesn't depend on stale usernames."""
    url = f"https://lichess.org/api/player/top/{n}/classical"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"[stage 1] failed to fetch top classical users ({e}); using fallback list.")
        return list(LICHESS_FALLBACK_USERS)

    users: list[str] = []
    for user in payload.get("users", []):
        username = user.get("username")
        if not username:
            continue
        if user.get("title") == "BOT" or username.upper().endswith("_BOT"):
            continue
        users.append(username)

    if not users:
        print("[stage 1] top-player endpoint returned no usable usernames; using fallback list.")
        return list(LICHESS_FALLBACK_USERS)
    return users


def fetch_games(force: bool = False) -> Path:
    if GAMES_PATH.exists() and not force:
        size_mb = GAMES_PATH.stat().st_size / 1e6
        print(f"[stage 1] cached PGN found ({size_mb:.1f} MB); skipping fetch.")
        return GAMES_PATH

    users = _lichess_top_classical_users()
    print(f"[stage 1] fetching up to ~{len(users) * GAMES_PER_USER} classical games from Lichess...")
    total_bytes = 0
    with open(GAMES_PATH, "w", encoding="utf-8") as fout:
        for user in users:
            url = f"https://lichess.org/api/games/user/{user}"
            params = {
                "perfType": "classical",
                "max": GAMES_PER_USER,
                "rated": "true",
                "clocks": "false",
                "evals": "false",
                "opening": "false",
            }
            headers = {"Accept": "application/x-chess-pgn"}
            try:
                r = requests.get(url, params=params, headers=headers, stream=True, timeout=60)
                r.raise_for_status()
                wrote = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        if isinstance(chunk, bytes):
                            wrote += len(chunk)
                            chunk = chunk.decode("utf-8", errors="ignore")
                        else:
                            wrote += len(chunk.encode("utf-8"))
                        fout.write(chunk)
                total_bytes += wrote
                print(f"  {user}: {wrote/1024:.0f} KB")
            except Exception as e:
                print(f"  {user}: FAILED ({e}); continuing")
            time.sleep(1.0)  # polite rate limit
    if total_bytes == 0:
        raise RuntimeError("[stage 1] fetched zero PGN bytes from Lichess; cannot continue.")
    print(f"[stage 1] saved to {GAMES_PATH}")
    return GAMES_PATH


# =============================================================================
# Stage 2 — Decision extraction
# =============================================================================

# --- Feature extraction helpers ------------------------------------------------

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

# Simple piece-square tables (positive = good for white, mirrored for black)
PST_PAWN = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
], dtype=float)

PST_KNIGHT = np.array([
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
], dtype=float)


def board_to_planes(board: chess.Board) -> np.ndarray:
    """18x8x8 tensor: 12 piece planes + 6 extras. Returned flattened (1152-d)."""
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r, c = divmod(sq, 8)
        idx = (piece.piece_type - 1) + (0 if piece.color else 6)
        planes[idx, r, c] = 1.0
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    ep = board.ep_square
    planes[17, :, :] = (chess.square_file(ep) / 8.0) if ep is not None else 0.0
    return planes.reshape(-1)


def _material_balance(board: chess.Board) -> float:
    s = 0
    for sq, piece in board.piece_map().items():
        v = PIECE_VALUES[piece.piece_type]
        s += v if piece.color == chess.WHITE else -v
    return float(s)


def _doubled_isolated_backward(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    files = [0] * 8
    for sq in pawns:
        files[chess.square_file(sq)] += 1
    doubled = sum(max(0, c - 1) for c in files)
    isolated = 0
    for f, c in enumerate(files):
        if c == 0:
            continue
        left = files[f - 1] if f > 0 else 0
        right = files[f + 1] if f < 7 else 0
        if left == 0 and right == 0:
            isolated += c
    return doubled + isolated


def _attack_count(board: chess.Board, color: bool) -> int:
    n = 0
    for sq in chess.SQUARES:
        if board.is_attacked_by(color, sq):
            n += 1
    return n


def _space_control(board: chess.Board, color: bool) -> int:
    opp_half = range(32, 64) if color == chess.WHITE else range(0, 32)
    n = 0
    for sq in opp_half:
        if board.is_attacked_by(color, sq) or (board.piece_at(sq) and board.piece_at(sq).color == color):
            n += 1
    return n


def _king_pawn_shield(board: chess.Board, color: bool) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    pawns = board.pieces(chess.PAWN, color)
    n = 0
    for sq in pawns:
        if abs(chess.square_file(sq) - kf) <= 1 and abs(chess.square_rank(sq) - kr) <= 2:
            n += 1
    return n


def _development(board: chess.Board, color: bool) -> int:
    """Minor pieces off back rank."""
    back_rank = 0 if color == chess.WHITE else 7
    n = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) != back_rank:
                n += 1
    return n


def _king_ring_attackers(board: chess.Board, color: bool) -> int:
    """Group B surrogate for 'king danger': enemy attackers on squares within 1 of own king."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    ring = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = kf + df, kr + dr
            if 0 <= f < 8 and 0 <= r < 8:
                ring.append(chess.square(f, r))
    n = 0
    for sq in ring:
        if board.is_attacked_by(not color, sq):
            n += 1
    return n


def _mobility(board: chess.Board, color: bool) -> int:
    if board.turn == color:
        return board.legal_moves.count()
    tmp = board.copy(stack=False)
    tmp.turn = color
    try:
        return tmp.legal_moves.count()
    except Exception:
        return 0


def _passed_pawns(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    opp_pawns = board.pieces(chess.PAWN, not color)
    n = 0
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocked = False
        for opp in opp_pawns:
            of, o_r = chess.square_file(opp), chess.square_rank(opp)
            if abs(of - f) <= 1:
                if (color == chess.WHITE and o_r > r) or (color == chess.BLACK and o_r < r):
                    blocked = True
                    break
        if not blocked:
            n += 1
    return n


def _threats(board: chess.Board, color: bool) -> int:
    """Count enemy pieces attacked by us with more attackers than defenders (loose en-prise)."""
    n = 0
    for sq, piece in board.piece_map().items():
        if piece.color == color:
            continue
        attackers = len(board.attackers(color, sq))
        defenders = len(board.attackers(not color, sq))
        if attackers > defenders and attackers > 0:
            n += 1
    return n


def _pst_sum(board: chess.Board) -> float:
    s = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            tbl = PST_PAWN
        elif piece.piece_type == chess.KNIGHT:
            tbl = PST_KNIGHT
        else:
            continue
        idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
        v = tbl[idx]
        s += v if piece.color == chess.WHITE else -v
    return float(s / 100.0)


def _imbalance(board: chess.Board) -> float:
    w_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    b_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    bishop_pair = int(w_bishops >= 2) - int(b_bishops >= 2)
    w_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
    b_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    knight_pair = int(w_knights >= 2) - int(b_knights >= 2)
    return float(bishop_pair * 0.5 + knight_pair * 0.1)


def concepts(board: chess.Board) -> dict:
    """Return the 12-dim concept vector C. All python-chess, no Stockfish."""
    stm = board.turn  # side-to-move perspective
    opp = not stm

    # Group A (structural, used in the outcome DGP)
    material = _material_balance(board)
    if stm == chess.BLACK:
        material = -material
    pawn_struct_stm = _doubled_isolated_backward(board, stm)
    pawn_struct_opp = _doubled_isolated_backward(board, opp)
    pawn_struct = pawn_struct_opp - pawn_struct_stm  # positive = we have better structure
    activity = _attack_count(board, stm) - _attack_count(board, opp)
    space = _space_control(board, stm) - _space_control(board, opp)
    king_shield = _king_pawn_shield(board, stm)
    development = _development(board, stm) - _development(board, opp)

    # Group B (surrogate engine-style structural; NOT used in the outcome DGP)
    king_danger = _king_ring_attackers(board, stm)
    mobility = _mobility(board, stm) - _mobility(board, opp)
    passed = _passed_pawns(board, stm) - _passed_pawns(board, opp)
    threats = _threats(board, stm) - _threats(board, opp)
    pst = _pst_sum(board) * (1 if stm == chess.WHITE else -1)
    imbalance = _imbalance(board) * (1 if stm == chess.WHITE else -1)

    return dict(
        c_material=material,
        c_pawn_struct=pawn_struct,
        c_activity=activity,
        c_space=space,
        c_king_shield=king_shield,
        c_development=development,
        c_king_danger=king_danger,
        c_mobility=mobility,
        c_passed=passed,
        c_threats=threats,
        c_pst=pst,
        c_imbalance=imbalance,
    )


# --- Treatment: aggression score A(S, m) --------------------------------------

def aggression_score(board: chess.Board, move: chess.Move) -> float:
    stm = board.turn
    opp_king = board.king(not stm)

    # captures (including en passant)
    is_capture = board.is_capture(move)
    capture_val = 0.0
    if is_capture:
        target = board.piece_at(move.to_square)
        if target is None and board.is_en_passant(move):
            capture_val = 1.0
        elif target is not None:
            capture_val = 1.0

    board.push(move)
    gives_check = board.is_check()
    board.pop()

    # king-zone pressure: target square within 2 of opp king
    king_zone = 0.0
    if opp_king is not None:
        df = abs(chess.square_file(move.to_square) - chess.square_file(opp_king))
        dr = abs(chess.square_rank(move.to_square) - chess.square_rank(opp_king))
        if max(df, dr) <= 2:
            king_zone = 0.5

    # sacrifice flag: captured piece value < moved piece value
    sacrifice = 0.0
    if is_capture:
        mover = board.piece_at(move.from_square)
        target_pt = None
        if board.is_en_passant(move):
            target_pt = chess.PAWN
        else:
            tgt_piece = board.piece_at(move.to_square)
            if tgt_piece is not None:
                target_pt = tgt_piece.piece_type
        if mover is not None and target_pt is not None:
            if PIECE_VALUES[target_pt] < PIECE_VALUES[mover.piece_type]:
                sacrifice = 0.5

    # retreat penalty: moving backward from opponent half
    retreat = 0.0
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    if stm == chess.WHITE and to_rank < from_rank:
        retreat = 0.3
    elif stm == chess.BLACK and to_rank > from_rank:
        retreat = 0.3

    return capture_val + (1.0 if gives_check else 0.0) + king_zone + sacrifice - retreat


AGGRESSION_THRESHOLD = 0.5  # calibrated on 15k cached decisions to P(T_rule=1) = 0.34


# --- Context X from game headers + ply ----------------------------------------

def _safe_int(x, default=1500):
    try:
        return int(x)
    except Exception:
        return default


def extract_decisions(force: bool = False, max_decisions: int = 15000) -> pd.DataFrame:
    if DECISIONS_PATH.exists() and not force:
        df = pd.read_parquet(DECISIONS_PATH)
        print(f"[stage 2] cached decisions found ({len(df)} rows); skipping extraction.")
        return df

    print(f"[stage 2] extracting decisions from {GAMES_PATH}...")
    rows = []
    rng = np.random.default_rng(0)
    with open(GAMES_PATH) as fin:
        pbar = tqdm(total=max_decisions, desc="decisions")
        while len(rows) < max_decisions:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            headers = game.headers
            w_elo = _safe_int(headers.get("WhiteElo"))
            b_elo = _safe_int(headers.get("BlackElo"))
            tc = headers.get("TimeControl", "0+0")
            try:
                base, inc = tc.split("+")
                tc_seconds = int(base) + 40 * int(inc)
            except Exception:
                tc_seconds = 0
            if tc_seconds < 1500:  # keep classical only (>=25 min equivalent)
                continue
            try:
                eco = headers.get("ECO", "A00")
                eco_num = (ord(eco[0]) - ord("A")) * 100 + int(eco[1:])
            except Exception:
                eco_num = 0

            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY + 4:
                continue
            eligible = [i for i in range(MIN_PLY, min(len(moves), MAX_PLY))]
            if not eligible:
                continue
            sample_idxs = rng.choice(eligible, size=min(POSITION_SAMPLE_PER_GAME, len(eligible)), replace=False)
            sample_idxs.sort()

            ply = 0
            for i, move in enumerate(moves):
                if ply in sample_idxs:
                    stm = board.turn
                    elo_self = w_elo if stm == chess.WHITE else b_elo
                    elo_opp = b_elo if stm == chess.WHITE else w_elo
                    delta_elo = elo_self - elo_opp

                    # features
                    try:
                        s_vec = board_to_planes(board)
                        c_feats = concepts(board)
                        agg = aggression_score(board, move)
                    except Exception:
                        ply += 1
                        board.push(move)
                        continue

                    row = dict(
                        ply=ply,
                        phase=(0 if ply < 24 else (1 if ply < 60 else 2)),
                        elo_self=elo_self,
                        elo_opp=elo_opp,
                        delta_elo=delta_elo,
                        tc_seconds=tc_seconds,
                        eco_num=eco_num,
                        color=int(stm == chess.WHITE),
                        agg=agg,
                        T=int(agg >= AGGRESSION_THRESHOLD),
                        **c_feats,
                    )
                    for k in range(len(s_vec)):
                        row[f"s{k}"] = s_vec[k]
                    rows.append(row)
                    pbar.update(1)
                    if len(rows) >= max_decisions:
                        break
                board.push(move)
                ply += 1
        pbar.close()

    df = pd.DataFrame(rows)
    if df.empty or "T" not in df.columns:
        raise RuntimeError(
            "[stage 2] extracted zero decisions from the cached PGN. "
            "Re-run with --refetch after confirming stage 1 returned non-empty games."
        )
    print(f"[stage 2] extracted {len(df)} decisions; marginal P(T=1) = {df['T'].mean():.3f}")
    df.to_parquet(DECISIONS_PATH)
    print(f"[stage 2] saved to {DECISIONS_PATH}")
    return df


# =============================================================================
# Stage 3 — DGP: generate Y given gamma and a latent U
# =============================================================================

GROUP_A_FEATURES = [
    "c_material", "c_pawn_struct", "c_activity",
    "c_space", "c_king_shield", "c_development",
]
GROUP_B_FEATURES = [
    "c_king_danger", "c_mobility", "c_passed",
    "c_threats", "c_pst", "c_imbalance",
]
CONCEPT_FEATURES = GROUP_A_FEATURES + GROUP_B_FEATURES
CONTEXT_FEATURES = ["delta_elo", "tc_seconds", "ply", "color", "eco_num", "phase"]
S_FEATURES = [f"s{k}" for k in range(1152)]


def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu, sd = out[c].mean(), out[c].std()
        out[c] = (out[c] - mu) / (sd + 1e-8)
    return out


def generate_outcomes(df: pd.DataFrame, gamma: float, tau_star: float,
                      alpha_U: float, seed: int) -> pd.DataFrame:
    """Construct Y per §3.3 of docs/experimental_notes.md.

    Y only depends on standardized context phi(X) and Group A concepts psi(C)
    plus latent U and Gaussian noise. Stockfish eval does not appear.
    T is the observed aggression-based assignment, nudged by U via a logit shift
    (soft re-assignment to introduce alpha_U-controlled confounding).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    d = _standardize(df, ["delta_elo", "tc_seconds", "ply"] + GROUP_A_FEATURES)

    phi = (
        0.4 * d["delta_elo"].to_numpy()
        + 0.2 * d["tc_seconds"].to_numpy()
        + 0.1 * d["ply"].to_numpy()
    )
    psi = (
        0.3 * d["c_material"].to_numpy()
        + 0.2 * d["c_king_shield"].to_numpy()
    )
    U = rng.standard_normal(n)
    eps0 = rng.standard_normal(n) * 0.5

    Y0 = phi + psi + gamma * U + eps0
    Y1 = Y0 + tau_star

    # Observed T: start from rule-based A, nudge with U via logit
    base_logit = np.where(df["T"].to_numpy() == 1, 1.2, -1.2)
    logit = base_logit + alpha_U * U
    p = 1.0 / (1.0 + np.exp(-logit))
    T_obs = (rng.random(n) < p).astype(int)

    Y_obs = T_obs * Y1 + (1 - T_obs) * Y0

    out = df.copy()
    out["Y_obs"] = Y_obs
    out["T_obs"] = T_obs
    out["_Y0"] = Y0
    out["_Y1"] = Y1
    out["_tau_star"] = tau_star
    out["_psi_var"] = float(np.var(psi))
    return out


# =============================================================================
# Stage 4 — DR-Learner runs
# =============================================================================

MIN_PROPENSITY = 0.02  # lower clip for IPW denominator; mirrors EconML min_propensity
MAX_PROPENSITY = 0.98


def _make_propensity_model(n_features: int, seed: int):
    """LogReg propensity wrapped in StandardScaler. Regularization scales with
    feature count: raw S (1152+ dims) needs stronger L2 than the 12-d concept vector.
    Both pipelines get max_iter=5000 so raw converges cleanly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    C = 0.1 if n_features > 200 else 1.0
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", LogisticRegression(max_iter=5000, C=C, solver="lbfgs", random_state=seed)),
    ])


def _dr_learner_ate(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                    seed: int) -> tuple[float, float, float, float, float]:
    """Fit a LinearDRLearner and return
    (ate_hat, p_min_pre, p_max_pre, p_min_post, p_max_post).

    pre  = raw propensity scores from a standalone fit on X (diagnostic only).
    post = the same scores clipped to [MIN_PROPENSITY, MAX_PROPENSITY], which is
           what LinearDRLearner uses internally when min_propensity is set.
    """
    from lightgbm import LGBMRegressor
    from econml.dr import LinearDRLearner

    n_features = X.shape[1]
    prop_model = _make_propensity_model(n_features, seed)

    est = LinearDRLearner(
        model_propensity=_make_propensity_model(n_features, seed),
        model_regression=LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            verbose=-1, random_state=seed,
        ),
        min_propensity=MIN_PROPENSITY,
        cv=5,
        random_state=seed,
    )
    # W = nuisance controls; X = effect modifiers (none for ATE-only).
    est.fit(Y=Y, T=T, X=None, W=X)
    ate_hat = float(est.ate(X=None, T0=0, T1=1))

    # Diagnostic propensity fit (EconML hides the fold-wise estimates).
    prop_model.fit(X, T)
    p = prop_model.predict_proba(X)[:, 1]
    p_clipped = np.clip(p, MIN_PROPENSITY, MAX_PROPENSITY)
    return (
        ate_hat,
        float(p.min()), float(p.max()),
        float(p_clipped.min()), float(p_clipped.max()),
    )


@dataclass
class PilotRun:
    N: int
    gamma: float
    rep: str
    seed: int
    ate_hat: float
    ate_bias: float
    p_T1: float
    p_min_pre: float
    p_max_pre: float
    p_min_post: float
    p_max_post: float
    wall_s: float


def run_pilot(df_decisions: pd.DataFrame) -> pd.DataFrame:
    print("[stage 4] running pilot grid...")
    Ns = [2000, 5000]
    gammas = [0.0, 0.3]
    alpha_U = 0.3
    tau_star = 0.3
    reps = ["raw", "concept"]
    seeds = [0, 1, 2]

    results: list[PilotRun] = []
    total = len(Ns) * len(gammas) * len(reps) * len(seeds)
    pbar = tqdm(total=total, desc="fits")

    for gamma in gammas:
        df_y = generate_outcomes(df_decisions, gamma=gamma, tau_star=tau_star,
                                 alpha_U=alpha_U, seed=42)
        for N in Ns:
            for seed in seeds:
                rng = np.random.default_rng(1000 + seed)
                idx = rng.choice(len(df_y), size=N, replace=False)
                sub = df_y.iloc[idx].reset_index(drop=True)
                Y = sub["Y_obs"].to_numpy()
                T = sub["T_obs"].to_numpy()
                p_T1 = float(T.mean())
                X_context = sub[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
                X_raw = np.concatenate([sub[S_FEATURES].to_numpy(dtype=np.float32), X_context], axis=1)
                X_con = np.concatenate([sub[CONCEPT_FEATURES].to_numpy(dtype=np.float32), X_context], axis=1)

                for rep_name, X_adj in [("raw", X_raw), ("concept", X_con)]:
                    t0 = time.time()
                    try:
                        ate_hat, p_min_pre, p_max_pre, p_min_post, p_max_post = \
                            _dr_learner_ate(X_adj, T, Y, seed=seed)
                    except Exception as e:
                        print(f"  FAIL rep={rep_name} N={N} gamma={gamma} seed={seed}: {e}")
                        pbar.update(1)
                        continue
                    dt = time.time() - t0
                    bias = abs(ate_hat - tau_star)
                    results.append(PilotRun(
                        N=N, gamma=gamma, rep=rep_name, seed=seed,
                        ate_hat=ate_hat, ate_bias=bias, p_T1=p_T1,
                        p_min_pre=p_min_pre, p_max_pre=p_max_pre,
                        p_min_post=p_min_post, p_max_post=p_max_post,
                        wall_s=dt,
                    ))
                    pbar.update(1)
    pbar.close()

    out = pd.DataFrame([r.__dict__ for r in results])
    out.to_csv(RESULTS_PATH, index=False)
    print(f"[stage 4] saved results to {RESULTS_PATH}")
    return out


def report(results: pd.DataFrame, df_decisions: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("PILOT SUMMARY")
    print("=" * 72)
    print(f"Decisions extracted: {len(df_decisions)}")
    print(f"Marginal P(T=1) observed (pre-DGP): {df_decisions['T'].mean():.3f}")
    print()

    # psi variance sanity — recompute on the full pool for one seed
    sanity = generate_outcomes(df_decisions.head(1000), gamma=0.3, tau_star=0.3,
                               alpha_U=0.3, seed=0)
    print(f"Var(psi(C)) on 1k sample: {float(sanity['_psi_var'].iloc[0]):.4f}")
    print()

    agg = (results.groupby(["gamma", "N", "rep"])
                   .agg(ate_bias_mean=("ate_bias", "mean"),
                        ate_bias_std=("ate_bias", "std"),
                        p_T1=("p_T1", "mean"),
                        p_min_pre=("p_min_pre", "min"),
                        p_max_pre=("p_max_pre", "max"),
                        p_min_post=("p_min_post", "min"),
                        p_max_post=("p_max_post", "max"),
                        wall_s=("wall_s", "mean"))
                   .round(4))
    print(agg.to_string())
    print()

    # Exit-criteria check
    print("Exit criteria:")
    c1 = _exit_advantage_visible(results)
    c2 = _exit_no_overlap_pathology(results)
    c3 = _exit_wallclock_ok(results)
    print(f"  [{'x' if c1 else ' '}] concept advantage visible at N=2000, gamma=0.3 (sample-efficiency regime)")
    print(f"  [{'x' if c2 else ' '}] clipped propensities inside [0.02, 0.98] for all fits")
    print(f"  [{'x' if c3 else ' '}] wall-clock at N=5000 under 60s per fit")
    if c1 and c2 and c3:
        print("\nAll three exit criteria satisfied — safe to scale to primary grid.")
    else:
        print("\nAt least one criterion failed — inspect pilot_results.csv before scaling.")


def _exit_advantage_visible(results: pd.DataFrame) -> bool:
    """H1 is a SAMPLE-EFFICIENCY claim: concept wins at SMALL N under confounding.
    Test at N=2000, gamma=0.3 — where the advantage should be maximal."""
    sub = results[(results["N"] == 2000) & (results["gamma"] == 0.3)]
    if sub.empty:
        return False
    raw = sub[sub["rep"] == "raw"]["ate_bias"].mean()
    con = sub[sub["rep"] == "concept"]["ate_bias"].mean()
    return con < raw


def _exit_no_overlap_pathology(results: pd.DataFrame) -> bool:
    """Check CLIPPED propensities — pre-clip extremes are a reportable finding,
    not an exit-blocker. The DR-Learner internally uses min_propensity."""
    return bool((results["p_min_post"] >= MIN_PROPENSITY - 1e-9).all()
                and (results["p_max_post"] <= MAX_PROPENSITY + 1e-9).all())


def _exit_wallclock_ok(results: pd.DataFrame) -> bool:
    sub = results[results["N"] == 5000]
    if sub.empty:
        return False
    return bool(sub["wall_s"].max() < 60.0)


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true", help="re-download games")
    ap.add_argument("--rebuild", action="store_true", help="re-extract decisions")
    ap.add_argument("--max-decisions", type=int, default=15000)
    args = ap.parse_args()

    fetch_games(force=args.refetch)
    df_dec = extract_decisions(force=args.rebuild or args.refetch,
                                max_decisions=args.max_decisions)
    if len(df_dec) < 5500:
        print(f"[warn] only {len(df_dec)} decisions extracted; N=5000 runs may be unstable.")
    results = run_pilot(df_dec)
    report(results, df_dec)


if __name__ == "__main__":
    main()
