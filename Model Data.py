"""
BetPredict — Dual-Model Master Predictor (Situational vs. Market)
═════════════════════════════════════════════════════════════════
Implements the Value-Gap strategy with the full technical output suite.

  Model A (Situational): Predicts FAIR ODDS based on Pure Cricket Logic.
  Model B (Momentum): Predicts PRICE DIRECTION based on Market History.

  Logic:
    - If Market Price > Fair Price + Edge -> VALUE detected (BACK)
    - If Market Price < Fair Price - Edge -> OVER-HYPED detected (LAY)
    - Model B acts as a 'Momentum Filter' to confirm the entry.

PATCHES APPLIED:
  ✓ FIX 1: Bookmaker odds conversion now uses 1+(rate/100) instead of rate/100
  ✓ FIX 2: Removed double-wrapping bug in /signal response
  ✓ FIX 3: Added end-of-match / unsafe-LAY safety vetos
  ✓ FIX 4: match_id now passed through to rule_engine for per-match state reset
  ✓ FIX 5: LAY/BACK vetos now use EV-based math (fair_price vs breakeven_prob),
           not a hardcoded price floor. Old hardcoded MIN_LAY_ODDS / MAX_BACK_ODDS
           kept only as fallback when fair_price is unavailable.
  ✓ FIX 6: Fair price now computed relative to MARKET FAVOURITE (not always batting
           team). Probability is flipped when bowling team is the favourite, so the
           value gap comparison is always apples-to-apples.
  ✓ FIX 7: Model A stats now loaded into unified MODEL_A_STATS dict (matches
           predict_engine.py pattern) — fixes None lookups causing wrong fair odds.
  ✓ FIX 8: get_full_features() aligned with predict_engine.py — consistent wickets
           source, toss_decision_field logic, projected_score / relative_projection,
           and batter/bowler quality lookups.
  ✓ FIX 9: Price shock state fields added (tick_shock_buffer, tick_shock,
           price_momentum_6b, price_volatility_12b) so run_dual_prediction() can
           compute them before snapshotting state, matching predict_engine.py.
  ✓ FIX 10: favourite_rid lookup now works correctly after _ingest_market_prices
            stores team names as keys; RUNNER_TEAM_MAP fallback no longer silently
            breaks the is_fav_batting check.
"""

import asyncio
import json
import os
import pickle
import threading
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

import patterns_engine
import rule_engine

# ── CONFIG ────────────────────────────────────────────────────────
from config import CRICBUZZ_ID, MARKET_ID
load_dotenv()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")

MODEL_A_PATH  = "model_a_cricket.pkl"   # Pure Cricket (Fair Odds)
MODEL_B_PATH  = "betpredict_model.pkl"  # Market Momentum (Unified)

HISTORICAL_DB = "cricsheet_parsed.csv"
BETFAIR_DB    = "betfair_ipl_only.csv"
LABELED_DB_CANDIDATES = [
    "labeled_dataset.csv",
    "labeled_dataset.csv",
    "labeled_dataset.csv",
]
PRICE_FILE       = "ws_price.json"
LIVE_REPORT_FILE = "live_report.json"
MATCH_LOG_FILE   = "match_history_log.json"

RUNNER_TEAM_MAP = {
    "22121561": "Delhi Capitals",
    "38528100": "Punjab Kings"
}
BANKROLL          = float(os.getenv("BANKROLL", "2000"))
KELLY_FRACTION    = float(os.getenv("KELLY_FRACTION", "1.0"))
MAX_EXPOSURE_PCT  = float(os.getenv("MAX_EXPOSURE_PCT", "0.30"))
MIN_STAKE         = float(os.getenv("MIN_STAKE", "50"))
TARGET_PROFIT_PCT = 0.10
STOP_LOSS_PCT     = 0.20
SIGNAL_TTL_S      = int(os.getenv("SIGNAL_TTL_S", "12"))
INVALIDATE_PRICE_MOVE_PCT = float(os.getenv("INVALIDATE_PRICE_MOVE_PCT", "0.06"))
INVALIDATE_RR_DELTA       = float(os.getenv("INVALIDATE_RR_DELTA", "2.5"))

# Use mathematical edge, not arbitrary confidence thresholds
MIN_VALUE_GAP = 0.05  # 5% gap between Fair Odds and Market Odds required

# ── SAFETY VETO THRESHOLDS ────────────────────────────────────────
EV_REQUIRED_EDGE_PP  = 0.05    # 5 percentage points

# Fallback price floors used ONLY when fair_price is unavailable.
FALLBACK_MIN_LAY_ODDS = 1.40
FALLBACK_MAX_BACK_ODDS = 6.00

# Situational vetos (non-math-based, kept as-is):
END_OF_MATCH_OVER    = 19.0   # Innings near complete after this
MATCH_DECIDED_RRR    = 0.5    # Innings 2 with RRR <= this = match decided

PRICE_STALE_SECS = 10
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))


# ── STATE ─────────────────────────────────────────────────────────
state = {
    "ltp": 2.0, "prev_ltp": 2.0,
    "runners": {}, "runner_names": RUNNER_TEAM_MAP.copy(),
    "score": "0/0", "bat_team": "", "bowl_team": "",
    "wickets": 0, "prev_wickets": 0,
    "over": 0.0, "crr": 0.0, "rrr": 0.0, "innings": 1,
    "recent": "", "last_ball_id": None, "stop": False,
    "price_buffer": [], "phase_stability_log": [],
    # FIX 9: price shock fields — required by run_dual_prediction()
    "tick_shock_buffer": [],
    "tick_shock": 0, "price_momentum_6b": 0, "price_volatility_12b": 0,
    "api_ball_by_ball":    [],
    "api_market_prices":   {},
    "api_last_post_epoch": 0.0,
    "match_id":            "",

    # ── Position Management ──
    "current_book": {},
    "total_exposure": 0.0,
    "available_capital": 0.0,
    "best_case_pnl": 0.0,
    "worst_case_pnl": 0.0,
    "trades_done_this_match": 0,
    "trades_remaining": 5,

    # ── Constraints ──
    "constraints": {
        "max_stake_pct": 1.0,
        "max_stake_amount": 1500.0,
        "profit_target_pct": 0.1,
        "confidence_threshold": 0.5,
    },
}
state_lock = threading.Lock()


# ── UTILITY FUNCTIONS ─────────────────────────────────────────────

def _over_bucket(over):
    if over <= 5:  return "PP (1-5)"
    if over <= 10: return "MID-EARLY (6-10)"
    if over <= 15: return "MID-LATE (11-15)"
    return "DEATH (16-20)"

def _fav_strength(p):
    if p < 1.15: return "MATCH_LOCKED"
    if p < 1.35: return "ULTRA_STRONG"
    if p < 1.65: return "STRONG"
    if p < 2.00: return "MODERATE"
    return "EVEN_OR_DOG"

def json_safe(obj):
    if isinstance(obj, (np.integer, np.floating)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _parse_over_to_balls(over_val):
    try:
        parts = str(over_val).split(".")
        o = int(parts[0])
        b = int(parts[1]) if len(parts) > 1 else 0
        return max(0, o * 6 + b), o, b
    except Exception:
        o = int(float(over_val))
        return max(0, o * 6), o, 0

def _current_phase(over_int: int):
    if over_int < 6:   return "POWERPLAY"
    if over_int < 15:  return "MIDDLE"
    return "DEATH"

def _balls_remaining_in_phase(over_int: int, ball_in_over: int):
    if over_int < 6:    end = 6
    elif over_int < 15: end = 15
    else:               end = 20
    return max(0, end * 6 - (over_int * 6 + ball_in_over))

def calculate_ticks(p1, p2):
    if p1 == p2: return 0
    if p1 is None or p2 is None: return 0
    try:
        start, end = min(p1, p2), max(p1, p2)
        ticks = 0
        curr = start
        while curr < end - 0.0001:
            ts = _tick_size(curr)
            curr = round(curr + ts, 4)
            ticks += 1
        return ticks if p2 > p1 else -ticks
    except Exception:
        return 0

def _kelly_stake(bankroll, odds, p_win, kelly_fraction=0.25,
                 max_exposure_pct=0.03, min_stake=50.0, is_lay=False):
    if bankroll <= 0:
        return {"recommended_stake": 0.0, "max_exposure": 0.0, "kelly": 0.0, "liability": 0.0}
    max_exposure = bankroll * max_exposure_pct
    eff_odds = (odds / (odds - 1.0)) if is_lay and odds > 1.01 else odds
    if eff_odds <= 1.01:
        return {"recommended_stake": 0.0, "max_exposure": max_exposure, "kelly": 0.0, "liability": 0.0}
    b = eff_odds - 1.0
    edge = p_win * b - (1.0 - p_win)
    if edge <= 0:
        return {"recommended_stake": 0.0, "max_exposure": max_exposure, "kelly": 0.0, "liability": 0.0}
    kelly = float(np.clip(edge / b, 0.0, 1.0))
    risk_amount = float(np.clip(bankroll * kelly * kelly_fraction, 0.0, max_exposure))
    if is_lay:
        liability = risk_amount
        stake = liability / (odds - 1.0)
    else:
        stake = risk_amount
        liability = stake
    if 0 < stake < min_stake:
        stake = min_stake
        if is_lay:
            liability = stake * (odds - 1.0)
            if liability > max_exposure:
                return {"recommended_stake": 0.0, "max_exposure": max_exposure, "kelly": kelly, "liability": 0.0}
        else:
            if stake > max_exposure:
                return {"recommended_stake": 0.0, "max_exposure": max_exposure, "kelly": kelly, "liability": 0.0}
    return {
        "recommended_stake": round(stake, 2),
        "max_exposure":      round(max_exposure, 2),
        "kelly":             round(kelly, 4),
        "liability":         round(liability, 2),
    }

def _tick_size(price: float) -> float:
    p = float(price)
    if p < 2:   return 0.01
    if p < 3:   return 0.02
    if p < 4:   return 0.05
    if p < 6:   return 0.1
    if p < 10:  return 0.2
    if p < 20:  return 0.5
    if p < 30:  return 1.0
    if p < 50:  return 2.0
    if p < 100: return 5.0
    return 10.0

def _move_ticks(price: float, n_ticks: int, direction: str) -> float:
    p = float(price)
    for _ in range(int(max(0, n_ticks))):
        ts = _tick_size(p)
        p = (p + ts) if direction.upper() == "UP" else max(1.01, p - ts)
    return round(p, 2)

def _forecast_next_ball_price(action, confidence, current_price):
    a    = (action or "WAIT").upper()
    conf = float(confidence or 0.0)
    p0   = float(current_price or 0.0)
    if p0 <= 1.01:
        return {"direction": "STABLE", "expected_ticks": 0, "price_now": p0,
                "price_expected": p0, "price_range": [p0, p0]}
    direction = "DOWN" if a == "BACK" else "UP" if a == "LAY" else "STABLE"
    if a == "WAIT": ticks = 0 if conf < 0.58 else 1
    else:           ticks = 1 if conf < 0.65 else 2 if conf < 0.75 else 3
    if direction == "STABLE" or ticks == 0:
        expected = p0
        lo = _move_ticks(p0, 1, "DOWN")
        hi = _move_ticks(p0, 1, "UP")
    else:
        expected = _move_ticks(p0, ticks, direction)
        lo = _move_ticks(p0, max(ticks - 1, 0), direction)
        hi = _move_ticks(p0, ticks + 1, direction)
    pr = sorted([float(lo), float(hi)])
    return {"direction": direction, "expected_ticks": int(ticks),
            "price_now": round(p0, 2), "price_expected": float(expected),
            "price_range": [round(pr[0], 2), round(pr[1], 2)]}

def _first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p): return p
    return None

def _explain_patterns(pattern_names):
    meta = getattr(patterns_engine, "PATTERN_META", {}) or {}
    cm = {"wicket": "red", "batting": "green", "pressure": "brown/orange",
          "price": "purple", "phase": "blue", "unknown": "grey", "none": "grey"}
    exp = []
    for name in pattern_names or []:
        sig, cat, col, desc = meta.get(str(name), ("WAIT", "unknown", "#888780", "No description available"))
        exp.append({"pattern": str(name), "signal": sig, "category": cat,
                    "color": col, "color_meaning": cm.get(str(cat), "grey"), "description": desc})
    return exp


# ── EV-BASED VETO HELPERS ─────────────────────────────────────────

def _lay_ev_check(market_price: float, fair_price: Optional[float],
                  required_edge_pp: float = EV_REQUIRED_EDGE_PP) -> Tuple[bool, str]:
    if market_price <= 1.0:
        return False, f"LAY at {market_price:.2f} — invalid price"

    if fair_price is None or fair_price <= 1.0:
        if market_price < FALLBACK_MIN_LAY_ODDS:
            return False, (f"LAY at {market_price:.2f} < {FALLBACK_MIN_LAY_ODDS} "
                           f"(no fair price for EV check)")
        return True, ""

    true_lose_prob = 1.0 - (1.0 / fair_price)
    breakeven_lose_prob = (market_price - 1.0) / market_price
    edge_pp = true_lose_prob - breakeven_lose_prob

    if edge_pp < required_edge_pp:
        return False, (f"LAY at {market_price:.2f}: edge {edge_pp*100:+.1f}pp insufficient "
                       f"(model_lose={true_lose_prob:.1%} vs breakeven={breakeven_lose_prob:.1%}, "
                       f"need ≥{required_edge_pp*100:.0f}pp cushion)")
    return True, ""


def _back_ev_check(market_price: float, fair_price: Optional[float],
                   required_edge_pp: float = EV_REQUIRED_EDGE_PP) -> Tuple[bool, str]:
    if market_price <= 1.0:
        return False, f"BACK at {market_price:.2f} — invalid price"

    if fair_price is None or fair_price <= 1.0:
        if market_price > FALLBACK_MAX_BACK_ODDS:
            return False, (f"BACK at {market_price:.2f} > {FALLBACK_MAX_BACK_ODDS} "
                           f"(no fair price for EV check)")
        return True, ""

    true_win_prob = 1.0 / fair_price
    breakeven_win_prob = 1.0 / market_price
    edge_pp = true_win_prob - breakeven_win_prob

    if edge_pp < required_edge_pp:
        return False, (f"BACK at {market_price:.2f}: edge {edge_pp*100:+.1f}pp insufficient "
                       f"(model_win={true_win_prob:.1%} vs breakeven={breakeven_win_prob:.1%}, "
                       f"need ≥{required_edge_pp*100:.0f}pp cushion)")
    return True, ""


# ── SAFETY VETO ENGINE (EV-based + situational) ───────────────────

def _safety_veto_check(action: str, market_price: float, snap: dict,
                       fair_price: Optional[float] = None) -> Tuple[str, List[str]]:
    vetos = []

    if action == "WAIT":
        return action, vetos

    over     = float(snap.get("over", 0.0))
    innings  = int(snap.get("innings", 1))
    rrr      = float(snap.get("rrr", 0.0))
    crr      = float(snap.get("crr", 0.0))

    # Veto 1: Innings 1 essentially over
    if innings == 1 and over >= END_OF_MATCH_OVER:
        vetos.append(f"Innings 1 near complete (over {over:.1f}) — no edge left")
        return "WAIT", vetos

    # Veto 2: Innings 2 — match decided
    if innings == 2 and rrr > 0 and rrr <= MATCH_DECIDED_RRR:
        vetos.append(f"Match decided in innings 2 (RRR {rrr:.2f} <= {MATCH_DECIDED_RRR})")
        return "WAIT", vetos

    # Veto 3: LAY EV check
    if action == "LAY":
        ok, reason = _lay_ev_check(market_price, fair_price)
        if not ok:
            vetos.append(reason)
            return "WAIT", vetos

    # Veto 4: BACK EV check
    if action == "BACK":
        ok, reason = _back_ev_check(market_price, fair_price)
        if not ok:
            vetos.append(reason)
            return "WAIT", vetos

    # Veto 5: Innings 2 with very high CRR (chase nearly done)
    if innings == 2 and crr > 0 and rrr > 0 and crr >= 2 * rrr and over >= 15:
        vetos.append(f"Late innings 2, CRR {crr:.1f} >> RRR {rrr:.1f} — chase essentially done")
        return "WAIT", vetos

    # Veto 6: No score data = stale state
    if snap.get("score", "0/0") == "0/0" and over == 0.0:
        vetos.append("No live match data — refusing to trade on stale state")
        return "WAIT", vetos

    return action, vetos


# ── POSITION MANAGEMENT FUNCTIONS ─────────────────────────────────

def _parse_position_data(api_payload: dict) -> Tuple[Dict, List[str]]:
    warnings = []
    position_state = {
        "current_book": api_payload.get("current_book", {}),
        "total_exposure": float(api_payload.get("total_exposure", 0.0)),
        "available_capital": float(api_payload.get("available_capital", 0.0)),
        "best_case_pnl": float(api_payload.get("best_case_pnl", 0.0)),
        "worst_case_pnl": float(api_payload.get("worst_case_pnl", 0.0)),
        "trades_done_this_match": int(api_payload.get("trades_done_this_match", 0)),
        "trades_remaining": int(api_payload.get("trades_remaining", 5)),
    }
    if position_state["total_exposure"] > position_state["available_capital"] + 1000:
        warnings.append(f"Exposure exceeds available capital")
    return position_state, warnings

def _validate_stake_against_constraints(
    proposed_stake: float,
    available_capital: float,
    constraints: Dict,
) -> Tuple[float, List[str]]:
    warnings = []
    adjusted_stake = proposed_stake

    max_stake_amount = constraints.get("max_stake_amount", 1500.0)
    if adjusted_stake > max_stake_amount:
        warnings.append(f"Stake exceeds max_stake_amount {max_stake_amount:.0f}")
        adjusted_stake = max_stake_amount

    max_stake_pct = constraints.get("max_stake_pct", 0.3)
    max_by_pct = available_capital * max_stake_pct
    if adjusted_stake > max_by_pct:
        warnings.append(f"Stake exceeds max_stake_pct ({max_stake_pct:.1%})")
        adjusted_stake = max_by_pct

    if adjusted_stake > available_capital:
        warnings.append(f"Insufficient capital")
        adjusted_stake = max(0, available_capital)

    return adjusted_stake, warnings

def _calculate_position_impact(
    proposed_stake: float,
    action: str,
    runner: str,
    odds: float,
    current_book: Dict,
    total_exposure: float,
) -> Dict:
    impact = {
        "trade_pnl": proposed_stake if action == "LAY" else -proposed_stake,
        "new_exposure": total_exposure + proposed_stake,
    }
    if action == "BACK":
        impact["simulated_book"] = current_book.copy()
        impact["simulated_book"][runner] = current_book.get(runner, 0) + (proposed_stake * (odds - 1))
    else:
        impact["simulated_book"] = current_book.copy()
        impact["simulated_book"][runner] = current_book.get(runner, 0) - (proposed_stake * (odds - 1))

    impact["new_worst_case"] = min(impact["simulated_book"].values() or [0])
    impact["new_best_case"] = max(impact["simulated_book"].values() or [0])
    return impact


# ── HISTORICAL ENRICHMENT ENGINES ─────────────────────────────────

class PriceTrendEngine:
    def __init__(self, db_path):
        self.phase_avg_odds = {}
        self.df_stats = pd.DataFrame()
        try:
            self.df = pd.read_csv(db_path)
            self.df['ph'] = self.df['over'].apply(_over_bucket)
            def get_fav_p(p):
                return p if p <= 2.0 else (1.0 / (1.0 - (1.0/p)) if p > 1.0 else 2.0)
            self.df['fav_p'] = self.df['betfair_price'].apply(get_fav_p)
            self.phase_avg_odds = self.df.groupby('ph')['fav_p'].mean().round(2).to_dict()
            self.df_stats = self.df
        except Exception as e:
            print(f"[PriceTrendEngine] WARN: {e}")

    def get_projection(self, ph, fav_price):
        if self.df_stats.empty: return "N/A"
        try:
            fp = float(fav_price)
            similar = self.df_stats[
                (self.df_stats["ph"] == ph) &
                (self.df_stats["fav_p"].between(fp - 0.15, fp + 0.15))
            ]["match_id"].astype(str).unique()
            phases = ["PP (1-5)", "MID-EARLY (6-10)", "MID-LATE (11-15)", "DEATH (16-20)"]
            idx = phases.index(ph)
            if idx >= 3: return "Final Phase"
            moves = self.df_stats[
                (self.df_stats['match_id'].isin(similar)) &
                (self.df_stats['ph'] == phases[idx+1])
            ]['price_direction_next'].value_counts(normalize=True)
            return (f"DOWN {round(moves.get('DOWN',0)*100)}%, "
                    f"UP {round(moves.get('UP',0)*100)}% (Matched {len(similar)})")
        except Exception:
            return "Projection N/A"


class HistoryEngine:
    def __init__(self, db_path):
        self.df = pd.DataFrame()
        self.bench_1, self.bench_2 = {}, {}
        try:
            full_df = pd.read_csv(db_path, low_memory=False)
            full_df['match_date'] = pd.to_datetime(full_df['match_date'], errors='coerce')
            self.df = full_df[full_df['match_date'].dt.year >= 2021].copy()
            self.df = self.df.sort_values(['match_id', 'innings', 'over', 'score_before']).reset_index(drop=True)
            print(f"[HistoryEngine] Loaded {len(self.df):,} modern deliveries (2021-2026)")
            self.bench_1 = self._calc(1)
            self.bench_2 = self._calc(2)
        except Exception as e:
            print(f"[HistoryEngine] ERROR: {e}")

    def _calc(self, inn):
        res = {}
        for s, e in [(0, 5), (5, 10), (10, 15), (15, 20)]:
            m = (self.df['innings'] == inn) & (self.df['over'] >= s) & (self.df['over'] < e)
            if not self.df[m].empty:
                phase_runs = self.df[m].groupby('match_id')['runs_total'].sum().mean()
                phase_wkts = self.df[m].groupby('match_id')['wickets_this_ball'].sum().mean()
                res[f"{s+1}-{e}_overs"] = f"Avg {round(phase_runs)} runs / {round(phase_wkts, 1)} wkts"
        return res

    def find_sim(self, ov, wk, inn):
        if self.df.empty: return 0, {"avg": 0.0}
        mask = (self.df['innings'] == inn) & (self.df['over'] == int(ov)) & (self.df['wickets_before'] == wk)
        sim_indices = self.df.index[mask].tolist()
        runs_achieved = []
        for idx in sim_indices[:40]:
            try:
                current_match = self.df.iloc[idx]['match_id']
                current_score = self.df.iloc[idx]['score_before']
                for lookahead in range(15, 25):
                    target_idx = idx + lookahead
                    if target_idx < len(self.df):
                        target_row = self.df.iloc[target_idx]
                        if target_row['match_id'] == current_match:
                            runs_achieved.append(target_row['score_before'] - current_score)
                            break
            except Exception:
                continue
        avg_projection = float(np.mean(runs_achieved)) if runs_achieved else 0.0
        return len(sim_indices), {"avg": round(avg_projection, 1)}


class OddsFlipTracker:
    def __init__(self, betfair_path):
        self.stats = {}
        try:
            df = pd.read_csv(betfair_path)
            df = df[df["in_play"] == 1].dropna(subset=["ltp", "winner"])
            rows = []
            for (m, r), g in df.groupby(["market_id", "runner_id"]):
                won = 1 if str(g["winner"].iloc[-1]).lower() in str(g["runner_name"].iloc[0]).lower() else 0
                for i, p in enumerate(g["ltp"].values):
                    rows.append({"ov": (i / len(g)) * 20, "won": won, "p": p})
            dfp = pd.DataFrame(rows)
            dfp['ph'] = dfp['ov'].apply(_over_bucket)
            dfp['b']  = dfp['p'].apply(lambda x: "FAV" if x < 2.0 else "DOG")
            self.stats = dfp.groupby(['ph', 'b'])['won'].mean().to_dict()
        except Exception as e:
            print(f"[OddsFlipTracker] WARN: {e}")

    def get_wr(self, ov, p):
        if p < 1.05: return 99
        return round(float(self.stats.get((_over_bucket(ov), "FAV" if p < 2.0 else "DOG"), 0.5)) * 100)


# ── INITIALIZE (load models + enrichment engines) ──────────────────
resolved_labeled = _first_existing_path(LABELED_DB_CANDIDATES)
trend_engine = (PriceTrendEngine(resolved_labeled) if resolved_labeled
                else PriceTrendEngine(LABELED_DB_CANDIDATES[0]))
hist_engine  = HistoryEngine(HISTORICAL_DB)
flip_tracker = OddsFlipTracker(BETFAIR_DB)

# FIX 7: Unified stats dict — same structure as predict_engine.py.
# Old code used four flat globals (MODEL_A_TEAM_STATS etc.) which were only
# populated if the pkl happened to include those exact keys. The unified dict
# with safe .get() defaults guarantees the feature builder always gets numbers.
MODEL_A_STATS = {
    "team_stats": {},
    "venue_avg_scores": {},
    "batter_avg": {},
    "bowler_avg": {},
}

try:
    with open(MODEL_A_PATH, "rb") as f:
        pkg_a = pickle.load(f)
        model_a = pkg_a["model"]
        feat_cols_a = pkg_a["feature_cols"]
        for key in MODEL_A_STATS.keys():
            if key in pkg_a:
                MODEL_A_STATS[key] = pkg_a[key]
    print(f"[LOAD] Model A loaded. Features: {len(feat_cols_a)}")
except Exception as e:
    print(f"[FATAL] Model A Load Fail: {e}")
    exit(1)


# ─────────────────────────────────────────────────────────────────
# FLASK API (Position-Aware /signal endpoint)
# ─────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

def _ingest_market_prices(mp: dict):
    """
    Convert market_prices to runners + ltp.

    Handles BOTH formats:
      - Decimal odds (e.g., 1.85)         → passed through unchanged
      - Bookmaker rates (e.g., 256, 37)   → converted via 1 + (rate/100)
    """
    runners = {}
    runner_names = {}
    for runner_key, entry in mp.items():
        try:
            back_p = float(entry.get("back", 200))
            lay_p  = float(entry.get("lay",  200))
        except Exception:
            continue

        if back_p >= 10:
            back_p = 1.0 + (back_p / 100.0)
        if lay_p >= 10:
            lay_p = 1.0 + (lay_p / 100.0)

        mid = round((back_p + lay_p) / 2.0, 3)

        team_name = RUNNER_TEAM_MAP.get(str(runner_key))
        if not team_name:
            team_name = str(runner_key)

        runners[team_name] = max(1.01, mid)
        runner_names[str(runner_key)] = team_name

    if runners:
        state["runners"] = runners
        state["runner_names"].update(runner_names)
        state["ltp"] = min(runners.values())
        print(f"[MARKET] Incoming raw: {mp}")
        print(f"[MARKET] Converted to decimal: {runners}")
        print(f"[MARKET] LTP (favourite): {state['ltp']:.2f}")

        team_names = list(runners.keys())
        if not state["bat_team"] and len(team_names) >= 1:
            state["bat_team"] = team_names[0]
        if not state["bowl_team"] and len(team_names) >= 2:
            state["bowl_team"] = team_names[1]


@flask_app.route("/signal", methods=["POST"])
def signal_endpoint():
    """Position-aware signal endpoint. Receives market + position data, returns prediction."""
    body = request.get_json(force=True) or {}

    with state_lock:
        state["match_id"] = body.get("match_id", "")

        # Extract team names from match_id (format: "Team A vs Team B")
        match_id = state["match_id"]
        if match_id and " vs " in match_id:
            teams = match_id.split(" vs ")
            state["bat_team"] = teams[0].strip()
            state["bowl_team"] = teams[1].strip()

        score_raw = body.get("score", "").strip()
        # Handle both "123/4" and "Team 123-4 (5.2) | Opp 0-0 (0.0)" formats
        if "/" in score_raw:
            state["score"] = score_raw
        elif "-" in score_raw:
            # Extract from "Team 123-4 (5.2)" format
            parts = score_raw.split("|")[0].strip()  # Get first team's part
            if "-" in parts:
                try:
                    score_part = parts.split()[-2]  # e.g., "123-4"
                    runs, wickets = score_part.split("-")
                    state["score"] = f"{runs}/{wickets}"
                    # Extract over from parentheses: (5.2)
                    over_str = parts.split()[-1].strip("()")
                    state["over"] = float(over_str)
                except:
                    state["score"] = "0/0"
        else:
            state["score"] = "0/0"

        mp = body.get("market_prices", {})
        if isinstance(mp, dict) and mp:
            state["api_market_prices"] = mp
            _ingest_market_prices(mp)

        bbb = body.get("ball_by_ball")
        if isinstance(bbb, list):
            state["api_ball_by_ball"] = bbb[-24:]

        state["api_last_post_epoch"] = time.time()

        pos_data, pos_warnings = _parse_position_data(body)
        state["current_book"] = pos_data["current_book"]
        state["total_exposure"] = pos_data["total_exposure"]
        state["available_capital"] = pos_data["available_capital"]
        state["best_case_pnl"] = pos_data["best_case_pnl"]
        state["worst_case_pnl"] = pos_data["worst_case_pnl"]
        state["trades_done_this_match"] = pos_data["trades_done_this_match"]
        state["trades_remaining"] = pos_data["trades_remaining"]

        state["constraints"] = body.get("constraints", state["constraints"])

    if state["score"] == "0/0":
        fetch_score()

    full_prediction = run_dual_prediction()
    print(json.dumps(full_prediction, indent=2, default=json_safe))

    if request.args.get("format") == "simple":
        return jsonify({
            "timestamp":    full_prediction["timestamp"],
            "action":       full_prediction["ai_prediction"]["action"],
            "confidence":   float(full_prediction["ai_prediction"]["confidence"]),
            "value_gap":    full_prediction["ai_prediction"]["value_gap"],
            "fair_price":   float(full_prediction["ai_prediction"]["fair_price_model_a"]),
            "market_price": float(full_prediction["stake_sizing"].get("market_price", state["ltp"])),
            "stake":        float(full_prediction.get("constraint_validation", {}).get("final_stake_constrained", 0)),
            "tp_price":     float(full_prediction["stake_sizing"].get("target_profit_price", 0)),
            "sl_price":     float(full_prediction["stake_sizing"].get("stop_loss_price", 0)),
        }), 200

    veto_list = full_prediction.get("rule_engine", {}).get("veto_reasons", []) or []
    safety_vetos = full_prediction.get("ai_prediction", {}).get("safety_vetos", []) or []
    all_vetos = list(safety_vetos) + list(veto_list)

    if all_vetos:
        reason_text = f"VETO: {all_vetos[0]}"
    else:
        reason_text = full_prediction.get("ai_prediction", {}).get("reasoning", "No edge detected")

    short_response = {
        "action":     full_prediction["ai_prediction"]["action"],
        "team":       full_prediction["odds_deep_dive"].get("favourite") or "",
        "confidence": round(float(full_prediction["ai_prediction"]["confidence"]), 4),
        "reason":     reason_text,
        "tp_price":   float(full_prediction["stake_sizing"].get("target_profit_price", 0)),
        "sl_price":   float(full_prediction["stake_sizing"].get("stop_loss_price", 0)),
        "raw":        json.loads(json.dumps(full_prediction, default=json_safe))
    }

    return jsonify(short_response), 200


@flask_app.route("/health", methods=["GET"])
def health_endpoint():
    with state_lock:
        capital = state["available_capital"]
        trades = state["trades_remaining"]
    return jsonify({
        "status": "ok",
        "model_a_path": MODEL_A_PATH,
        "edge_threshold": f"{MIN_VALUE_GAP:.1%}",
        "ev_required_edge_pp": f"{EV_REQUIRED_EDGE_PP*100:.0f}pp",
        "signal_ttl_s": SIGNAL_TTL_S,
        "capital": capital,
        "trades_remaining": trades,
        "safety_vetos": {
            "ev_required_edge_pp": EV_REQUIRED_EDGE_PP,
            "fallback_min_lay_odds": FALLBACK_MIN_LAY_ODDS,
            "fallback_max_back_odds": FALLBACK_MAX_BACK_ODDS,
            "end_of_match_over": END_OF_MATCH_OVER,
            "match_decided_rrr": MATCH_DECIDED_RRR,
        },
    }), 200


def start_flask_thread():
    from werkzeug.serving import make_server
    srv = make_server(FLASK_HOST, FLASK_PORT, flask_app)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[FLASK] /signal listening on http://{FLASK_HOST}:{FLASK_PORT}")
    return t


# ─────────────────────────────────────────────────────────────────
# LIVE INGEST LOOPS (Price & Score)
# ─────────────────────────────────────────────────────────────────

async def price_loop():
    while not state["stop"]:
        try:
            if (os.path.exists(PRICE_FILE) and
                    time.time() - os.path.getmtime(PRICE_FILE) < PRICE_STALE_SECS):
                with open(PRICE_FILE, "r") as f:
                    data = json.load(f)
                with state_lock:
                    state["runners"] = data.get("runners", {})
                    if state["runners"]:
                        state["ltp"] = min(state["runners"].values())
                        state["price_buffer"].append(float(state["ltp"]))
                    if len(state["price_buffer"]) > 30:
                        state["price_buffer"].pop(0)
        except Exception as e: pass
        await asyncio.sleep(0.5)

def fetch_score():
    try:
        r = requests.get(f"https://www.cricbuzz.com/api/mcenter/livescore/{CRICBUZZ_ID}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            ms = data.get("miniscore", {})
            if ms:
                with state_lock:
                    state["bat_team"]  = ms.get('batTeam', {}).get('teamName') or ""
                    state["bowl_team"] = ms.get('bowlTeam', {}).get('teamName') or ""
                    new_wickets = int(ms.get('batTeam', {}).get('teamWkts', 0))
                    state["prev_wickets"], state["wickets"] = state["wickets"], new_wickets
                    state["score"]   = f"{ms.get('batTeam', {}).get('teamScore', 0)}/{state['wickets']}"
                    state["over"]    = float(ms.get("overs", 0))
                    state["crr"]     = float(ms.get("currentRunRate", 0))
                    state["rrr"]     = float(ms.get("requiredRunRate", 0))
                    state["innings"] = 2 if state["rrr"] > 0 else 1
                    state["recent"]  = ms.get("recentOvsStats", "") or ""
    except Exception: pass


# ─────────────────────────────────────────────────────────────────
# FEATURE BUILDER — aligned with predict_engine.py (FIX 8)
# ─────────────────────────────────────────────────────────────────

def _parse_score_string(score: str):
    try:
        p = score.split("/")
        return int(p[0]), int(p[1])
    except: return 0, 0

def _parse_recent_balls(recent: str):
    balls = []
    if not recent: return balls
    for tok in str(recent).replace("|", " ").split():
        upper = tok.upper()
        digits = "".join(filter(str.isdigit, tok))
        runs = int(digits) if digits else 0
        wicket = 1 if "W" in upper else 0
        balls.append({"runs": runs, "wicket": wicket,
                      "boundary": 1 if runs in (4, 6) else 0,
                      "dot": 1 if (runs == 0 and wicket == 0) else 0})
    return balls

def get_full_features(detected_patterns, snap):
    try: ctx = rule_engine.load_match_context()
    except: ctx = {"target_score": 180, "batting_won_toss": 0}

    over_val = float(snap.get("over", 0.0))
    _, over_int, ball_in_over = _parse_over_to_balls(over_val)
    ball_number = over_int * 6 + ball_in_over
    score_runs, _ = _parse_score_string(snap.get("score", "0/0"))

    # FIX 8a: always read wickets from state["wickets"], not from the score
    # string — the score string may lag by one delivery.
    wickets = int(snap.get("wickets", 0))

    # ── ROLLING BALL METRICS ──
    # FIX 8b: fall back to recent-string parse when api_ball_by_ball is empty,
    # matching predict_engine.py behaviour.
    bbb = snap.get("api_ball_by_ball", [])
    if not bbb and snap.get("recent"):
        bbb = _parse_recent_balls(snap["recent"])
    bbb = [b for b in bbb if isinstance(b, dict)]

    last_18 = bbb[-18:] if len(bbb) >= 18 else bbb
    runs_last_18 = sum(b.get("runs", 0) for b in last_18)
    wkts_last_18 = sum(b.get("wicket", 0) or b.get("is_wicket", 0) for b in last_18)

    last_12 = bbb[-12:] if len(bbb) >= 12 else bbb
    bowler_wkts_last_12 = sum(b.get("wicket", 0) or b.get("is_wicket", 0) for b in last_12)

    # ── TEAM & VENUE STATS — FIX 7/8c ──
    bat_team  = snap.get("bat_team", "")
    bowl_team = snap.get("bowl_team", "")
    venue     = ctx.get("venue", "neutral")

    ts_map = MODEL_A_STATS.get("team_stats", {})
    bat_win_rate  = float(ts_map.get(bat_team, 0.5))
    bowl_win_rate = float(ts_map.get(bowl_team, 0.5))

    vs_map = MODEL_A_STATS.get("venue_avg_scores", {})
    venue_avg = float(vs_map.get(venue, 160.0))

    # ── RATE CALCULATIONS ──
    crr = float(snap.get("crr", 0.0))
    rrr = float(snap.get("rrr", 0.0))
    is_inn2 = 1 if int(snap.get("innings", 1)) == 2 else 0

    rr_pressure = (rrr / max(crr, 0.1)) if is_inn2 else (crr / 8.0)
    rr_pressure = max(0.0, min(5.0, rr_pressure))

    target_score = int(ctx.get("target_score", 180))
    runs_rem  = max(0, target_score - score_runs) if is_inn2 else 0
    balls_rem = max(0, 120 - ball_number)

    # ── PROJECTIONS — FIX 8d ──
    # predict_engine.py sets projected_score = target_score in innings 2
    # (the target is known, so projecting via CRR is redundant/misleading).
    if is_inn2:
        projected_score = float(target_score)
        relative_projection = (
            (target_score - score_runs) / max(rrr * balls_rem / 6.0, 1.0)
            if rrr > 0 else 0.0
        )
    else:
        projected_score = score_runs + (crr * (balls_rem / 6.0)) if crr > 0 else float(score_runs)
        relative_projection = 0.0

    # ── BATTER / BOWLER QUALITY — FIX 8e ──
    batter_avg_map = MODEL_A_STATS.get("batter_avg", {})
    bowler_avg_map = MODEL_A_STATS.get("bowler_avg", {})
    batter_quality = float(batter_avg_map.get(bat_team, 30.0)) / 100.0
    bowler_quality = float(bowler_avg_map.get(bowl_team, 8.0)) / 50.0

    # ── TOSS — FIX 8f ──
    batting_won_toss = int(ctx.get("batting_won_toss", 0))
    # predict_engine.py: toss_decision_field = 1 when toss_decision == "field"
    toss_decision_field = 1 if ctx.get("toss_decision") == "field" else 0

    return {
        "innings":               int(snap.get("innings", 1)),
        "over":                  float(over_int),
        "over_norm":             float(over_int) / 19.0,
        "ball_number":           int(ball_number),
        "score_before":          int(score_runs),
        "wickets_before":        wickets,
        "wickets_in_hand":       max(0, 10 - wickets),
        "current_run_rate":      crr,
        "required_rate":         rrr,
        "run_rate_pressure":     rr_pressure,
        "runs_remaining":        runs_rem,
        "balls_remaining":       balls_rem,
        "is_innings_2":          is_inn2,
        "runs_last_18":          int(runs_last_18),
        "wickets_last_18":       int(wkts_last_18),
        "batting_team_win_rate": bat_win_rate,
        "bowling_team_win_rate": bowl_win_rate,
        "team_strength_diff":    bat_win_rate - bowl_win_rate,
        "venue_avg_score":       venue_avg,
        "batting_team_won_toss": batting_won_toss,
        "toss_decision_field":   toss_decision_field,
        "is_powerplay":          1 if over_int < 6 else 0,
        "is_death_overs":        1 if over_int >= 15 else 0,
        "run_rate_gap":          (rrr - crr) if is_inn2 else 0.0,
        "relative_score":        float(score_runs) / max(venue_avg, 1.0),
        "wickets_ratio":         float(wickets) / 10.0,
        "balls_ratio":           float(ball_number) / 120.0,
        "pressure_index":        (wickets * 2.0) + (rr_pressure * 0.5) + (ball_number / 120.0 * 0.5),
        "momentum_score":        (runs_last_18 / 3.0) if runs_last_18 > 0 else -1.0,
        "recent_wickets_pressure": wkts_last_18 / 3.0 if last_18 else 0.0,
        "batter_quality":        batter_quality,
        "bowler_quality":        bowler_quality,
        "projected_score":       projected_score,
        "relative_projection":   relative_projection,
        "bowler_wickets_last_12": int(bowler_wkts_last_12),
    }


# ─────────────────────────────────────────────────────────────────
# DUAL MODEL PREDICTION (Logic Core)
# ─────────────────────────────────────────────────────────────────

def run_dual_prediction() -> dict:
    with state_lock:
        # FIX 9: compute price-shock features inside the lock, same as
        # predict_engine.py, before we snapshot state.
        current_ltp = float(state["ltp"])
        prev_ltp    = float(state["prev_ltp"])
        ts = calculate_ticks(prev_ltp, current_ltp)

        state["tick_shock_buffer"].append(ts)
        if len(state["tick_shock_buffer"]) > 30:
            state["tick_shock_buffer"].pop(0)

        mom = sum(state["tick_shock_buffer"][-6:])
        vol = (np.std(state["tick_shock_buffer"][-12:])
               if len(state["tick_shock_buffer"]) >= 2 else 0)

        state["tick_shock"]          = ts
        state["price_momentum_6b"]   = mom
        state["price_volatility_12b"] = vol

        snap = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in state.items()}

    detected = patterns_engine.detect_patterns({
        "over": snap["over"], "wickets_before": snap["wickets"], "innings": snap["innings"],
        "rrr": snap["rrr"], "crr": snap["crr"], "recent": snap["recent"],
        "betfair_price": snap["ltp"], "price_change": float(snap["ltp"]) - float(snap["prev_ltp"]),
        "wickets_this_ball": 1 if snap["wickets"] > snap["prev_wickets"] else 0
    })

    feats = get_full_features(detected, snap)

    # ── MODEL A: FAIR ODDS ──
    X_a = np.array([[feats.get(c, 0) for c in feat_cols_a]])
    p_batting_win = float(model_a.predict_proba(X_a)[0][1])
    fair_odds_batting = float(1.0 / max(p_batting_win, 0.01))

    # FIX 10: after _ingest_market_prices, snap["runners"] keys are team
    # names (not runner IDs). min() on team-name keys gives the favourite's
    # team name directly — no RUNNER_TEAM_MAP lookup needed here.
    market_price  = snap["ltp"]
    favorite_team = ""
    is_fav_batting = False

    if snap["runners"]:
        # snap["runners"] = {team_name: decimal_price, ...}
        favorite_team = min(snap["runners"], key=snap["runners"].get)
        if favorite_team == snap.get("bat_team"):
            is_fav_batting = True

    # Flip fair odds to the favourite's perspective if needed.
    if is_fav_batting:
        fair_odds = fair_odds_batting
        p_fair_win = p_batting_win
    else:
        p_fair_win = 1.0 - p_batting_win
        fair_odds  = float(1.0 / max(p_fair_win, 0.01))

    # ── VALUE GAP ANALYSIS ──
    value_gap = (market_price - fair_odds) / fair_odds

    action, wait_reason = "WAIT", "Market aligns with situation"

    if value_gap > MIN_VALUE_GAP:
        action = "BACK"
        wait_reason = (f"VALUE DETECTED (+{value_gap:.1%}). "
                       f"Market {favorite_team} {market_price:.2f} > Fair {fair_odds:.2f}")
    elif value_gap < -MIN_VALUE_GAP:
        action = "LAY"
        wait_reason = (f"OVER-HYPED ({value_gap:.1%}). "
                       f"Market {favorite_team} {market_price:.2f} < Fair {fair_odds:.2f}")
    else:
        wait_reason = f"Gap: {value_gap:+.1%}. Need >{MIN_VALUE_GAP*100:.0f}% value gap."

    # ── SAFETY VETOS (EV-based, fair_price passed through) ──
    pre_veto_action = action
    action, safety_vetos = _safety_veto_check(action, market_price, snap, fair_price=fair_odds)
    if safety_vetos and action == "WAIT":
        wait_reason = f"SAFETY VETO ({pre_veto_action} blocked): {'; '.join(safety_vetos)}"
        print(f"[VETO] {pre_veto_action} blocked: {safety_vetos}")

    # ── MONEY MANAGEMENT (rule_engine layer) ──
    rule_res = rule_engine.evaluate(action, p_fair_win, snap, value_gap=value_gap)
    action     = rule_res["final_action"]
    final_conf = rule_res["final_confidence"]

    # ── FIXED STAKE & TP/SL CALCULATION (30-10-20 Strategy) ──
    fixed_stake = BANKROLL * MAX_EXPOSURE_PCT
    tp_price, sl_price = 0.0, 0.0

    if action == "BACK":
        tp_price = round(market_price / (1.0 + TARGET_PROFIT_PCT), 2)
        sl_price = round(market_price / (1.0 - STOP_LOSS_PCT), 2)
    elif action == "LAY":
        tp_price = round(market_price / (1.0 - TARGET_PROFIT_PCT), 2)
        sl_price = round(market_price / (1.0 + STOP_LOSS_PCT), 2)

    stake_res = {
        "recommended_stake": round(fixed_stake, 2),
        "max_exposure": round(fixed_stake, 2),
        "kelly": 1.0,
        "liability": round(fixed_stake * (market_price - 1) if action == "LAY" else fixed_stake, 2),
        "target_profit_price": tp_price,
        "stop_loss_price": sl_price
    }

    # ── CONSTRAINT VALIDATION ──
    constraint_warnings = []
    final_stake = stake_res["recommended_stake"]

    if action != "WAIT":
        adjusted_stake, constraint_msgs = _validate_stake_against_constraints(
            final_stake,
            snap["available_capital"],
            snap["constraints"]
        )
        constraint_warnings.extend(constraint_msgs)
        final_stake = adjusted_stake

        if snap["trades_remaining"] <= 0:
            constraint_warnings.append("No trades remaining")
            final_stake = 0
        if snap["available_capital"] <= 0:
            constraint_warnings.append("No available capital")
            final_stake = 0

        if final_stake == 0:
            action = "WAIT"
            wait_reason = f"Constraints blocked trade: {', '.join(constraint_warnings)}"

    final_stake = int(final_stake)

    _, o_int, b_o = _parse_over_to_balls(snap["over"])

    # Prefer team names from API (extracted from match_id), fall back to runners
    bat_team  = snap.get('bat_team', '').strip()
    bowl_team = snap.get('bowl_team', '').strip()

    if not bat_team or not bowl_team:
        team_names = list(snap["runners"].keys()) if snap["runners"] else []
        if not bat_team:
            bat_team = team_names[0] if len(team_names) > 0 else 'Team A'
        if not bowl_team:
            bowl_team = team_names[1] if len(team_names) > 1 else 'Team B'

    # Resolve team names through map only if they look like runner IDs
    bat_team  = RUNNER_TEAM_MAP.get(str(bat_team),  bat_team)
    bowl_team = RUNNER_TEAM_MAP.get(str(bowl_team), bowl_team)

    position_impact = {}
    if action != "WAIT" and final_stake > 0:
        traded_team = favorite_team if favorite_team else snap.get("bat_team", "Team")
        position_impact = _calculate_position_impact(
            final_stake, action, traded_team, market_price,
            snap["current_book"], snap["total_exposure"]
        )

    # snap["runners"] already has team names as keys after _ingest_market_prices
    normalized_runners = dict(snap["runners"])

    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "match_info": f"{bat_team} vs {bowl_team} | {snap['score']} ({snap['over']} ov)",
        "live_market_rates": normalized_runners,
        "ai_prediction": {
            "action": action,
            "recommendation": f"{action} {favorite_team}" if action not in ["WAIT", "HOLD"] else "WAIT - No edge",
            "fair_price_model_a": round(fair_odds, 2),
            "momentum_signal_model_b": "DISABLED (Value-Gap Only Mode)",
            "value_gap": f"{value_gap:+.1%}",
            "confidence": round(final_conf, 2),
            "reasoning": wait_reason,
            "active_patterns": detected,
            "safety_vetos": safety_vetos,
            "pre_veto_action": pre_veto_action,
        },
        "next_ball_price_forecast": _forecast_next_ball_price(action, final_conf, market_price),
        "stake_sizing": {**stake_res, "market_price": market_price,
                         "recommended_stake": int(stake_res["recommended_stake"])},
        "constraint_validation": {
            "warnings": constraint_warnings,
            "final_stake_constrained": final_stake,
            "respects_all_constraints": len(constraint_warnings) == 0,
        },
        "position_management": {
            "current_book": snap["current_book"],
            "total_exposure": snap["total_exposure"],
            "available_capital": snap["available_capital"],
            "best_case_pnl": snap["best_case_pnl"],
            "worst_case_pnl": snap["worst_case_pnl"],
            "trades_done": snap["trades_done_this_match"],
            "trades_remaining": snap["trades_remaining"],
        },
        "position_impact": position_impact,
        "historical_context": {
            "similar_matches": hist_engine.find_sim(snap["over"], snap["wickets"], snap["innings"])[0],
            "avg_odds_history": trend_engine.phase_avg_odds,
        },
        "live_analysis": {
            "current_phase": _current_phase(o_int),
            "rrr": snap["rrr"], "crr": snap["crr"],
            "stability": f"{round((snap['phase_stability_log'].count('SAME')/max(1,len(snap['phase_stability_log'])))*100)}%"
        },
        "odds_deep_dive": {
            "favourite": favorite_team,
            "live_rate": market_price,
            "historical_win_prob": f"{flip_tracker.get_wr(snap['over'], market_price)}%"
        },
        "rule_engine": rule_res
    }


# ─────────────────────────────────────────────────────────────────
# LOGGING & SERVER
# ─────────────────────────────────────────────────────────────────

async def predict_tick():
    # Fetch score BEFORE ball_id check so state is always fresh
    fetch_score()

    ball_id = f"{state['over']}_{state['score']}"
    if ball_id == state["last_ball_id"]: return
    state["last_ball_id"] = ball_id

    output = run_dual_prediction()

    print(json.dumps(output, indent=2, default=json_safe))
    with open(LIVE_REPORT_FILE, "w") as f: json.dump(output, f, indent=2, default=json_safe)
    state['prev_ltp'] = state['ltp']

async def main():
    print("=" * 65)
    print("  BetPredict Value-Gap Predictor — Model A Mode (Position-Aware)")
    print(f"  Model A: {MODEL_A_PATH} (Fair Value)")
    print("  Model B: DISABLED (Value-Gap Only)")
    print(f"  API: http://{FLASK_HOST}:{FLASK_PORT}/signal")
    print("  Position Management: ENABLED")
    print("  Constraint Validation: ENABLED")
    print(f"  Fair Price: Favourite-relative (FIX 6 applied)")
    print(f"  Safety Vetos: EV-BASED (required edge {EV_REQUIRED_EDGE_PP*100:.0f}pp, "
          f"fallback floor {FALLBACK_MIN_LAY_ODDS}/{FALLBACK_MAX_BACK_ODDS})")
    print("=" * 65)

    start_flask_thread()
    asyncio.create_task(price_loop())

    while not state["stop"]:
        await asyncio.sleep(2)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nStopped.")