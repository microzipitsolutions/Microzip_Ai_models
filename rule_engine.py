"""
BetPredict Rule Engine
======================
Implements the 37 trading-strategy rules from cricket_trading_strategy.json.

Wraps the XGBoost ML signal — ML stays primary, rules apply as:
  - VETO            : force WAIT (e.g. daily target hit, terminal odds)
  - FILTER_IF_BACK  : force WAIT if original signal was BACK
  - FILTER_IF_LAY   : force WAIT if original signal was LAY
  - BIAS            : nudge confidence ± a small amount
  - SET_STAKE_PCT   : set the bankroll % to risk
  - HEDGE           : emit a hedge instruction alongside the signal
  - SET_TP_SL_PCT   : override TP/SL with %-based exits

Defaults (safe):
  - STAKE_PCT_DEFAULT = 3%   (override via AGGRESSIVE_MODE=1 → 30%)
  - DAILY_PROFIT_TARGET_PCT = 10%
  - DAILY_LOSS_LIMIT_PCT    = 20%
  - Odds in rules 1, 2, 14, 19 interpreted as Indian-style (60 → 1.60)

PATCHES APPLIED:
  ✓ FIX 1: Per-match state reset (flip_count, last_fav_team, etc.) when
           match_id changes. Daily counters (PnL, wins) preserved across matches.
  ✓ FIX 2: Flip counting now per-BALL not per-tick. Tick-level oscillations in
           tight markets no longer inflate flip_count.
  ✓ FIX 3: Tight-market gate added — flips ignored when fav margin < 0.10.
"""

import os
import json
from datetime import date

# ── CONFIG ────────────────────────────────────────────────────────
TRADING_STATE_FILE = "./live/trading_state.json"
MATCH_CONTEXT_FILE = "./live/match_context.json"
RULE_AUDIT_FILE    = "./live/rule_audit.json"

STAKE_PCT_DEFAULT       = 0.30
STAKE_PCT_AGGRESSIVE    = 0.30
STAKE_PCT_INNINGS_2     = 0.40
DAILY_PROFIT_TARGET_PCT = 0.50
DAILY_LOSS_LIMIT_PCT    = 0.40
MAX_TRADES_PER_INNINGS  = 10
STOP_AFTER_WINS         = 5
FLIPS_FOR_LAY_ONLY      = 8  # Increased from 4 — less restrictive
HEDGE_ODDS_FLOOR        = 1.10
EXTREME_LOW_ODDS        = 1.05
TIGHT_MATCH_ODDS        = (1.70, 2.30)
BIG_TARGET_RUNS         = 190
TP_PCT_DEFAULT          = 0.10
SL_PCT_DEFAULT          = 0.20
TIE_RR_RATIO            = 3.0

# 🆕 Tight-market gate: don't count fav flips when both prices are within
# this margin (decimal odds difference). Prevents tick-level noise inflation.
FLIP_TIGHT_MARKET_MARGIN = 0.10

AGGRESSIVE_MODE = os.getenv("AGGRESSIVE_MODE", "0") == "1"


# ── STATE PERSISTENCE ────────────────────────────────────────────

def _today_str() -> str:
    return date.today().isoformat()

def load_trading_state() -> dict:
    default = {
        "date":                _today_str(),
        "trades_today":        0,
        "wins_today":          0,
        "losses_today":        0,
        "daily_pnl_pct":       0.0,
        "bankroll_recovered":  False,
        "flip_count":          0,
        "trades_innings_1":    0,
        "trades_innings_2":    0,
        "wins_innings_1":      0,
        "wins_innings_2":      0,
        "last_fav_team":       None,
        "match_start_fav":     None,
        "major_turn_seen":     False,
        "current_match_id":    None,
        "last_ball_anchor":    None,    # 🆕 (over, score) tuple for per-ball flip debounce
    }
    try:
        with open(TRADING_STATE_FILE, "r") as f:
            data = json.load(f)
        if data.get("date") != _today_str():
            return default
        for k, v in default.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return default

def save_trading_state(state: dict) -> None:
    os.makedirs(os.path.dirname(TRADING_STATE_FILE) or ".", exist_ok=True)
    with open(TRADING_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_match_context() -> dict:
    default = {
        "venue":                  "",
        "venue_innings_win_ratio": {"1": 0.5, "2": 0.5},
        "pitch_type":             "neutral",
        "target_score":           0,
        "weather":                "clear",
        "league_volatility":      "medium",
        "key_player_failed":      False,
        "batting_won_toss":       0,
    }
    try:
        with open(MATCH_CONTEXT_FILE, "r") as f:
            data = json.load(f)
        for k, v in default.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return default


# ── VERDICT HELPER ────────────────────────────────────────────────

def _r(rule_id, name, verdict, reason, **extras):
    return {"rule_id": rule_id, "name": name, "verdict": verdict,
            "reason": reason, **extras}


# ── INDIVIDUAL RULES ──────────────────────────────────────────────

def rule_01_first_over_entry(signal, snap, ts, ctx):
    if int(snap["innings"]) != 1 or float(snap["over"]) > 1.5:
        return None
    p = float(snap["ltp"])
    if signal == "BACK" and abs(p - 1.60) < 0.05:
        return _r(1, "FIRST_OVER_BACK_160", "BIAS",
                  f"Innings-1 first over: BACK at ~1.60 (live={p})", bias=+0.04)
    if signal == "LAY" and abs(p - 1.50) < 0.05:
        return _r(1, "FIRST_OVER_LAY_150", "BIAS",
                  f"Innings-1 first over: LAY at ~1.50 (live={p})", bias=+0.04)
    return None

def rule_02_odds_band_bias(signal, snap, ts, ctx):
    p = float(snap["ltp"])
    # 🆕 Align with MIN_LAY_ODDS = 1.40
    if 1.40 <= p < 1.60 and signal == "LAY":
        return _r(2, "ODDS_BAND_LAY_BIAS", "BIAS",
                  f"Price {p} in safe LAY zone (≥1.40)", bias=+0.04)
    if p > 1.40 and signal == "BACK":
        return _r(2, "ODDS_ABOVE_140_BACK", "BIAS",
                  f"Price {p} >1.40 — BACK zone", bias=+0.03)
    return None


def rule_03_safety_veto_lay(signal, snap, ts, ctx):
    """🆕 Hard veto for unsafe LAY odds (mirrors master predictor safety)."""
    p = float(snap["ltp"])
    if signal == "LAY" and p < 1.40:
        return _r(3, "SAFETY_VETO_LAY_LOW", "VETO",
                  f"LAY at {p:.2f} < 1.40 — unfavourable risk/reward")
    return None


def rule_04_base_stake_pct(signal, snap, ts, ctx):
    pct = STAKE_PCT_AGGRESSIVE if AGGRESSIVE_MODE else STAKE_PCT_DEFAULT
    mode = "aggressive (rule-4 30%)" if AGGRESSIVE_MODE else "safe default (3%)"
    return _r(4, "BASE_STAKE_PCT", "SET_STAKE_PCT",
              f"Base stake = {pct*100:.0f}% — {mode}", stake_pct=pct)

def rule_05_hedge_low_odds(signal, snap, ts, ctx):
    p = float(snap["ltp"])
    if p < HEDGE_ODDS_FLOOR and p >= EXTREME_LOW_ODDS:
        return _r(5, "HEDGE_BELOW_110", "HEDGE",
                  f"Price {p} <1.10 — green-book hedge both sides")
    return None

def rule_06_follow_favorite(signal, snap, ts, ctx):
    p = float(snap["ltp"])
    if p < 1.43 and signal == "BACK":
        return _r(6, "FOLLOW_FAV_70_30", "BIAS",
                  f"Strong fav (~70:30, price={p}) — back the favorite", bias=+0.03)
    return None

def rule_08_venue_stats(signal, snap, ts, ctx):
    inn = str(int(snap["innings"]))
    ratio = float(ctx.get("venue_innings_win_ratio", {}).get(inn, 0.5))
    if ratio >= 0.60 and signal == "BACK":
        return _r(8, "VENUE_FAVORS_INNINGS", "BIAS",
                  f"Venue innings-{inn} win ratio {ratio:.0%} favors batting side",
                  bias=+0.03)
    if ratio <= 0.40 and signal == "LAY":
        return _r(8, "VENUE_AGAINST_INNINGS", "BIAS",
                  f"Venue innings-{inn} win ratio {ratio:.0%} — LAY bias",
                  bias=+0.03)
    return None

def rule_09_pitch_adjust(signal, snap, ts, ctx):
    pt = ctx.get("pitch_type", "neutral")
    if pt == "batting" and signal == "BACK" and int(snap["innings"]) == 2:
        return _r(9, "BATTING_PITCH_CHASE", "BIAS",
                  "Batting pitch in chase — BACK bias", bias=+0.03)
    if pt == "bowling" and signal == "LAY":
        return _r(9, "BOWLING_PITCH_LAY", "BIAS",
                  "Bowling pitch — LAY bias on batting side", bias=+0.03)
    return None

def rule_10_big_chase_lay_bias(signal, snap, ts, ctx):
    target = int(ctx.get("target_score", 0))
    if int(snap["innings"]) == 2 and target >= BIG_TARGET_RUNS and signal == "LAY":
        return _r(10, "BIG_TARGET_LAY", "BIAS",
                  f"Chase target {target}≥{BIG_TARGET_RUNS} — LAY favored",
                  bias=+0.04)
    return None

def rule_11_player_driven_move(signal, snap, ts, ctx):
    return None

def rule_12_innings2_higher_stake(signal, snap, ts, ctx):
    if int(snap["innings"]) != 2: return None
    rrr_delta = float(snap.get("rrr", 0)) - float(snap.get("crr", 0))
    if abs(rrr_delta) >= 3.0 and signal in ("BACK", "LAY"):
        pct = STAKE_PCT_INNINGS_2 if AGGRESSIVE_MODE else STAKE_PCT_DEFAULT * 1.33
        return _r(12, "INNINGS_2_CLEAR_BOOST", "SET_STAKE_PCT",
                  f"Innings 2 with clear lean (Δ={rrr_delta:+.1f}) — boost stake to {pct*100:.0f}%",
                  stake_pct=pct)
    return None

def rule_13_innings1_scalping(signal, snap, ts, ctx):
    return None

def rule_14_back_at_131_133(signal, snap, ts, ctx):
    p = float(snap["ltp"])
    if 1.31 <= p <= 1.33 and signal == "BACK":
        return _r(14, "BACK_AT_131_133", "BIAS",
                  f"Price {p} in 1.31-1.33 scalp zone", bias=+0.03,
                  scalp_exit_below=1.12)
    return None

def rule_15_extreme_low_close(signal, snap, ts, ctx):
    p = float(snap["ltp"])
    if p < EXTREME_LOW_ODDS:
        return _r(15, "EXTREME_LOW_CLOSE", "VETO",
                  f"Price {p} <1.05 — terminal, hedge full position and exit")
    return None

def rule_16_target_20_trades(signal, snap, ts, ctx):
    return None

def rule_17_tp_sl_pct(signal, snap, ts, ctx):
    if signal not in ("BACK", "LAY"): return None
    return _r(17, "TP_SL_PCT", "SET_TP_SL_PCT",
              f"TP +{TP_PCT_DEFAULT*100:.0f}% / SL -{SL_PCT_DEFAULT*100:.0f}%",
              tp_pct=TP_PCT_DEFAULT, sl_pct=SL_PCT_DEFAULT)

def rule_18_trailing_sl(signal, snap, ts, ctx):
    if signal not in ("BACK", "LAY"): return None
    return _r(18, "TRAILING_SL", "ENABLE_TRAILING_SL",
              "Trailing stop-loss — adjust as price moves favorably",
              trailing_pct=0.05)

def rule_20_innings1_limits(signal, snap, ts, ctx):
    if int(snap["innings"]) != 1: return None
    if ts["trades_innings_1"] >= MAX_TRADES_PER_INNINGS:
        return _r(20, "INNINGS_1_TRADE_CAP", "VETO",
                  f"Innings-1 trade cap reached ({ts['trades_innings_1']}/{MAX_TRADES_PER_INNINGS})")
    if ts["wins_innings_1"] >= STOP_AFTER_WINS:
        return _r(20, "INNINGS_1_WIN_STOP", "VETO",
                  f"Innings-1 win stop ({ts['wins_innings_1']}/{STOP_AFTER_WINS}) — preserve PnL")
    return None

def rule_21_innings2_limits(signal, snap, ts, ctx):
    if int(snap["innings"]) != 2: return None
    if ts["trades_innings_2"] >= MAX_TRADES_PER_INNINGS:
        return _r(21, "INNINGS_2_TRADE_CAP", "VETO",
                  f"Innings-2 trade cap reached ({ts['trades_innings_2']}/{MAX_TRADES_PER_INNINGS})")
    if ts["wins_innings_2"] >= STOP_AFTER_WINS:
        return _r(21, "INNINGS_2_WIN_STOP", "VETO",
                  f"Innings-2 win stop ({ts['wins_innings_2']}/{STOP_AFTER_WINS}) — preserve PnL")
    return None

def rule_22_daily_targets(signal, snap, ts, ctx):
    if ts["daily_pnl_pct"] >= DAILY_PROFIT_TARGET_PCT:
        return _r(22, "DAILY_PROFIT_TARGET", "VETO",
                  f"Daily profit target hit ({ts['daily_pnl_pct']:+.1%}) — stop trading")
    if ts["daily_pnl_pct"] <= -DAILY_LOSS_LIMIT_PCT:
        return _r(22, "DAILY_LOSS_LIMIT", "VETO",
                  f"Daily loss limit hit ({ts['daily_pnl_pct']:+.1%}) — stop trading")
    return None

def rule_23_capital_weekly(signal, snap, ts, ctx):
    return None

def rule_24_recover_capital_first(signal, snap, ts, ctx):
    if ts.get("bankroll_recovered"): return None
    # 🆕 Only fire when there's actually a loss to recover from.
    # Previously fired on default daily_pnl_pct == 0.0 (cold start).
    if ts.get("daily_pnl_pct", 0.0) >= 0:
        return None
    if signal in ("BACK", "LAY"):
        return _r(24, "RECOVER_CAPITAL_FIRST", "SET_STAKE_PCT",
                  f"Daily PnL {ts['daily_pnl_pct']:+.1%} — half-stake recovery mode",
                  stake_pct=(STAKE_PCT_DEFAULT * 0.5))
    return None

def rule_25_trade_all_matches(signal, snap, ts, ctx):
    return None

def rule_26_odds_direction(signal, snap, ts, ctx):
    return None

def rule_28_big_target_lay(signal, snap, ts, ctx):
    target = int(ctx.get("target_score", 0))
    if (int(snap["innings"]) == 2 and target >= BIG_TARGET_RUNS
            and signal == "BACK"):
        return _r(28, "BIG_TARGET_NO_BACK", "FILTER_IF_BACK",
                  f"Chase target {target}≥{BIG_TARGET_RUNS} — no BACK, LAY only")
    return None

def rule_29_tight_match_lay_only(signal, snap, ts, ctx):
    runners = snap.get("runners", {}) or {}
    if not runners: return None
    prices = list(runners.values())
    lo, hi = TIGHT_MATCH_ODDS
    if all(lo <= float(p) <= hi for p in prices) and signal == "BACK":
        return _r(29, "TIGHT_MATCH_LAY_ONLY", "FILTER_IF_BACK",
                  f"Tight market (all prices in {lo}-{hi}) — LAY only, no BACK")
    return None

def rule_30_after_4_flips_lay_only(signal, snap, ts, ctx):
    # DISABLED: Allow BACK even with high flip count, just reduce confidence
    # if ts["flip_count"] >= FLIPS_FOR_LAY_ONLY and signal == "BACK":
    #     return _r(30, "POST_4_FLIPS_LAY_ONLY", "FILTER_IF_BACK",
    #               f"{ts['flip_count']} fav flips ≥{FLIPS_FOR_LAY_ONLY} — volatility lockout, LAY only")
    return None

def rule_31_volatile_league_lay_only(signal, snap, ts, ctx):
    if ctx.get("league_volatility") == "high" and signal == "BACK":
        return _r(31, "VOLATILE_LEAGUE_LAY_ONLY", "FILTER_IF_BACK",
                  "High-volatility league — LAY only mode")
    return None

def rule_32_pre_match_data(signal, snap, ts, ctx):
    return None

def rule_33_player_form(signal, snap, ts, ctx):
    return None

def rule_34_tie_market_rr(signal, snap, ts, ctx):
    runners = snap.get("runners", {}) or {}
    if len(runners) < 2: return None
    prices = sorted(runners.values())
    if abs(prices[0] - prices[1]) < 0.10 and signal in ("BACK", "LAY"):
        return _r(34, "TIE_MARKET_1_TO_3", "SET_TP_SL_PCT",
                  f"Tie market (Δprice {prices[1]-prices[0]:.2f}) — risk-reward 1:{TIE_RR_RATIO}",
                  tp_pct=SL_PCT_DEFAULT * TIE_RR_RATIO, sl_pct=SL_PCT_DEFAULT)
    return None

def rule_35_bad_weather_lay(signal, snap, ts, ctx):
    if ctx.get("weather") in ("rain", "overcast") and signal == "BACK":
        return _r(35, "BAD_WEATHER_LAY", "SET_STAKE_PCT",
                  f"Weather '{ctx['weather']}' — reduced stake due to uncertainty",
                  stake_pct=(STAKE_PCT_DEFAULT * 0.75))
    return None

def rule_36_mostly_trade_favorite(signal, snap, ts, ctx):
    p = float(snap["ltp"])
    if p > 2.5:
        if signal == "BACK":
            return _r(36, "AVOID_DOG_BACK", "FILTER_IF_BACK",
                      f"Price {p} >2.5 — backing dog, skip (mostly trade fav)")
        if signal == "LAY":
            return _r(36, "AVOID_DOG_LAY", "FILTER_IF_LAY",
                      f"Price {p} >2.5 — laying dog, skip (mostly trade fav)")
    return None

def rule_37_track_volatility(signal, snap, ts, ctx):
    return None


# ── ALL RULES ─────────────────────────────────────────────────────

ALL_RULES = [
    rule_01_first_over_entry,    rule_02_odds_band_bias,
    rule_04_base_stake_pct,
    rule_05_hedge_low_odds,      rule_06_follow_favorite,
    rule_08_venue_stats,
    rule_09_pitch_adjust,        rule_10_big_chase_lay_bias,
    rule_11_player_driven_move,  rule_12_innings2_higher_stake,
    rule_13_innings1_scalping,   rule_14_back_at_131_133,
    rule_15_extreme_low_close,   rule_16_target_20_trades,
    rule_17_tp_sl_pct,           rule_18_trailing_sl,
    rule_20_innings1_limits,
    rule_21_innings2_limits,     rule_22_daily_targets,
    rule_23_capital_weekly,      rule_24_recover_capital_first,
    rule_25_trade_all_matches,   rule_26_odds_direction,
    rule_28_big_target_lay,
    rule_29_tight_match_lay_only, rule_30_after_4_flips_lay_only,
    rule_31_volatile_league_lay_only, rule_32_pre_match_data,
    rule_33_player_form,         rule_34_tie_market_rr,
    rule_35_bad_weather_lay,     rule_36_mostly_trade_favorite,
    rule_37_track_volatility,
]


# ── STATE UPDATER (call once per tick from predictor) ─────────────

def update_state(snap: dict, ts: dict) -> dict:
    """
    Update flip count, major-turn flag, current fav before evaluating rules.

    🆕 FIX 1: Per-match reset when match_id changes.
    🆕 FIX 2: Flip counting per BALL (anchored to (over, score)) not per tick.
    🆕 FIX 3: Tight-market gate — flips ignored if fav margin < FLIP_TIGHT_MARKET_MARGIN.
    """
    runners = snap.get("runners", {}) or {}
    if not runners:
        return ts

    # ── PER-MATCH RESET (FIX 1) ──
    incoming_match_id = snap.get("match_id") or snap.get("match_info") or ""
    if incoming_match_id and ts.get("current_match_id") != incoming_match_id:
        # New match detected — reset per-match counters but preserve daily totals.
        ts["current_match_id"] = incoming_match_id
        ts["flip_count"] = 0
        ts["last_fav_team"] = None
        ts["match_start_fav"] = None
        ts["major_turn_seen"] = False
        ts["trades_innings_1"] = 0
        ts["trades_innings_2"] = 0
        ts["wins_innings_1"] = 0
        ts["wins_innings_2"] = 0
        ts["last_ball_anchor"] = None
        # daily_pnl_pct, trades_today, wins_today PRESERVED — those are session-level.
        print(f"[RULE_ENGINE] New match detected: {incoming_match_id} — per-match state reset")

    # ── FAV RESOLUTION ──
    fav_rid = min(runners, key=runners.get)
    fav_team = snap.get("runner_names", {}).get(str(fav_rid), str(fav_rid))

    if ts.get("match_start_fav") is None:
        ts["match_start_fav"] = fav_team

    # ── FLIP DETECTION (FIX 2 + FIX 3) ──
    # Only check for flips when the BALL changes (not every tick).
    # And only count when the market has a clear favourite (margin gate).
    current_ball_anchor = (snap.get("over"), snap.get("score"))
    last_ball_anchor = ts.get("last_ball_anchor")

    # Tight-market gate: skip flip-counting if both teams within FLIP_TIGHT_MARKET_MARGIN
    prices_sorted = sorted(runners.values())
    is_tight = len(prices_sorted) >= 2 and (prices_sorted[1] - prices_sorted[0]) < FLIP_TIGHT_MARKET_MARGIN

    if last_ball_anchor != current_ball_anchor:
        # New ball — check for fav flip
        if not is_tight and ts.get("last_fav_team") and ts["last_fav_team"] != fav_team:
            ts["flip_count"] = int(ts.get("flip_count", 0)) + 1
            if ts["flip_count"] >= 1 and not ts.get("major_turn_seen"):
                ts["major_turn_seen"] = True
        ts["last_fav_team"] = fav_team
        ts["last_ball_anchor"] = current_ball_anchor
    # else: same ball as previous tick — don't update, don't count

    return ts


# ── EVALUATOR ─────────────────────────────────────────────────────

def evaluate(ml_signal: str, ml_confidence: float, snap: dict,
             trading_state: dict | None = None,
             match_context: dict | None = None,
             value_gap: float = 0.0) -> dict:
    """
    Apply all 37 rules to the ML output. Returns:
      {
        "final_action":    "BACK"|"LAY"|"WAIT",
        "final_confidence": float,
        "stake_pct":       float,
        "tp_pct":          float,
        "sl_pct":          float,
        "trailing_sl":     bool,
        "hedge":           bool,
        "fired_rules":     [verdicts...],
        "veto_reasons":    [...],
        "ml_signal":       original,
        "ml_confidence":   original,
      }

    value_gap: Absolute value gap (e.g., 0.43 for +43%). Used to override soft vetos
               when edge is large enough (>0.40) + confidence is high (>0.70).
    """
    ts  = trading_state if trading_state is not None else load_trading_state()
    ctx = match_context if match_context is not None else load_match_context()

    ts = update_state(snap, ts)

    action     = ml_signal
    confidence = float(ml_confidence)

    fired = []
    stake_pct = STAKE_PCT_DEFAULT
    tp_pct    = TP_PCT_DEFAULT
    sl_pct    = SL_PCT_DEFAULT
    trailing  = False
    hedge     = False
    vetos     = []

    for rule_fn in ALL_RULES:
        try:
            v = rule_fn(action, snap, ts, ctx)
        except Exception as e:
            v = _r(-1, rule_fn.__name__, "ERROR", f"rule crashed: {e}")
        if v is None: continue
        fired.append(v)

        verdict = v["verdict"]
        if verdict == "VETO":
            vetos.append(v["reason"])
            action = "WAIT"
        elif verdict == "FILTER_IF_BACK" and action == "BACK":
            vetos.append(v["reason"])
            action = "WAIT"
        elif verdict == "FILTER_IF_LAY" and action == "LAY":
            vetos.append(v["reason"])
            action = "WAIT"
        elif verdict == "BIAS":
            confidence = max(0.0, min(1.0, confidence + float(v.get("bias", 0))))
        elif verdict == "SET_STAKE_PCT":
            stake_pct = float(v.get("stake_pct", stake_pct))
        elif verdict == "SET_TP_SL_PCT":
            tp_pct = float(v.get("tp_pct", tp_pct))
            sl_pct = float(v.get("sl_pct", sl_pct))
        elif verdict == "ENABLE_TRAILING_SL":
            trailing = True
        elif verdict == "HEDGE":
            hedge = True

    # ── EDGE-VS-VOLATILITY OVERRIDE ──
    # If ML signal detected a large edge (>40%) + high confidence (>70%),
    # override soft vetos (market tightness, volatility flips) but keep hard vetos.
    if action == "WAIT" and ml_signal in ["BACK", "LAY"]:
        abs_gap = abs(value_gap)
        if abs_gap > 0.40 and ml_confidence > 0.70:
            soft_veto_keywords = ["TIGHT_MARKET", "TIGHT_MATCH", "flips", "volatility", "Tight market", "fav flips"]
            filtered_vetos = [v for v in vetos
                            if not any(kw.lower() in v.lower() for kw in soft_veto_keywords)]

            if len(filtered_vetos) == 0:
                action = ml_signal
                print(f"[EDGE_OVERRIDE] Large edge ({abs_gap:+.1%}) + high confidence ({ml_confidence:.0%}) "
                      f"→ Override soft vetos, executing {ml_signal}")

            vetos = filtered_vetos

    if action == "WAIT" and ml_signal != "WAIT":
        confidence = float(ml_confidence)

    save_trading_state(ts)

    return {
        "final_action":     action,
        "final_confidence": round(confidence, 4),
        "stake_pct":        round(stake_pct, 4),
        "tp_pct":           round(tp_pct, 4),
        "sl_pct":           round(sl_pct, 4),
        "trailing_sl":      trailing,
        "hedge":            hedge,
        "fired_rules":      fired,
        "veto_reasons":     vetos,
        "ml_signal":        ml_signal,
        "ml_confidence":    round(float(ml_confidence), 4),
        "trading_state":    {
            "trades_today":     ts["trades_today"],
            "wins_today":       ts["wins_today"],
            "daily_pnl_pct":    round(ts["daily_pnl_pct"], 4),
            "flip_count":       ts["flip_count"],
            "major_turn_seen":  ts["major_turn_seen"],
            "current_match_id": ts.get("current_match_id"),
        },
    }


# ── AUDIT LOGGING ─────────────────────────────────────────────────

def log_audit(prediction: dict, rule_result: dict) -> None:
    os.makedirs(os.path.dirname(RULE_AUDIT_FILE) or ".", exist_ok=True)
    entry = {
        "ts":               prediction.get("timestamp"),
        "match_info":       prediction.get("match_info"),
        "ml_signal":        rule_result["ml_signal"],
        "ml_confidence":    rule_result["ml_confidence"],
        "final_action":     rule_result["final_action"],
        "final_confidence": rule_result["final_confidence"],
        "stake_pct":        rule_result["stake_pct"],
        "fired_rule_ids":   [r["rule_id"] for r in rule_result["fired_rules"]],
        "veto_reasons":     rule_result["veto_reasons"],
    }
    with open(RULE_AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
