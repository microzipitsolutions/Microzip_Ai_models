"""
BetPredict Unified Pattern Engine
=================================
Centralized logic for 22 Match Patterns affecting Betfair Prices.
"""

import numpy as np

# Pattern Definitions: (Signal, Category, UI_Color, Description)
PATTERN_META = {
    # PRICE UP (LAY)
    "WICKET_SPIKE":       ("LAY",  "wicket",   "#E24B4A", "Sudden price spike after wicket"),
    "POWERPLAY_WICKET":   ("LAY",  "wicket",   "#E24B4A", "Wicket in first 6 overs"),
    "MIDDLE_COLLAPSE":    ("LAY",  "wicket",   "#E24B4A", "3+ wickets in overs 7-14"),
    "WICKET_COLLAPSE":    ("LAY",  "wicket",   "#E24B4A", "3+ wickets in 3 overs"),
    "TAIL_EXPOSED":       ("LAY",  "wicket",   "#E24B4A", "7+ wickets down late in game"),
    "CHASE_CRISIS":       ("LAY",  "pressure", "#854F0B", "RRR climbing 2.5+ above CRR"),
    "RUN_RATE_DANGER":    ("LAY",  "pressure", "#854F0B", "RRR > 12 in middle overs"),
    "DOT_BALL_TRAP":      ("LAY",  "pressure", "#854F0B", "5+ dot balls in last 6"),
    "PRESSURE_BUILD":     ("LAY",  "pressure", "#854F0B", "4+ dot balls in last 6"),
    "FAVOURITE_DRIFT":    ("LAY",  "price",    "#534AB7", "Fav price drifting above 2.5"),
    "MIDDLE_GRIND":       ("LAY",  "phase",    "#185FA5", "Slow scoring with 2-4 wkts down"),

    # PRICE DOWN (BACK)
    "SIX_STORM":          ("BACK", "batting",  "#3B6D11", "Six hit on last ball"),
    "BOUNDARY_BURST":     ("BACK", "batting",  "#3B6D11", "3+ boundaries in last 6 balls"),
    "BATTING_SURGE":      ("BACK", "batting",  "#3B6D11", "15+ runs in last 3 overs"),
    "POWERPLAY_SURGE":    ("BACK", "batting",  "#3B6D11", "3+ boundaries in powerplay"),
    "POWERPLAY_DOMINANT": ("BACK", "batting",  "#3B6D11", "50+ runs with <2 wkts by over 6"),
    "DEATH_CARNAGE":      ("BACK", "batting",  "#3B6D11", "15+ runs late (Over 17+)"),
    "DEATH_REVIVAL":      ("BACK", "batting",  "#3B6D11", "Scoring runs with 5+ wkts down"),
    "DEATH_SLOG":         ("BACK", "batting",  "#3B6D11", "High momentum with wkts in hand"),
    "CHASE_CONTROLLED":   ("BACK", "pressure", "#854F0B", "CRR well above RRR in chase"),
    "UNDERDOG_SURGE":     ("BACK", "price",    "#534AB7", "Dog price dropping below 2.0"),

    # STABLE
    "PRICE_STABILIZE":    ("WAIT", "price",    "#888780", "Price stable - no signal")
}


def _parse_recent(recent: str):
    """
    Parse Cricbuzz recentOvsStats string into a list of integer run values.
    Wicket balls ('W', 'Wd', 'Nb', etc.) are treated as 0 runs so the
    window length stays accurate (a wicket ball is not a dot ball, but
    it contributes 0 batting runs for surge/carnage calculations).
    Returns (recent_list, recent_runs, recent_wickets_count).
    """
    tokens = recent.replace('|', ' ').split()
    runs = []
    wickets_seen = 0
    for tok in tokens:
        upper = tok.upper()
        # Extract digits from token (handles "4W", "1Wd", etc.)
        digits = ''.join(filter(str.isdigit, tok))
        if 'W' in upper:
            wickets_seen += 1
            runs.append(int(digits) if digits else 0)
        elif digits:
            runs.append(int(digits))
        else:
            # Wide/no-ball symbols with no digit — treat as 0 for window integrity
            runs.append(0)
    return tokens, runs, wickets_seen


def detect_patterns(row) -> list:
    """Returns a list of active pattern names for a given ball/state row."""
    p = []

    over       = float(row.get('over', 0))
    over_int   = int(over)
    wickets    = int(row.get('wickets_before', 0))
    innings    = int(row.get('innings', 1))
    rrr        = float(row.get('rrr', 0))
    crr        = float(row.get('crr', 0))
    recent_raw = str(row.get('recent', ""))
    price      = float(row.get('betfair_price', 2.0))
    change     = float(row.get('price_change', 0))

    recent_list, recent_runs, recent_wkts_count = _parse_recent(recent_raw)

    last6_runs   = recent_runs[-6:]  if len(recent_runs) >= 6  else recent_runs
    last6_sum    = sum(last6_runs)
    last6_dots   = last6_runs.count(0)
    last6_bounds = [r for r in last6_runs if r >= 4]

    # ------------------------------------------------------------------ #
    # 1. WICKET PATTERNS
    # ------------------------------------------------------------------ #
    if wickets >= 7 and over >= 15:
        p.append("TAIL_EXPOSED")

    recent_wickets_tokens = [x for x in recent_list if 'W' in x.upper()]
    if len(recent_wickets_tokens) >= 3:
        p.append("WICKET_COLLAPSE")
    if 7 <= over_int <= 14 and len(recent_wickets_tokens) >= 2:
        p.append("MIDDLE_COLLAPSE")

    if recent_list:
        last_tok = recent_list[-1].upper()
        if 'W' in last_tok:
            if over_int <= 5:
                p.append("POWERPLAY_WICKET")
            if change > 0.05:
                p.append("WICKET_SPIKE")

    # ------------------------------------------------------------------ #
    # 2. PRESSURE PATTERNS
    # ------------------------------------------------------------------ #
    if innings == 2:
        if rrr - crr >= 2.5:
            p.append("CHASE_CRISIS")
        if crr - rrr >= 2.5:
            p.append("CHASE_CONTROLLED")
        if 6 <= over_int <= 14 and rrr > 12:
            p.append("RUN_RATE_DANGER")

    if len(last6_runs) >= 6:
        if last6_dots >= 5:
            p.append("DOT_BALL_TRAP")
        elif last6_dots >= 4:
            p.append("PRESSURE_BUILD")

    # Middle grind: slow scoring with mid-wickets (2-4 down, overs 6-14)
    if 6 <= over_int <= 14 and 2 <= wickets <= 4 and last6_sum <= 4 and len(last6_runs) >= 6:
        p.append("MIDDLE_GRIND")

    # ------------------------------------------------------------------ #
    # 3. BATTING SURGE PATTERNS
    # ------------------------------------------------------------------ #
    # Last ball was a six
    if recent_list and '6' in recent_list[-1]:
        p.append("SIX_STORM")

    # 3+ boundaries in last 6 balls
    if len(last6_bounds) >= 3:
        p.append("BOUNDARY_BURST")

    # 15+ runs in available recent window (last ~3 overs worth of data)
    last18_runs = recent_runs[-18:] if len(recent_runs) >= 6 else recent_runs
    if sum(last18_runs) >= 15 and len(last18_runs) >= 6:
        p.append("BATTING_SURGE")

    # Powerplay-specific batting
    if over_int <= 5:
        if len(last6_bounds) >= 3:
            p.append("POWERPLAY_SURGE")
        # Dominant powerplay: crr implies 50+ by over 6 with <2 wickets
        if over >= 2.0 and crr >= 9.0 and wickets < 2:
            p.append("POWERPLAY_DOMINANT")

    # Death over patterns (over 17+)
    if over >= 17:
        if last6_sum >= 15 and len(last6_runs) >= 6:
            p.append("DEATH_CARNAGE")
        if wickets >= 5 and last6_sum >= 8 and len(last6_runs) >= 6:
            p.append("DEATH_REVIVAL")
        if wickets <= 4 and last6_sum >= 10 and len(last6_runs) >= 6:
            p.append("DEATH_SLOG")

    # ------------------------------------------------------------------ #
    # 4. PRICE PATTERNS
    # ------------------------------------------------------------------ #
    if over >= 1.0:
        if price > 2.5 and change > 0.05:
            p.append("FAVOURITE_DRIFT")
        if price < 2.0 and change < -0.05:
            p.append("UNDERDOG_SURGE")

    # ------------------------------------------------------------------ #
    # 5. STABILITY (only when nothing else fires)
    # ------------------------------------------------------------------ #
    if not p:
        p.append("PRICE_STABILIZE")

    return p
