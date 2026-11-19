#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a future target time.

Reads one case per file (schema "makefun/gta6-trophies-case/v1") from the
input directory and writes, for each case, a JSON forecast of five
quantiles per trophy to the output directory.

Model, per trophy (all work happens on the logit scale, where a trophy's
earn rate r percent becomes logit(r) = ln(r / (100 - r))):

1.  Trend.  The log of the earn rate is roughly linear in log time since
    the release day began: both the number of players Exophase tracks and
    the number who have earned a given trophy grow like powers of the
    time since launch, so their ratio -- the earn rate -- is a power law
    in time, i.e. linear in log time on the logit scale (for the small
    rates most trophies have, logit and log agree).  But a trophy's rate
    cannot rise like that forever: it levels off as the players who will
    ever earn it have done so, so the true curve is concave in log time.
    We therefore fit two shapes per trophy and blend them by in-sample
    fit: a recency-weighted least-squares line of logit(rate) against
    ln(1 + hours since 2026-11-19T00:00:00Z) (weights decaying with a
    24-hour half-life, extrapolated move damped to 80% and capped at
    2.5 logit units), and a saturating hyperbola r = R * u / (u + K)
    with u = 1 + hours, fitted on the logit scale by grid search over
    the ceiling R and the timescale K (coarse grid plus a pattern-search
    refinement), using the exact form logit(R * u / (u + K)) and only
    reads from 12 hours after the release-day start, where the
    hyperbola's shape is cleanest.

2.  Spread.  The predictive spread (one logit standard deviation, turned
    into quantiles with normal z-scores) combines four honest sources of
    doubt, added in quadrature:
      a. the standard error of the trend line's prediction at the target,
      b. the weighted residual scatter of past reads around the line,
      c. refresh jumps: Exophase does not update its page continuously;
         it refreshes in jumps, hours apart, and a jump moves every rate
         at once (newly tracked players dilute the rates).  We measure
         the typical per-refresh logit jump from the history's own
         changes and scale it by the square root of the number of
         refreshes expected before the target.  Only half the measured
         jump is counted: part of every jump is the trend's own step
         between refreshes, which the trend fit already forecasts, and
         refresh shocks partly mean-revert as progress catches up, so
         the full measured jump would double-count the trend as noise.
         When the history never moved at all, we fall back to a prior
         jump size of 1.0 logit and a 4-hour refresh interval, so a flat
         launch night still gets a band wide enough to survive the first
         refresh.
      d. a horizon floor that grows with the square root of the horizon
         in days, so the forecast is never overconfident.
    The total spread is clamped to [0.30, 1.80] logit units.

3.  Edge cases.  Rates are clamped to [0.01, 99.99] before the logit
    (the page prints two decimals; 0% and 100% have no logit).  A trophy
    with fewer than three usable reads, or no usable read at all, falls
    back to persistence -- its last rate as the median, with the
    horizon-floor spread.  Output quantiles are rounded to four decimals
    and forced non-decreasing.

The program is deterministic: no randomness, no wall-clock reads, no
network.  Standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# The release day began at 2026-11-19T00:00:00Z; log time is measured
# from there, matching the house's own curve baseline convention.
RELEASE_START = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Standard-normal quantiles for the five requested levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

RATE_LO = 0.01
RATE_HI = 99.99

# Recency half-life of a history read, in hours.
WEIGHT_HALF_LIFE_H = 24.0
# How much of the fitted move from the last rate we trust.
DAMPING = 0.8
# Largest median move (logit units) we allow away from the last rate.
MAX_MEDIAN_SHIFT = 2.5
# Prior refresh-jump size (logit) and refresh interval (hours), used
# only when the history never changed, so a flat launch night still
# expects its first refresh.
PRIOR_JUMP_LOGIT = 1.0
PRIOR_REFRESH_INTERVAL_H = 4.0
# Spread clamps (logit standard deviation).
SPREAD_MIN = 0.30
SPREAD_MAX = 1.80
# Horizon floor: 0.25 + 0.10 * sqrt(horizon in days).
FLOOR_BASE = 0.25
FLOOR_PER_SQRT_DAY = 0.10


def parse_time(text):
    """Parse an ISO 8601 timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def logit(rate_pct):
    """Logit of a percent rate, with the rate clamped to (0, 100)."""
    p = min(max(rate_pct, RATE_LO), RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    """Inverse logit, returned as a percent in [0.01, 99.99]."""
    if value >= 0.0:
        p = 1.0 / (1.0 + math.exp(-value))
    else:
        e = math.exp(value)
        p = e / (1.0 + e)
    return min(max(p * 100.0, RATE_LO), RATE_HI)


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def log_time(at):
    """ln(1 + hours from the release-day start to `at`)."""
    hours = (at - RELEASE_START).total_seconds() / 3600.0
    return math.log(1.0 + max(hours, -0.99))


def saturating_fit(xs, ys, ws, target_x):
    """Saturating (hyperbolic) fit of logit(rate) against log time.

    Model: r(u) = R * u / (u + K), where u = 1 + hours since the
    release-day start and x = ln(u).  A trophy's earn rate rises like a
    power of time early on and levels off at R as the players who will
    ever earn it have done so -- the shape a cumulative completion curve
    takes.  The fit is done on the logit scale with the exact form
    logit(R * u / (u + K)): fitting log(rate) instead would miss how the
    logit keeps steepening while the rate is large.  R and K are picked
    by a grid search (R above the largest observed rate, K log-spaced);
    for K far larger than u the model reduces to power-law growth, so
    it nests the no-saturation case.  Returns (y_at_target, sse,
    resid_sd) or None when the fit is degenerate.
    """
    n = len(xs)
    if n < 3:
        return None
    total_w = sum(ws)
    if total_w <= 0.0:
        return None
    # Fit only the reads from 12 hours after the release-day start
    # (u >= 13).  The hyperbola is written in u = 1 + hours, while a
    # real trophy curve is ~0 at the release instant; the two shapes
    # differ most in the first hours, so those reads would pull the
    # fit away from the recent curvature that matters for the
    # extrapolation.  (A case whose history is all earlier than that
    # has no saturating fit and falls back to the line.)
    kept = [(x, y, w) for x, y, w in zip(xs, ys, ws)
            if math.exp(x) >= 13.0]
    if len(kept) < 3:
        return None
    xs, ys, ws = (list(part) for part in zip(*kept))
    n = len(xs)
    total_w = sum(ws)
    us = [math.exp(x) for x in xs]
    r_max = max(inv_logit(y) for y in ys)
    # Candidate ceilings: a little above the largest observed rate up to
    # (nearly) everybody earning the trophy.
    r_grid = []
    for mult in (1.05, 1.15, 1.3, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0,
                 8.0, 12.0, 20.0, 40.0):
        cand = r_max * mult
        if cand >= 99.99:
            cand = 99.99
        if not r_grid or cand > r_grid[-1] * 1.02:
            r_grid.append(cand)
        if cand >= 99.99:
            break
    u_target = math.exp(target_x)
    log = math.log

    def sse_for(r_ceil, k):
        """Weighted SSE of the logit curve logit(R*u/(u+K))."""
        total = 0.0
        for u, y, w in zip(us, ys, ws):
            rate = r_ceil * u / (u + k)
            if rate >= 99.99:
                rate = 99.99
            p = rate / 100.0
            diff = y - log(p / (1.0 - p))
            total += w * diff * diff
        return total

    def y_at(r_ceil, k):
        rate = r_ceil * u_target / (u_target + k)
        if rate >= 99.99:
            rate = 99.99
        p = rate / 100.0
        return log(p / (1.0 - p))

    best = None  # (sse, R, K)
    k = 0.5
    while k <= 4096.0:
        for r_ceil in r_grid:
            sse = sse_for(r_ceil, k)
            if best is None or sse < best[0]:
                best = (sse, r_ceil, k)
        k *= 2.0
    if best is None:
        return None
    # Refine: the SSE surface is sharp in R (a few percent of R can
    # change it several-fold), far sharper than the coarse grids, so
    # walk the best point to a local minimum in (R, K).  The valley is
    # diagonal -- a lower ceiling wants a faster approach -- so each
    # round tries coordinate steps and diagonal steps together.
    for _round in range(4):
        _sse, r_best, k_best = best
        for mult in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.18):
            cand_r = min(max(r_best * mult, r_max * 1.001), 99.99)
            sse = sse_for(cand_r, k_best)
            if sse < best[0]:
                best = (sse, cand_r, k_best)
        _sse, r_best, k_best = best
        for mult in (0.6, 0.75, 0.9, 1.0, 1.1, 1.33, 1.67):
            cand_k = k_best * mult
            sse = sse_for(r_best, cand_k)
            if sse < best[0]:
                best = (sse, r_best, cand_k)
        _sse, r_best, k_best = best
        for r_mult, k_mult in ((0.85, 0.75), (0.85, 1.33),
                               (1.18, 0.75), (1.18, 1.33),
                               (0.9, 0.9), (1.1, 1.1)):
            cand_r = min(max(r_best * r_mult, r_max * 1.001), 99.99)
            cand_k = k_best * k_mult
            sse = sse_for(cand_r, cand_k)
            if sse < best[0]:
                best = (sse, cand_r, cand_k)
    sse, r_best, k_best = best
    resid_sd = math.sqrt(max(sse / total_w, 0.0))
    return y_at(r_best, k_best), sse, resid_sd


def weighted_line(xs, ys, ws):
    """Weighted least squares of y on x.

    Returns (intercept, slope, pred_se_fn, resid_sd) where pred_se_fn
    maps an x value to the standard error of the fitted mean there and
    resid_sd is the weighted residual standard deviation.  Returns None
    when the fit is degenerate (fewer than 3 points or no spread in x).
    """
    n = len(xs)
    if n < 3:
        return None
    total_w = sum(ws)
    if total_w <= 0.0:
        return None
    x_bar = sum(w * x for x, w in zip(xs, ws)) / total_w
    y_bar = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar)
              for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    # Weighted residual variance, corrected for the two fitted params.
    s2 = (sum(w * r * r for r, w in zip(residuals, ws)) / total_w
          * n / max(n - 2, 1))
    resid_sd = math.sqrt(max(s2, 0.0))
    var_slope = s2 / sxx
    var_intercept = s2 * (1.0 / total_w + x_bar * x_bar / sxx)
    cov = -s2 * x_bar / sxx

    def pred_se(x_new):
        var = (var_intercept + 2.0 * x_new * cov
               + x_new * x_new * var_slope)
        return math.sqrt(max(var, 0.0))

    return intercept, slope, pred_se, resid_sd


def forecast_trophy(series, target_x, horizon_hours):
    """Forecast one trophy's five quantiles.

    `series` is a list of (log_time, logit_rate, weight) triples, oldest
    first, for the reads where this trophy had a rate.

    Returns ("fit", centre, pred_se, resid_sd, jump_size, floor) for a
    trophy with a usable trend fit, or ("done", quantiles) when the
    trophy falls back to persistence / a neutral prior.
    """
    horizon_days = max(horizon_hours, 0.0) / 24.0
    floor = FLOOR_BASE + FLOOR_PER_SQRT_DAY * math.sqrt(horizon_days)

    if not series:
        # No information at all: a wide, neutral prior centred at 1%.
        centre = logit(1.0)
        spread = max(1.5, floor)
        return ("done", quantiles(centre, spread))

    last_logit = series[-1][1]
    xs = [s[0] for s in series]
    ys = [s[1] for s in series]
    ws = [s[2] for s in series]
    fit = weighted_line(xs, ys, ws)
    sat = saturating_fit(xs, ys, ws, target_x)
    if fit is None and sat is None:
        # Too little history to fit a line: persist the last rate.
        return ("done", quantiles(last_logit,
                                  max(floor, 0.5 * math.sqrt(horizon_days))))

    # Two shapes for the same curve: a straight line of logit(rate) on
    # log time (keeps rising) and a saturating hyperbola (levels off).
    # Blend them by in-sample fit, so the shape the history actually
    # follows gets the weight; a flat history makes both agree on the
    # last rate anyway.
    total_w = sum(ws)
    n = len(xs)
    y_lin = None
    se = 0.0
    resid_lin = 0.0
    sse_lin = None
    if fit is not None:
        intercept, _slope, pred_se, resid_lin = fit
        fitted_target = intercept + _slope * target_x
        y_lin = last_logit + DAMPING * (fitted_target - last_logit)
        se = pred_se(target_x)
        sse_lin = resid_lin * resid_lin * total_w * max(n - 2, 1) / n
    y_sat = None
    resid_sat = 0.0
    sse_sat = None
    if sat is not None:
        y_sat, sse_sat, resid_sat = sat
    if y_lin is not None and y_sat is not None:
        denom = (sse_lin or 0.0) + (sse_sat or 0.0)
        w_sat = (sse_lin or 0.0) / denom if denom > 1e-12 else 0.0
        centre_raw = w_sat * y_sat + (1.0 - w_sat) * y_lin
        resid_sd = math.sqrt(w_sat * resid_sat * resid_sat
                             + (1.0 - w_sat) * resid_lin * resid_lin)
    elif y_sat is not None:
        centre_raw = y_sat
        resid_sd = resid_sat
    else:
        centre_raw = y_lin
        resid_sd = resid_lin
    shift = centre_raw - last_logit
    shift = min(max(shift, -MAX_MEDIAN_SHIFT), MAX_MEDIAN_SHIFT)
    centre = last_logit + shift

    # Refresh jumps: typical |logit change| between consecutive reads
    # where this trophy's rate moved at all.
    jumps = [abs(series[i][1] - series[i - 1][1])
             for i in range(1, len(series))
             if series[i][1] != series[i - 1][1]]
    jump_size = median(jumps) if jumps else None
    return ("fit", centre, se, resid_sd, jump_size, floor)


def quantiles(centre, spread):
    """Five quantiles around `centre` (logit) with sd `spread`."""
    spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
    out = []
    prev = None
    for z in Z_SCORES:
        q = round(inv_logit(centre + z * spread), 4)
        q = min(max(q, RATE_LO), RATE_HI)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    """Forecast every trophy of one case; returns the output object."""
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_hours = max((target - checkpoint).total_seconds(), 0.0) / 3600.0
    target_x = log_time(target)

    history = case.get("history", [])
    trophies = case["trophies"]
    n_trophies = len(trophies)

    # Per-read log times and recency weights (half-life 24 h).
    read_times = []
    for read in history:
        at = parse_time(read["at"])
        read_times.append(at)
    if read_times:
        last_time = read_times[-1]
        weights = [0.5 ** ((last_time - at).total_seconds() / 3600.0
                           / WEIGHT_HALF_LIFE_H)
                   for at in read_times]
    else:
        weights = []

    # Refresh interval: median gap between consecutive reads where the
    # page changed at all (any rate, or the players-tracked count).
    # The gap is measured from the previous CHANGE, not the previous
    # read: reads between refreshes are identical, so measuring from
    # the previous read would just return the read cadence.
    change_gaps = []
    prev_rates = None
    prev_players = None
    last_change_at = None
    for read, at in zip(history, read_times):
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if prev_rates is not None:
            changed = (players != prev_players or len(rates) != len(prev_rates)
                       or any(a != b for a, b in zip(rates, prev_rates)))
            if changed:
                if last_change_at is not None:
                    change_gaps.append(
                        (at - last_change_at).total_seconds() / 3600.0)
                last_change_at = at
        prev_rates = rates
        prev_players = players
    refresh_interval = (median(change_gaps) if change_gaps
                        else PRIOR_REFRESH_INTERVAL_H)
    if refresh_interval <= 0.0:
        refresh_interval = PRIOR_REFRESH_INTERVAL_H
    expected_refreshes = horizon_hours / refresh_interval

    # Global fallback jump size, from any trophy that ever moved.
    all_jumps = []

    # Build one series per trophy first, so the global jump fallback is
    # known before any trophy is forecast.
    series_by_trophy = []
    for idx in range(n_trophies):
        series = []
        for read, at, w in zip(history, read_times, weights):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                series.append((log_time(at), logit(float(rates[idx])), w))
        series_by_trophy.append(series)
        for i in range(1, len(series)):
            if series[i][1] != series[i - 1][1]:
                all_jumps.append(abs(series[i][1] - series[i - 1][1]))
    global_jump = median(all_jumps) if all_jumps else PRIOR_JUMP_LOGIT

    rows = []
    for trophy, series in zip(trophies, series_by_trophy):
        kind, *rest = forecast_trophy(series, target_x, horizon_hours)
        if kind == "fit":
            # ("fit", centre, pred_se, resid_sd, jump_size, floor).
            centre, se, resid_sd, jump_size, floor = rest
            jump = jump_size if jump_size is not None else global_jump
            jump_spread = 0.5 * jump * math.sqrt(expected_refreshes)
            jump_spread = min(jump_spread, 1.2)
            spread = math.sqrt(se * se + resid_sd * resid_sd
                               + jump_spread * jump_spread + floor * floor)
            rows.append({"id": trophy["id"],
                         "q": quantiles(centre, spread)})
        else:
            # ("done", quantiles) -- fallback path, already finished.
            rows.append({"id": trophy["id"], "q": rest[0]})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)

    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    if not names:
        print("no .in case files in %s" % input_dir, file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"))
        if len(text.encode("utf-8")) > 65536:
            raise ValueError("forecast for %s exceeds 65536 bytes" % case_id)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s (%d trophies, %d bytes)"
              % (out_path, len(out["trophies"]), len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
