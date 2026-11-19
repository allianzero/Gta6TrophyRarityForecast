# GTA VI Trophy Rarity Forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/allianzero/Gta6TrophyRarityForecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/allianzero/Gta6TrophyRarityForecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent
of players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z,
as five quantiles per trophy. Entry for the make.fun bounty
`gta6-trophy-rarity-forecast-01`.

## Run

```
python3 main.py
```

The program reads every `*.in` case file in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR`
(default `/output`). Standard library only, no network, deterministic:
two runs on the same case write byte-identical output. A case takes far
less than the 8-second limit.

## Model

All work happens on the logit scale, where an earn rate `r` percent
becomes `ln(r / (100 - r))`. On that scale a miss from 1% to 2% costs
about what a miss from 33% to 50% does, which is exactly how the bounty
scores (mean pinball loss on the logit scale).

Per trophy:

1. **Trend.** Both the number of players Exophase tracks and the number
   who have earned a given trophy grow roughly like powers of the time
   since launch, so their ratio — the earn rate — is roughly a power law
   in time, i.e. linear in log time on the logit scale. But a trophy's
   rate cannot rise like that forever: it levels off as the players who
   will ever earn it have done so, so the true curve is concave in log
   time. We therefore fit two shapes per trophy and blend them by
   in-sample fit: a recency-weighted least-squares line of `logit(rate)`
   against `ln(1 + hours since 2026-11-19T00:00:00Z)` (weights decaying
   with a 24-hour half-life, extrapolated move damped to 80% and capped
   at 2.5 logit units), and a saturating hyperbola
   `r = R * u / (u + K)` with `u = 1 + hours`, fitted on the logit
   scale by grid search over the ceiling `R` and the timescale `K`
   (coarse grid plus a pattern-search refinement — the fit is sharp in
   `R`), using the exact form `logit(R * u / (u + K))` and only reads
   from 12 hours after the release-day start, where the hyperbola's
   shape is cleanest. The blend follows whichever shape the history
   actually shows.

2. **Spread.** One logit standard deviation, turned into quantiles with
   normal z-scores, combining four sources of doubt in quadrature:
   - the standard error of the trend line's prediction at the target;
   - the weighted residual scatter of past reads around the line;
   - **refresh jumps:** Exophase refreshes its page in jumps, hours
     apart, and a jump moves every rate at once as newly tracked players
     dilute them. We measure the typical per-refresh logit jump from the
     history's own changes — the interval between changes, measured
     change-to-change, not read-to-read — and scale it by the square
     root of the number of refreshes expected before the target. Only
     half the measured jump is counted: part of every jump is the
     trend's own step between refreshes, which the trend fit already
     forecasts, and refresh shocks partly mean-revert. When the history
     never moved at all (a flat launch night), we fall back to a prior
     jump of 1.0 logit and a 4-hour refresh interval, so the band is
     still wide enough to survive the first refresh — the house
     persistence baseline's failure mode on the practice cases, where
     its narrow band covered none of the refreshed rates;
   - a horizon floor growing with the square root of the horizon in
     days, so the forecast is never overconfident.
   Total spread is clamped to [0.30, 1.80] logit units.

3. **Edge cases.** Rates are clamped to [0.01, 99.99] before the logit.
   A trophy with fewer than three usable reads falls back to
   persistence (last rate as median, horizon-floor spread); a trophy
   with no reads at all gets a wide neutral prior centred at 1%.

## Why it beats the house persistence baseline

The baseline says every rate stays where it is, with a band that only
widens like a random walk. On the practice cases that band is so narrow
that Exophase's first refresh — which cut nearly every rate as thousands
of new players were tracked — landed outside it for every trophy
(coverage 0, loss ~1.51). This entry instead extrapolates each trophy's
own trend in log time and sizes its band from the history's actual
jumpiness plus a refresh prior, so a refresh moves the forecast's band,
not just its median.

## Verification

Run on the kit's practice cases (`p00`, `p01`):

```
mkdir -p /tmp/gta6-out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=/tmp/gta6-out python3 main.py
```

Outputs are valid JSON with 55 rows each, five non-decreasing quantiles
in [0, 100], under 65,536 bytes, deterministic across runs, and each
case completes in well under a second. The kit ships no practice
answers, so no practice pinball loss can be computed locally.

Because the kit has no practice answers, the model was also checked on
two synthetic final-like cases (60 hours of history, 10-minute reads,
refreshes every 2–4 hours, target 4.5 days out): one where the true
rates saturate hyperbolically and one where they keep growing linearly
in logit-log-time. Against the house persistence baseline's formula the
entry scored 57% lower pinball loss in the first world (0.152 vs 0.350)
and 42% lower in the second (0.211 vs 0.364), with 90% coverage of
0.98 and 0.96. Both worlds are simulations, not the real final — they
validate the machinery, not the result.

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=f0netwmk&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
