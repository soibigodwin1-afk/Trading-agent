# Harmonic Pattern Telegram Agent

Scans forex, crypto, and indices (plus a couple of individual stocks)
for harmonic patterns (Gartley, Deep Gartley, Bat, Alternate Bat,
Butterfly, Crab, Deep Crab, Shark) and classic chart patterns (Double
Top/Bottom, Head & Shoulders, Broadening, Rounding, Triangles, Selling
Climax), sends an annotated chart to Telegram when one completes and
passes confirmation, then tracks the outcome and replies under the
original alert once it resolves. No auto-trading -- this is
identification and tracking only.

Runs free on GitHub Actions.

## Design notes (read before changing thresholds)

This isn't a generic pattern scanner -- the ratio tables, tolerances,
stop-loss placement, RSI rules, and PRZ logic are pulled directly from
Scott Carney's *Harmonic Trading* trilogy and H.M. Gartley's original
*Profits in the Stock Market* (1935), not approximated. See
`patterns.py`, `chart_patterns.py`, and `indicators.py` docstrings for
the specific source and chapter behind each rule.

Key design decisions worth knowing about before touching the code:

- **RSI confirmation is a hard gate, not a scoring bonus.** A pattern
  with clean price geometry but no valid RSI BAMM divergence will NOT
  fire a completion alert (`rsi.require_confirmation` in config.yaml).
- **RSI BAMM is checked on the Proximate timeframe, not Primary**
  (Vol. 3 Ch. 8) -- the pattern is identified on the Primary timeframe,
  but confirmation is read one timeframe down, timed around the
  Primary pattern's own B and D pivots mapped onto Proximate bars.
- **RSI BAMM is split into Type-I and Type-II**, because they're
  genuinely different structures: Type-I is the pattern's own B-to-D
  swing (confirms at the same moment the pattern completes -- nothing
  is "missed"); Type-II is a real retest of the same PRZ after an
  initial reaction didn't commit (costs real time by definition, which
  is why Type-II trades get bigger objectives and more discretionary
  management). `check_rsi_bamm()` tries Type-I first, falls back to
  Type-II.
- **Trend alignment is a tag, never a block.** A pattern against the
  higher-timeframe trend still alerts -- it's just labeled
  "counter-trend" rather than silently dropped, since real reversals
  have to start as counter-trend signals somewhere.
- **5m is held to a stricter confluence bar** than other timeframes,
  reflecting Carney's own stated view that 15m is his reliability
  floor for harmonic execution.
- **Charts default to a log (ratio) price scale** for crypto and any
  symbol whose lookback window spans a wide price range -- matching
  Gartley's own charting convention (Ch. II), since Fibonacci ratios
  measure percentage moves, which only render consistently on a log
  axis. See `charting.should_use_log_scale()`.
- **Deriv forex/synthetics get a tick-count volume proxy** on 5m/15m/
  1H (not 4H/1D, where it would mean an unreasonable number of extra
  tick-history calls for little benefit) -- a standard substitute
  since these markets have no single real traded-volume figure. This
  is what feeds the Selling Climax and decisive-penetration volume
  checks for Deriv symbols; previously those only worked on yfinance
  stock data.
- **Outcome tracking uses a simple binary model** (target vs. stop,
  whichever hits first) -- not the full partial-exit/trailing-stop
  system described in the books. Simpler to build and to trust the
  resulting stats.
- **Rounding and Triangle detection are flagged as experimental** in
  the alert caption, since they're built on geometric heuristics
  (quadratic curve fit, trendline slope) rather than a precisely
  sourced ratio definition. Their own detection thresholds
  (`rounding_min_r2`, `triangle_flat_slope_frac`) -- not just
  confluence -- are eligible for monthly calibration proposals.

## Setup

### 1. Telegram bot
Message **@BotFather** on Telegram, `/newbot`, note the token. Send
the bot a message, then visit
`https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat ID.

### 2. Market data
No token needed for market data -- Deriv's public endpoint
(`wss://api.derivws.com/trading/v1/options/ws/public`) covers forex,
crypto, and indices with no auth. Individual stocks fall back to
`yfinance`. Confirm the exact Deriv symbol codes in `config.yaml`
against Deriv's `active_symbols` list before relying on them --
the ones shipped here are illustrative.

### 3. Repo setup
1. Push this to a new GitHub repo (public = unlimited free Actions minutes).
2. Settings -> Secrets and variables -> Actions -> add
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
3. Settings -> Actions -> General -> Workflow permissions ->
   "Read and write permissions" (needed for the state-commit heartbeat).
4. Two workflows run automatically: `scan.yml` (every 30 min) and
   `report.yml` (daily -- internally decides if a weekly/monthly
   report is actually due).

## How a signal reaches you

1. **Completion alert**: chart shows the pattern's pivots, the PRZ as
   a shaded zone with the converging ratios labeled, the Terminal
   Price Bar highlighted, and entry/stop/target as labeled lines.
   Only sent if confluence, PRZ-width sanity, and RSI BAMM all pass.
2. **Outcome reply**: threaded under the original alert once price
   hits the target, hits the stop, or the pattern's resolution window
   (`resolution_timeout_bars`) expires unresolved.
3. **Weekly digest**: plain win/loss/pending tally, skipped if nothing
   new happened.
4. **Monthly analysis + calibration proposals**: per-bucket win rates
   once enough samples exist (`calibration_sample_min`). Any proposed
   parameter change arrives with Accept/Decline buttons -- nothing
   changes until you respond, and unanswered proposals expire after
   `calibration_expiry_days` (default: no change).

## Files

- `data.py` -- Deriv (public, no token) + yfinance fallback for stocks
- `patterns.py` -- harmonic detection: ZigZag pivots, the standardized
  XABCD ratio table, confluence scoring, Shark/5-0's separate skeleton
- `chart_patterns.py` -- classic Gartley chart patterns + Selling Climax
- `indicators.py` -- Wilder RSI + RSI BAMM (the 50-level crossing rule)
- `trend.py` -- HTF trend classification + ATR-based regime filter
- `charting.py` -- renders the annotated "trade plan" chart
- `scan.py` -- the 30-minute orchestrator (Primary/Proximate/Distal)
- `report.py` -- weekly digest, monthly analysis, calibration proposals
- `state.py` / `state.json` -- dedup, outcome log, open watches,
  pending calibrations, all committed back each run

## Known limitations

- Deriv doesn't offer individual equities -- stocks route through
  yfinance, which is free but unofficial and occasionally flaky.
- Deriv's tick-count volume proxy is an approximation, not real traded
  volume -- reasonable for forex/synthetics (which have no single true
  volume figure anyway), but still a proxy. It's also only fetched for
  5m/15m/1H to keep the extra API round-trip reasonable.
- Rounding Top/Bottom and Triangle detection use reasonable geometric
  heuristics (quadratic curve fit; trendline slope), not something
  with as precise a source-material definition as the harmonic ratios
  -- expect these two to need more real-world tuning than the rest,
  which is why they're labeled "experimental" in alerts and have their
  own calibratable thresholds.
- This is alerting and tracking only. No orders are ever placed.
