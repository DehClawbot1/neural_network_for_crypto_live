# Alpha Thesis — BTC Polymarket Price Markets

**Status**: v1.0 — baseline thesis used to justify every live BTC trade.
**Owner**: trading_agent_n1
**Last reviewed**: 2026-04-16

---

## 1. Why this market exists (counterparty asymmetry)

Polymarket BTC price markets (e.g. "Will BTC be above $X on YYYY-MM-DD?") are
priced by a mixed population of:

1. **Retail sentiment traders** who anchor on round numbers and recent price
   action, often mis-estimating tail probabilities on short-dated binaries.
2. **Reflexive social-media flow** where prices move on headlines before the
   underlying spot market fully repriced the same information.
3. **Liquidity-providing MMs** who quote tight but widen sharply on catalysts
   (FOMC, CPI, options expiry, ETF flow prints).

The bot's edge is NOT predicting BTC direction better than the CME/Binance
order book. It is exploiting **probability-translation errors** — the gap
between the fair binary probability implied by a well-calibrated forecast of
the underlying BTC distribution and the quoted Polymarket probability.

## 2. Signal sources and why each should carry information

| Source | Mechanism | Horizon |
|---|---|---|
| `btc_mtf_forecaster` (15m / 1h / 4h) | Direction + vol forecast → converts to binary prob via cumulative normal over strike distance | 15m – 4h |
| `order_flow_analyzer` (whale_pressure, imbalance) | Large-lot trades precede mean-reverting fills on Polymarket by 30s–5min | < 10 min |
| `technical_analyzer` (trend bias, fractal breakout) | Confluence filter — blocks trades against dominant trend | intraday |
| `BTCSentimentFeatures` | Funding, OI changes, social → regime tilt | 1h – 1d |
| `macro_context` (CPI/FOMC windows) | Event-window vol expansion; size down pre-event | event-based |
| `onchain` (exchange flows, stablecoin mint) | Slower flow signal, mostly daily regime | 1d+ |

Each feature is independently predictive in the training set (see
`research/feature_ic_report.json`). The stage2 temporal model is the
*composer* — its only job is to learn how these signals interact
conditional on time-of-day and regime.

## 3. Why we believe this edge does not arbitrage away

1. **Small notional**: Polymarket BTC market size (low $M / day) is below the
   threshold that high-frequency market makers target.
2. **Settlement friction**: USDC on Polygon + resolution delay deters
   delta-hedgers; retail cannot cheaply arb vs CME.
3. **Funding constraint**: Capital locked until resolution — professional
   funds require IRR that binary events rarely clear.

These conditions can erode. **Decay monitor**: if `p_tp_before_sl` rolling
Brier score over the last 500 live trades exceeds `0.22`, freeze and rerun
the thesis review.

## 4. Kill criteria (when to stop trading BTC)

- Live rolling win-rate (after fees) < 50% over 200 trades.
- Realised Sharpe (annualised) < 0.3 over 90 days.
- Calibration ECE > 0.08 on stage2 outputs.
- Polymarket quoted spread doubles vs 30-day median (liquidity shift).
- Any of the above → stage demotion via `deployment_gate`.

## 5. Known failure modes

- **Resolution ambiguity**: rare but catastrophic. Filter: refuse markets
  whose resolution source is not already in the whitelisted oracle set.
- **Liquidity crunch**: quoted size < `min_notional_for_size`. Handled by
  `risk_service.approve_size`.
- **Model drift during catalysts**: macro_context sets pre-event freeze.

## 6. Explicit out-of-scope

This thesis does **not** claim alpha on:
- Spot BTC execution.
- Options volatility surface.
- Cross-venue basis.

The bot trades Polymarket binaries *only*.
