# Alpha Thesis — Polymarket Weather (Temperature) Markets

**Status**: v1.0 — baseline thesis for live weather trades.
**Owner**: trading_agent_n1
**Last reviewed**: 2026-04-16

---

## 1. Why this market exists

Polymarket weather markets resolve against a specific station/source (e.g.
NWS La Guardia daily high). Pricing is dominated by:

1. **Casual participants** who quote from consumer apps (Apple Weather,
   Google) that display a single-point forecast with no distribution.
2. **Anchoring on the prior day** ("yesterday was 72°F, so it will be ~72°F").
3. **Stale prices** — Polymarket doesn't reprice every ensemble update, so
   post-12z GFS/ECMWF updates often create a 4–10 hour window of mispricing.

Our edge is running a probabilistic temperature forecast (ensemble + local
station bias correction) and taking the Polymarket quote when it sits
outside the ensemble's inner 60% interval.

## 2. Signal sources

| Source | Mechanism | Horizon |
|---|---|---|
| NWS gridded forecast (hourly update) | Authoritative distribution for the resolution source | 1h – 7d |
| GFS / ECMWF ensemble spread | Uncertainty proxy → sizes position via conformal interval | 6h – 10d |
| Station bias-correction (last 30d residuals) | Persistent local bias vs NWS grid | daily |
| Time-to-resolution decay | Forecast skill collapses < 24h → smaller positions further out | dynamic |

## 3. Why we believe the edge persists

1. **Specialised data moat**: NWS ensemble products are free but require
   domain-specific decoding (GRIB2, station metadata). Retail won't do this.
2. **Low notional per market**: $500–$5k daily volume — below pro desks.
3. **No offsetting hedge**: no efficient weather derivative retail can use
   to arb Polymarket quotes.

**Decay monitor**: Brier score vs NWS rolling > 0.20 → thesis review.

## 4. Kill criteria

- Live win-rate after fees < 52% over 150 trades.
- Calibration ECE > 0.06 (tighter than BTC — we expect weather to be
  better calibrated because the underlying is stationary).
- NWS API unavailable > 2h → automatic freeze.
- Station-specific residual bias shifts > 2°F week-over-week.

## 5. Known failure modes

- **Resolution source ambiguity**: "temperature be between X and Y" markets
  need exact station + time-window match. Filter in
  `weather_temperature_markets.is_weather_temperature_market`.
- **Severe weather / forecast model regime change**: freeze on any NWS
  severe weather alert for the resolution region.
- **DST / timezone edge cases** at midnight resolution.

## 6. Explicit out-of-scope

This thesis does **not** claim alpha on:
- Precipitation / snowfall markets (separate thesis required).
- Long-range climate markets (> 10 days).
- Non-US weather markets (station bias model is US-only).
