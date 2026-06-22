# CME Futures Contract Specifications

This document captures the official CME contract specifications for the 35 products in the ML4T futures universe. Data sourced directly from CME Group website (December 2025).

## Contract Month Patterns

CME uses three main listing patterns:

1. **Monthly** - Contracts for all 12 calendar months (26 consecutive months typical)
2. **Quarterly** - Mar (H), Jun (M), Sep (U), Dec (Z) plus 3 serial months
3. **Bi-monthly** - Specific months based on harvest/delivery cycles

### CME Month Codes
| Code | Month | Code | Month |
|------|-------|------|-------|
| F | January | N | July |
| G | February | Q | August |
| H | March | U | September |
| J | April | V | October |
| K | May | X | November |
| M | June | Z | December |

---

## Equity Index Futures (4 products)

All quarterly: **H, M, U, Z**

| Product | Name | Listed Contracts |
|---------|------|------------------|
| ES | E-mini S&P 500 | Quarterly (H,M,U,Z) for 5+ years |
| NQ | E-mini Nasdaq 100 | Quarterly (H,M,U,Z) for 5+ years |
| YM | E-mini Dow | Quarterly (H,M,U,Z) for 5+ years |
| RTY | E-mini Russell 2000 | Quarterly (H,M,U,Z) for 5+ years |

---

## Treasury Futures (4 products)

All quarterly: **H, M, U, Z**

| Product | Name | Listed Contracts |
|---------|------|------------------|
| ZN | 10-Year T-Note | Quarterly (H,M,U,Z) |
| ZB | 30-Year T-Bond | Quarterly (H,M,U,Z) |
| ZF | 5-Year T-Note | Quarterly (H,M,U,Z) |
| ZT | 2-Year T-Note | Quarterly (H,M,U,Z) |

---

## Energy Futures (4 products)

All monthly: **All 12 months**

| Product | Name | Listed Contracts |
|---------|------|------------------|
| CL | Crude Oil WTI | Monthly for 9+ years |
| NG | Natural Gas | Monthly for 12+ years |
| RB | RBOB Gasoline | Monthly for 3+ years |
| HO | Heating Oil | Monthly for 3+ years |

---

## Metals Futures (5 products)

### Base Metals (3 products) - Monthly

| Product | Name | Listed Contracts | Source |
|---------|------|------------------|--------|
| GC | Gold | "Monthly contracts listed for 26 consecutive months and any Jun and Dec in the nearest 72 months" | CME Verified |
| SI | Silver | "Monthly contracts listed for 26 consecutive months and any Jul and Dec in the nearest 60 months" | CME Verified |
| HG | Copper | "Monthly contracts listed for 24 consecutive months and any Mar, May, Jul, Sep, and Dec in the nearest 63 months" | CME Verified |

### PGM (2 products) - Quarterly

| Product | Name | Listed Contracts | Pattern | Source |
|---------|------|------------------|---------|--------|
| PL | Platinum | "Monthly contracts listed for 3 consecutive months and any Jan, Apr, Jul, and Oct in the nearest 36 months" | F, J, N, V | CME Verified |
| PA | Palladium | "Monthly contracts listed for 3 consecutive months and any Mar, Jun, Sep, Dec in the nearest 36 months" | H, M, U, Z | CME Verified |

**Note**: PL and PA use quarterly patterns, NOT monthly like GC/SI/HG. Near-term monthly contracts have limited liquidity.

---

## Currency Futures (7 products)

### G10 Currencies (6 products)
Quarterly: **H, M, U, Z** (plus 3 serial months for near-term)

| Product | Name | Listed Contracts | Source |
|---------|------|------------------|--------|
| 6E | Euro FX | "Quarterly contracts (Mar, Jun, Sep, Dec) listed for 20 consecutive quarters and serial contracts listed for 3 months" | CME Verified |
| 6J | Japanese Yen | Same pattern as 6E | Inferred |
| 6B | British Pound | Same pattern as 6E | Inferred |
| 6A | Australian Dollar | Same pattern as 6E | Inferred |
| 6C | Canadian Dollar | Same pattern as 6E | Inferred |
| 6S | Swiss Franc | Same pattern as 6E | Inferred |

**Note**: Serial months (non-quarterly) have limited historical data. For backtesting, use quarterly contracts only.

### Emerging Market Currencies (1 product)
Monthly: **All 12 months**

| Product | Name | Listed Contracts | Source |
|---------|------|------------------|--------|
| 6M | Mexican Peso | "Monthly contracts listed for 13 consecutive months and 2 additional quarterly contracts (Mar, Jun, Sep, Dec)" | CME Verified |

---

## Interest Rate Futures (1 product)

| Product | Name | Listed Contracts |
|---------|------|------------------|
| SR3 | Three-Month SOFR | Monthly (all 12 months) - IMM quarterly + serial months |

---

## Agriculture Futures (5 products)

| Product | Name | Contract Months | Pattern |
|---------|------|-----------------|---------|
| ZC | Corn | H, K, N, U, Z | Mar, May, Jul, Sep, Dec |
| ZS | Soybeans | F, H, K, N, Q, U, X | Jan, Mar, May, Jul, Aug, Sep, Nov |
| ZW | Wheat | H, K, N, U, Z | Mar, May, Jul, Sep, Dec |
| ZM | Soybean Meal | F, H, K, N, Q, U, V, Z | Jan, Mar, May, Jul, Aug, Sep, Oct, Dec |
| ZL | Soybean Oil | F, H, K, N, Q, U, V, Z | Jan, Mar, May, Jul, Aug, Sep, Oct, Dec |

---

## Livestock Futures (3 products)

| Product | Name | Contract Months | Pattern |
|---------|------|-----------------|---------|
| LE | Live Cattle | G, J, M, Q, V, Z | Feb, Apr, Jun, Aug, Oct, Dec |
| HE | Lean Hogs | G, J, K, M, N, Q, V, Z | Feb, Apr, May, Jun, Jul, Aug, Oct, Dec |
| GF | Feeder Cattle | F, H, J, K, Q, U, V, X | Jan, Mar, Apr, May, Aug, Sep, Oct, Nov |

---

## Crypto Futures (2 products)

Monthly: **All 12 months**

| Product | Name | Listed Contracts |
|---------|------|------------------|
| BTC | Bitcoin | Monthly for nearest months + quarterly |
| ETH | Ether | Monthly for nearest months + quarterly |

---

## Data Collection Notes

### Source
- Primary: CME Group website contract specs pages
- Tool: Chrome DevTools MCP for JavaScript-rendered content
- Date: December 20, 2025

### Important Distinction
**CME Listed Contracts** vs **Databento Available Data**:
- CME lists what contracts CAN trade
- Databento definition data shows what contracts ARE actively trading
- Far-deferred contracts may be listed but have no trades (no data)
- For historical backtesting, use the pattern that matches liquid contracts

### Serial vs Quarterly Months
For FX futures (6E, 6J, etc.):
- **Quarterly** (H, M, U, Z): Full historical depth, most liquid
- **Serial** (non-quarterly): Only near-term, limited history
- **Recommendation**: Download quarterly only for backtesting

---

## Configuration Summary

For the `individual_contracts.yaml` download config:

| Category | Products | Pattern | Months |
|----------|----------|---------|--------|
| Equity Index | ES, NQ, YM, RTY | Quarterly | H, M, U, Z |
| Treasury | ZN, ZB, ZF, ZT | Quarterly | H, M, U, Z |
| Energy | CL, NG, RB, HO | Monthly | All 12 |
| Base Metals | GC, SI, HG | Monthly | All 12 |
| PGM | PL | Quarterly | F, J, N, V |
| PGM | PA | Quarterly | H, M, U, Z |
| G10 FX | 6E, 6J, 6B, 6A, 6C, 6S | Quarterly | H, M, U, Z |
| EM FX | 6M | Monthly | All 12 |
| Rates | SR3 | Monthly | All 12 |
| Grains | ZC, ZS, ZW | Bi-monthly | Specific |
| Oilseeds | ZM, ZL | Bi-monthly | Specific |
| Livestock | LE, HE, GF | Bi-monthly | Specific |
| Crypto | BTC, ETH | Monthly | All 12 |

---

*Last updated: 2025-12-20*
