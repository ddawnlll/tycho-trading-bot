# jesse_momentum_v2: Jesse momentum research report

**NO_ECONOMIC_CLAIM. Research and simulation only.** Jesse is the research engine; Tycho/NautilusTrader is untouched and remains the execution system. Nothing was activated, deployed or connected to an exchange account.

## Bottom line

No momentum candidate survived. The three preregistered rules (H1 pullback, H2 breakout, H3 relative strength) were rejected in walk-forward on Binance USDT-M perpetuals 2021-H1..2026-H2 (H2 narrowly: every check passed except the block bootstrap, whose 5th percentile was -2.7e-7/day). The logistic-regression filter on H2 passed every walk-forward criterion on the 9 gated chunks (Sharpe 0.82, CAGR 2.0%, DD 2.5%, null p 0.001), but that is in-sample-inspected data and one of five candidates; on the untouched holdout it lost money (return -0.1%, Sharpe -0.03, 57 trades) and was rejected. The holdout's power is low (3 half-years, 3-31 coins, spot 2018-19), so this is not proof that the H2+LR effect is absent either. Nothing is admitted.

## Verdicts

| hypothesis | walk-forward verdict | OOS CAGR | Sharpe | max DD | trades | 3x-cost CAGR | null p | ML | holdout |
|---|---|---|---|---|---|---|---|---|---|
| H1 rule | **REJECT** | -0.2% | -0.18 | 3.6% | 695 | -0.6% | 0.887 | REJECT | not read (rule rejected) |
| H1 + LR filter | **REJECT** | 0.5% | 0.39 | 2.5% | 237 | 0.2% | n/a | (is the ML variant) | not read |
| H2 rule | **REJECT** | 1.9% | 0.63 | 4.4% | 759 | 1.6% | 0.016 | PASS_WALK_FORWARD_ML | not read (rule rejected) |
| H2 + LR filter | **PASS_WALK_FORWARD_ML** | 2.0% | 0.82 | 2.5% | 386 | 1.8% | 0.001 | (is the ML variant) | **REJECT** |
| H3 rule | **REJECT** | -0.1% | 0.12 | 44.7% | 789 | -3.7% | 0.133 | EXCLUDED_BY_PREREGISTRATION | not read (rule rejected) |

LR rows cover only the walk-forward chunks whose training fold passed the independence gate (see the ML section), compared with the rule on the same chunks.

Holdout: read once for H2+LR

## Setup and isolation

- jesse_image: salehmir/jesse@sha256:4c48ffdfa26f19de53c6c832a7e4bccd7f8e698e8fc59aded24aa99f03f2c092
- jesse_version: 3.5.3 (Python 3.11.17 inside the image)
- postgres_image_for_dashboard: timescale/timescaledb@sha256:4c16bc1477e87128710e9846fd3f41aa961fce94a6f5d24c4825947ff706b8af
- research_isolation: ./run.sh: docker --network none (bridge only for fetch_data.py), non-root uid, 1.5 GB memory cap, no host env/home; mounts only jesse_research/ (code read-only), data/, results/, trials.jsonl
- mcp: Jesse built-in MCP (Streamable HTTP) at http://127.0.0.1:9000/mcp, registered in Claude Code as 'jesse-mcp' (local scope, status Connected); tools/list returned 104 tools; list_project_files verified; get_live_sessions returned live_plugin_unavailable
- mcp_safety: dashboard container started with 'jesse run' only (no 'jesse install-live'), LICENSE_API_TOKEN empty, no exchange keys; port bound to 127.0.0.1; .claude/settings.json denies the MCP key/credential/live-session/config tools; the dashboard was stopped during batch runs to free memory
- tycho_untouched: no file under tycho/, scripts/, tests/, research/, pyproject.toml or the paper bot changed; work lives in jesse_research/ on branch research/jesse-momentum-v2
- host: Debian 13, 2 vCPU, 7.6 GB RAM (about 1 GB free), Docker 26.1.5

## Data

- Source: https://data.binance.vision public monthly archives, sha256 CHECKSUM verified per file; hard cutoff month 2026-09 (Tycho frozen window 2026-10-26 20:00 UTC onward never read).
- Symbols kept: {'perp': 680, 'spot': 694}; archived symbols whose data ends before the period end (delisted/renamed): {'perp': 29, 'spot': 4}.
- 1m files: 1405; absent monthly files: ['1m/perp/LUNAUSDT/2022-06', '1m/spot/BCCUSDT/2018-12', '1m/spot/BCHABCUSDT/2019-12']; missing minutes inside files (flat-filled): 41097 (all on spot; perp files had none); OHLC anomalies: 0; absent funding files: ['funding/LUNAUSDT/2022-06'].
- Survivorship: the S3 archive listing includes delisted symbols and membership is fixed point-in-time at each half-year, so delisted coins (e.g. LUNA 2022-H1) are traded until their last archived minute. Residual biases: archives start 2020-01 (2020-H1 empty), symbols absent from the 2026-10-26 exchangeInfo snapshot are assumed crypto, and positions in a delisted coin close at the last archived 1m close rather than Binance's settlement price.

Universe (top 15 by prior 60-day median daily quote volume):

- perp 2020-H1 (eligible 0): (none)
- perp 2020-H2 (eligible 24): BTCUSDT ETHUSDT BCHUSDT LINKUSDT ADAUSDT XTZUSDT EOSUSDT XRPUSDT LTCUSDT ETCUSDT VETUSDT BNBUSDT TRXUSDT XLMUSDT IOSTUSDT
- perp 2021-H1 (eligible 69): BTCUSDT ETHUSDT XRPUSDT LTCUSDT LINKUSDT YFIUSDT BCHUSDT SUSHIUSDT EOSUSDT ADAUSDT XLMUSDT BNBUSDT UNIUSDT AAVEUSDT YFIIUSDT
- perp 2021-H2 (eligible 110): BTCUSDT ETHUSDT DOGEUSDT XRPUSDT BNBUSDT ADAUSDT MATICUSDT ETCUSDT EOSUSDT LTCUSDT DOTUSDT LINKUSDT SOLUSDT BCHUSDT FILUSDT
- perp 2022-H1 (eligible 130): BTCUSDT ETHUSDT SANDUSDT 1000SHIBUSDT SOLUSDT MANAUSDT LUNAUSDT GALAUSDT AVAXUSDT BNBUSDT MATICUSDT XRPUSDT DOTUSDT ADAUSDT DOGEUSDT
- perp 2022-H2 (eligible 139): BTCUSDT ETHUSDT GMTUSDT SOLUSDT AVAXUSDT APEUSDT ADAUSDT BNBUSDT FTMUSDT NEARUSDT WAVESUSDT XRPUSDT TRXUSDT DOTUSDT SANDUSDT
- perp 2023-H1 (eligible 150): BTCUSDT ETHUSDT DOGEUSDT BNBUSDT XRPUSDT LTCUSDT SOLUSDT MATICUSDT MASKUSDT LINKUSDT CHZUSDT APEUSDT ETCUSDT DYDXUSDT ADAUSDT
- perp 2023-H2 (eligible 181): BTCUSDT ETHUSDT XRPUSDT SOLUSDT LTCUSDT ARBUSDT BNBUSDT MATICUSDT TOMOUSDT CFXUSDT LINAUSDT DOGEUSDT OPUSDT ADAUSDT INJUSDT
- perp 2024-H1 (eligible 223): BTCUSDT ETHUSDT SOLUSDT AVAXUSDT XRPUSDT DOGEUSDT LINKUSDT TIAUSDT MATICUSDT BNBUSDT TRBUSDT ADAUSDT ARBUSDT OPUSDT DOTUSDT
- perp 2024-H2 (eligible 270): BTCUSDT ETHUSDT SOLUSDT 1000PEPEUSDT WIFUSDT DOGEUSDT PEOPLEUSDT BNBUSDT ORDIUSDT XRPUSDT WLDUSDT 1000SHIBUSDT ENAUSDT BOMEUSDT 1000BONKUSDT
- perp 2025-H1 (eligible 318): BTCUSDT ETHUSDT DOGEUSDT SOLUSDT XRPUSDT 1000PEPEUSDT SUIUSDT ADAUSDT BNBUSDT WIFUSDT NEIROUSDT 1000SHIBUSDT AVAXUSDT LINKUSDT WLDUSDT
- perp 2025-H2 (eligible 440): BTCUSDT ETHUSDT SOLUSDT 1000PEPEUSDT XRPUSDT DOGEUSDT SUIUSDT FARTCOINUSDT TRUMPUSDT ADAUSDT WIFUSDT ENAUSDT BNBUSDT AAVEUSDT VIRTUALUSDT
- perp 2026-H1 (eligible 578): ETHUSDT BTCUSDT SOLUSDT ZECUSDT XRPUSDT DOGEUSDT BNBUSDT SUIUSDT HYPEUSDT ASTERUSDT 1000PEPEUSDT PIPPINUSDT ADAUSDT LINKUSDT AVAXUSDT
- perp 2026-H2 (eligible 634): BTCUSDT ETHUSDT SOLUSDT ZECUSDT HYPEUSDT XRPUSDT DOGEUSDT LABUSDT BNBUSDT WLDUSDT SUIUSDT NEARUSDT 1000PEPEUSDT ADAUSDT TAOUSDT
- spot 2018-H1 (eligible 0): (none)
- spot 2018-H2 (eligible 3): BCCUSDT NEOUSDT QTUMUSDT
- spot 2019-H1 (eligible 10): EOSUSDT XLMUSDT NEOUSDT ETCUSDT ONTUSDT VETUSDT ICXUSDT IOTAUSDT QTUMUSDT NULSUSDT
- spot 2019-H2 (eligible 31): BCHABCUSDT EOSUSDT MATICUSDT BTTUSDT FETUSDT NEOUSDT CELRUSDT ETCUSDT ATOMUSDT ONTUSDT XLMUSDT IOTAUSDT VETUSDT ICXUSDT ZILUSDT

## Baselines (stitched OOS 2021-H1..2026-H2)

| baseline | CAGR | Sharpe | max DD | by year |
|---|---|---|---|---|
| BTC_buy_hold | 4.0% | 0.36 | 82.7% | 2021: -0.2%, 2022: -66.3%, 2023: 137.6%, 2024: 93.0%, 2025: -12.0%, 2026: -7.8% |
| equal_weight_buy_hold | -7.5% | 0.35 | 92.5% | 2021: 136.5%, 2022: -80.8%, 2023: 127.6%, 2024: 17.0%, 2025: -56.7%, 2026: 21.5% |

## Every grid cell at fixed parameters (same OOS period; base / 3x cost / delayed entry)

| cell | CAGR | Sharpe | max DD | trades | win rate | funding USD | 3x CAGR | delay CAGR |
|---|---|---|---|---|---|---|---|---|
| H1:rsi_entry35_stop_atr2.0 | -0.5% | -0.29 | 5.5% | 584 | 25.9% | 902 | -0.9% | -0.4% |
| H1:rsi_entry35_stop_atr3.0 | -0.2% | -0.21 | 3.4% | 504 | 29.8% | 604 | -0.5% | -0.1% |
| H1:rsi_entry45_stop_atr2.0 | -1.2% | -0.34 | 11.7% | 1756 | 26.8% | 3253 | -2.5% | -1.2% |
| H1:rsi_entry45_stop_atr3.0 | -0.7% | -0.30 | 7.5% | 1525 | 30.5% | 2160 | -1.5% | -0.7% |
| H2:n30_k3.0 | 1.2% | 0.32 | 6.7% | 1446 | 31.9% | 4191 | 0.3% | 1.1% |
| H2:n30_k5.0 | 2.5% | 0.68 | 4.7% | 1019 | 34.0% | 4979 | 2.1% | 2.3% |
| H2:n90_k3.0 | 0.8% | 0.28 | 3.9% | 1011 | 32.1% | 3056 | 0.2% | 0.8% |
| H2:n90_k5.0 | 2.0% | 0.67 | 3.3% | 722 | 36.1% | 3829 | 1.7% | 1.8% |
| H3:L180_k3 | -1.5% | 0.03 | 34.1% | 719 | 49.9% | -213 | -4.7% | -2.7% |
| H3:L180_k5 | -4.6% | -0.08 | 40.0% | 990 | 48.9% | -1621 | -8.7% | -8.2% |
| H3:L300_k3 | 4.6% | 0.32 | 35.6% | 612 | 47.7% | 108 | 1.8% | 2.7% |
| H3:L300_k5 | 9.3% | 0.48 | 37.3% | 772 | 47.5% | 10391 | 5.6% | 9.1% |

## Walk-forward detail

### H1: REJECT

Selected cell per test half-year (expanding window, train Sharpe / trades): 2021-H1: {'rsi_entry': 35, 'stop_atr': 3.0} (0.54/68); 2021-H2: {'rsi_entry': 35, 'stop_atr': 3.0} (0.49/123); 2022-H1: {'rsi_entry': 35, 'stop_atr': 3.0} (0.38/174); 2022-H2: {'rsi_entry': 35, 'stop_atr': 3.0} (0.05/191); 2023-H1: {'rsi_entry': 45, 'stop_atr': 3.0} (-0.13/670); 2023-H2: {'rsi_entry': 45, 'stop_atr': 3.0} (-0.16/822); 2024-H1: {'rsi_entry': 35, 'stop_atr': 3.0} (0.00/324); 2024-H2: {'rsi_entry': 35, 'stop_atr': 3.0} (-0.10/380); 2025-H1: {'rsi_entry': 35, 'stop_atr': 3.0} (-0.16/441); 2025-H2: {'rsi_entry': 35, 'stop_atr': 3.0} (-0.21/476); 2026-H1: {'rsi_entry': 35, 'stop_atr': 3.0} (-0.20/511); 2026-H2: {'rsi_entry': 35, 'stop_atr': 3.0} (-0.25/542)

- OOS net: CAGR -0.2%, total -1.4%, Sharpe -0.18, max DD 3.6%, trades 695, win rate 31.1%, fees 1655 USD, funding 778 USD (portfolio of 15 x 10,000 USD slots)
- By year: 2021: 0.3%, 2022: -1.4%, 2023: 0.4%, 2024: -0.9%, 2025: -0.4%, 2026: 0.6%
- 3x cost CAGR -0.6%; delayed entry CAGR -0.1%; without best symbol (UNIUSDT) CAGR -0.3%; positive cells 0/4
- Block bootstrap 5th pct of mean daily return: -0.000030
- Null: {"observed_mean_trade_return": 0.00284504091632558, "null_mean": 0.007412472477321354, "null_q95": 0.013698814155597868, "p": 0.8871128871128872, "trades": 695}
- Checks: net_cagr_gt_0 FAIL, sharpe_ge_0.5 FAIL, max_dd_le_0.30 PASS, trades_ge_100 PASS, stress_3x_cagr_gt_0 FAIL, bootstrap_q05_gt_0 FAIL, null_p_le_0.0167 FAIL, cells_positive_ge_3of4 FAIL, delay_cagr_gt_0 FAIL, drop_best_symbol_cagr_gt_0 FAIL, half_years_nonneg PASS

### H2: REJECT

Selected cell per test half-year (expanding window, train Sharpe / trades): 2021-H1: {'n': 90, 'k': 5.0} (0.77/86); 2021-H2: {'n': 30, 'k': 5.0} (0.90/233); 2022-H1: {'n': 30, 'k': 5.0} (0.93/321); 2022-H2: {'n': 90, 'k': 5.0} (0.76/269); 2023-H1: {'n': 90, 'k': 5.0} (0.54/322); 2023-H2: {'n': 90, 'k': 5.0} (0.61/388); 2024-H1: {'n': 90, 'k': 5.0} (0.74/461); 2024-H2: {'n': 90, 'k': 5.0} (0.72/515); 2025-H1: {'n': 90, 'k': 5.0} (0.75/599); 2025-H2: {'n': 90, 'k': 5.0} (0.69/652); 2026-H1: {'n': 90, 'k': 5.0} (0.69/705); 2026-H2: {'n': 90, 'k': 5.0} (0.62/765)

- OOS net: CAGR 1.9%, total 11.5%, Sharpe 0.63, max DD 4.4%, trades 759, win rate 35.4%, fees 1256 USD, funding 3934 USD (portfolio of 15 x 10,000 USD slots)
- By year: 2021: 4.1%, 2022: -2.1%, 2023: 4.2%, 2024: 2.4%, 2025: 0.9%, 2026: 1.6%
- 3x cost CAGR 1.6%; delayed entry CAGR 1.7%; without best symbol (BNBUSDT) CAGR 1.4%; positive cells 4/4
- Block bootstrap 5th pct of mean daily return: -0.000000
- Null: {"observed_mean_trade_return": 0.03232767130041466, "null_mean": 0.018339264543790153, "null_q95": 0.028916566379263396, "p": 0.015984015984015984, "trades": 759}
- Checks: net_cagr_gt_0 PASS, sharpe_ge_0.5 PASS, max_dd_le_0.30 PASS, trades_ge_100 PASS, stress_3x_cagr_gt_0 PASS, bootstrap_q05_gt_0 FAIL, null_p_le_0.0167 PASS, cells_positive_ge_3of4 PASS, delay_cagr_gt_0 PASS, drop_best_symbol_cagr_gt_0 PASS, half_years_nonneg PASS

### H3: REJECT

Selected cell per test half-year (expanding window, train Sharpe / trades): 2021-H1: {'L': 180, 'k': 5} (1.41/85); 2021-H2: {'L': 300, 'k': 5} (1.11/128); 2022-H1: {'L': 300, 'k': 5} (1.02/196); 2022-H2: {'L': 300, 'k': 5} (0.68/264); 2023-H1: {'L': 300, 'k': 5} (0.56/333); 2023-H2: {'L': 300, 'k': 5} (0.52/391); 2024-H1: {'L': 300, 'k': 5} (0.72/461); 2024-H2: {'L': 300, 'k': 5} (0.76/526); 2025-H1: {'L': 300, 'k': 5} (0.73/605); 2025-H2: {'L': 300, 'k': 5} (0.67/667); 2026-H1: {'L': 300, 'k': 5} (0.53/736); 2026-H2: {'L': 300, 'k': 5} (0.45/802)

- OOS net: CAGR -0.1%, total -0.6%, Sharpe 0.12, max DD 44.7%, trades 789, win rate 46.6%, fees 15084 USD, funding -2702 USD (portfolio of 15 x 10,000 USD slots)
- By year: 2021: -3.2%, 2022: -19.0%, 2023: 30.3%, 2024: 15.7%, 2025: -15.4%, 2026: -0.7%
- 3x cost CAGR -3.7%; delayed entry CAGR -3.4%; without best symbol (INJUSDT) CAGR -2.9%; positive cells 3/4
- Block bootstrap 5th pct of mean daily return: -0.000347
- Null: {"observed_weekly_sharpe": 0.22111293845569252, "null_mean": -0.2532683041164797, "null_q95": 0.4587042365409753, "p": 0.13286713286713286, "weeks": 300}
- Checks: net_cagr_gt_0 FAIL, sharpe_ge_0.5 FAIL, max_dd_le_0.30 FAIL, trades_ge_100 PASS, stress_3x_cagr_gt_0 FAIL, bootstrap_q05_gt_0 FAIL, null_p_le_0.0167 FAIL, cells_positive_ge_3of4 PASS, delay_cagr_gt_0 FAIL, drop_best_symbol_cagr_gt_0 FAIL, half_years_nonneg FAIL

## Logistic regression vs. non-ML rule

### H1: REJECT

| test chunk | train trades | distinct entry days | minority | gate |
|---|---|---|---|---|
| 2021-H1 | 61 | 31 | 29 | no |
| 2021-H2 | 123 | 53 | 50 | no |
| 2022-H1 | 174 | 73 | 69 | no |
| 2022-H2 | 191 | 80 | 69 | no |
| 2023-H1 | 670 | 262 | 210 | yes |
| 2023-H2 | 819 | 311 | 259 | yes |
| 2024-H1 | 319 | 142 | 105 | no |
| 2024-H2 | 380 | 172 | 125 | yes |
| 2025-H1 | 438 | 195 | 140 | yes |
| 2025-H2 | 476 | 210 | 149 | yes |
| 2026-H1 | 511 | 231 | 158 | yes |
| 2026-H2 | 542 | 250 | 164 | yes |

On the 7 gated chunks: LR filter CAGR 0.5%, Sharpe 0.39, DD 2.5%, trades 237 vs rule CAGR 0.0%, Sharpe 0.04, DD 3.5%, trades 491. 3x-cost LR CAGR 0.2%. Checks: {'net_cagr_gt_0': True, 'sharpe_ge_0.5': False, 'max_dd_le_0.30': True, 'trades_ge_100': True, 'stress_3x_cagr_gt_0': True, 'bootstrap_q05_gt_0': False, 'beats_rule_sharpe': True}. random-entry null and bounded robustness were not re-run for the ML variant; a PASS here would still need them before any holdout use

### H2: PASS_WALK_FORWARD_ML

| test chunk | train trades | distinct entry days | minority | gate |
|---|---|---|---|---|
| 2021-H1 | 76 | 45 | 28 | no |
| 2021-H2 | 232 | 119 | 82 | no |
| 2022-H1 | 315 | 168 | 115 | yes |
| 2022-H2 | 267 | 162 | 101 | no |
| 2023-H1 | 322 | 194 | 113 | yes |
| 2023-H2 | 381 | 229 | 135 | yes |
| 2024-H1 | 452 | 266 | 164 | yes |
| 2024-H2 | 515 | 310 | 190 | yes |
| 2025-H1 | 599 | 348 | 217 | yes |
| 2025-H2 | 652 | 369 | 234 | yes |
| 2026-H1 | 705 | 395 | 253 | yes |
| 2026-H2 | 765 | 429 | 272 | yes |

On the 9 gated chunks: LR filter CAGR 2.0%, Sharpe 0.82, DD 2.5%, trades 386 vs rule CAGR 1.9%, Sharpe 0.67, DD 3.3%, trades 535. 3x-cost LR CAGR 1.8%. Checks: {'net_cagr_gt_0': True, 'sharpe_ge_0.5': True, 'max_dd_le_0.30': True, 'trades_ge_100': True, 'stress_3x_cagr_gt_0': True, 'bootstrap_q05_gt_0': True, 'beats_rule_sharpe': True}. all preregistered walk-forward criteria evaluated for the ML variant (stage mlcheck)

Full walk-forward criteria for the LR variant: null {"observed_mean_trade_return": 0.03560043207423162, "null_mean": 0.009185047544408149, "null_q95": 0.020880527834280693, "p": 0.000999000999000999, "trades": 386}; fixed cells with their own fold models n30_k3.0: Sharpe 0.68 on 9 chunks, n30_k5.0: Sharpe 0.74 on 9 chunks, n90_k3.0: Sharpe 0.53 on 9 chunks, n90_k5.0: Sharpe 0.94 on 8 chunks; delayed entry CAGR 1.9%; without XRPUSDT CAGR 1.5%; checks {'net_cagr_gt_0': True, 'sharpe_ge_0.5': True, 'max_dd_le_0.30': True, 'trades_ge_100': True, 'stress_3x_cagr_gt_0': True, 'bootstrap_q05_gt_0': True, 'beats_rule_sharpe': True, 'null_p_le_0.0167': True, 'cells_positive_ge_3of4': True, 'delay_cagr_gt_0': True, 'drop_best_symbol_cagr_gt_0': True, 'half_years_nonneg': True} -> **PASS_WALK_FORWARD_ML**

### H3: EXCLUDED_BY_PREREGISTRATION


## Holdout

Read exactly once (Binance spot USDT pairs outside Tycho holdout B, 2018-H2..2019-H2, 2018-H1 had no eligible coin; perp-like costs, assumed 0.01%/8h funding). It is now INSPECTED and cannot serve as an untouched holdout again.

- H2+LR: LR refit on 808 dev trades of the frozen cell before the read; coefficients [-0.085, -0.337, -0.009, 0.074, -0.102], threshold 0.361
- H2+LR {'n': 90, 'k': 5.0}: return -0.1%, Sharpe -0.03, DD 1.3%, trades 57, 3x cost return -0.3%: REJECT {'net_return_gt_0': False, 'sharpe_ge_0.3': False, 'max_dd_le_0.30': True, 'stress_return_gt_0': False, 'trades_ge_30': True}

## Trial ledger

`trials.jsonl`: 23 rows, hash chain verified: preregistration, both baselines, every grid cell with its base/3x-cost/delay variants, each walk-forward selection, the ML comparisons, the ML full checks, the holdout parameter freeze and the holdout read. Every individual Jesse backtest (8276 runs: market, chunk, symbol, hypothesis, parameters, variant, trades, net PnL, fees, funding) is in `results/jesse_runs.csv` and `results/report.json`.

## Data status after this study

| dataset | status |
|---|---|
| Binance USDT-M perps 2020-01..2026-09 | INSPECTED (Tycho v1 and this study's walk-forward) |
| Binance spot 2017-08..2019-12, Tycho holdout-B coins | INSPECTED (Tycho v1) |
| Binance spot 2018-H2..2019-H2, other coins (this holdout) | INSPECTED since this study (read once for H2+LR) |
| Binance perps from 2026-10-26 20:00 UTC (Tycho frozen window) | UNTOUCHED (never requested) |

## Failures and incidents

- fetch run 1 stopped by the author after universe membership showed TradFi perps (XAU, XAG, CL, SNDK, MU); amendment recorded, fetch re-run (cached files reused). 1m files of the excluded symbols were downloaded but are never loaded.
- dev run 1 crashed at 2021-H1 XRPUSDT: an H3 1x short slot reached negative equity and Jesse rejected the next order (InsufficientMargin, value -1926 USD). Fix committed before resuming: a slot with equity <= 0 opens nothing new (its loss stays in the portfolio). Runs cached before the crash were unaffected by the change.
- Data: 1m archives absent for LUNAUSDT 2022-06 (delisted 2022-05), BCCUSDT 2018-12 (delisted), BCHABCUSDT 2019-12; funding archive absent for LUNAUSDT 2022-06.
- Data: 2020-H1 perp chunk empty (archives start 2020-01, 60-day history rule); spot 2018-H1 empty, 2018-H2 has 3 eligible coins.
- dev run 2 crashed at 2024-H1 TRBUSDT: after the Dec-2023 squeeze close - k*ATR14 was negative, an invalid Jesse stop price. Rule added before any result was inspected: no entry without a placeable protective stop (stop > 0). Earlier cached runs never hit this path.
- ml run 1 crashed (LogisticRegression: NaN input) before any model was fit: offline feature extraction ran ta.ema(300) outside a backtest, where Jesse slices to its default 240 candles -> NaN for every row. Fixed with sequential=True on the 330-bar window (identical values in-backtest); test added.

## Uncertainties

- The author read docs/MOMENTUM_V1_RESULTS.md (Tycho v1) before writing this protocol: long-only alt breakouts degraded on holdout A, trend rules were not distinguishable from noise.
- Binance USDT-M perpetual history 2020-01..2026-09 was used by Tycho v1 (12-coin dev universe and holdout A of 538 other perps). It is INSPECTED data: the walk-forward below is a historical diagnostic, never an untouched holdout.
- Binance spot 2017-08..2019-12 for BTC ETH BNB XRP ADA LTC LINK TRX DOGE was Tycho v1 holdout B: INSPECTED, excluded from this holdout.
- Dev data (perps 2020-H2..2026-H2) was already used by Tycho v1 and read by the author: walk-forward numbers are diagnostics, not clean evidence.
- Fills are Jesse bar fills at 4h closes and stop prices (gap-aware); queue position, market impact and liquidation engine behaviour are not modelled.
- Funding is applied after each Jesse run from actual settlements (spot holdout: assumed 0.01%/8h); sizing inside Jesse does not see funding.
- 15 equal slots rebalanced each half-year; 1x H3 shorts can lose more than a slot's capital, which is kept as portfolio loss.
- Correlated coins: trades and days are not independent; bootstrap blocks of 30 days may be short for multi-week holds.
- Indicators use a fixed 330-bar window (Jesse warm-up slicing), so EMA300 is a truncated EMA.
- Multiplicity: 3 rules plus 2 LR variants were tested; the null threshold 0.0167 was set for 3 hypotheses.
- The LR comparison covers only gated chunks (H2: 2022-H1 and 2023-H1..2026-H2; 2021 and 2022-H2 had too few training trades).
- Holdout power is low and its era/instrument differ (2018-19 spot, assumed funding, 1m gaps flat-filled: 47,225 minutes in the H2+LR runs).
- Absolute returns are small by construction (1% risk per trade per 1/15 slot); Sharpe, not CAGR, is the comparable statistic.

## Next steps

- A hypothesis that failed here is closed; any revision needs a new pre-registration (no re-tuning on these results).
- The only clean evidence left for perpetuals is prospective: after Tycho's frozen window, or a new forward window registered before it starts, evaluated in paper mode with owner approval.
- If a future candidate passes walk-forward, run its holdout once and then confirm it in NautilusTrader with in-venue funding before any paper trial.

Reproduce: see `README.md` in this directory.
