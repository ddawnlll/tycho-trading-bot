# Selective migration

Source inspected: `ddawnlll/tycho`, commit
`0a309c4939dcf75a1bff4e844015bfb35c51d2ec` (2026-10-26).
Destination and only development repository: `ddawnlll/tycho-trading-bot`.

The owner request supersedes the source DEV-only/synthetic-only scope for this
paper bot. Nothing was changed in the source repository. No secrets or protected
holdouts were accessed. This repository is not the V8 contract corpus and does
not inherit its unrelated monograph build system.

Selected ideas and concrete destinations:

| Source | Selected mechanism | Destination |
|---|---|---|
| `src/tycho/runtime.py` | Manifest identity, defensive input copies, reject unavailable snapshots | `tycho/safety.py`, `tycho/experts.py`, `tycho/nautilus_live.py` |
| `src/tycho/trials.py` | Append-only hash chain with integrity verification | `tycho/evidence.py`, `research/trials.jsonl` |
| `src/tycho/gates.py` | INVALID/BLOCKED/FAIL precedence; no automatic activation | `tycho/evidence.py` |
| `src/tycho/stats.py`, `config/tycho.toml` | Explicit uncertainty/multiplicity policy, deterministic research, costs | `tycho/evidence.py`, `research/protocol.json` |
| `src/tycho/tape.py` | Causal availability, data quality, gaps/staleness rejection | `tycho/safety.py` |

Adapted mechanisms are small and local. The source's E-001-specific four-MA
criteria, PENDING configuration corpus and owner holdout attestation are not
pretended to apply to these different experts. DSR/HAC-alpha are not implemented
or claimed. The current fixed-family null and block-bootstrap outputs are
provisional diagnostics and cannot replace independent prospective evidence.

Intentionally not migrated: custom reference ledger/execution, vectorized engine,
scanner orchestration, dispatch boards, hook framework, monograph registry,
synthetic-only data ban, unrelated task factories or meta-learning. NautilusTrader
is the supported bot/research execution engine. Existing legacy engines are
retained as forensic/regression code and do not launch a second bot.

Historical returns in the former README and `runs/` are archived context, never
admission evidence. First priority is the two owner-authorized paper experts;
three additional research candidates remain disabled. No API execution factory
is constructed. No real-money mode can be selected through the supported API.
