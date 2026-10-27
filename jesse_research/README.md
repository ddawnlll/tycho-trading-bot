# jesse_momentum_v2: Jesse momentum research (simulation only)

**NO_ECONOMIC_CLAIM.** Jesse 3.5.3 is the research engine here. Tycho/NautilusTrader in the parent
repository remains the execution system and was not modified: its dependencies, frozen protocol
(`research/protocol.json`), momentum v1 ledger and paper safety controls are unchanged. No exchange
credentials, orders, deployment or expert activation are involved.

- `preregistration.json`: hypotheses, grids, data split, costs, criteria and budget, committed before
  any market-data backtest (amendments are listed in the file, each made before results).
- `trials.jsonl`: append-only, hash-chained ledger of every trial (verified by `mv2_report.py`).
- `REPORT.md` / `results/report.json`: final report (all trials, net metrics, drawdowns, trades, failures).
- `failures.json`: incidents and data problems.

## Reproduce

Requires Docker and about 3 GB of disk for data.

```bash
docker pull salehmir/jesse:3.5.3          # digest in results/environment.json
NET=bridge ./run.sh research_scripts/fetch_data.py /data   # public archives only; needs data/exchange_info_20261026.json
./run.sh research_scripts/test_mv2.py      # synthetic self-checks
./run.sh research_scripts/mv2_run.py dev   # Jesse backtests, cached per run in results/cache (resumable)
./run.sh research_scripts/mv2_run.py analyze
./run.sh research_scripts/mv2_run.py ml
./run.sh research_scripts/mv2_run.py holdout   # refuses a second read
./run.sh research_scripts/mv2_report.py && cp results/REPORT.md REPORT.md
```

`run.sh` runs each script in the pinned Jesse image with no network (bridge only for the fetcher),
a non-root user, a 1.5 GB memory cap and no host environment or home directory. Only this directory
is mounted. The exchangeInfo snapshot that defines the non-crypto exclusion was fetched from
`https://fapi.binance.com/fapi/v1/exchangeInfo` on 2026-10-26; its sha256 is in the preregistration.
Re-running `fetch_data.py` today would request the same archive months (cut off at 2026-09).

## Jesse dashboard and MCP (optional)

```bash
docker network create jesse-net
docker run -d --name jesse-postgres --network jesse-net --memory 256m -e POSTGRES_USER=jesse_user \
  -e POSTGRES_PASSWORD=<pg password> -e POSTGRES_DB=jesse_db -v jesse-pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:2.19.3-pg17
docker run -d --name jesse-dashboard --network jesse-net --memory 900m -p 127.0.0.1:9000:9000 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/project:/home" -w /home salehmir/jesse:3.5.3 jesse run
claude mcp add --transport http jesse-mcp http://127.0.0.1:9000/mcp --header "Authorization: Bearer <sha256(PASSWORD)>"
```

`project/.env` (gitignored) holds the random dashboard and Postgres passwords; no exchange keys
and no `LICENSE_API_TOKEN`. The live plugin is not installed (`jesse install-live` is never run),
so MCP live-session tools return `live_plugin_unavailable`. `.claude/settings.json` additionally
denies the MCP key, credential, live-session and config tools.
