# How we write and document code

Read this once after you pull. Every function in this repo has an **ID tag**, so anyone
can find the code, its explanation and its spec rule with one search.

## 1. Find anything: Ctrl+Shift+F

In VS Code, **Ctrl+Shift+F** searches every file in the repo. Search an ID such as
`STR-2.1c` and you'll see:

- the function in the code (its docstring starts with the tag)
- its entry in `docs/code_reference/` (what it does, inputs, outputs, example)
- its rule in `strategy/universe_and_parameters.txt` (the `Code:` lines), for strategy functions

Search just the prefix (`STR-2.1`, `ING-`) to see a whole group.

## 2. The ID system

| Prefix | Folder | What lives there | Example |
|---|---|---|---|
| `ING-nn` | `src/ingest/` | downloading from LSEG | `ING-06` fetch_headlines |
| `DAT-nn` | `src/data/` | cleaning raw data into what the strategy expects | (planned) |
| `OPT-nn` | `src/options/` | option RICs, Black-Scholes / Greeks | (planned) |
| `STR-x.xa` | `strategy/strategy_logic.py` | the strategy rules | `STR-2.1c` is_shock |
| `EVT-nn` | `src/events/` | building the shock event set | (planned) |
| `BT-nn` | `src/backtest/` | simulating trades, fills, metrics | (planned) |
| `LIVE-nn` | `src/live/` | paper / live trading | (planned) |

- **STR IDs use the spec section number** from `strategy/universe_and_parameters.txt`, plus a letter
  per function: `STR-2.1a`, `STR-2.1b`, `STR-2.1c` are the three functions for spec 2.1.
- **All other prefixes count up** in the order functions are written: `ING-01`, `ING-02`, ...
- IDs are never reused. If a function is deleted, its ID is retired.
- To find the next free number, check the index in `docs/code_reference/README.md`.

## 3. When you add or change a function

Do all three **in the same commit**:

1. **Tag it in the code.** The docstring's first line starts with the ID and says what it does:
   ```python
   def fetch_prices(ric, start, end, interval="daily"):
       """[ING-02] daily price bars for one RIC, starting 1 day early so the first return can be computed"""
   ```
2. **Document it** in the matching file in `docs/code_reference/` (`ING_ingest.md`,
   `STR_strategy.md`, ...). Copy the template at the bottom of
   `docs/code_reference/README.md`, and add a row to the index table there.
3. **If it's a strategy rule**, add or update the `Code:` line under its spec section in
   `strategy/universe_and_parameters.txt`.

Planned functions that aren't written yet get a `# TODO [ID] name: ...` comment in the code
and an entry marked `**Status:** ⬜ planned` in the docs.

## 4. Check before you push

```
.venv/Scripts/python.exe -m pytest tests/test_docs.py
```

It fails if a function has an ID but no docs entry, if the docs list an ID that no longer
exists in the code, or if two functions share an ID. Fix what it prints, then push.

## 5. Other team rules

- **Keys stay local.** Put them in `.env` (copy `.env.example`). Never commit `.env` or
  `lseg-data.config.json`; a deleted file still lives in git history.
- **Pull before you push:** `git pull --rebase`, then `git push`.
- **Run scripts from the repo root**, e.g. `python scripts/smoke_test.py`.
- **Returns are decimals** (`0.012` = 1.2%). Option prices and credits are **per share**
  (x100 for one contract).
