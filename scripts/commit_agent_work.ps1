# Commit the agent + index-branch work to a new branch and push it, without touching anyone else's changes.
# Run from the repo folder in PowerShell:   .\scripts\commit_agent_work.ps1
# It only adds the files listed below. Your other uncommitted edits stay uncommitted, exactly as they are now.
# It never adds .env (your keys stay on your laptop).

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Branch = "agent-index-branch"

$Files = @(
    "src/agent/__init__.py", "src/agent/evidence.py", "src/agent/llm.py", "src/agent/screen.py",
    "src/agent/trade_desk.py", "src/agent/valuation.py",
    "src/backtest/overfit.py",
    "src/ingest/lseg_fundamentals.py", "src/ingest/lseg_bulk.py",
    "scripts/__init__.py", "scripts/agent_diag.py", "scripts/daily.py", "scripts/download_fundamentals.py",
    "scripts/download_market_context.py", "scripts/download_spy_history.py", "scripts/index_branch.py",
    "scripts/index_signal.py", "scripts/live_scan.py", "scripts/run_agent.py", "scripts/score_signals.py",
    "scripts/spy_backtest_quotes.py", "scripts/spy_benchmark.py", "scripts/trade_desk.py", "scripts/fill_sensitivity.py",
    "scripts/commit_agent_work.ps1",
    "tests/test_agent.py", "tests/test_research.py",
    "docs/code_reference/AG_agent.md",
    "strategy/preregistered_spy.md", "strategy/preregistered_index_branch.md",
    "strategy/preregistered_index_branch_small.md"
)
# backtest results live under data/, which .gitignore skips, so they are added with -f (results only, no raw data)
$Results = "data/results/index_branch"

# 1. stop if a file is missing
$missing = $Files | Where-Object { -not (Test-Path $_) }
if ($missing) { Write-Host "Missing files, nothing committed:`n  $($missing -join "`n  ")" -ForegroundColor Red; exit 1 }

# 2. tests must pass before anything is committed
python -m pytest tests/test_agent.py tests/test_research.py -q
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed, nothing committed." -ForegroundColor Red; exit 1 }

# 3. new branch from where you are now (your uncommitted edits come along, untouched)
$current = git branch --show-current
if ($current -ne $Branch) {
    git show-ref --verify --quiet "refs/heads/$Branch"
    if ($LASTEXITCODE -eq 0) { git switch $Branch } else { git switch -c $Branch }
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not switch to $Branch." -ForegroundColor Red; exit 1 }
}

# 4. add only our files
git add -- $Files
git add -f -- $Results
Write-Host "`nAbout to commit these files:" -ForegroundColor Cyan
git diff --cached --name-status
$ok = Read-Host "`nCommit and push them to origin/$Branch? (y/n)"
if ($ok -ne "y") { git reset -q -- $Files $Results; Write-Host "Cancelled. Files unstaged, nothing committed (you are on branch $Branch)."; exit 0 }

# 5. commit and push
git commit -m "Add AI agent (buy + sell plans) and tested index branch with capital-based sizing" `
           -m "Agent: shock screen with point-in-time news, valuation gate and entry/exit plans (src/agent, scripts/run_agent, live_scan, trade_desk, daily). Index branch: pre-registered walk-forward backtest 2010-2026 on SPY options (scripts/index_branch.py, strategy/preregistered_index_branch*.md), daily signal (scripts/index_signal.py), capital test (--stage capital). Results in data/results/index_branch." `
           -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_01BUJgskgE8PYZC8TithhRpg"
if ($LASTEXITCODE -ne 0) { Write-Host "Commit failed." -ForegroundColor Red; exit 1 }
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) { Write-Host "Push failed (check your GitHub sign-in). The commit is saved locally." -ForegroundColor Red; exit 1 }

Write-Host "`nDone. Open the pull request here:" -ForegroundColor Green
Write-Host "https://github.com/AggieInvestmentClub/sellingputsQuantFund/compare/main...$($Branch)?expand=1"
