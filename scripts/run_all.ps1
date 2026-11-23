# One-command reproduction (Windows / PowerShell).
#   .\scripts\run_all.ps1            # real backbone
#   .\scripts\run_all.ps1 mock       # simulated backbone, no weights
param([string]$Backend = "real", [int]$Seed = 42)
$ErrorActionPreference = "Stop"
python -u -m src.verifier_audit
foreach ($g in 4, 8) {
  python -u -m src.experiments --backend $Backend --arm all --grid $g --seed $Seed
}
python -u -m src.tables  --backend $Backend --grid 4
python -u -m src.figures --backend $Backend --grid 4
Write-Host "[run_all] done -> paper/table1_main.md, paper/figs/"