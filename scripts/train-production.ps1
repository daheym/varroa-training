[CmdletBinding()]
param(
    [ValidateRange(1, [int]::MaxValue)]
    [int]$Epochs = 50,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$Patience = 10,

    [string]$BaseConfig = "configs/labeling.json",

    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$RunPrefix = "training-production"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

Push-Location $repoRoot
try {
    $baseConfigPath = (Resolve-Path -LiteralPath $BaseConfig).Path

    # Validation does not write generated data, so run it before reserving a number.
    & $python -m src.cli --config $baseConfigPath validate
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validation failed with exit code $LASTEXITCODE"
    }

    $existingNames = @()
    foreach ($directory in @("runs", "data/processed")) {
        if (Test-Path -LiteralPath $directory) {
            $existingNames += Get-ChildItem -LiteralPath $directory -Directory | ForEach-Object { $_.Name }
        }
    }
    foreach ($directory in @("data/splits", "runs/.configs")) {
        if (Test-Path -LiteralPath $directory) {
            $existingNames += Get-ChildItem -LiteralPath $directory -File -Filter "*.json" | ForEach-Object { $_.BaseName }
        }
    }
    $namePattern = "^{0}-(\d+)$" -f [regex]::Escape($RunPrefix)
    $existingNumbers = @($existingNames | ForEach-Object {
        if ($_ -match $namePattern) {
            [int]$Matches[1]
        }
    })
    $number = if ($existingNumbers.Count -gt 0) {
        [int](($existingNumbers | Measure-Object -Maximum).Maximum + 1)
    } else {
        1
    }

    do {
        $runName = "{0}-{1:D2}" -f $RunPrefix, $number
        $runPath = Join-Path $repoRoot "runs/$runName"
        $processedPath = Join-Path $repoRoot "data/processed/$runName"
        $splitPath = Join-Path $repoRoot "data/splits/$runName.json"
        $generatedConfigPath = Join-Path $repoRoot "runs/.configs/$runName.json"
        $collision = ((Test-Path -LiteralPath $runPath) -or
                      (Test-Path -LiteralPath $processedPath) -or
                      (Test-Path -LiteralPath $splitPath) -or
                      (Test-Path -LiteralPath $generatedConfigPath))
        $number++
    } while ($collision)

    $config = Get-Content -LiteralPath $baseConfigPath -Raw | ConvertFrom-Json
    $config.processed = "data/processed/$runName"
    $config.splits = "data/splits/$runName.json"

    $generatedConfigDirectory = Split-Path -Parent $generatedConfigPath
    New-Item -ItemType Directory -Path $generatedConfigDirectory -Force | Out-Null
    $json = $config | ConvertTo-Json -Depth 100
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($generatedConfigPath, $json + [Environment]::NewLine, $utf8WithoutBom)

    Write-Host "Preparing $runName"
    Write-Host "  dataset: data/processed/$runName"
    Write-Host "  split:   data/splits/$runName.json"
    Write-Host "  run:     runs/$runName"

    & $python -m src.cli --config $generatedConfigPath prepare
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset preparation failed with exit code $LASTEXITCODE"
    }

    & $python -m src.cli --config $generatedConfigPath train --output $runPath --epochs $Epochs --patience $Patience
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }

    Write-Host "Completed $runName"
    Write-Host "Generated config: $generatedConfigPath"
}
finally {
    Pop-Location
}
