$envPath = Join-Path $PSScriptRoot ".conda"
if (-not (Test-Path $envPath)) {
    Write-Error "Conda environment not found: $envPath"
    exit 1
}

Write-Host "Activate with:"
Write-Host "conda activate $envPath"
Write-Host ""
Write-Host "One-shot examples:"
Write-Host "conda run -p $envPath python train_LPRNet.py --help"
Write-Host "conda run -p $envPath python test_LPRNet.py --help"
