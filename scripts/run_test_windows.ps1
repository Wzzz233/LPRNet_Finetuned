param(
    [Parameter(Mandatory = $true)]
    [string]$TestDir,

    [Parameter(Mandatory = $true)]
    [string]$LabelFile,

    [string]$Weights = ".\weights_red_stage3\Final_LPRNet_model.pth",
    [int]$BatchSize = 120,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$envPath = Join-Path $PSScriptRoot ".conda"
if (-not (Test-Path $envPath)) {
    throw "Conda environment not found: $envPath"
}
if (-not (Test-Path $TestDir)) {
    throw "TestDir not found: $TestDir"
}
if (-not (Test-Path $LabelFile)) {
    throw "LabelFile not found: $LabelFile"
}

Write-Host "Note: test_LPRNet.py currently uses load_data.py defaults for txt_file."
Write-Host "For strict custom labels, align the default in load_data.py or run infer_single.py."

$cmd = @(
    "conda", "run", "-p", $envPath, "python", "test_LPRNet.py",
    "--test_img_dirs", $TestDir,
    "--test_batch_size", $BatchSize,
    "--pretrained_model", $Weights,
    "--cuda", ($(if ($Cuda) { "true" } else { "false" }))
)

Write-Host "Running:" ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length-1)]
