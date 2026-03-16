param(
    [Parameter(Mandatory = $true)]
    [string]$TrainDir,

    [string]$LabelFile = "",
    [string]$TrainLabelFile = "",
    [string]$TestLabelFile = "",

    [string]$TestDir = "",
    [string]$Pretrained = ".\weights_red_stage3\Final_LPRNet_model.pth",
    [int]$MaxEpoch = 15,
    [int]$TrainBatch = 64,
    [int]$TestBatch = 120,
    [double]$LearningRate = 0.001,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$envPath = Join-Path $PSScriptRoot ".conda"
if (-not (Test-Path $envPath)) {
    throw "Conda environment not found: $envPath"
}
if (-not (Test-Path $TrainDir)) {
    throw "TrainDir not found: $TrainDir"
}
if ([string]::IsNullOrWhiteSpace($TestDir)) {
    $TestDir = $TrainDir
}
if ([string]::IsNullOrWhiteSpace($TrainLabelFile)) {
    if ([string]::IsNullOrWhiteSpace($LabelFile)) {
        throw "TrainLabelFile (or legacy LabelFile) not provided."
    }
    $TrainLabelFile = $LabelFile
}
if ([string]::IsNullOrWhiteSpace($TestLabelFile)) {
    $TestLabelFile = $TrainLabelFile
}
if (-not (Test-Path $TrainLabelFile)) {
    throw "TrainLabelFile not found: $TrainLabelFile"
}
if (-not (Test-Path $TestLabelFile)) {
    throw "TestLabelFile not found: $TestLabelFile"
}

$cmd = @(
    "conda", "run", "-p", $envPath, "python", "train_LPRNet.py",
    "--train_img_dirs", $TrainDir,
    "--test_img_dirs", $TestDir,
    "--train_txt_file", $TrainLabelFile,
    "--test_txt_file", $TestLabelFile,
    "--pretrained_model", $Pretrained,
    "--save_folder", ".\weights_local\",
    "--max_epoch", $MaxEpoch,
    "--train_batch_size", $TrainBatch,
    "--test_batch_size", $TestBatch,
    "--learning_rate", $LearningRate,
    "--cuda", ($(if ($Cuda) { "true" } else { "false" }))
)

Write-Host "Running:" ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length-1)]
