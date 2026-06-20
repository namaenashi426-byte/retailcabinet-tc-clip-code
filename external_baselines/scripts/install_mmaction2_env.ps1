param(
  [string]$EnvName = "retail_mmaction2",
  [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating conda environment: $EnvName"
conda create -y -n $EnvName python=$PythonVersion

Write-Host "Installing PyTorch. Adjust CUDA channel/package if your driver requires a different build."
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing OpenMMLab dependencies via MIM."
conda run -n $EnvName python -m pip install -U openmim
conda run -n $EnvName mim install mmengine
conda run -n $EnvName mim install mmcv

Write-Host "Installing MMAction2 editable package."
$RepoRoot = Resolve-Path -LiteralPath "$PSScriptRoot\..\repos\mmaction2"
conda run -n $EnvName python -m pip install -v -e $RepoRoot

Write-Host "Installing video IO helpers."
conda run -n $EnvName python -m pip install decord opencv-contrib-python pandas scikit-learn

Write-Host "Validation command:"
Write-Host "conda run -n $EnvName python -c `"import torch, mmcv, mmaction; print(torch.__version__, mmcv.__version__, mmaction.__version__)`""
