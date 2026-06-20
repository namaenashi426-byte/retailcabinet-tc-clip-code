param(
  [string]$EnvName = "retail_vificlip",
  [string]$PythonVersion = "3.8"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating conda environment: $EnvName"
conda create -y -n $EnvName python=$PythonVersion

Write-Host "Installing ViFi-CLIP dependencies."
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r "$PSScriptRoot\..\repos\ViFi-CLIP\requirements.txt"

Write-Host "Validation command:"
Write-Host "conda run -n $EnvName python -c `"import torch, decord; print(torch.__version__, decord.__version__)`""
