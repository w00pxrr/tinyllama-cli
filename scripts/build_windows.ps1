$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DistDir = Join-Path $RootDir "dist/windows"
$BuildDir = Join-Path $RootDir "build/windows"
$CacheDir = Join-Path $RootDir ".pyinstaller-cache"
$SpecFile = Join-Path $RootDir "packaging/pyinstaller-cli.spec"
$PythonExe = Join-Path $RootDir ".venv/Scripts/python.exe"

New-Item -ItemType Directory -Force -Path $DistDir, $BuildDir, $CacheDir | Out-Null
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $DistDir "tinyllama-cli")
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $DistDir "tinyllama-cli-windows.zip")
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $BuildDir

$env:PYINSTALLER_CONFIG_DIR = $CacheDir
& $PythonExe -m PyInstaller --noconfirm --clean --distpath $DistDir --workpath $BuildDir $SpecFile

Compress-Archive -Path (Join-Path $DistDir "tinyllama-cli\*") -DestinationPath (Join-Path $DistDir "tinyllama-cli-windows.zip") -Force
Write-Host "Windows artifacts ready:"
Write-Host "  $DistDir\tinyllama-cli"
Write-Host "  $DistDir\tinyllama-cli-windows.zip"
