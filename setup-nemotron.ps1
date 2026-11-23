$ErrorActionPreference = 'Stop'
$name = 'nemo-speech-0.3.0-windows-x86_64-cuda.zip'
$sha256 = 'd0d32ae1bf6ee0534767a47b2aa0e9e2ff6fe39643465c0baae9301d04f6ed42'
$dest = Join-Path $PSScriptRoot '.tools/nemo-speech'
$cache = Join-Path $PSScriptRoot '.cache/nemo-speech'
New-Item -ItemType Directory -Force -Path $dest, $cache | Out-Null
$zip = Join-Path $cache $name
if (-not (Test-Path -LiteralPath $zip) -or (Get-FileHash $zip).Hash -ne $sha256) {
    Invoke-WebRequest "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.3.0/$name" -OutFile $zip
}
if ((Get-FileHash $zip).Hash -ne $sha256) { throw 'NeMo-Speech.cpp archive digest mismatch' }
Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
Get-ChildItem -Path $dest -Recurse -Filter nemo-speech.exe | Select-Object -ExpandProperty FullName
