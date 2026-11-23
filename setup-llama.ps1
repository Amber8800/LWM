$ErrorActionPreference = 'Stop'
$release = 'b12651'
$platform = 'win-cuda-12.4-x64.zip'
$assets = (Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$release").assets
$dest = Join-Path $PSScriptRoot '.tools/llama'
$cache = Join-Path $PSScriptRoot '.cache/llama'
New-Item -ItemType Directory -Force -Path $dest, $cache | Out-Null
foreach ($name in @("llama-$release-bin-$platform", "cudart-llama-bin-$platform")) {
    $asset = $assets | Where-Object name -eq $name
    if (-not $asset.digest.StartsWith('sha256:')) { throw "Missing official digest for $name" }
    $zip = Join-Path $cache $name
    if (-not (Test-Path -LiteralPath $zip) -or (Get-FileHash $zip).Hash -ne $asset.digest.Substring(7)) {
        Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    }
    if ((Get-FileHash $zip).Hash -ne $asset.digest.Substring(7)) { throw "Digest mismatch for $name" }
    Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
}
Get-ChildItem -LiteralPath $dest -Recurse -Filter llama-server.exe | Select-Object -ExpandProperty FullName
