param([string]$Python = $env:NAOMIGEN_PYTHON)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
$requirements = Join-Path $projectRoot 'naomi_zenodo_stage1\requirements.txt'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $candidates = @()
    if ($Python) { $candidates += $Python }
    # The bundled runtime is an optional local discovery path, never a repository dependency.
    $candidates += (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    foreach ($name in @('python3.12','python3','python')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and $command.Source -notlike '*\WindowsApps\*') { $candidates += $command.Source }
    }
    $selectedPython = $null
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        & $candidate -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $selectedPython = $candidate; break }
    }
    if (-not $selectedPython) {
        throw 'Python 3.12 is required. Install it and put python on PATH, or set NAOMIGEN_PYTHON to its executable.'
    }
    Write-Output 'Creating the ignored .venv environment...'
    & $selectedPython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
}
$sha = [Security.Cryptography.SHA256]::Create()
try { $digest = [BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($requirements))).Replace('-','') }
finally { $sha.Dispose() }
$stamp = Join-Path $venvRoot 'naomigen-requirements.sha256'
if (-not (Test-Path -LiteralPath $stamp) -or (Get-Content -LiteralPath $stamp -Raw).Trim() -ne $digest) {
    & $pythonExe -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install pinned Python dependencies.' }
    [IO.File]::WriteAllText($stamp,$digest)
}
Write-Output 'NaomiGen Python environment is ready.'
