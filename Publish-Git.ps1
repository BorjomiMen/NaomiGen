param(
    [Parameter(Mandatory=$true)][string]$RemoteUrl,
    [string]$Branch = 'codex/baseline-64x64',
    [string]$Message = 'Preserve reproducible NAOMi 64x64 baseline'
)
$ErrorActionPreference = 'Stop'
$gitCommand = Get-Command git -ErrorAction Stop
$gitExe = $gitCommand.Source
function Invoke-GitChecked([string[]]$GitArguments) {
    & $gitExe @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArguments -join ' ')" }
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.git'))) {
        Invoke-GitChecked -GitArguments @('init','-b',$Branch)
    }
    $activeBranch = (& $gitExe branch --show-current).Trim()
    if ($activeBranch -ne $Branch) { throw "Current branch is '$activeBranch'. Pass -Branch '$activeBranch' explicitly or select '$Branch'." }
    $remotes = @(& $gitExe remote)
    if ($remotes -contains 'origin') {
        $existingRemote = (& $gitExe remote get-url origin).Trim()
        if ($existingRemote -ne $RemoteUrl) { throw 'origin points to another repository; it was not changed.' }
    } else { Invoke-GitChecked -GitArguments @('remote','add','origin',$RemoteUrl) }
    Invoke-GitChecked -GitArguments @('add','--all')
    & $gitExe diff --cached --quiet
    $differenceCode = $LASTEXITCODE
    if ($differenceCode -eq 1) { Invoke-GitChecked -GitArguments @('commit','-m',$Message) }
    elseif ($differenceCode -ne 0) { throw 'Could not inspect staged changes.' }
    Invoke-GitChecked -GitArguments @('push','--set-upstream','origin',$Branch)
} finally { Pop-Location }
