param(
    [ValidateRange(0.05, 24)][double]$Hours = 7,
    [string]$Output = 'results/research_pilot'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $taskRoot '.runtime\py311\python.exe'
$outputPath = [IO.Path]::GetFullPath((Join-Path $taskRoot $Output))
if (-not $outputPath.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The pilot output must stay inside this workspace.'
}
foreach ($required in @($pythonExe, (Join-Path $taskRoot 'checkpoints\smolvla_libero\model.safetensors'), (Join-Path $taskRoot 'data\libero_pilot.npz'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing local prerequisite: $required" }
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$statusPath = Join-Path $outputPath 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $previous = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $activeWorker = Get-CimInstance Win32_Process -Filter "ProcessId = $($previous.pid)"
    if ($activeWorker -and $activeWorker.CommandLine -like '*research_pilot.py*') {
        Write-Output "Evaluation already running: PID $($previous.pid). Results: $outputPath\RESULTS.html"
        return
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $outputPath "worker.$stamp.stdout.log"
$stderr = Join-Path $outputPath "worker.$stamp.stderr.log"
$workerArgs = @('-X', 'utf8', '-u', 'scripts/research_pilot.py', '--output', ('"' + $outputPath + '"'), '--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture))
$worker = Start-Process -FilePath $pythonExe -ArgumentList $workerArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$dashboard = Start-Process -FilePath $pythonExe -ArgumentList @('-X','utf8','-u','scripts/pilot_dashboard.py','--output',('"' + $outputPath + '"'),'--watch') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputPath "dashboard.$stamp.stdout.log") -RedirectStandardError (Join-Path $outputPath "dashboard.$stamp.stderr.log") -PassThru
$record = [ordered]@{
    pid = $worker.Id; dashboard_pid = $dashboard.Id; started = (Get-Date).ToString('o');
    hours = $Hours; output = $outputPath; stdout = $stdout; stderr = $stderr;
    results = (Join-Path $outputPath 'RESULTS.html'); device = 'CPU; isolated Python 3.11 runtime'
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Encoding UTF8
$record | ConvertTo-Json
