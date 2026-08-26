$ErrorActionPreference = "Continue"
$Distro = "Ubuntu-22.04"
$LinuxUser = "ding"
$IntervalSeconds = 300
$MaximumConsecutiveRecoveries = 2
$Mutex = [System.Threading.Mutex]::new($false, "Local\RecsysWSLKeepalive")
$ConsecutiveRecoveries = 0

if (-not $Mutex.WaitOne(0, $false)) {
    exit 0
}

try {
    while ($true) {
        try {
            & "$env:WINDIR\System32\wsl.exe" -d $Distro -u $LinuxUser -- `
                /bin/bash -lc `
                'probe=$(mktemp /tmp/recsys-wsl-write.XXXXXX) && rm -f "$probe"' `
                | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $ConsecutiveRecoveries = 0
            }
            elseif ($ConsecutiveRecoveries -lt $MaximumConsecutiveRecoveries) {
                $ConsecutiveRecoveries += 1
                $timestamp = Get-Date -Format o
                "$timestamp unhealthy WSL root; bounded recovery $ConsecutiveRecoveries" `
                    | Add-Content "$env:USERPROFILE\.recsys\wsl-keepalive-error.log"
                & "$env:WINDIR\System32\wsl.exe" --terminate $Distro | Out-Null
                Start-Sleep -Seconds 10
                & "$env:WINDIR\System32\wsl.exe" `
                    -d $Distro -u $LinuxUser -- /bin/true | Out-Null
            }
        }
        catch {
            $timestamp = Get-Date -Format o
            "$timestamp $($_.Exception.Message)" | Add-Content `
                "$env:USERPROFILE\.recsys\wsl-keepalive-error.log"
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    $Mutex.ReleaseMutex()
    $Mutex.Dispose()
}
