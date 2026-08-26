$ErrorActionPreference = "Continue"
$Distro = "Ubuntu-22.04"
$LinuxUser = "ding"
$IntervalSeconds = 300
$Mutex = [System.Threading.Mutex]::new($false, "Local\RecsysWSLKeepalive")

if (-not $Mutex.WaitOne(0, $false)) {
    exit 0
}

try {
    while ($true) {
        try {
            & "$env:WINDIR\System32\wsl.exe" `
                -d $Distro -u $LinuxUser -- /bin/true | Out-Null
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
