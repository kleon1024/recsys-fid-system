$ErrorActionPreference = "Continue"
$Distro = "Ubuntu-22.04"
$LinuxUser = "ding"
$IntervalSeconds = 300

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
