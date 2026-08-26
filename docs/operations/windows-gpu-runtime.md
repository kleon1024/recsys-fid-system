# Windows RTX 4090 runtime

This is the operating contract for the WSL2 GPU worker. It protects the host
from unbounded recommendation simulations without turning a failed experiment
into an infinite restart loop.

## Host boundary

The Windows host has 32 GB RAM. `%UserProfile%/.wslconfig` owns the WSL ceiling:

```ini
[wsl2]
memory=24GB
swap=4GB

[experimental]
autoMemoryReclaim=gradual
```

This leaves approximately 8 GB for Windows. Increasing WSL memory or swap is not
an OOM fix; the simulator must remain bounded. The Linux VHD has ample free
space, but the Windows C drive is separately monitored because WSL swap and VHD
growth ultimately depend on it.

## Service recovery

WSL uses systemd. `ssh.service` and `tailscaled.service` are enabled and already
use `Restart=on-failure`. The host policy rejects user-created Scheduled Tasks,
so an HKCU login startup entry launches the checked-in hidden PowerShell
keepalive and probes WSL every five minutes. User lingering keeps user-level job
and health units alive without an SSH session.

The Windows keepalive performs an actual Linux-root write probe, not only
`/bin/true`. A Hyper-V storage interruption can leave the distro reachable while
ext4 has aborted its journal and rejects every write. The keepalive terminates
and starts that distro at most twice consecutively; persistent storage failure
remains stopped for operator inspection instead of entering a reboot loop.

Install the per-user Windows keepalive from WSL:

```bash
mkdir -p /mnt/c/Users/1995d/.recsys
install -m 0644 ops/windows-wsl/wsl-keepalive.ps1 \
  /mnt/c/Users/1995d/.recsys/wsl-keepalive.ps1
/mnt/c/Windows/System32/reg.exe add \
  'HKCU\Software\Microsoft\Windows\CurrentVersion\Run' \
  /v RecsysWSLKeepalive /t REG_SZ \
  /d 'powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\Users\1995d\.recsys\wsl-keepalive.ps1' /f
```

Install the checked-in units:

```bash
mkdir -p ~/.config/systemd/user ~/.config/recsys/jobs
install -m 0644 ops/windows-wsl/recsys-gpu-job@.service ~/.config/systemd/user/
install -m 0644 ops/windows-wsl/recsys-gpu-health.service ~/.config/systemd/user/
install -m 0644 ops/windows-wsl/recsys-gpu-health.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now recsys-gpu-health.timer
```

## Bounded jobs

One job owns one environment file. Example:

```bash
cat > ~/.config/recsys/jobs/feed-aa.env <<'EOF'
RECSYS_COMMAND='exec ~/miniconda3/envs/llm-training/bin/python -m fid_lab.simulation.digital_twin.experiments.launch_cli --plan experiments/plans/F-AA-00.json --runtime-root ~/.local/share/recsys-fid-system/feed-standard-v2 --device cuda'
EOF
systemctl --user start recsys-gpu-job@feed-aa.service
```

The unit applies an 18 GB soft memory boundary, 20 GB hard boundary and 2 GB swap
boundary. It also raises the inherited soft file-descriptor limit to 65,536.
WSL exposes CUDA allocations through DXG descriptors; leaving the process soft
limit at 1,024 causes `get_unused_fd_flags` / `CUDA driver error: unknown error`
under tensor-heavy worlds even when RAM and VRAM are healthy. CUDA or CPU OOM
exits may restart at most twice per 30 minutes. A third failure remains failed
and must be diagnosed from:

```bash
systemctl --user status recsys-gpu-job@feed-aa.service
journalctl --user -u recsys-gpu-job@feed-aa.service
```

Do not enable experiment jobs at boot. Only infrastructure and health checks
restart automatically; a factual LR always starts from its immutable checkpoint
and pre-registered plan.
