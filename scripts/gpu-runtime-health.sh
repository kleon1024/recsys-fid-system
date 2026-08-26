#!/usr/bin/env bash
set -euo pipefail

minimum_ram_kib=$((2 * 1024 * 1024))
minimum_root_kib=$((20 * 1024 * 1024))
minimum_windows_kib=$((10 * 1024 * 1024))
minimum_gpu_mib=2048

available_ram_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
swap_free_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
root_free_kib="$(df --output=avail / | tail -1 | tr -d ' ')"
windows_free_kib="$(df --output=avail /mnt/c | tail -1 | tr -d ' ')"
gpu_row="$(/usr/lib/wsl/lib/nvidia-smi \
  --query-gpu=memory.free,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits | head -1)"
IFS=',' read -r gpu_free_mib gpu_total_mib gpu_utilization gpu_temperature \
  <<< "${gpu_row// /}"

ssh_state="$(systemctl is-active ssh || true)"
tailscale_state="$(systemctl is-active tailscaled || true)"
status=pass

if (( available_ram_kib < minimum_ram_kib \
      || root_free_kib < minimum_root_kib \
      || windows_free_kib < minimum_windows_kib \
      || gpu_free_mib < minimum_gpu_mib )); then
  status=fail
fi
if [[ "${ssh_state}" != active || "${tailscale_state}" != active ]]; then
  status=fail
fi

printf '{"status":"%s","ram_available_kib":%s,"swap_free_kib":%s,' \
  "${status}" "${available_ram_kib}" "${swap_free_kib}"
printf '"root_free_kib":%s,"windows_free_kib":%s,' \
  "${root_free_kib}" "${windows_free_kib}"
printf '"gpu_free_mib":%s,"gpu_total_mib":%s,"gpu_utilization":%s,' \
  "${gpu_free_mib}" "${gpu_total_mib}" "${gpu_utilization}"
printf '"gpu_temperature_c":%s,"ssh":"%s","tailscale":"%s"}\n' \
  "${gpu_temperature}" "${ssh_state}" "${tailscale_state}"

[[ "${status}" == pass ]]
