# Lay phien chay tu Pi ve may tinh (PowerShell). Chay trong thu muc uav_perception_pi tren may tinh:
#   .\tools\fetch_run.ps1 -Pi <user>@<ip-pi>                                  # phien moi nhat (moi loai)
#   .\tools\fetch_run.ps1 -Pi <user>@<ip-pi> -Kind real_time_perception       # phien moi nhat cua 1 loai
#   .\tools\fetch_run.ps1 -Pi <user>@<ip-pi> -Name 20261114-101530_real_time_perception
#   .\tools\fetch_run.ps1 -Pi <user>@<ip-pi> -All                             # ca thu muc runs/
# Ket qua: runs_pi\<ten phien>\ (object.jpg, final.jpg, frame.jpg, selected_mask.png, result.json, timing.csv,
# hw_usage.csv, run.log, ...). Lan dau ssh / scp se hoi mat khau Pi.
param(
    [Parameter(Mandatory = $true)][string]$Pi,       # <user>@<ip-pi>
    [string]$Kind = "",                              # real_time_perception | real_time | sensor_check | offline
    [string]$Name = "",                              # ten thu muc phien cu the
    [string]$RemoteDir = "~/uav_perception_pi/runs",
    [string]$Out = "runs_pi",
    [switch]$All
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Out | Out-Null
if ($All) {
    scp -r "${Pi}:$RemoteDir/*" $Out
    Write-Host "Da chep ca $RemoteDir vao $Out"
    exit 0
}
if ($Pi -match "[<>]") { throw "Thay <user>@<ip-pi> bang gia tri that, vd pi@192.168.1.50 (xem IP tren Pi: hostname -I)" }
if ($Name -eq "") {
    $pattern = if ($Kind -eq "") { "*" } else { "*_$Kind" }
    $found = ssh -o ConnectTimeout=8 $Pi "cd $RemoteDir && ls -1td $pattern 2>/dev/null | head -n 1"
    if ($LASTEXITCODE -eq 255) { throw "Khong ssh duoc toi $Pi (sai IP / Pi chua bat / khac mang). Thu: ssh $Pi" }
    $Name = if ($null -eq $found) { "" } else { "$found".Trim() }
    if ($Name -eq "") { throw "Khong co phien nao khop '$pattern' trong $RemoteDir tren Pi" }
}
scp -r "${Pi}:$RemoteDir/$Name" $Out
if ($LASTEXITCODE -ne 0) { throw "scp loi (ma $LASTEXITCODE)" }
Write-Host "Da chep phien $Name vao $Out\$Name"
Get-ChildItem "$Out\$Name" | Select-Object Name, Length | Format-Table -AutoSize
