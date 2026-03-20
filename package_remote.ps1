param(
    [string]$OutputZip = "comfyui_pose_redirect_align.zip"
)

$source = Join-Path $PSScriptRoot "comfyui_pose_redirect_align"
$zipPath = Join-Path $PSScriptRoot $OutputZip

if (-not (Test-Path $source)) {
    throw "Source folder not found: $source"
}

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path $source -DestinationPath $zipPath
Write-Host "Created package: $zipPath"
