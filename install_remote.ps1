param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot
)

$source = Join-Path $PSScriptRoot "comfyui_pose_redirect_align"
$targetRoot = Join-Path $ComfyUIRoot "custom_nodes"
$target = Join-Path $targetRoot "comfyui_pose_redirect_align"

if (-not (Test-Path $source)) {
    throw "Source folder not found: $source"
}

if (-not (Test-Path $targetRoot)) {
    throw "ComfyUI custom_nodes folder not found: $targetRoot"
}

if (Test-Path $target) {
    Write-Host "Updating existing node at $target"
    Remove-Item -Recurse -Force $target
}

Copy-Item -Recurse -Force $source $target
Write-Host "Installed comfyui_pose_redirect_align to $target"
Write-Host "Restart ComfyUI to load the node."
