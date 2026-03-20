# ComfyUI Pose Redirect Align

This custom node aligns `source_pose` to the composition of `reference_pose`.

What it does:

1. Detects the person foreground area in `reference_pose`.
2. Detects the person foreground area in `source_pose`.
3. Scales `source_pose` so its person height matches the person height in `reference_pose`.
4. Estimates a head anchor in both poses using the upper body region.
5. Translates the scaled `source_pose` so the head position matches the reference pose.

Outputs:

- `aligned_pose`: the remapped pose image, ready for ControlNet.
- `aligned_mask`: the foreground mask after alignment.
- `scale`: applied scale factor.
- `offset_x` / `offset_y`: applied translation.

## Install

Copy the whole `comfyui_pose_redirect_align` folder into:

```text
ComfyUI/custom_nodes/
```

Then restart ComfyUI.

## Node Name

```text
姿态重定向对齐
```

Category:

```text
pose/redirect
```

## Remote Deployment

Important: a custom node must be installed on the machine that is actually running ComfyUI.

- If you control the remote server, upload this folder to the server's `ComfyUI/custom_nodes/` directory and restart ComfyUI.
- If the remote platform does not allow custom nodes, this node cannot be used there.
- If the remote platform has a custom node manager, package this folder as a zip or git repo and install it from the server side.

## Tuning

- Increase `background_threshold` if the pose image background is not clean.
- Adjust `head_search_ratio` if the head anchor is detected too high or too low.
- Widen `min_scale` and `max_scale` if the two poses differ a lot in size.
