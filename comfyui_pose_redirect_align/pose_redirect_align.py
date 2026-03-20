from typing import Tuple

import cv2
import numpy as np
import torch


def _to_uint8_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image tensor, got shape {array.shape}")

    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] > 3:
        array = array[:, :, :3]

    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _border_background_color(image: np.ndarray, border: int = 4) -> np.ndarray:
    h, w = image.shape[:2]
    border = max(1, min(border, h // 2 or 1, w // 2 or 1))
    strips = [
        image[:border, :, :].reshape(-1, 3),
        image[h - border :, :, :].reshape(-1, 3),
        image[:, :border, :].reshape(-1, 3),
        image[:, w - border :, :].reshape(-1, 3),
    ]
    return np.median(np.concatenate(strips, axis=0), axis=0).astype(np.uint8)


def _foreground_mask(image: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    bg = _border_background_color(image)
    diff = np.abs(image.astype(np.int16) - bg.astype(np.int16))
    mask = np.max(diff, axis=2) > threshold

    if mask.sum() == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bg_gray = int(np.mean(bg))
        mask = np.abs(gray.astype(np.int16) - bg_gray) > threshold

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool), bg


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _head_anchor(mask: np.ndarray, bbox: Tuple[int, int, int, int], head_ratio: float) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    head_h = max(1, int(round(height * head_ratio)))
    top_mask = mask[y0 : min(y0 + head_h, y1), x0:x1].astype(np.uint8)

    if top_mask.sum() == 0:
        return x0 + width / 2.0, float(y0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(top_mask, connectivity=8)
    if num_labels <= 1:
        ys, xs = np.where(top_mask > 0)
        return x0 + float(xs.mean()), y0 + float(ys.mean())

    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == best_label)
    if len(xs) == 0:
        return x0 + width / 2.0, float(y0)

    return x0 + float(xs.mean()), y0 + float(ys.mean())


def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    height, width = mask.shape
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def _paste_with_mask(
    canvas: np.ndarray,
    canvas_mask: np.ndarray,
    image: np.ndarray,
    image_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> Tuple[np.ndarray, np.ndarray]:
    out = canvas.copy()
    out_mask = canvas_mask.copy()

    src_h, src_w = image.shape[:2]
    dst_h, dst_w = canvas.shape[:2]

    x0 = max(0, offset_x)
    y0 = max(0, offset_y)
    x1 = min(dst_w, offset_x + src_w)
    y1 = min(dst_h, offset_y + src_h)

    if x0 >= x1 or y0 >= y1:
        return out, out_mask

    src_x0 = x0 - offset_x
    src_y0 = y0 - offset_y
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)

    patch = image[src_y0:src_y1, src_x0:src_x1]
    patch_mask = image_mask[src_y0:src_y1, src_x0:src_x1]

    region = out[y0:y1, x0:x1]
    region[patch_mask] = patch[patch_mask]
    out[y0:y1, x0:x1] = region
    out_mask[y0:y1, x0:x1] |= patch_mask
    return out, out_mask


def _align_single(
    reference_pose: torch.Tensor,
    source_pose: torch.Tensor,
    threshold: int,
    head_ratio: float,
    min_scale: float,
    max_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, int, int]:
    ref_image = _to_uint8_image(reference_pose)
    src_image = _to_uint8_image(source_pose)

    ref_mask, ref_bg = _foreground_mask(ref_image, threshold)
    src_mask, _ = _foreground_mask(src_image, threshold)

    ref_bbox = _mask_bbox(ref_mask)
    src_bbox = _mask_bbox(src_mask)

    ref_height = max(1, ref_bbox[3] - ref_bbox[1])
    src_height = max(1, src_bbox[3] - src_bbox[1])
    scale = float(np.clip(ref_height / float(src_height), min_scale, max_scale))

    ref_head = _head_anchor(ref_mask, ref_bbox, head_ratio)
    src_head = _head_anchor(src_mask, src_bbox, head_ratio)

    scaled_src = _resize_image(src_image, scale)
    scaled_mask = _resize_mask(src_mask, scale)

    offset_x = int(round(ref_head[0] - src_head[0] * scale))
    offset_y = int(round(ref_head[1] - src_head[1] * scale))

    canvas = np.full_like(ref_image, ref_bg, dtype=np.uint8)
    canvas_mask = np.zeros(ref_mask.shape, dtype=bool)
    aligned, aligned_mask = _paste_with_mask(canvas, canvas_mask, scaled_src, scaled_mask, offset_x, offset_y)

    aligned_tensor = torch.from_numpy(aligned.astype(np.float32) / 255.0).unsqueeze(0)
    mask_tensor = torch.from_numpy(aligned_mask.astype(np.float32)).unsqueeze(0)
    return aligned_tensor, mask_tensor, scale, offset_x, offset_y


class PoseRedirectAlignByHead:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "\u53c2\u8003\u59ff\u6001\u56fe": ("IMAGE",),
                "\u6e90\u59ff\u6001\u56fe": ("IMAGE",),
                "\u80cc\u666f\u9608\u503c": (
                    "INT",
                    {
                        "default": 18,
                        "min": 1,
                        "max": 255,
                        "step": 1,
                        "tooltip": "\u4e0e\u8fb9\u754c\u80cc\u666f\u5dee\u5f02\u8db3\u591f\u5927\u7684\u50cf\u7d20\u4f1a\u88ab\u89c6\u4e3a pose \u524d\u666f\u3002",
                    },
                ),
                "\u5934\u90e8\u641c\u7d22\u6bd4\u4f8b": (
                    "FLOAT",
                    {
                        "default": 0.22,
                        "min": 0.05,
                        "max": 0.45,
                        "step": 0.01,
                        "tooltip": "\u7528\u4e8e\u4f30\u8ba1\u5934\u90e8\u951a\u70b9\u7684\u4e0a\u534a\u90e8\u533a\u57df\u6bd4\u4f8b\u3002",
                    },
                ),
                "\u6700\u5c0f\u7f29\u653e": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                    },
                ),
                "\u6700\u5927\u7f29\u653e": (
                    "FLOAT",
                    {
                        "default": 4.00,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "\u5bf9\u9f50\u540e\u59ff\u6001\u56fe",
        "\u5bf9\u9f50\u540e\u906e\u7f69",
        "\u7f29\u653e\u500d\u7387",
        "\u6a2a\u5411\u504f\u79fb",
        "\u7eb5\u5411\u504f\u79fb",
    )
    FUNCTION = "align_pose"
    CATEGORY = "\u59ff\u6001/\u91cd\u5b9a\u5411"

    def align_pose(self, **kwargs):
        reference_pose = kwargs["\u53c2\u8003\u59ff\u6001\u56fe"]
        source_pose = kwargs["\u6e90\u59ff\u6001\u56fe"]
        background_threshold = kwargs["\u80cc\u666f\u9608\u503c"]
        head_search_ratio = kwargs["\u5934\u90e8\u641c\u7d22\u6bd4\u4f8b"]
        min_scale = kwargs["\u6700\u5c0f\u7f29\u653e"]
        max_scale = kwargs["\u6700\u5927\u7f29\u653e"]

        ref_batch = reference_pose.shape[0]
        src_batch = source_pose.shape[0]
        batch = max(ref_batch, src_batch)

        if ref_batch not in (1, batch):
            raise ValueError("reference_pose batch must be 1 or equal to source_pose batch size")
        if src_batch not in (1, batch):
            raise ValueError("source_pose batch must be 1 or equal to reference_pose batch size")
        if min_scale > max_scale:
            raise ValueError("min_scale cannot be larger than max_scale")

        aligned_images = []
        aligned_masks = []
        scales = []
        offset_xs = []
        offset_ys = []

        for index in range(batch):
            ref_img = reference_pose[0 if ref_batch == 1 else index]
            src_img = source_pose[0 if src_batch == 1 else index]
            aligned, mask, scale, offset_x, offset_y = _align_single(
                ref_img,
                src_img,
                background_threshold,
                head_search_ratio,
                min_scale,
                max_scale,
            )
            aligned_images.append(aligned)
            aligned_masks.append(mask)
            scales.append(scale)
            offset_xs.append(offset_x)
            offset_ys.append(offset_y)

        aligned_batch = torch.cat(aligned_images, dim=0)
        mask_batch = torch.cat(aligned_masks, dim=0)

        return (
            aligned_batch,
            mask_batch,
            float(scales[0]),
            int(offset_xs[0]),
            int(offset_ys[0]),
        )


NODE_CLASS_MAPPINGS = {
    "PoseRedirectAlignByHead": PoseRedirectAlignByHead,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseRedirectAlignByHead": "\u59ff\u6001\u91cd\u5b9a\u5411\u5bf9\u9f50",
}
