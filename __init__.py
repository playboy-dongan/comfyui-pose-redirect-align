import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "comfyui_pose_redirect_align" / "pose_redirect_align.py"
_SPEC = importlib.util.spec_from_file_location("pose_redirect_align_impl", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load node module from {_MODULE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

NODE_CLASS_MAPPINGS = _MODULE.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _MODULE.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
