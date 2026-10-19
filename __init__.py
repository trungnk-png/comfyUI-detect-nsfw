"""
Detect NSFW for ComfyUI.

One node:
- image -> Freepik NSFWDetector
- string -> Qwen prompt guard
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
