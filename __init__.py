"""DB9_Ultimate — tiled upscale 6K-11K cho mọi checkpoint (Flux 2 Klein, Qwen-Image 2.1, ...)."""
__version__ = "0.1.0"

from .db9u.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
