"""GGUF model loader for comfygo.

Provides direct loading of GGUF-quantized CLIP/text-encoder models
without going through ComfyUI's node registry.
"""
import logging

log = logging.getLogger(__name__)


def load_clip_gguf(path: str, clip_type_str: str):
    """Load a GGUF-quantized CLIP / text encoder model.

    Mirrors what CLIPLoaderGGUF.load_clip() does internally, but takes a
    full filesystem path instead of a bare filename resolved via folder_paths.

    Args:
        path:          Absolute path to the ``.gguf`` file.
        clip_type_str: One of the ComfyUI CLIPType names (e.g. ``"lumina2"``).

    Returns:
        Tuple ``(clip,)``.
    """
    import comfy.sd
    import comfy.model_management
    import folder_paths

    from .loader import gguf_clip_loader
    from .ops import GGMLOps
    from .patcher import GGUFModelPatcher

    clip_type = getattr(
        comfy.sd.CLIPType,
        clip_type_str.upper(),
        comfy.sd.CLIPType.STABLE_DIFFUSION,
    )

    # Load raw state dict from GGUF file
    clip_data = gguf_clip_loader(path)

    # Build CLIP model with GGMLOps (handles dequantization on the fly)
    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=clip_type,
        state_dicts=[clip_data],
        model_options={
            "custom_operations": GGMLOps,
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        },
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    # Replace the standard ModelPatcher with GGUFModelPatcher
    clip.patcher = GGUFModelPatcher.clone(clip.patcher)
    return (clip,)
