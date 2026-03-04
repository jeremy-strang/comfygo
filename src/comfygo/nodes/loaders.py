"""Direct model loaders — replacing ComfyUI node wrappers.

All functions take full filesystem paths (not bare filenames).
Path resolution (bare name → full path) is done in workflow.py before
calling these functions.
"""
import logging

import torch

log = logging.getLogger(__name__)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: str):
    """Load a full checkpoint (model + CLIP + VAE).

    Args:
        path: Absolute path to the safetensors / ckpt file.

    Returns:
        Tuple ``(model, clip, vae)``.
    """
    import comfy.sd
    import folder_paths

    out = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return out[:3]


# ── CLIP ──────────────────────────────────────────────────────────────────────

def load_clip(path: str, clip_type_str: str, device: str = "default"):
    """Load a text encoder / CLIP model.

    Args:
        path: Absolute path to the model file.
        clip_type_str: One of the ComfyUI CLIPType names (e.g. ``"lumina2"``).
        device: ``"default"`` or ``"cpu"``.

    Returns:
        Tuple ``(clip,)``.
    """
    import comfy.sd
    import folder_paths

    clip_type = getattr(
        comfy.sd.CLIPType,
        clip_type_str.upper(),
        comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")

    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options=model_options,
    )
    return (clip,)


# ── VAE ───────────────────────────────────────────────────────────────────────

def load_vae(path: str):
    """Load a VAE model.

    Args:
        path: Absolute path to the VAE file.

    Returns:
        Tuple ``(vae,)``.
    """
    import comfy.sd
    import comfy.utils

    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return (vae,)


# ── Diffusion model (UNET) ────────────────────────────────────────────────────

def load_unet(path: str, weight_dtype: str = "default"):
    """Load a standalone diffusion model (UNET).

    Args:
        path: Absolute path to the diffusion model file.
        weight_dtype: One of ``"default"``, ``"fp8_e4m3fn"``,
                      ``"fp8_e4m3fn_fast"``, ``"fp8_e5m2"``.

    Returns:
        Tuple ``(model,)``.
    """
    import comfy.sd

    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    return (model,)


# ── LoRA ──────────────────────────────────────────────────────────────────────

def load_lora(model, clip, path: str, strength_model: float, strength_clip: float):
    """Apply a LoRA to both model and CLIP.

    Args:
        model: Diffusion model patcher.
        clip: CLIP model.
        path: Absolute path to the LoRA file.
        strength_model: LoRA strength applied to the diffusion model.
        strength_clip: LoRA strength applied to the CLIP model.

    Returns:
        Tuple ``(model_lora, clip_lora)``.
    """
    import comfy.sd
    import comfy.utils

    lora = comfy.utils.load_torch_file(path, safe_load=True)
    model_lora, clip_lora = comfy.sd.load_lora_for_models(
        model, clip, lora, strength_model, strength_clip
    )
    return (model_lora, clip_lora)


def load_lora_model_only(model, path: str, strength_model: float):
    """Apply a LoRA to the diffusion model only (no CLIP).

    Args:
        model: Diffusion model patcher.
        path: Absolute path to the LoRA file.
        strength_model: LoRA strength.

    Returns:
        Tuple ``(model_lora,)``.
    """
    model_lora, _ = load_lora(model, None, path, strength_model, 0.0)
    return (model_lora,)
