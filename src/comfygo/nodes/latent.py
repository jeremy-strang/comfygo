"""Latent image creation and VAE decode — replacing ComfyUI node wrappers."""
import logging

import torch

log = logging.getLogger(__name__)


def empty_sd3_latent(width: int, height: int, batch_size: int = 1):
    """Create a zeroed SD3/AuraFlow/Lumina2-style 16-channel latent tensor.

    Mirrors ComfyUI's ``EmptySD3LatentImage`` node (nodes_sd3.py).

    Args:
        width:  Image width in pixels (must be divisible by 8).
        height: Image height in pixels (must be divisible by 8).
        batch_size: Number of images in the batch.

    Returns:
        Tuple ``(latent_dict,)`` where the dict has ``"samples"`` and
        ``"downscale_ratio_spacial"`` keys.
    """
    import comfy.model_management

    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    return ({"samples": latent, "downscale_ratio_spacial": 8},)


def vae_decode(vae, samples: dict):
    """Decode a latent dict back to pixel-space images.

    Mirrors ComfyUI's ``VAEDecode`` node (nodes.py).

    Args:
        vae:     ComfyUI VAE model object.
        samples: Latent dict with a ``"samples"`` key.

    Returns:
        Tuple ``(images,)`` — float32 tensor of shape
        ``[B, H, W, C]`` in [0, 1] range.
    """
    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    images = vae.decode(latent)
    if len(images.shape) == 5:
        # Combine temporal batch dimension
        images = images.reshape(
            -1, images.shape[-3], images.shape[-2], images.shape[-1]
        )
    return (images,)
