"""Sampling helpers — replacing ComfyUI node wrappers."""
import logging

import torch

log = logging.getLogger(__name__)


def model_sampling_auraflow(model, shift: float):
    """Patch *model* with AuraFlow / Lumina2-style discrete-flow sampling.

    Mirrors ComfyUI's ``ModelSamplingAuraFlow.patch_aura()``
    (comfy_extras/nodes_model_advanced.py).

    Args:
        model: ComfyUI model patcher.
        shift: Sigma shift value (default 3.0 for Lumina2).

    Returns:
        Tuple ``(patched_model,)``.
    """
    import comfy.model_sampling

    m = model.clone()

    class ModelSamplingAdvanced(
        comfy.model_sampling.ModelSamplingDiscreteFlow,
        comfy.model_sampling.CONST,
    ):
        pass

    ms = ModelSamplingAdvanced(model.model.model_config)
    ms.set_parameters(shift=shift, multiplier=1.0)
    m.add_object_patch("model_sampling", ms)
    return (m,)


def ksampler_advanced(
    model,
    add_noise: str,
    noise_seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent_image: dict,
    start_at_step: int,
    end_at_step: int,
    return_with_leftover_noise: str,
    denoise: float = 1.0,
):
    """Run KSamplerAdvanced.

    Mirrors ComfyUI's ``KSamplerAdvanced.sample()`` / ``common_ksampler()``
    (nodes.py).

    Args:
        model: Diffusion model patcher.
        add_noise: ``"enable"`` to add noise, ``"disable"`` to skip.
        noise_seed: RNG seed for noise generation.
        steps: Total number of denoising steps.
        cfg: Classifier-free guidance scale.
        sampler_name: K-diffusion sampler name (e.g. ``"euler"``).
        scheduler: Noise schedule name (e.g. ``"simple"``).
        positive: Positive conditioning.
        negative: Negative conditioning.
        latent_image: Latent dict with ``"samples"`` key.
        start_at_step: First denoising step index (inclusive).
        end_at_step:   Last denoising step index (exclusive).
        return_with_leftover_noise: ``"enable"`` keeps leftover noise.
        denoise: Overall denoising strength (default 1.0).

    Returns:
        Tuple ``(out_latent,)`` — updated latent dict.
    """
    import comfy.sample
    import comfy.utils

    force_full_denoise = return_with_leftover_noise != "enable"
    disable_noise = add_noise == "disable"

    latent_samples = latent_image["samples"]
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model, latent_samples, latent_image.get("downscale_ratio_spacial", None)
    )

    if disable_noise:
        noise = torch.zeros(
            latent_samples.size(),
            dtype=latent_samples.dtype,
            layout=latent_samples.layout,
            device="cpu",
        )
    else:
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)

    noise_mask = latent_image.get("noise_mask", None)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_samples,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_at_step,
        last_step=end_at_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=None,
        disable_pbar=disable_pbar,
        seed=noise_seed,
    )
    out = latent_image.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)
