"""Model patch helpers — replacing ComfyUI node wrappers."""
import logging

log = logging.getLogger(__name__)


def patch_model_add_downscale(
    model,
    block_number: int,
    downscale_factor: float,
    start_percent: float,
    end_percent: float,
    downscale_after_skip: bool,
    downscale_method: str,
    upscale_method: str,
):
    """Add Kohya Deep Shrink (downscale + upscale patch) to *model*.

    Mirrors ComfyUI's ``PatchModelAddDownscale.execute()``
    (comfy_extras/nodes_model_downscale.py).

    Args:
        model:                Diffusion model patcher to clone and patch.
        block_number:         U-Net input block index at which to downscale.
        downscale_factor:     Scale factor for downscaling (e.g. 2.0 = half size).
        start_percent:        Sigma percentage at which patching begins.
        end_percent:          Sigma percentage at which patching ends.
        downscale_after_skip: Whether to apply the patch after skip connections.
        downscale_method:     Interpolation method for downscaling.
        upscale_method:       Interpolation method for upscaling in output block.

    Returns:
        Tuple ``(patched_model,)``.
    """
    import comfy.utils

    model_sampling = model.get_model_object("model_sampling")
    sigma_start = model_sampling.percent_to_sigma(start_percent)
    sigma_end = model_sampling.percent_to_sigma(end_percent)

    def input_block_patch(h, transformer_options):
        if transformer_options["block"][1] == block_number:
            sigma = transformer_options["sigmas"][0].item()
            if sigma_end <= sigma <= sigma_start:
                h = comfy.utils.common_upscale(
                    h,
                    round(h.shape[-1] * (1.0 / downscale_factor)),
                    round(h.shape[-2] * (1.0 / downscale_factor)),
                    downscale_method,
                    "disabled",
                )
        return h

    def output_block_patch(h, hsp, transformer_options):
        if h.shape[2] != hsp.shape[2]:
            h = comfy.utils.common_upscale(
                h, hsp.shape[-1], hsp.shape[-2], upscale_method, "disabled"
            )
        return h, hsp

    m = model.clone()
    if downscale_after_skip:
        m.set_model_input_block_patch_after_skip(input_block_patch)
    else:
        m.set_model_input_block_patch(input_block_patch)
    m.set_model_output_block_patch(output_block_patch)
    return (m,)
