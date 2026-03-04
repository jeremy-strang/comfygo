"""Impact Pack wrappers — direct function calls without ComfyUI node registry.

Requires ``setup_comfy_env()`` to have been called first (adds Impact Pack's
``modules/`` directory to sys.path so ``import impact.*`` resolves correctly).
"""
import logging
import os

log = logging.getLogger(__name__)


def _model_kind_from_name(model_path: str) -> str:
    """Infer SAM model kind from filename."""
    name = os.path.basename(model_path).lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    return "vit_b"


def load_sam_model(path: str, device_mode: str = "AUTO"):
    """Load a SAM (Segment Anything) model from a full filesystem path.

    Args:
        path:        Absolute path to the SAM ``.pth`` / ``.pt`` file.
        device_mode: ``"AUTO"``, ``"Prefer GPU"``, or ``"CPU"``.

    Returns:
        Tuple ``(sam,)``.
    """
    from segment_anything import sam_model_registry
    import comfy.model_management
    from impact import core

    model_kind = _model_kind_from_name(path)
    sam = sam_model_registry[model_kind](checkpoint=path)

    size = os.path.getsize(path)
    safe_to = core.SafeToGPU(size)

    if device_mode == "Prefer GPU":
        device = comfy.model_management.get_torch_device()
        safe_to.to_device(sam, device)

    is_auto_mode = device_mode == "AUTO"
    sam_obj = core.SAMWrapper(sam, is_auto_mode=is_auto_mode, safe_to_gpu=safe_to)
    sam.sam_wrapper = sam_obj

    log.info("Loaded SAM model: %s (device_mode=%s)", path, device_mode)
    return (sam,)


def load_ultralytics_detector(path: str):
    """Load a YOLO/Ultralytics detection model from a full filesystem path.

    Args:
        path: Absolute path to the YOLO ``.pt`` model file.

    Returns:
        Tuple ``(bbox_detector, segm_detector)`` where *segm_detector* is a
        ``NO_SEGM_DETECTOR`` placeholder.
    """
    from .subcore import load_yolo, UltraBBoxDetector, NO_SEGM_DETECTOR

    model = load_yolo(path)
    log.info("Loaded YOLO model: %s", path)
    return (UltraBBoxDetector(model), NO_SEGM_DETECTOR())


def face_detail(
    image,
    model,
    clip,
    vae,
    positive,
    negative,
    bbox_detector,
    sam_model_opt=None,
    **kwargs,
):
    """Run FaceDetailer on *image*.

    Thin wrapper around ``impact.impact_pack.FaceDetailer.doit()``.  All
    FaceDetailer keyword arguments are forwarded via ``**kwargs``.

    Args:
        image:         Input image tensor ``[B, H, W, C]``.
        model:         Diffusion model patcher.
        clip:          CLIP model.
        vae:           VAE model.
        positive:      Positive conditioning.
        negative:      Negative conditioning.
        bbox_detector: Bounding-box detector (e.g. ``UltraBBoxDetector``).
        sam_model_opt: Optional SAM model for mask refinement.
        **kwargs:      Additional FaceDetailer parameters (guide_size, steps,
                       cfg, sampler_name, …).

    Returns:
        Tuple ``(enhanced_image, cropped_enhanced, cropped_enhanced_alpha,
        mask, detailer_pipe, cnet_images)``.
        Index ``[0]`` gives the final enhanced image tensor.
    """
    from impact.impact_pack import FaceDetailer

    fd = FaceDetailer()
    return fd.doit(
        image=image,
        model=model,
        clip=clip,
        vae=vae,
        positive=positive,
        negative=negative,
        bbox_detector=bbox_detector,
        sam_model_opt=sam_model_opt,
        **kwargs,
    )
