"""Conditioning helpers — replacing ComfyUI node wrappers."""
import logging

import torch

log = logging.getLogger(__name__)


def encode_text(clip, text: str):
    """Encode *text* with *clip* into a conditioning tensor.

    Args:
        clip: ComfyUI CLIP model object.
        text: Prompt string.

    Returns:
        Tuple ``(conditioning,)`` — a list of ``[tensor, dict]`` pairs.
    """
    if clip is None:
        raise RuntimeError(
            "encode_text: clip is None — check that the CLIP model loaded correctly."
        )
    tokens = clip.tokenize(text)
    return (clip.encode_from_tokens_scheduled(tokens),)


def set_clip_last_layer(clip, stop_at_clip_layer: int):
    """Clone *clip* and set the last active CLIP layer.

    Args:
        clip: ComfyUI CLIP model object.
        stop_at_clip_layer: Negative integer (e.g. ``-2``).

    Returns:
        Tuple ``(clip,)`` with the cloned, layer-limited CLIP.
    """
    clip = clip.clone()
    clip.clip_layer(stop_at_clip_layer)
    return (clip,)


def concat_conditioning(conditioning_to, conditioning_from):
    """Concatenate two conditioning tensors along dim=1.

    Matches ComfyUI's ConditioningConcat node behaviour exactly.

    Args:
        conditioning_to: Primary conditioning (list of ``[tensor, dict]``).
        conditioning_from: Conditioning to append.

    Returns:
        Tuple ``(conditioning,)`` with concatenated result.
    """
    import logging as _logging
    if len(conditioning_from) > 1:
        _logging.getLogger(__name__).warning(
            "concat_conditioning: conditioning_from has >1 entry; "
            "only the first will be used."
        )
    cond_from = conditioning_from[0][0]
    out = []
    for t1, meta in conditioning_to:
        tw = torch.cat((t1, cond_from), dim=1)
        out.append([tw, meta.copy()])
    return (out,)
