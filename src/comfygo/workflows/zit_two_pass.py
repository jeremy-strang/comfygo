"""ZImage Turbo two-pass workflow for AuraFlow-based models."""
import os
import random
import time
import traceback
from math import ceil, floor
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from rich.console import Console

from ..config.schema import CheckpointEntry, LoraConfig, PromptsConfig, WorkflowConfig
from ..prompt.prompter import Prompter
from ..utils.console import SmartHighlighter, pad_msg
from ..utils.files import save_images_with_metadata

import logging
log = logging.getLogger(__name__)
console = Console(highlighter=SmartHighlighter(), force_terminal=True, color_system="truecolor")


def _print(obj, **kwargs) -> None:
    """Route strings to the logger and Rich renderables to the console."""
    if isinstance(obj, str):
        log.info(obj)
        return
    console.print(obj, **kwargs)


def _resolve(name: str, base_dir: Optional[Path]) -> str:
    """Resolve a model name to a full filesystem path."""
    if not name:
        return name
    if os.path.isabs(name):
        return name
    if base_dir is not None:
        return str(base_dir / name)
    return name


def run_workflow(config: WorkflowConfig) -> None:
    """Execute the workflow based on configuration."""
    alphabet = string.ascii_letters + string.digits
    runner_id = "".join(secrets.choice(alphabet) for _ in range(length)).upper()

    # Initialise ML environment (no PromptServer, no node registry)
    from ..nodes import setup_comfy_env
    setup_comfy_env(config.paths)

    # Import direct node functions (after env setup so comfy.* is on sys.path)
    from ..nodes.loaders import (
        load_checkpoint, load_clip, load_lora,
        load_lora_model_only, load_unet, load_vae,
    )
    from ..nodes.conditioning import concat_conditioning, encode_text, set_clip_last_layer
    from ..nodes.latent import empty_sd3_latent, vae_decode
    from ..nodes.sampling import ksampler_advanced, model_sampling_auraflow
    from ..nodes.model_patches import patch_model_add_downscale
    from ..nodes.gguf import load_clip_gguf
    from ..nodes.impact import face_detail, load_sam_model, load_ultralytics_detector

    paths = config.paths

    checkpoint_list = config.models.get_checkpoint_list()
    if config.models.shuffle:
        random.shuffle(checkpoint_list)
        _print("Checkpoint order shuffled")

    total_runs = len(checkpoint_list) * config.generation.runs
    _print(
        f"Starting workflow: {len(checkpoint_list)} checkpoint(s) x "
        f"{config.generation.runs} run(s) = {total_runs} total"
    )

    with torch.inference_mode():
        # Detailing checkpoint
        detailing_ckpt = None
        detail_model = None
        detail_clip = None
        if config.detailing:
            _print(f"Loading detailing checkpoint: {config.detailing.checkpoint}")
            detailing_ckpt = load_checkpoint(
                _resolve(config.detailing.checkpoint, paths.get_checkpoints())
            )
            detail_clip = detailing_ckpt[1]

            detail_clip_skip = config.detailing.get_clip_skip()
            if detail_clip_skip < 0:
                detail_clip = set_clip_last_layer(detailing_ckpt[1], detail_clip_skip)[0]

        # CLIP for main generation
        clip_name = config.models.clip.name
        _print(f"Loading CLIP: {clip_name}")

        if clip_name[clip_name.rfind('.'):].lower() == ".gguf":
            clip_result = load_clip_gguf(
                _resolve(clip_name, paths.get_text_encoders()),
                config.models.clip.type,
            )
        else:
            clip_result = load_clip(
                _resolve(clip_name, paths.get_text_encoders()),
                config.models.clip.type,
                config.models.clip.device,
            )

        # VAE
        _print(f"Loading VAE: {config.models.vae}")
        vae_result = load_vae(_resolve(config.models.vae, paths.get_vae()))

        # Detailer providers
        detailer_providers = {}
        if config.detailing and config.detailing.detailers:
            for detailer in config.detailing.detailers:
                _print(
                    f"Loading detailer '{detailer.detector_model}': "
                    f"{detailer.detector_model}, {detailer.sam_model}"
                )
                detailer_providers[detailer.detector_model] = {
                    "bbox": load_ultralytics_detector(
                        _resolve(detailer.detector_model, paths.get_ultralytics())
                    ),
                    "sam": load_sam_model(
                        _resolve(detailer.sam_model, paths.get_sams()),
                        detailer.sam_device,
                    ),
                }

        # Detailing model patches and LoRAs
        if config.detailing:
            ds = config.detailing.downscale
            patched_model = patch_model_add_downscale(
                model=detailing_ckpt[0],
                block_number=ds.block_number,
                downscale_factor=ds.downscale_factor,
                start_percent=ds.start_percent,
                end_percent=ds.end_percent,
                downscale_after_skip=ds.downscale_after_skip,
                downscale_method=ds.downscale_method,
                upscale_method=ds.upscale_method,
            )
            detail_model = patched_model[0]

            for lora in config.detailing.loras:
                strength = lora.get_strength()
                strength_clip = lora.get_strength_clip()
                _print(
                    f"Loading detailing LoRA: {lora.name} "
                    f"(strength={strength}, clip={strength_clip})"
                )
                lora_result = load_lora(
                    detail_model, detail_clip,
                    _resolve(lora.name, paths.get_loras()),
                    strength, strength_clip,
                )
                detail_model = lora_result[0]
                detail_clip = lora_result[1]

        global_run_idx = 0

        # CHECKPOINT LOOP
        for ckpt_idx, checkpoint_entry in enumerate(checkpoint_list):
            try:
                global_run_idx += 1

                log.info(
                    f"Checkpoint {ckpt_idx + 1}/{len(checkpoint_list)}: "
                    f"{checkpoint_entry.checkpoint}"
                )

                unet_result = load_unet(
                    _resolve(checkpoint_entry.checkpoint, paths.get_diffusion_models()),
                    weight_dtype="default",
                )

                effective_loras: List[LoraConfig] = (
                    checkpoint_entry.loras if checkpoint_entry.loras is not None
                    else config.loras
                )
                effective_loras_pass1: List[LoraConfig] = (
                    checkpoint_entry.loras_pass1 if checkpoint_entry.loras_pass1 is not None
                    else config.loras_pass1
                )
                effective_loras_pass2: List[LoraConfig] = (
                    checkpoint_entry.loras_pass2 if checkpoint_entry.loras_pass2 is not None
                    else config.loras_pass2
                )

                model_initial = unet_result[0]

                _print("Applying LoRAs (Both Passes)")
                for lora in effective_loras:
                    strength = lora.get_strength()
                    lora_name = lora.get_name()
                    _print(f"Loading LoRA: {lora_name} (strength={strength})")
                    _print(f'    + Loaded LoRA "{lora_name}" (model: {round(strength, 2)})')
                    lora_result = load_lora_model_only(
                        model_initial,
                        _resolve(lora_name, paths.get_loras()),
                        strength,
                    )
                    model_initial = lora_result[0]

                effective_prompts: PromptsConfig = (
                    checkpoint_entry.prompts if checkpoint_entry.prompts is not None
                    else config.prompts
                )

                _print("Applying LoRAs (First Pass)")
                model_pass1 = model_initial
                for lora in effective_loras_pass1:
                    strength = lora.get_strength()
                    lora_name = lora.get_name()
                    _print(f"Loading LoRA: {lora_name} (strength={strength})")
                    _print(f'    + Loaded LoRA "{lora_name}" (model: {round(strength, 2)})')
                    lora_result = load_lora_model_only(
                        model_pass1,
                        _resolve(lora_name, paths.get_loras()),
                        strength,
                    )
                    model_pass1 = lora_result[0]

                _print("Applying LoRAs (Second Pass)")
                model_pass2 = model_initial
                for lora in effective_loras_pass2:
                    strength = lora.get_strength()
                    lora_name = lora.get_name()
                    _print(f"Loading LoRA: {lora_name} (strength={strength})")
                    _print(f'    + Loaded LoRA "{lora_name}" (model: {round(strength, 2)})')
                    lora_result = load_lora_model_only(
                        model_pass2,
                        _resolve(lora_name, paths.get_loras()),
                        strength,
                    )
                    model_pass2 = lora_result[0]

                seed_for_prompter = config.generation.get_seed()
                prompter = Prompter(effective_prompts, seed=seed_for_prompter)
                _print(f"Prompts:     {prompter.get_num_prompts()} from {effective_prompts.prompts_file}")

                # GENERATION LOOP
                for r in range(config.generation.runs):
                    ti = time.perf_counter()

                    metadata = {
                        "RunID": runner_id,
                        "Set": ckpt_idx,
                        "Run": r,
                        "TotalRuns": global_run_idx,
                        "Checkpoint": checkpoint_entry.checkpoint,
                        "Started": ti,
                    }
                    set_msg = f" Set {ckpt_idx} / {len(checkpoint_list)}, Run {r + 1} / {config.generation.runs} "
                    pchar = "=" if r == 0 else " "
                    padl = pchar * floor((80 - len(set_msg)) / 2)
                    padr = pchar * ceil((80 - len(set_msg)) / 2)
                    _print(f"\n\n{padl}{set_msg}{padr}")
                    if r == 0:
                        _print(f"Checkpoint: {checkpoint_entry.checkpoint}")

                    eff_width, eff_height = checkpoint_entry.get_dimensions(
                        config.generation.dimensions, config.generation.width, config.generation.height
                    )
                    eff_steps = checkpoint_entry.get_steps(config.generation.steps)
                    eff_end_at_step = checkpoint_entry.get_end_at_step(config.generation.end_at_step)
                    eff_cfg = checkpoint_entry.get_cfg(config.generation.cfg)
                    eff_sampler, eff_scheduler = checkpoint_entry.get_sampler_scheduler(
                        config.generation.sampler_scheduler, config.generation.sampler, config.generation.scheduler
                    )
                    eff_shift = checkpoint_entry.get_shift(config.generation.shift)
                    eff_clip_skip = checkpoint_entry.get_clip_skip(config.models.clip_skip)
                    if eff_clip_skip < 0:
                        clip_with_skip = set_clip_last_layer(clip_result[0], eff_clip_skip)
                    else:
                        clip_with_skip = clip_result

                    seed = config.generation.get_seed()
                    prompt, subject = prompter.get_random_prompt()

                    _print(
                        f"Run {global_run_idx}/{total_runs} "
                        f"(checkpoint {ckpt_idx + 1}, run {r + 1}): "
                        f"seed={seed}, {eff_width}x{eff_height}, "
                        f"steps={eff_steps}, end_at_step={eff_end_at_step}, cfg={eff_cfg}"
                    )

                    metadata.update({
                        "Latent": f"{eff_width} x {eff_height}",
                        "Seed": seed, "Steps": eff_steps, "CFG": eff_cfg,
                        "Shift": eff_shift, "Sampler": eff_sampler,
                        "Scheduler": eff_scheduler, "End at Step": eff_end_at_step,
                    })

                    latent_result = empty_sd3_latent(width=eff_width, height=eff_height, batch_size=1)

                    eff_prompt = prompter.substitute(prompt)
                    positive_cond = None
                    negative_cond = None
                    concat_cond = None

                    if detailing_ckpt:
                        positive_cond = encode_text(detailing_ckpt[1], eff_prompt)
                        negative_cond = encode_text(
                            detailing_ckpt[1],
                            config.detailing.prompt_negative if config.detailing else "",
                        )
                        detail_pos_cond = encode_text(
                            detailing_ckpt[1],
                            prompter.substitute(config.detailing.prompt_positive) if config.detailing else "",
                        )
                        concat_cond = concat_conditioning(
                            conditioning_to=detail_pos_cond[0],
                            conditioning_from=positive_cond[0],
                        )

                    eff_prompt = prompter.substitute(prompt)
                    main_positive = encode_text(clip_with_skip[0], eff_prompt)
                    main_negative = encode_text(clip_with_skip[0], prompter.negative_prompt)

                    auraflow_model_pass1 = model_sampling_auraflow(model_pass1, eff_shift)

                    _print(pad_msg(f"First Pass:", f"{eff_sampler} / {eff_scheduler}, CFG: {eff_cfg}, steps: {eff_steps} (end at {eff_end_at_step})"))
                    _print(f'"{prompt}"')
                    first_pass = ksampler_advanced(
                        model=auraflow_model_pass1[0],
                        add_noise="enable",
                        noise_seed=seed,
                        steps=eff_steps,
                        cfg=eff_cfg,
                        sampler_name=eff_sampler,
                        scheduler=eff_scheduler,
                        positive=main_positive[0],
                        negative=main_negative[0],
                        latent_image=latent_result[0],
                        start_at_step=0,
                        end_at_step=eff_end_at_step,
                        return_with_leftover_noise="enable",
                    )

                    auraflow_model_pass2 = model_sampling_auraflow(model_pass2, eff_shift)

                    _print(pad_msg(f"Second Pass:", f"{eff_sampler} / {eff_scheduler}, CFG: {eff_cfg}, steps: {eff_steps} (start at {eff_end_at_step})"))
                    second_pass = ksampler_advanced(
                        model=auraflow_model_pass2[0],
                        add_noise="disable",
                        noise_seed=seed,
                        steps=eff_steps,
                        cfg=eff_cfg,
                        sampler_name=eff_sampler,
                        scheduler=eff_scheduler,
                        positive=main_positive[0],
                        negative=main_negative[0],
                        latent_image=first_pass[0],
                        start_at_step=eff_end_at_step,
                        end_at_step=10000,
                        return_with_leftover_noise="enable",
                    )

                    decoded = vae_decode(vae_result[0], second_pass[0])
                    current_image = decoded[0]

                    if config.detailing and config.detailing.detailers:
                        for detailer_cfg in config.detailing.detailers:
                            providers = detailer_providers[detailer_cfg.detector_model]

                            det_steps = detailer_cfg.get_steps(config.detailing.steps)
                            det_cfg = detailer_cfg.get_cfg(config.detailing.cfg)
                            det_sampler = detailer_cfg.get_sampler(config.detailing.sampler)
                            det_scheduler = detailer_cfg.get_scheduler(config.detailing.scheduler)
                            det_denoise = detailer_cfg.get_denoise(config.detailing.denoise)

                            _print(pad_msg(
                                f"Applying detailer '{detailer_cfg.detector_model}': ",
                                f"{det_sampler} / {det_scheduler}, CFG: {det_cfg}, steps: {det_steps}, denoise: {det_denoise}",
                            ))

                            detail_result = face_detail(
                                image=current_image,
                                model=detail_model,
                                clip=detail_clip,
                                vae=detailing_ckpt[2],
                                positive=concat_cond[0],
                                negative=negative_cond[0],
                                bbox_detector=providers["bbox"][0],
                                sam_model_opt=providers["sam"][0],
                                guide_size=detailer_cfg.guide_size,
                                guide_size_for=detailer_cfg.guide_size_for,
                                max_size=detailer_cfg.max_size,
                                seed=seed,
                                steps=det_steps,
                                cfg=det_cfg,
                                sampler_name=det_sampler,
                                scheduler=det_scheduler,
                                denoise=det_denoise,
                                feather=detailer_cfg.feather,
                                noise_mask=detailer_cfg.noise_mask,
                                force_inpaint=detailer_cfg.force_inpaint,
                                bbox_threshold=detailer_cfg.bbox_threshold,
                                bbox_dilation=detailer_cfg.bbox_dilation,
                                bbox_crop_factor=detailer_cfg.bbox_crop_factor,
                                sam_detection_hint=detailer_cfg.sam_detection_hint,
                                sam_dilation=detailer_cfg.sam_dilation,
                                sam_threshold=detailer_cfg.sam_threshold,
                                sam_bbox_expansion=detailer_cfg.sam_bbox_expansion,
                                sam_mask_hint_threshold=detailer_cfg.sam_mask_hint_threshold,
                                sam_mask_hint_use_negative=detailer_cfg.sam_mask_hint_use_negative,
                                drop_size=detailer_cfg.drop_size,
                                wildcard=detailer_cfg.wildcard,
                                cycle=detailer_cfg.cycle,
                                inpaint_model=detailer_cfg.inpaint_model,
                                noise_mask_feather=detailer_cfg.noise_mask_feather,
                                tiled_encode=detailer_cfg.tiled_encode,
                                tiled_decode=detailer_cfg.tiled_decode,
                            )
                            current_image = detail_result[0]

                    tf = time.perf_counter()
                    metadata.update({
                        "Elapsed": tf - ti,
                        "Seed": seed,
                        "Checkpoint": checkpoint_entry.checkpoint,
                        "Steps": eff_steps,
                        "EndAtStep": eff_end_at_step,
                        "CFG": eff_cfg,
                        "Sampler": eff_sampler,
                        "Scheduler": eff_scheduler,
                        "Shift": eff_shift,
                        "Latent": f"{round(eff_width)}x{round(eff_height)}",
                        "Loras": "[" + (
                            ", ".join(f"{l.name} ({l.get_strength()})" for l in effective_loras)
                            if effective_loras else ""
                        ) + "]",
                        "LorasPass1": "[" + (
                            ", ".join(f"{l.name} ({l.get_strength()})" for l in effective_loras_pass1)
                            if effective_loras_pass1 else ""
                        ) + "]",
                        "LorasPass2": "[" + (
                            ", ".join(f"{l.name} ({l.get_strength()})" for l in effective_loras_pass2)
                            if effective_loras_pass2 else ""
                        ) + "]",
                        "Prompt": prompt,
                        "PromptTokens": Prompter.count_tokens(prompt),
                        "Subject": subject,
                        "SubjectTokens": Prompter.count_tokens(subject),
                        "Negative": prompter.negative_prompt,
                        "NegativeTokens": Prompter.count_tokens(prompter.negative_prompt),
                    })

                    ckpt_id = checkpoint_entry.get_id()
                    filename_code = os.path.basename(
                        checkpoint_entry.checkpoint.replace(".safetensors", "")[0:100]
                    )

                    save_images_with_metadata(
                        current_image,
                        config.output.path,
                        filename_prefix=f"{ckpt_id}_{runner_id}_S{ckpt_idx}R{r}",
                        metadata=None,
                        extension="png",
                        dpi=300,
                        quality=90,
                        optimize=True,
                        number_padding=2,
                    )

                    _print(f"Saved image: {config.output.path}")
            except Exception as e:
                _print(f"Exception loading face detailer: {e}")
                traceback.print_exc(e)

    _print("Workflow complete")
