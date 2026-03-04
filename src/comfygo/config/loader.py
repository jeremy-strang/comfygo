"""YAML configuration loading, validation, and default generation."""

import logging
import os
from typing import Any, Dict, Optional

import yaml

from .schema import CheckpointEntry, PathsConfig, WorkflowConfig

log = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


def load_checkpoint_config(config_path: str) -> Dict[str, Any]:
    """Load a checkpoint-specific configuration file.

    Used when a CheckpointEntry specifies config_file.
    Only generation-related settings are extracted.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with applicable settings (ignores nested checkpoints).

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Checkpoint config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Extract only applicable fields (not nested checkpoints or models list)
    result = {}

    # Fields to extract (including paired properties)
    gen_fields = [
        "steps", "cfg", "sampler", "scheduler", "shift", "width", "height",
        "sampler_scheduler", "dimensions",
    ]

    # Generation overrides
    if "generation" in data:
        gen = data["generation"]
        for field in gen_fields:
            if field in gen:
                result[field] = gen[field]

    # Direct overrides (at top level)
    for field in gen_fields:
        if field in data:
            result[field] = data[field]

    # Prompts override
    if "prompts" in data:
        result["prompts"] = data["prompts"]

    # Loras override
    if "loras" in data:
        result["loras"] = data["loras"]

    return result


def _resolve_checkpoint_config_files(config: WorkflowConfig, base_dir: str) -> None:
    """Resolve config_file references in checkpoint entries.

    Loads external config files and merges values into checkpoint entries.
    Inline overrides take precedence over config_file values.

    Args:
        config: WorkflowConfig to process.
        base_dir: Base directory for resolving relative config_file paths.
    """
    from .randomizable import (
        parse_dimensions,
        parse_randomizable,
        parse_randomizable_paired,
        parse_sampler_scheduler,
    )
    from .schema import LoraConfig, PromptsConfig

    for entry in config.models.checkpoints:
        if not entry.config_file:
            continue

        # Resolve relative paths
        config_file_path = entry.config_file
        if not os.path.isabs(config_file_path):
            config_file_path = os.path.join(base_dir, config_file_path)

        # Load external config
        external = load_checkpoint_config(config_file_path)

        # Merge values (external values only used if inline not set)
        if entry.steps is None and "steps" in external:
            entry.steps = parse_randomizable(external["steps"], int)
        if entry.cfg is None and "cfg" in external:
            entry.cfg = parse_randomizable(external["cfg"], float)
        if entry.sampler is None and "sampler" in external:
            entry.sampler = parse_randomizable(external["sampler"])
        if entry.scheduler is None and "scheduler" in external:
            entry.scheduler = parse_randomizable(external["scheduler"])
        if entry.shift is None and "shift" in external:
            entry.shift = parse_randomizable(external["shift"], float)
        if entry.width is None and "width" in external:
            entry.width = parse_randomizable(external["width"], int)
        if entry.height is None and "height" in external:
            entry.height = parse_randomizable(external["height"], int)

        # Paired properties
        if entry.sampler_scheduler is None and "sampler_scheduler" in external:
            entry.sampler_scheduler = parse_randomizable_paired(
                external["sampler_scheduler"], parse_sampler_scheduler
            )
        if entry.dimensions is None and "dimensions" in external:
            entry.dimensions = parse_randomizable_paired(
                external["dimensions"], parse_dimensions
            )

        if entry.prompts is None and "prompts" in external:
            entry.prompts = PromptsConfig.from_dict(external["prompts"])
        if entry.loras is None and "loras" in external:
            entry.loras = [LoraConfig.from_dict(l) for l in external["loras"]]


def load_config(config_path: str) -> WorkflowConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated WorkflowConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ConfigValidationError: If config is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigValidationError("Config must be a YAML dictionary")

    config = WorkflowConfig.from_dict(data)

    # Resolve config_file references in checkpoint entries
    base_dir = os.path.dirname(os.path.abspath(config_path))
    _resolve_checkpoint_config_files(config, base_dir)

    validate_config(config)
    return config


def validate_config(config: WorkflowConfig) -> None:
    """Validate the loaded configuration.

    Args:
        config: WorkflowConfig to validate.

    Raises:
        ConfigValidationError: If validation fails.
    """
    errors = []
    warnings = []

    # Checkpoint validation - need at least one checkpoint
    checkpoint_list = config.models.get_checkpoint_list()
    if not checkpoint_list:
        errors.append(
            "At least one checkpoint is required: set models.checkpoint or models.checkpoints"
        )

    # Warn if both formats are set
    if config.models.checkpoint and config.models.checkpoints:
        warnings.append(
            "Both models.checkpoint and models.checkpoints are set; "
            "using models.checkpoints (models.checkpoint is ignored)"
        )

    # Validate each checkpoint entry
    for i, entry in enumerate(checkpoint_list):
        if not entry.checkpoint:
            errors.append(f"checkpoints[{i}].checkpoint is required")

        # Validate config_file if specified
        if entry.config_file and not os.path.exists(entry.config_file):
            errors.append(
                f"checkpoints[{i}].config_file not found: {entry.config_file}"
            )

    # Prompt source validation
    if config.prompts.prompts_file is None and config.prompts.prompt_override is None:
        errors.append("Either prompts.prompts_file or prompts.prompt_override must be set")

    if config.prompts.prompts_file and not os.path.exists(config.prompts.prompts_file):
        errors.append(f"prompts_file not found: {config.prompts.prompts_file}")

    # Subject source validation
    if config.prompts.subjects_file and not os.path.exists(config.prompts.subjects_file):
        errors.append(f"subjects_file not found: {config.prompts.subjects_file}")

    # Detailing validation
    if config.detailing:
        if not config.detailing.checkpoint:
            errors.append("detailing.checkpoint is required when detailing is configured")

        if not config.detailing.detailers:
            errors.append("detailing.detailers must have at least one detailer configured")

    # Log warnings
    for warning in warnings:
        log.warning(warning)

    if errors:
        raise ConfigValidationError("\n".join(errors))


def generate_default_config() -> str:
    """Generate default config.yaml content matching runner_ref.py values.

    Returns:
        YAML string with default configuration.
    """
    return '''# ComfyGo Workflow Configuration
# Generated from runner_ref.py default values



paths:
  base: ~/models

output:
  path: ~/Pictures

logging:
  level: INFO
  files:
    info: { path: .logs/runner_info.log, level: INFO }
    error: { path: .logs/runner_errors.log, level: ERROR }

loras: []
loras_pass1: []
loras_pass2: []

prompts:
  prompts_file: configs/prompts.txt
  subjects_file: configs/subjects.txt
  prefix:
  suffix: correct human anatomy, no text, no watermark
  negative: ""
  tags: {}
  substitutions:
  - { search: "some text", replace: "", reversible: false }

detailing:
  checkpoint: stabilityai/stable-diffusion-xl-base-1.0.safetensors
  prompt_positive: realistic textures
  prompt_negative: ''
  steps: 8
  cfg: 1.0
  sampler: lcm
  scheduler: exponential
  denoise: 0.3
  clip_skip: 0
  loras:
  - { name: dmd2_sdxl_4step_lora_fp16.safetensors, strength: 0.75 }
  detailers:
  - detector_model: bbox/face.pt
    denoise: 0.25
    steps: 6
    prompt_positive: perfect teeth, realistic skin
    bbox_crop_factor: 2.5
    bbox_dilation: 20


paths:
  # Set `base` to resolve all relative model names from a single root directory.
  # Individual overrides take precedence over the base-derived path.
  # base: "/models"

  # Or set directories per model type explicitly:
  # checkpoints: "/models/checkpoints"
  # diffusion_models: "/models/diffusion_models"
  # text_encoders: "/models/text_encoders"
  # vae: "/models/vae"
  # loras: "/models/loras"
  # embeddings: "/models/embeddings"
  # sams: "/models/sams"
  # ultralytics: "/models/ultralytics"
  # impact_pack: "/opt/ComfyUI-Impact-Pack/modules"

output:
  path: ~/Pictures

generation:
  runs: 10
  dimensions: [1024x1024, 1152x896, 1216x832, 832x1216, 896x1152]
  steps: 10
  cfg: 1.0
  sampler_scheduler: [euler/simple, res_multistep/beta]
  shift: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
  seed_override: null  # Set to integer for reproducible results (e.g., 22)
  shift: 3.0
  clip_skip: -2

models:
  vae: "ae.safetensors"
  clip:
    name: "qwen_3_4b.safetensors"
    type: "lumina2"
    device: "gpu"

  checkpoints:
  - checkpoint: Tongyi-MAI/Z-Image-Turbo.safetensors


loras: []
loras_pass1: []
loras_pass2: []

prompts:
  # Source for main prompts - set ONE of these:
  prompts_file: null  # Path to .txt file with one prompt per line
  prompt_override: null  # Direct prompt string (overrides prompts_file)

  # Composition options
  prefix: ""  # Prepended to each prompt
  suffix: ""  # Appended to each prompt

  # Negative prompt
  negative: "text, watermark, cum, water"

  # Tagged prompt composition - define tags inline, reference with {tag_name}
  # Example: "A {style} photo of {subject}, {lighting}"
  tags: {}
  # tags:
  #   style: ["realistic", "photographic", "cinematic"]
  #   subject: ["a woman", "a man"]
  #   lighting: ["natural light", "studio lighting"]

detailing:
  checkpoint: stabilityai/stable-diffusion-xl-base-1.0.safetensors
  prompt_positive: realistic textures
  prompt_negative: ""

  loras:
  - name: dmd2_sdxl_4step_lora_fp16.safetensors
    strength: 1.0

  downscale:
    block_number: 3
    downscale_factor: 2
    start_percent: 0.0
    end_percent: 0.35
    downscale_after_skip: true
    downscale_method: "bicubic"
    upscale_method: "bicubic"

  # Default detailer settings (can be overridden per-detailer)
  steps: 8
  cfg: 1.0
  sampler: "lcm"
  scheduler:"exponential
  denoise: 0.30
  clip_skip: 0

  detailers:
    - detector_model: "bbox/face.pt"
      denoise: 0.25
      steps: 6
      prompt_positive: perfect teeth, realistic skin
      bbox_crop_factor: 2.5
      bbox_dilation: 20
      sam_model: "sam_vit_b_01ec64.pth"
      sam_device: "Prefer GPU"
      guide_size: 368
      guide_size_for: true
      max_size: 768
      feather: 5
      noise_mask: true
      force_inpaint: true
      bbox_threshold: 0.5
      bbox_dilation: 10
      bbox_crop_factor: 1.9
      sam_detection_hint: "center-1"
      sam_dilation: 0
      sam_threshold: 0.8
      sam_bbox_expansion: 0
      sam_mask_hint_threshold: 0.7
      sam_mask_hint_use_negative: "False"
      drop_size: 10
      wildcard: ""
      cycle: 1
      inpaint_model: false
      noise_mask_feather: 20
      tiled_encode: false
      tiled_decode: false
'''
