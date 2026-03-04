"""Configuration schema dataclasses for ComfyGo workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from hashlib import sha256
from pathlib import Path
import random

from .randomizable import (
    parse_dimensions,
    parse_randomizable,
    parse_randomizable_paired,
    parse_sampler_scheduler,
    resolve_optional,
    resolve_paired,
    resolve_value,
)


T = TypeVar("T")
def dataclass_from_dict(cls: Type[T], data: Dict[str, Any], *, required: Set[str] | None = None) -> T:
    """Create dataclass instance from dict, ignoring unknown keys.

    If `required` is provided, raise ValueError when any required key is missing.
    """
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}

    if required:
        missing = [k for k in required if k not in filtered]
        if missing:
            raise ValueError(f"{cls.__name__}: missing required field(s): {', '.join(missing)}")

    return cls(**filtered)


@dataclass
class LoraConfig:
    """Configuration for a single LoRA.

    Both strength and strength_clip support randomization via lists.
    """

    name: Union[str, List[str]] = ""
    strength: Union[float, List[float]] = 1.0
    strength_clip: Optional[Union[float, List[float]]] = None  # Defaults to strength

    def get_name(self) -> str:
        """Get name, resolving random selection if list."""
        return resolve_value(self.name)

    def get_strength(self) -> float:
        """Get strength, resolving random selection if list."""
        return resolve_value(self.strength)

    def get_strength_clip(self) -> float:
        """Get strength_clip, resolving random selection if list."""
        val = self.strength_clip if self.strength_clip is not None else self.strength
        return resolve_value(val)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> LoraConfig:
        return LoraConfig(
            name=parse_randomizable(data.get("name"), str),
            strength=parse_randomizable(data.get("strength", 1.0), float),
            strength_clip=parse_randomizable(data.get("strength_clip"), float),
        )


@dataclass
class ClipConfig:
    """Configuration for CLIP model."""

    name: str = "qwen_3_4b.safetensors"
    type: str = "lumina2"
    device: str = "gpu"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClipConfig":
        return dataclass_from_dict(ClipConfig, data)


@dataclass
class ModelsConfig:
    """Configuration for models (UNET, VAE, CLIP).

    Supports multiple checkpoints via the checkpoints list.
    For backward compatibility, single checkpoint field is also supported.
    """

    checkpoint: str = ""
    vae: str = "ae.safetensors"
    clip: ClipConfig = field(default_factory=ClipConfig)
    clip_skip: Union[int, List[int]] = -2
    shuffle: bool = False  # If True, randomize checkpoint order
    checkpoints: List[CheckpointEntry] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelsConfig":
        data = dict(data)  # copy so we can safely mutate

        clip_data = data.get("clip")
        if isinstance(clip_data, dict):
            data["clip"] = ClipConfig.from_dict(clip_data)
        else:
            data["clip"] = ClipConfig()

        cps = data.get("checkpoints")
        if isinstance(cps, list):
            data["checkpoints"] = [
                CheckpointEntry.from_dict(c) if isinstance(c, dict) else c
                for c in cps
            ]
        else:
            data.pop("checkpoints", None)

        if "clip_skip" in data:
            data["clip_skip"] = parse_randomizable(data["clip_skip"], int)

        if "shuffle" in data:
            data["shuffle"] = bool(data["shuffle"])
        return dataclass_from_dict(ModelsConfig, data)

    def get_clip_skip(self) -> int:
        """Get clip_skip, resolving random selection if list."""
        return -1 * abs(resolve_value(self.clip_skip))

    def get_checkpoint_list(self) -> List[CheckpointEntry]:
        """Return list of checkpoints (handles backward compat)."""
        if self.checkpoints:
            return self.checkpoints
        elif self.checkpoint:
            return [CheckpointEntry(checkpoint=self.checkpoint)]
        return []


@dataclass
class SubstitutionConfig:
    """Configuration for substituting text."""
    search: str
    replace: str = ""
    reverse: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SubstitutionConfig":
        return dataclass_from_dict(SubstitutionConfig, data, required={"search"})


@dataclass
class PromptsConfig:
    """Configuration for prompts including randomization and tags."""

    prompts_file: Optional[str] = None
    subjects_file: Optional[str] = None
    prompt_override: Optional[str] = None
    prefix: str = ""
    suffix: str = ""
    negative: str = ""
    tags: Dict[str, List[str]] = field(default_factory=dict)
    substitutions: Optional[List[SubstitutionConfig]] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptsConfig":
        data = dict(data)  # copy so we can mutate safely
        subs = data.get("substitutions")
        if isinstance(subs, list):
            data["substitutions"] = [
                SubstitutionConfig.from_dict(s) if isinstance(s, dict) else s
                for s in subs
            ]
        else:
            data["substitutions"] = []

        # Normalize tags -> dict (otherwise default_factory applies)
        if "tags" in data and not isinstance(data["tags"], dict):
            data.pop("tags", None)

        return dataclass_from_dict(PromptsConfig, data)


@dataclass
class CheckpointEntry:
    """A checkpoint with optional per-checkpoint overrides.

    Each checkpoint can override generation parameters, prompts, and loras.
    Supports randomization via lists for numeric/string parameters.
    Paired properties (sampler_scheduler, dimensions) also supported.
    """
    checkpoint: str
    config_file: Optional[str] = None  # Path to external config file

    # Override generation params (any can be randomizable)
    steps: Optional[Union[int, List[int]]] = None
    end_at_step: Optional[Union[int, List[int]]] = None
    cfg: Optional[Union[float, List[float]]] = None
    sampler: Optional[Union[str, List[str]]] = None
    scheduler: Optional[Union[str, List[str]]] = None
    shift: Optional[Union[float, List[float]]] = None
    width: Optional[Union[int, List[int]]] = None
    height: Optional[Union[int, List[int]]] = None
    clip_skip: Optional[Union[int, List[int]]] = None
    _id: str = ""

    # Paired properties
    sampler_scheduler: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None
    dimensions: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None

    # Override prompts config
    prompts: Optional[PromptsConfig] = None

    # Override loras (replaces main loras list for this checkpoint)
    loras: Optional[List[LoraConfig]] = None
    loras_pass1: Optional[List[LoraConfig]] = None
    loras_pass2: Optional[List[LoraConfig]] = None

    def get_id(self) -> str:
        if self._id:
            return self._id
        self._id = sha256(self.checkpoint.encode('utf-8')).hexdigest()[0:8].upper()
        return self._id

    def get_steps(self, fallback: Union[int, List[int]]) -> int:
        """Get steps, using fallback if not overridden."""
        return resolve_optional(self.steps, fallback)

    def get_end_at_step(self, fallback: Union[int, List[int]]) -> int:
        """Get end_at_step, using fallback if not overridden."""
        return resolve_optional(self.end_at_step, fallback)

    def get_cfg(self, fallback: Union[float, List[float]]) -> float:
        """Get cfg, using fallback if not overridden."""
        return resolve_optional(self.cfg, fallback)

    def get_clip_skip(self, fallback: Union[int, List[int]]) -> int:
        """Get clip_skip, using fallback if not overridden."""
        return -1 * abs(resolve_optional(self.clip_skip, fallback))

    def get_sampler_scheduler(
        self,
        fallback_paired: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]],
        fallback_individual_sampler: Union[str, List[str]],
        fallback_individual_scheduler: Union[str, List[str]],
    ) -> Tuple[str, str]:
        """Get (sampler, scheduler) atomically — one random.choice call for pairs.

        Priority: checkpoint individual > checkpoint paired > main paired > main individual.
        When only one individual is set, the other comes from a paired source.
        """
        if self.sampler is not None and self.scheduler is not None:
            return (resolve_value(self.sampler), resolve_value(self.scheduler))
        if self.sampler is not None:
            paired = resolve_paired(self.sampler_scheduler) or resolve_paired(fallback_paired)
            scheduler = paired[1] if paired else resolve_value(fallback_individual_scheduler)
            return (resolve_value(self.sampler), scheduler)
        if self.scheduler is not None:
            paired = resolve_paired(self.sampler_scheduler) or resolve_paired(fallback_paired)
            sampler = paired[0] if paired else resolve_value(fallback_individual_sampler)
            return (sampler, resolve_value(self.scheduler))
        paired = resolve_paired(self.sampler_scheduler)
        if paired:
            return paired
        main_paired = resolve_paired(fallback_paired)
        if main_paired:
            return main_paired
        return (resolve_value(fallback_individual_sampler), resolve_value(fallback_individual_scheduler))

    def get_shift(self, fallback: Union[float, List[float]]) -> float:
        """Get shift, using fallback if not overridden."""
        return resolve_optional(self.shift, fallback)

    def get_dimensions(
        self,
        fallback_paired: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]],
        fallback_individual_width: Union[int, List[int]],
        fallback_individual_height: Union[int, List[int]],
    ) -> Tuple[int, int]:
        """Get (width, height) atomically — one random.choice call for pairs.

        Priority: checkpoint individual > checkpoint paired > main paired > main individual.
        When only one individual is set, the other comes from a paired source.
        """
        if self.width is not None and self.height is not None:
            return (resolve_value(self.width), resolve_value(self.height))
        if self.width is not None:
            paired = resolve_paired(self.dimensions) or resolve_paired(fallback_paired)
            height = paired[1] if paired else resolve_value(fallback_individual_height)
            return (resolve_value(self.width), height)
        if self.height is not None:
            paired = resolve_paired(self.dimensions) or resolve_paired(fallback_paired)
            width = paired[0] if paired else resolve_value(fallback_individual_width)
            return (width, resolve_value(self.height))
        paired = resolve_paired(self.dimensions)
        if paired:
            return paired
        main_paired = resolve_paired(fallback_paired)
        if main_paired:
            return main_paired
        return (resolve_value(fallback_individual_width), resolve_value(fallback_individual_height))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CheckpointEntry:
        # Parse prompts override if present
        prompts = None
        if "prompts" in data:
            prompts = PromptsConfig.from_dict(data["prompts"])

        # Parse loras override if present
        loras = None
        if "loras" in data:
            loras = [LoraConfig.from_dict(l) for l in data["loras"]]

        loras_pass1 = None
        if "loras_pass1" in data:
            loras_pass1 = [LoraConfig.from_dict(l) for l in data["loras_pass1"]]

        loras_pass2 = None
        if "loras_pass2" in data:
            loras_pass2 = [LoraConfig.from_dict(l) for l in data["loras_pass2"]]

        checkpoint = data.get("checkpoint")
        return CheckpointEntry(
            checkpoint=checkpoint,
            config_file=data.get("config_file"),
            steps=parse_randomizable(data.get("steps"), int),
            end_at_step=parse_randomizable(data.get("end_at_step"), int),
            cfg=parse_randomizable(data.get("cfg"), float),
            sampler=parse_randomizable(data.get("sampler")),
            scheduler=parse_randomizable(data.get("scheduler")),
            shift=parse_randomizable(data.get("shift"), float),
            width=parse_randomizable(data.get("width"), int),
            height=parse_randomizable(data.get("height"), int),
            clip_skip=parse_randomizable(data.get("clip_skip"), int),
            sampler_scheduler=parse_randomizable_paired(
                data.get("sampler_scheduler"), parse_sampler_scheduler
            ),
            dimensions=parse_randomizable_paired(
                data.get("dimensions"), parse_dimensions
            ),
            prompts=prompts,
            loras=loras,
            loras_pass1=loras_pass1,
            loras_pass2=loras_pass2,
        )


@dataclass
class GenerationConfig:
    """Configuration for generation parameters.

    Most fields support randomization via lists (except runs, seed_override).
    Paired properties (sampler_scheduler, dimensions) take precedence over individual.
    """

    runs: int = 10
    width: Union[int, List[int]] = 832
    height: Union[int, List[int]] = 1216
    steps: Union[int, List[int]] = 10
    end_at_step: Union[int, List[int]] = 3
    cfg: Union[float, List[float]] = 1.0
    sampler: Union[str, List[str]] = "euler"
    scheduler: Union[str, List[str]] = "simple"
    seed_override: Optional[int] = None
    shift: Union[float, List[float]] = 3.0

    # Paired properties (take precedence over individual)
    sampler_scheduler: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None
    dimensions: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None

    def get_seed(self) -> int:
        """Get seed (override or random)."""
        if self.seed_override is not None:
            return self.seed_override
        return random.randint(1, 2**64)

    def get_dimensions(self) -> Tuple[int, int]:
        """Get (width, height) atomically — one random.choice call for pairs."""
        paired = resolve_paired(self.dimensions)
        if paired:
            return paired
        return (resolve_value(self.width), resolve_value(self.height))

    def get_steps(self) -> int:
        """Get steps, resolving random selection if list."""
        return resolve_value(self.steps)

    def get_end_at_step(self) -> int:
        """Get end_at_step, resolving random selection if list."""
        return resolve_value(self.end_at_step)

    def get_cfg(self) -> float:
        """Get cfg, resolving random selection if list."""
        return resolve_value(self.cfg)

    def get_sampler_scheduler(self) -> Tuple[str, str]:
        """Get (sampler, scheduler) atomically — one random.choice call for pairs."""
        paired = resolve_paired(self.sampler_scheduler)
        if paired:
            return paired
        return (resolve_value(self.sampler), resolve_value(self.scheduler))

    def get_shift(self) -> float:
        """Get shift, resolving random selection if list."""
        return resolve_value(self.shift)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GenerationConfig":
        return GenerationConfig(
            runs=int(data.get("runs", 1)),
            width=parse_randomizable(data.get("width", 832), int),
            height=parse_randomizable(data.get("height", 1216), int),
            steps=parse_randomizable(data.get("steps", 10), int),
            end_at_step=parse_randomizable(data.get("end_at_step", 3), int),
            cfg=parse_randomizable(data.get("cfg", 1.0), float),
            sampler=parse_randomizable(data.get("sampler", "euler")),
            scheduler=parse_randomizable(data.get("scheduler", "simple")),
            seed_override=(
                int(data["seed_override"])
                if data.get("seed_override") is not None
                else None
            ),
            shift=parse_randomizable(data.get("shift", 3.0), float),
            sampler_scheduler=parse_randomizable_paired(
                data.get("sampler_scheduler"), parse_sampler_scheduler
            ),
            dimensions=parse_randomizable_paired(
                data.get("dimensions"), parse_dimensions
            ),
        )


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    path: str = "ComfyUI"  # filename_prefix for SaveImage
    encrypt: bool = False
    private_key: Optional[str] = None  # Path to external config file
    public_key: Optional[str] = None  # Path to external config file

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> OutputConfig:
        return OutputConfig(
            path=data.get("path", "ComfyUI"),
            encrypt=data.get("private_key", False),
            private_key=data.get("private_key"),
            public_key=data.get("public_key"),
        )


@dataclass
class DownscaleConfig:
    """Configuration for PatchModelAddDownscale."""

    block_number: int = 3
    downscale_factor: float = 2.0
    start_percent: float = 0.0
    end_percent: float = 0.35
    downscale_after_skip: bool = True
    downscale_method: str = "bicubic"
    upscale_method: str = "bicubic"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DownscaleConfig:
        return DownscaleConfig(
            block_number=int(data.get("block_number", 3)),
            downscale_factor=float(data.get("downscale_factor", 2.0)),
            start_percent=float(data.get("start_percent", 0.0)),
            end_percent=float(data.get("end_percent", 0.35)),
            downscale_after_skip=bool(data.get("downscale_after_skip", True)),
            downscale_method=data.get("downscale_method", "bicubic"),
            upscale_method=data.get("upscale_method", "bicubic"),
        )


@dataclass
class DetailerInstance:
    detector_model: str = ""
    sam_model: str = "sam_vit_b_01ec64.pth"
    sam_device: str = "Prefer GPU"

    steps: Optional[Union[int, List[int]]] = None
    cfg: Optional[Union[float, List[float]]] = None
    sampler: Optional[Union[str, List[str]]] = None
    scheduler: Optional[Union[str, List[str]]] = None
    denoise: Optional[Union[float, List[float]]] = None

    guide_size: int = 360
    guide_size_for: bool = True
    max_size: int = 768
    feather: int = 20
    noise_mask: bool = True
    force_inpaint: bool = True
    bbox_threshold: float = 0.5
    bbox_dilation: int = 10
    bbox_crop_factor: float = 3.0
    sam_detection_hint: str = "center-1"
    sam_dilation: int = 0
    sam_threshold: float = 0.8
    sam_bbox_expansion: int = 0
    sam_mask_hint_threshold: float = 0.9
    sam_mask_hint_use_negative: str = "False"
    drop_size: int = 10
    wildcard: str = ""
    cycle: int = 1
    inpaint_model: bool = False
    noise_mask_feather: int = 20
    tiled_encode: bool = False
    tiled_decode: bool = False
    clip_skip: int = 0

    def get_steps(self, fallback: Union[int, List[int]]) -> int:
        return resolve_optional(self.steps, fallback)

    def get_cfg(self, fallback: Union[float, List[float]]) -> float:
        return resolve_optional(self.cfg, fallback)

    def get_sampler(self, fallback: Union[str, List[str]]) -> str:
        return resolve_optional(self.sampler, fallback)

    def get_scheduler(self, fallback: Union[str, List[str]]) -> str:
        return resolve_optional(self.scheduler, fallback)

    def get_denoise(self, fallback: Union[float, List[float]]) -> float:
        return resolve_optional(self.denoise, fallback)

    def get_clip_skip(self) -> int:
        return -1 * abs(self.clip_skip)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DetailerInstance":
        d = dict(data)  # copy; unknown keys don't matter because we won't read them
        defaults = DetailerInstance()

        # Randomizable overrides
        steps = parse_randomizable(d.get("steps"), int) if "steps" in d else defaults.steps
        cfg = parse_randomizable(d.get("cfg"), float) if "cfg" in d else defaults.cfg
        sampler = parse_randomizable(d.get("sampler")) if "sampler" in d else defaults.sampler
        scheduler = parse_randomizable(d.get("scheduler")) if "scheduler" in d else defaults.scheduler
        denoise = parse_randomizable(d.get("denoise"), float) if "denoise" in d else defaults.denoise

        return DetailerInstance(
            detector_model=d.get("detector_model", defaults.detector_model),
            sam_model=d.get("sam_model", defaults.sam_model),
            sam_device=d.get("sam_device", defaults.sam_device),

            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            denoise=denoise,

            guide_size=int(d.get("guide_size", defaults.guide_size)),
            guide_size_for=bool(d.get("guide_size_for", defaults.guide_size_for)),
            max_size=int(d.get("max_size", defaults.max_size)),
            feather=int(d.get("feather", defaults.feather)),
            noise_mask=bool(d.get("noise_mask", defaults.noise_mask)),
            force_inpaint=bool(d.get("force_inpaint", defaults.force_inpaint)),
            bbox_threshold=float(d.get("bbox_threshold", defaults.bbox_threshold)),
            bbox_dilation=int(d.get("bbox_dilation", defaults.bbox_dilation)),
            bbox_crop_factor=float(d.get("bbox_crop_factor", defaults.bbox_crop_factor)),
            sam_detection_hint=d.get("sam_detection_hint", defaults.sam_detection_hint),
            sam_dilation=int(d.get("sam_dilation", defaults.sam_dilation)),
            sam_threshold=float(d.get("sam_threshold", defaults.sam_threshold)),
            sam_bbox_expansion=int(d.get("sam_bbox_expansion", defaults.sam_bbox_expansion)),
            sam_mask_hint_threshold=float(d.get("sam_mask_hint_threshold", defaults.sam_mask_hint_threshold)),
            sam_mask_hint_use_negative=d.get("sam_mask_hint_use_negative", defaults.sam_mask_hint_use_negative),
            drop_size=int(d.get("drop_size", defaults.drop_size)),
            wildcard=d.get("wildcard", defaults.wildcard),
            cycle=int(d.get("cycle", defaults.cycle)),
            inpaint_model=bool(d.get("inpaint_model", defaults.inpaint_model)),
            noise_mask_feather=int(d.get("noise_mask_feather", defaults.noise_mask_feather)),
            tiled_encode=bool(d.get("tiled_encode", defaults.tiled_encode)),
            tiled_decode=bool(d.get("tiled_decode", defaults.tiled_decode)),
        )


@dataclass
class DetailingConfig:
    """Configuration for face/area detailing.

    Default detailer settings support randomization via lists.
    """

    checkpoint: str
    prompt_positive: str = "realistic skin detail"
    prompt_negative: str = ""
    loras: List[LoraConfig] = field(default_factory=list)
    downscale: DownscaleConfig = field(default_factory=DownscaleConfig)
    clip_skip: int = 0

    # Default detailer settings (support randomization via lists)
    steps: Union[int, List[int]] = 7
    cfg: Union[float, List[float]] = 1.0
    sampler: Union[str, List[str]] = "lcm"
    scheduler: Union[str, List[str]] = "exponential"
    denoise: Union[float, List[float]] = 0.3

    # List of detailers (chained)
    detailers: List[DetailerInstance] = field(default_factory=list)

    def get_steps(self) -> int:
        """Get steps, resolving random selection if list."""
        return resolve_value(self.steps)

    def get_cfg(self) -> float:
        """Get cfg, resolving random selection if list."""
        return resolve_value(self.cfg)

    def get_sampler(self) -> str:
        """Get sampler, resolving random selection if list."""
        return resolve_value(self.sampler)

    def get_scheduler(self) -> str:
        """Get scheduler, resolving random selection if list."""
        return resolve_value(self.scheduler)

    def get_denoise(self) -> float:
        """Get denoise, resolving random selection if list."""
        return resolve_value(self.denoise)

    def get_clip_skip(self) -> int:
        return -1 * abs(self.clip_skip)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DetailingConfig:
        loras = [LoraConfig.from_dict(l) for l in data.get("loras", [])]
        downscale = DownscaleConfig.from_dict(data.get("downscale", {}))
        detailers = [DetailerInstance.from_dict(d) for d in data.get("detailers", [])]

        return DetailingConfig(
            checkpoint=data["checkpoint"],
            prompt_positive=data.get("prompt_positive", "realistic skin detail"),
            prompt_negative=data.get("prompt_negative", ""),
            loras=loras,
            downscale=downscale,
            steps=parse_randomizable(data.get("steps", 7), int),
            cfg=parse_randomizable(data.get("cfg", 1.0), float),
            sampler=parse_randomizable(data.get("sampler", "lcm")),
            scheduler=parse_randomizable(data.get("scheduler", "karras")),
            denoise=parse_randomizable(data.get("denoise", 0.3), float),
            detailers=detailers,
            clip_skip=int(data.get("clip_skip", 0)),
        )


@dataclass
class PathsConfig:
    """Filesystem paths for model directories.

    If a specific path is not provided, it defaults to base/<folder>.
    If it is provided, it overrides the base-derived path.
    """
    base: Optional[str] = None

    checkpoints: Optional[str] = None
    diffusion_models: Optional[str] = None
    text_encoders: Optional[str] = None
    vae: Optional[str] = None
    loras: Optional[str] = None
    embeddings: Optional[str] = None
    sams: Optional[str] = None
    ultralytics: Optional[str] = None
    impact_pack: Optional[str] = None

    def _resolve(self, value: Optional[str], default_subdir: str) -> Optional[Path]:
        if value:
            return Path(value)

        if self.base:
            return Path(self.base) / default_subdir

        return None

    def get_checkpoints(self) -> Optional[Path]:
        return self._resolve(self.checkpoints, "checkpoints")

    def get_diffusion_models(self) -> Optional[Path]:
        return self._resolve(self.diffusion_models, "diffusion_models")

    def get_text_encoders(self) -> Optional[Path]:
        return self._resolve(self.text_encoders, "text_encoders")

    def get_vae(self) -> Optional[Path]:
        return self._resolve(self.vae, "vae")

    def get_loras(self) -> Optional[Path]:
        return self._resolve(self.loras, "loras")

    def get_embeddings(self) -> Optional[Path]:
        return self._resolve(self.embeddings, "embeddings")

    def get_sams(self) -> Optional[Path]:
        return self._resolve(self.sams, "sams")

    def get_ultralytics(self) -> Optional[Path]:
        return self._resolve(self.ultralytics, "ultralytics")

    def get_impact_pack(self) -> Optional[Path]:
        return self._resolve(self.impact_pack, "impact_pack")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PathsConfig":
        return PathsConfig(
            base=data.get("base"),
            checkpoints=data.get("checkpoints"),
            diffusion_models=data.get("diffusion_models"),
            text_encoders=data.get("text_encoders"),
            vae=data.get("vae"),
            loras=data.get("loras"),
            embeddings=data.get("embeddings"),
            sams=data.get("sams"),
            ultralytics=data.get("ultralytics"),
            impact_pack=data.get("impact_pack"),
        )


@dataclass
class WorkflowConfig:
    """Root configuration object for a ComfyGo workflow."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    loras: List[LoraConfig] = field(default_factory=list)
    loras_pass1: List[LoraConfig] = field(default_factory=list)
    loras_pass2: List[LoraConfig] = field(default_factory=list)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    detailing: Optional[DetailingConfig] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> WorkflowConfig:
        paths_data = data.get("paths", {})
        if not isinstance(paths_data, dict):
            paths_data = {}
        return WorkflowConfig(
            paths=PathsConfig.from_dict(paths_data),
            output=OutputConfig.from_dict(data.get("output", {})),
            generation=GenerationConfig.from_dict(data.get("generation", {})),
            models=ModelsConfig.from_dict(data.get("models", {})),
            loras=[LoraConfig.from_dict(l) for l in data.get("loras", [])],
            loras_pass1=[LoraConfig.from_dict(l) for l in data.get("loras_pass1", [])],
            loras_pass2=[LoraConfig.from_dict(l) for l in data.get("loras_pass2", [])],
            prompts=PromptsConfig.from_dict(data.get("prompts", {})),
            detailing=(
                DetailingConfig.from_dict(data["detailing"])
                if "detailing" in data
                else None
            ),
        )
