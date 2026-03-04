# ComfyGo

A fast, headless CLI for running Stable Diffusion image generation workflows without a ComfyUI server. Designed for automated batch generation with randomized parameters — currently targeting the [ZImage Turbo](https://civitai.com/models/618526/zimage-turbo) (Aura Flow) architecture, with plans to generalize to other workflows with support for other architectures (SDXL, Flux, Qwen).

## Why ComfyGo?

Running ComfyUI through its server or via `comfy-cli` has significant overhead: it starts a full HTTP server, loads a node registry, and routes every generation through a JSON workflow API. ComfyGo bypasses all of that. It imports ComfyUI's core model-loading and sampling code directly as a Python library, running inference in-process with no server, no queueing, and no round-trip latency.

In practice this means:
- **Faster startup** — no server to spin up, no workflow compilation step
- **Less memory overhead** — no web server or node registry resident in RAM
- **Script-friendly** — runs from a single YAML config, fits naturally into cron jobs, batch scripts, or CI pipelines
- **Randomization built in** — any parameter (sampler, scheduler, resolution, CFG, LoRA strength, etc.) can be a list; values are randomly sampled each run. Paired parameters like `sampler/scheduler` or `width×height` are always selected together, so you never get mismatched pairs.

## Installation

```bash
pip install -e .
```

Python 3.10+ required. ComfyUI vendored dependencies are included; no separate ComfyUI installation needed.

## Quick Start

```bash
# Generate a default config
comfygo init

# Edit config.yaml, then run
comfygo run config.yaml

# Validate config without running
comfygo run config.yaml --dry-run
```

## CLI Reference

```
comfygo run <config>          Run a workflow
  --seed INT                  Override random seed
  --prompt TEXT               Override prompt (ignores prompts_file)
  --runs INT                  Override number of runs
  --dry-run                   Validate config only, no inference
  -v, --verbose               Verbose logging

comfygo init [-o PATH] [-f]   Generate a default config.yaml
comfygo validate <config>     Validate a config file
comfygo setup                 First-time setup helper
```

## Configuration

ComfyGo is driven entirely by a YAML config file. Run `comfygo init` to generate a fully annotated template.

### Paths

```yaml
paths:
  base: "/models"           # Root for all relative model names
  # Override any subdir individually:
  # loras: "/fast-nvme/loras"
```

When `base` is set, relative model names resolve to `base/<type>/<name>`. Individual overrides take precedence.

### Generation

```yaml
generation:
  runs: 20
  steps: 10
  cfg: 1.0
  shift: 3.0
  seed_override: null       # Fix seed for reproducibility

  # Single values or lists (randomly sampled each run):
  sampler: "euler"
  scheduler: "simple"
  # sampler: ["euler", "res_multistep"]
  # scheduler: ["simple", "beta"]

  # Paired sampling — sampler and scheduler always selected together:
  # sampler_scheduler:
  #   - "euler/simple"
  #   - "res_multistep/beta"

  # Paired dimensions — width and height always selected together:
  # dimensions:
  #   - "832x1216"
  #   - "1024x1024"
```

### Models

```yaml
models:
  vae: "ae.safetensors"
  clip_skip: -2             # Supports lists for randomization

  clip:
    name: "qwen_3_4b.safetensors"
    type: "lumina2"
    device: "cpu"

  # Single checkpoint (simple):
  checkpoint: "mymodel.safetensors"

  # Or multiple checkpoints with per-checkpoint overrides:
  checkpoints:
    - checkpoint: "modelA.safetensors"
      sampler_scheduler: "euler/simple"
      clip_skip: -1
    - checkpoint: "modelB.safetensors"
      dimensions: ["832x1216", "1024x1024"]
      loras:
        - name: "detail.safetensors"
          strength: 0.5
```

Per-checkpoint overrides accept the same randomizable fields as `generation` and take precedence over global values.

### Prompts

```yaml
prompts:
  prompts_file: "prompts.txt"   # One prompt per line
  # prompt_override: "a portrait"  # Direct prompt, ignores file
  prefix: "masterpiece, "
  suffix: ", photorealistic"
  negative: "blurry, text"

  # Named tag pools — reference with {tag_name} in prompts:
  tags:
    style: ["cinematic", "editorial", "documentary"]
    lighting: ["natural light", "golden hour", "overcast"]
```

### LoRAs

```yaml
loras:
  - name: "style.safetensors"
    strength: 0.4
    strength_clip: 0.4      # Defaults to strength if omitted

# Split LoRAs across the two-pass sampling:
loras_pass1:
  - name: "pass1only.safetensors"
    strength: 0.3
loras_pass2:
  - name: "pass2only.safetensors"
    strength: 0.3
```

LoRA `name` and `strength` both support lists for per-run random selection.

### Detailing (FaceDetailer)

Optional chained detailing pass using ComfyUI-Impact-Pack's FaceDetailer:

```yaml
detailing:
  checkpoint: "sdxl_base.safetensors"
  prompt_positive: "realistic skin detail"
  steps: 7
  cfg: 1.2
  sampler: "lcm"
  scheduler: "karras"
  denoise: 0.3
  clip_skip: -1

  detailers:
    - name: "face"
      detector_model: "bbox/face_yolov8m.pt"
      sam_model: "sam_vit_b_01ec64.pth"
```

## Project Structure

```
src/comfygo/
├── cli/          # CLI entrypoint and subcommands
├── config/       # Schema, loader, and randomizable value helpers
├── nodes/        # Thin wrappers around ComfyUI node functions
├── prompt/       # Prompt loading and tag substitution
├── runner/       # Main workflow execution logic
├── utils/        # Logging, console, file utilities
└── vendor/       # Vendored ComfyUI core (no server dependency)
```

## License

GPL-3.0