import os
os.environ["TQDM_DISABLE"] = "1"

import argparse
import logging
import sys
import yaml

from ..utils.logger import configure_logging
log = logging.getLogger(__name__)

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}

def setup_logging_from_config_file(config_path: str) -> None:
    cfg = {}
    if config_path and os.path.exists(config_path):
        cfg = _load_yaml(config_path)

    log_cfg = cfg.get("logging") or {}
    if not isinstance(log_cfg, dict):
        log_cfg = {}

    level = log_cfg.get("level", "INFO")

    files_cfg = log_cfg.get("files") or {}
    if not isinstance(files_cfg, dict):
        files_cfg = {}

    info_cfg = files_cfg.get("info") or {}
    if not isinstance(info_cfg, dict):
        info_cfg = {}

    error_cfg = files_cfg.get("error") or {}
    if not isinstance(error_cfg, dict):
        error_cfg = {}

    configure_logging(
        level=level,
        info_log=info_cfg.get("path"),
        info_level=info_cfg.get("level", "INFO"),
        error_log=error_cfg.get("path"),
        error_level=error_cfg.get("level", "ERROR"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="comfygo",
        description="ComfyUI workflow automation with YAML configuration",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a workflow from configuration",
    )
    run_parser.add_argument(
        "config_file",
        help="Path to YAML configuration file",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed for reproducibility",
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt (ignores prompts_file)",
    )
    run_parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override number of runs",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without executing",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Init command - generate default config
    init_parser = subparsers.add_parser(
        "init",
        help="Generate a default configuration file",
    )
    init_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="config.yaml",
        help="Output path for config file (default: config.yaml)",
    )
    init_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )

    # Setup command
    from . import setup_cmd
    setup_cmd.add_parser(subparsers)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
    )
    validate_parser.add_argument(
        "config_file",
        help="Path to YAML configuration file",
    )
    validate_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def cmd_run(args) -> int:
    """Execute the run command."""
    from ..config.loader import ConfigValidationError, load_config
    from ..workflows.zit_two_pass import run_workflow

    config_path = getattr(args, "config_file", None)
    log = logging.getLogger("comfygo")
    setup_logging_from_config_file(config_path)

    log.info("Logging configured")

    try:
        config = load_config(args.config_file)

        # Apply CLI overrides
        if args.seed is not None:
            config.generation.seed_override = args.seed
            log.info(f"Using seed override: {args.seed}")

        if args.prompt is not None:
            config.prompts.prompt_override = args.prompt
            log.info("Using prompt override from CLI")

        if args.runs is not None:
            config.generation.runs = args.runs
            log.info(f"Running {args.runs} iteration(s)")

        if args.dry_run:
            log.info("Configuration valid. Dry run complete.")
            log.info(f"  Checkpoint: {config.models.checkpoint}")
            log.info(f"  Size: {config.generation.width}x{config.generation.height}")
            log.info(f"  Steps: {config.generation.steps}, CFG: {config.generation.cfg}")
            if config.detailing:
                log.info(f"  Detailing: {len(config.detailing.detailers)} detailer(s)")
            return 0

        run_workflow(config)
        return 0

    except ConfigValidationError as e:
        log.error(f"Configuration error:\n{e}")
        return 1
    except FileNotFoundError as e:
        log.error(str(e))
        return 1
    except Exception as e:
        log.exception(f"Error running workflow: {e}")
        return 1


def cmd_init(args) -> int:
    """Execute the init command."""
    from ..config.loader import generate_default_config

    log = logging.getLogger("comfygo")

    if os.path.exists(args.output) and not args.force:
        log.error(f"File exists: {args.output}. Use --force to overwrite.")
        return 1

    content = generate_default_config()
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)

    log.info(f"Created config file: {args.output}")
    return 0


def cmd_validate(args) -> int:
    """Execute the validate command."""
    from ..config.loader import ConfigValidationError, load_config

    log = logging.getLogger("comfygo")

    try:
        config = load_config(args.config_file)
        log.info(f"Configuration valid: {args.config_file}")
        log.info(f"  UNET checkpoint: {config.models.checkpoint}")
        log.info(f"  VAE: {config.models.vae}")
        log.info(f"  CLIP: {config.models.clip.name} ({config.models.clip.type})")
        log.info(f"  Size: {config.generation.width}x{config.generation.height}")
        log.info(
            f"  Generation: {config.generation.steps} steps, "
            f"CFG {config.generation.cfg}, {config.generation.sampler}"
        )
        log.info(f"  LoRAs: {len(config.loras)}")

        if config.detailing:
            log.info(f"  Detailing checkpoint: {config.detailing.checkpoint}")
            log.info(f"  Detailing LoRAs: {len(config.detailing.loras)}")
            log.info(f"  Detailers: {len(config.detailing.detailers)}")
            for d in config.detailing.detailers:
                log.info(f"    - {d.name}: {d.detector_model}")

        if config.prompts.prompt_override:
            log.info("  Prompt: override set")
        elif config.prompts.prompts_file:
            log.info(f"  Prompts file: {config.prompts.prompts_file}")

        if config.prompts.tags:
            log.info(f"  Tags defined: {list(config.prompts.tags.keys())}")

        return 0

    except ConfigValidationError as e:
        log.error(f"Validation failed:\n{e}")
        return 1
    except Exception as e:
        log.error(f"Error: {e}")
        return 1


def main() -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if getattr(args, "verbose", False) else "INFO"

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "setup":
        from . import setup_cmd
        sys.exit(setup_cmd.run(args))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
