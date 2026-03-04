"""comfygo setup — register the ComfyUI ML library with the current venv."""
import subprocess
import sys
import sysconfig
from pathlib import Path


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "setup",
        help="Register the ComfyUI ML library with the current Python environment",
    )
    p.add_argument(
        "--comfyui-path",
        required=True,
        metavar="PATH",
        help="Path to the ComfyUI base directory (the one containing comfy/)",
    )
    p.set_defaults(func=run)


def run(args) -> int:
    comfyui_path = Path(args.comfyui_path).resolve()
    if not (comfyui_path / "comfy").is_dir():
        print(
            f"Error: {comfyui_path} does not contain a comfy/ subdirectory",
            file=sys.stderr,
        )
        return 1

    site_packages = Path(sysconfig.get_path("purelib"))
    pth_file = site_packages / "_comfyui_lib.pth"
    pth_file.write_text(str(comfyui_path) + "\n")

    result = subprocess.run(
        [sys.executable, "-c", "import comfy"],
        capture_output=True,
    )
    if result.returncode != 0:
        pth_file.unlink(missing_ok=True)
        print(f"Verification failed:\n{result.stderr.decode()}", file=sys.stderr)
        return 1

    print(f"Written: {pth_file}")
    print("comfy.* is now importable from this venv.")
    return 0
