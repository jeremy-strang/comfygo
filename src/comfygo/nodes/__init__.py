"""ComfyGo nodes package.

Provides direct ML function calls without ComfyUI's node registry overhead.

Usage::

    from comfygo.nodes import setup_comfy_env
    setup_comfy_env(config.paths)  # must be called before any comfy.* imports

    from comfygo.nodes.loaders import load_unet, load_clip, load_vae
    from comfygo.nodes.sampling import ksampler_advanced
    ...
"""
import logging
import os
import sys
import types

log = logging.getLogger(__name__)


def setup_comfy_env(paths) -> None:
    """Initialize the ML environment without ComfyUI's server/node registry.

    Installs lightweight stubs for ``folder_paths``, ``nodes``, ``server``,
    and ``execution``, then adds the ComfyUI library directory to sys.path so
    that ``comfy.*`` imports resolve correctly.

    **MUST be called before any** ``comfy.*`` **import** (including transitive
    imports from loaders.py, sampling.py, etc.).

    Args:
        paths: :class:`~comfygo.config.schema.PathsConfig` with model paths.
    """
    if "folder_paths" in sys.modules:
        log.debug("ComfyUI env already initialised — skipping setup_comfy_env()")
        return

    from .shims import FolderPathsShim

    # ── 1. folder_paths shim ──────────────────────────────────────────────
    shim = FolderPathsShim(paths)
    sys.modules["folder_paths"] = shim  # type: ignore[assignment]

    # ── 2. nodes stub ─────────────────────────────────────────────────────
    nodes_stub = types.ModuleType("nodes")
    nodes_stub.NODE_CLASS_MAPPINGS = {}          # type: ignore[attr-defined]
    nodes_stub.NODE_DISPLAY_NAME_MAPPINGS = {}   # type: ignore[attr-defined]
    nodes_stub.MAX_RESOLUTION = 8192             # type: ignore[attr-defined]

    # CLIPLoaderGGUF introspects nodes.CLIPLoader.INPUT_TYPES() to reuse the
    # "type" combo list.  Provide minimal stubs so that doesn't crash.
    class _CLIPLoaderStub:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "clip_name": ([], {}),
                    "type": (
                        [
                            "stable_diffusion", "stable_cascade", "sd3",
                            "stable_audio", "mochi", "ltxv", "pixart",
                            "cosmos", "lumina2", "wan", "hidream", "chroma",
                            "ace", "omnigen2", "qwen_image", "hunyuan_image",
                            "flux2", "ovis",
                        ],
                        {},
                    ),
                }
            }

    class _DualCLIPLoaderStub:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "clip_name1": ([], {}),
                    "clip_name2": ([], {}),
                    "type": (["sdxl", "sd3", "flux"], {}),
                }
            }

    class _TripleCLIPLoaderStub:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "clip_name1": ([], {}),
                    "clip_name2": ([], {}),
                    "clip_name3": ([], {}),
                }
            }

    nodes_stub.CLIPLoader = _CLIPLoaderStub          # type: ignore[attr-defined]
    nodes_stub.DualCLIPLoader = _DualCLIPLoaderStub  # type: ignore[attr-defined]
    nodes_stub.TripleCLIPLoader = _TripleCLIPLoaderStub  # type: ignore[attr-defined]

    # impact_pack.py uses these as base classes at module level:
    # class ImageSender(nodes.PreviewImage) and class LatentSender(nodes.SaveLatent)
    class _PreviewImage:
        pass

    class _SaveImage:
        pass

    class _SaveLatent:
        pass

    nodes_stub.PreviewImage = _PreviewImage  # type: ignore[attr-defined]
    nodes_stub.SaveImage = _SaveImage        # type: ignore[attr-defined]
    nodes_stub.SaveLatent = _SaveLatent      # type: ignore[attr-defined]

    # Functional VAE wrappers used by impact/utils.py at runtime
    class _VAEEncode:
        def encode(self, vae, pixels):
            t = vae.encode(pixels)
            return ({"samples": t},)

    class _VAEEncodeTiled:
        def encode(self, vae, pixels, tile_size=512, overlap=64, **kwargs):
            t = vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
            return ({"samples": t},)

    class _VAEDecode:
        def decode(self, vae, samples):
            images = vae.decode(samples["samples"])
            return (images,)

    class _VAEDecodeTiled:
        def decode(self, vae, samples, tile_size=512, **kwargs):
            images = vae.decode_tiled(samples["samples"])
            return (images,)

    nodes_stub.VAEEncode = _VAEEncode          # type: ignore[attr-defined]
    nodes_stub.VAEEncodeTiled = _VAEEncodeTiled  # type: ignore[attr-defined]
    nodes_stub.VAEDecode = _VAEDecode          # type: ignore[attr-defined]
    nodes_stub.VAEDecodeTiled = _VAEDecodeTiled  # type: ignore[attr-defined]
    sys.modules["nodes"] = nodes_stub

    # ── 3. server stub ────────────────────────────────────────────────────
    # Impact Pack's core.py calls update_node_status() which checks
    # PromptServer.instance.client_id; with client_id=None it returns early.
    server_stub = types.ModuleType("server")

    class _PSInstance:
        client_id = None

    class _PromptServer:
        instance = _PSInstance()

    server_stub.PromptServer = _PromptServer  # type: ignore[attr-defined]
    sys.modules["server"] = server_stub

    # ── 4. execution stub ─────────────────────────────────────────────────
    execution_stub = types.ModuleType("execution")
    sys.modules["execution"] = execution_stub

    # ── 5. Ensure vendored comfy.* / comfy_extras.* is on sys.path ───────
    # comfygo vendors comfy/ and comfy_extras/ in src/comfygo/vendor/.
    # Insert vendor/ at sys.path[0] so it takes precedence over any other
    # comfy installation (e.g. _comfyui_lib.pth pointing at a ComfyUI dir).
    _vendor_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "vendor")
    )
    if _vendor_dir not in sys.path:
        sys.path.insert(0, _vendor_dir)

    # vendor/impact/ contains the vendored impact package at vendor/impact/impact/.
    # Adding vendor/impact/ to sys.path makes `import impact` resolve to that
    # subdirectory rather than resolving vendor/impact/ itself as a namespace package.
    _vendor_impact_dir = os.path.join(_vendor_dir, "impact")
    if _vendor_impact_dir not in sys.path:
        sys.path.append(_vendor_impact_dir)

    try:
        import comfy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "comfy.* is not importable. The vendored comfy/ directory may "
            f"be missing from {_vendor_dir}."
        ) from exc

    # ── 6. Add Impact Pack modules dir to sys.path (for import impact.*) ──
    # User-configured path takes precedence over the vendored fallback above.
    _impact_pack_path = paths.get_impact_pack() if hasattr(paths, "get_impact_pack") else None
    if _impact_pack_path is not None:
        _impact_pack_str = str(_impact_pack_path)
        if _impact_pack_str not in sys.path:
            sys.path.insert(0, _impact_pack_str)
            log.debug("Added Impact Pack to sys.path: %s", _impact_pack_str)

    # ── 7. impact.impact_server stub ──────────────────────────────────────
    # impact/segs_nodes.py imports impact.impact_server (ComfyUI web routes
    # we don't need).  Pre-install a stub to break the circular import and
    # avoid the aiohttp/PromptServer dependency.
    impact_server_stub = types.ModuleType("impact.impact_server")
    impact_server_stub.segs_picker_map = {}  # type: ignore[attr-defined]
    sys.modules["impact.impact_server"] = impact_server_stub

    log.info(
        "ComfyUI ML environment ready (no PromptServer / node registry / "
        "init_extra_nodes)"
    )
