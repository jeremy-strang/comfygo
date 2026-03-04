"""Shims for ComfyUI modules that comfygo replaces.

Injected into sys.modules before any comfy.* imports so that custom node
code (Impact Pack, GGUF) resolves ComfyUI imports to our stubs.
"""
import os
import logging
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

_SUPPORTED_PT_EXTENSIONS = {
    '.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.pkl', '.sft', '.gguf'
}


class FolderPathsShim:
    """Drop-in replacement for ComfyUI's folder_paths module.

    Maps ComfyUI folder-name constants ("checkpoints", "sams", …) to explicit
    directory paths from PathsConfig.  Injected as sys.modules['folder_paths'].
    """

    supported_pt_extensions: Set[str] = _SUPPORTED_PT_EXTENSIONS

    def __init__(self, paths) -> None:
        """
        Args:
            paths: PathsConfig with model directory fields.
        """
        self._paths = paths
        # folder -> single directory path string
        self._folder_map: Dict[str, str] = {}
        # ComfyUI-style: folder -> ([path, ...], {ext, ...})
        self.folder_names_and_paths: Dict[str, Tuple[List[str], Set[str]]] = {}
        self._build_folder_map()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_folder_map(self) -> None:
        p = self._paths
        mappings: Dict[str, str] = {
            "checkpoints": p.checkpoints,
            "diffusion_models": p.diffusion_models,
            "unet": p.diffusion_models,
            "unet_gguf": p.diffusion_models,
            "text_encoders": p.text_encoders,
            "clip": p.text_encoders,
            "clip_gguf": p.text_encoders,
            "vae": p.vae,
            "loras": p.loras,
            "embeddings": p.embeddings,
            "sams": p.sams,
            "ultralytics": p.ultralytics,
        }
        for folder, path in mappings.items():
            if path:
                self._folder_map[folder] = path
                ext: Set[str] = {".gguf"} if folder in ("clip_gguf", "unet_gguf") \
                    else _SUPPORTED_PT_EXTENSIONS
                self.folder_names_and_paths[folder] = ([path], ext)

    # ------------------------------------------------------------------
    # Public API (mirrors folder_paths module interface)
    # ------------------------------------------------------------------

    @property
    def models_dir(self) -> str:
        """Best-effort models root directory (parent of the first configured model dir)."""
        for attr in ("checkpoints", "diffusion_models", "vae", "loras"):
            path = getattr(self._paths, attr, "")
            if path:
                return os.path.dirname(path.rstrip("/\\"))
        return ""

    def get_full_path(self, folder: str, name: str) -> Optional[str]:
        """Resolve *name* to a full filesystem path, or None if not found."""
        # Already absolute
        if os.path.isabs(name) and os.path.exists(name):
            return name
        # Look in the configured base for this folder
        base = self._folder_map.get(folder, "")
        if base:
            candidate = os.path.join(base, name)
            if os.path.exists(candidate):
                return candidate
        # Fallback: search all configured folders
        for fb_path in self._folder_map.values():
            if fb_path:
                candidate = os.path.join(fb_path, name)
                if os.path.exists(candidate):
                    log.debug("Resolved '%s' via fallback folder search: %s", name, candidate)
                    return candidate
        return None

    def get_full_path_or_raise(self, folder: str, name: str) -> str:
        """Like get_full_path but raises FileNotFoundError on miss."""
        path = self.get_full_path(folder, name)
        if path is None:
            configured = self._folder_map.get(folder, "not configured")
            raise FileNotFoundError(
                f"Model not found: '{name}' (folder='{folder}', base='{configured}')"
            )
        return path

    def get_folder_paths(self, folder: str) -> List[str]:
        """Return list of base directories for *folder*."""
        base = self._folder_map.get(folder, "")
        return [base] if base else []

    def get_filename_list(self, folder: str) -> List[str]:
        """Return sorted list of relative filenames found under *folder*."""
        base = self._folder_map.get(folder, "")
        if not base or not os.path.isdir(base):
            return []
        results: List[str] = []
        try:
            for root, _dirs, files in os.walk(base, followlinks=True):
                for fname in files:
                    rel = os.path.relpath(os.path.join(root, fname), base)
                    results.append(rel)
        except OSError as exc:
            log.warning("Error listing folder '%s': %s", folder, exc)
        return sorted(results)

    def add_model_folder_path(
        self, folder: str, path: str, is_default: bool = False
    ) -> None:
        """Register an extra search directory for *folder*."""
        if folder not in self._folder_map and path:
            self._folder_map[folder] = path
        entry = self.folder_names_and_paths.get(
            folder, ([], _SUPPORTED_PT_EXTENSIONS)
        )
        paths_list, exts = entry
        if path not in paths_list:
            paths_list.append(path)
        self.folder_names_and_paths[folder] = (paths_list, exts)

    # Stubs for ComfyUI methods comfygo doesn't need but nodes call anyway
    def get_output_directory(self) -> str:
        return "/tmp/comfygo_output"

    def get_input_directory(self) -> str:
        return "/tmp/comfygo_input"

    def get_user_directory(self) -> str:
        return "/tmp/comfygo_user"

    def get_annotated_filepath(self, name: str) -> str:
        return name

    def get_save_image_path(self, filename_prefix, output_dir, *args):
        return output_dir, filename_prefix, 0, "", ""
