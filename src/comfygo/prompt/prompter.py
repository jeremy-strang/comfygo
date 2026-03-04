"""Prompt loading, randomization, and tag expansion."""

import os
import random
import re
import tiktoken
from typing import Dict, List, Optional, ClassVar
from ..config.schema import PromptsConfig


class Prompter:
    """Handles prompt loading, randomization, tag expansion, and composition."""
    # Precompiled regex for clean_prompt
    _CLEAN_RULES: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"[\r\n]"), " "),
        (re.compile(r"\s+"), " "),
        (re.compile(r",\s*,+"), ","),
        (re.compile(r",+"), ","),
        (re.compile(r"\s*,\s*"), ", "),
        (re.compile(r"\(+"), "("),
        (re.compile(r"\)+"), ")"),
        (re.compile(r"\[+"), "["),
        (re.compile(r"\]+"), "]"),
        (re.compile(r"[^a-zA-Z0-9,\s]\s+[^a-zA-Z0-9\[\(\s]"), ""),
        (re.compile(r"\.,"), ","),
        (re.compile(r",\s*,+|,,+"), ","),
    ]


    def __init__(self, config: PromptsConfig, seed: Optional[int] = None):
        """Initialize the Prompter.

        Args:
            config: PromptsConfig with prompt settings.
            seed: Optional random seed for reproducibility.
        """
        self.config = config
        self._prompts: List[str] = []
        self._subjects: List[str] = []
        self._index = 0

        if seed is not None:
            random.seed(seed)

        self._load_prompts()


    @staticmethod
    def count_tokens(prompt: str, model: str = "gpt-4") -> int:
        enc = tiktoken.encoding_for_model(model)
        token_ids = enc.encode(prompt)
        return len(token_ids)


    @classmethod
    def clean_prompt(cls, prompt: str) -> str:
        """Clean and normalize prompt (remove duplicate commas, extra whitespace).

        Args:
            prompt: Raw prompt string.

        Returns:
            Cleaned prompt string.
        """
        s = str(prompt)

        # Apply regex normalization rules
        for pat, repl in cls._CLEAN_RULES:
            s = pat.sub(repl, s)

        s = s.strip(", ").strip()

        # --- dedupe logic ---
        parts = (p.strip() for p in s.split(",") if p.strip())

        seen = set()
        deduped = []

        for p in parts:
            key = p[: p.rfind(":")] if ":" in p else p
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        return ", ".join(deduped)


    def _load_prompts(self) -> None:
        """Load prompts from file or use override."""
        if self.config.prompt_override:
            self._prompts = [self.config.prompt_override]
        elif self.config.prompts_file and os.path.exists(self.config.prompts_file):
            with open(self.config.prompts_file, "r", encoding="utf-8") as f:
                self._prompts = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
        if self.config.subjects_file and os.path.exists(self.config.subjects_file):
            with open(self.config.subjects_file, "r", encoding="utf-8") as f:
                self._subjects = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip() and not line.strip().startswith("#")
                ]

        if not self._prompts:
            raise ValueError(
                "No prompts available: set prompts.prompts_file or prompts.prompt_override"
            )


    def get_num_prompts(self) -> int:
        if not self._prompts:
            return 0
        return len(self._prompts)


    def get_random_prompt(self) -> (str, str | None):
        """Get a random prompt with tags expanded and prefix/suffix applied.

        Returns:
            Composed prompt string.
        """
        if not self._prompts:
            raise ValueError("No prompts loaded")

        base_prompt = random.choice(self._prompts)
        expanded = self._expand_tags(base_prompt)
        subject = None
        if self._subjects:
            subject = random.choice(self._subjects)
        return (self._compose_prompt(expanded, subject=subject), subject)


    def get_next_prompt(self) -> (str, str | None):
        """Get next prompt in sequence (rotating) with tags expanded and prefix/suffix.

        Returns:
            Composed prompt string.
        """
        if not self._prompts:
            raise ValueError("No prompts loaded")

        base_prompt = self._prompts[self._index % len(self._prompts)]
        self._index += 1
        expanded = self._expand_tags(base_prompt)
        subject = None
        if self._subjects:
            subject = self._subjects[self._index % len(self._subjects)]
        return (self._compose_prompt(expanded, subject=subject), subject)


    def _expand_tags(self, prompt: str) -> str:
        """Expand {tag_name} placeholders with random selections from tags dict.

        Args:
            prompt: Prompt string with optional {tag} placeholders.

        Returns:
            Prompt with tags replaced by random selections.

        Example:
            tags = {"style": ["realistic", "anime"], "pose": ["standing", "sitting"]}
            prompt = "A {style} photo, woman {pose}"
            result = "A realistic photo, woman standing"  (random selections)
        """
        if not self.config.tags:
            return prompt


        def replace_tag(match: re.Match) -> str:
            tag_name = match.group(1)
            if tag_name in self.config.tags:
                options = self.config.tags[tag_name]
                if options:
                    return random.choice(options)
            # Return original if tag not found
            return match.group(0)

        # Match {tag_name} patterns
        return re.sub(r"\{(\w+)\}", replace_tag, prompt)


    def substitute(self, prompt: str, reverse: bool = False, *, max_passes: int = 3) -> str:
        """Apply configured substitutions until the prompt stabilizes.

        Args:
            prompt: Input prompt string.
            reverse: Whether or not to apply the substitutions in reverse
            max_passes: The max number of times to apply the substitution (prevents cycles)
        Returns:
            The prompt with the configured substitutions applied.
        """
        subs_cfg = self.config.substitutions
        if not subs_cfg: return Prompter.clean_prompt(str(prompt))

        subs = subs_cfg if isinstance(subs_cfg, list) else [subs_cfg]
        seen: set[str] = set()
        s = str(prompt)
        for _ in range(max_passes):
            if s in seen: break
            seen.add(s)
            before = s
            for pr in subs:
                if reverse and not getattr(pr, "reversible", False):
                    continue
                search = pr.replace if reverse else pr.search
                replace = pr.search if reverse else pr.replace

                if search: s = s.replace(search, replace)
            if s == before: break
        return Prompter.clean_prompt(s)


    def _compose_prompt(self, base: str, subject=None) -> str:
        """Apply prefix and suffix to a base prompt.

        Args:
            base: Base prompt string.

        Returns:
            Composed prompt with prefix and suffix.
        """
        parts = []
        if self.config.prefix:
            parts.append(self.config.prefix.strip())

        parts.append(base.strip())
        if self.config.suffix:
            parts.append(self.config.suffix.strip())

        composed = ", ".join(p for p in parts if p)
        if subject:
            composed = composed.replace("__subject__", subject)
        return Prompter.clean_prompt(composed)


    @property
    def negative_prompt(self) -> str:
        """Get the negative prompt.

        Returns:
            Negative prompt string.
        """
        return self.config.negative or ""


    @property
    def prompt_count(self) -> int:
        """Get the number of loaded prompts.

        Returns:
            Number of prompts available.
        """
        return len(self._prompts)

