"""Helpers for randomizable configuration values.

Any primitive config value can be a list; resolve_value() selects randomly.
Paired properties (sampler_scheduler, dimensions) allow coupled randomization.
"""

import random
from typing import Callable, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T", int, float, str)


def parse_dimensions(value: str) -> Tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' format to (width, height) tuple.

    Args:
        value: String in format "WIDTHxHEIGHT" (e.g., "896x1200").

    Returns:
        Tuple of (width, height) as integers.
    """
    parts = value.lower().split("x")
    return (int(parts[0]), int(parts[1]))


def parse_sampler_scheduler(value: str) -> Tuple[str, str]:
    """Parse 'sampler/scheduler' format to (sampler, scheduler) tuple.

    Args:
        value: String in format "sampler/scheduler" (e.g., "euler/simple").

    Returns:
        Tuple of (sampler, scheduler) as strings.
    """
    parts = value.split("/")
    return (parts[0], parts[1])


def parse_randomizable_paired(
    value, parser: Callable
) -> Optional[Union[Tuple, List[Tuple]]]:
    """Parse a paired value that may be single or list.

    Args:
        value: Raw value from YAML (single string or list of strings).
        parser: Function to parse each string (e.g., parse_dimensions).

    Returns:
        Parsed tuple or list of tuples, or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [parser(v) for v in value]
    return parser(value)


def resolve_paired(
    value: Optional[Union[Tuple, List[Tuple]]]
) -> Optional[Tuple]:
    """Resolve a paired value, randomly selecting if list.

    Args:
        value: Either a single tuple or a list of tuples.

    Returns:
        If list, a random selection. If single tuple, returns it. If None, returns None.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return random.choice(value)
    return value


def resolve_value(value: Union[T, List[T]]) -> T:
    """Resolve a value that may be a list (random selection) or single value.

    Args:
        value: Either a single value or a list of values.

    Returns:
        If list, a random selection. Otherwise, the value itself.
    """
    if isinstance(value, list):
        return random.choice(value)
    return value


def parse_randomizable(
    value,
    converter: Optional[Callable] = None,
) -> Union[T, List[T]]:
    """Parse a YAML value that may be single or list, optionally converting.

    Args:
        value: Raw value from YAML (single or list).
        converter: Optional function to convert each value (e.g., int, float).

    Returns:
        Parsed value (single or list with converter applied).
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [converter(v) if converter else v for v in value]
    return converter(value) if converter else value


def resolve_optional(
    override: Optional[Union[T, List[T]]],
    fallback: Union[T, List[T]],
) -> T:
    """Resolve an optional override, falling back to a default.

    Args:
        override: Optional value that may override the fallback.
        fallback: Default value to use if override is None.

    Returns:
        Resolved value from override if set, otherwise from fallback.
    """
    if override is not None:
        return resolve_value(override)
    return resolve_value(fallback)
