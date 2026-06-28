"""Exception hierarchy for harken.

All errors raised by the library derive from :class:`HarkenError`, so callers
can catch everything from the package with a single ``except``.
"""

from __future__ import annotations


class HarkenError(Exception):
    """Base class for all errors raised by harken."""


class AudioLoadError(HarkenError):
    """Raised when an audio file cannot be read or decoded."""


class ConfigError(HarkenError):
    """Raised when a configuration is inconsistent or incomplete."""


class RegistryError(HarkenError, KeyError):
    """Raised when a name is missing from (or already in) a registry."""


class DependencyError(HarkenError, ImportError):
    """Raised when an optional dependency is required but not installed."""
