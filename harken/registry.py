"""A tiny name -> object registry used for encoders and projectors.

The registry doubles as a decorator so plugins can self-register::

    audio_encoders = Registry("audio encoder")

    @audio_encoders.register("clap")
    class ClapEncoder(AudioEncoder):
        ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar, overload

from harken.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    @overload
    def register(self, key: str) -> Callable[[T], T]: ...
    @overload
    def register(self, key: str, obj: T) -> T: ...

    def register(self, key: str, obj: T | None = None) -> T | Callable[[T], T]:
        """Register ``obj`` under ``key``; usable directly or as a decorator."""

        def _add(value: T) -> T:
            if key in self._items:
                raise RegistryError(
                    f"{self.name} '{key}' is already registered"
                )
            self._items[key] = value
            return value

        if obj is None:
            return _add
        return _add(obj)

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(
                f"unknown {self.name} '{key}'. Available: {known}"
            ) from None

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._items)
