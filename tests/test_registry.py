import pytest
from harken.exceptions import RegistryError
from harken.registry import Registry


def test_register_and_get():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    assert reg.get("a") == 1
    assert "a" in reg


def test_register_as_decorator():
    reg: Registry[type] = Registry("cls")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo


def test_duplicate_registration_raises():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(RegistryError):
        reg.register("a", 2)


def test_unknown_key_lists_available():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(RegistryError) as exc:
        reg.get("b")
    assert "a" in str(exc.value)


def test_keys_and_len():
    reg: Registry[int] = Registry("thing")
    reg.register("b", 2)
    reg.register("a", 1)
    assert reg.keys() == ["a", "b"]
    assert len(reg) == 2
    assert list(reg) == ["a", "b"]
