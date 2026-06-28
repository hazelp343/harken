import pytest

from harken.exceptions import DependencyError
from harken.utils import require, seed_everything


def test_require_returns_module():
    mod = require("json")
    assert hasattr(mod, "loads")


def test_require_missing_raises_with_hint():
    with pytest.raises(DependencyError) as exc:
        require("definitely_not_a_real_module_xyz", extra="hf")
    assert "harken[hf]" in str(exc.value)


def test_seed_everything_is_deterministic():
    import random

    seed_everything(123)
    a = [random.random() for _ in range(3)]
    seed_everything(123)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_seed_everything_returns_seed():
    assert seed_everything(7) == 7
