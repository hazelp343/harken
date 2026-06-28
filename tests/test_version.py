import re

import harken


def test_version_is_exposed():
    assert isinstance(harken.__version__, str)
    assert harken.__version__


def test_version_is_pep440_ish():
    # major.minor.patch with optional pre-release suffix.
    assert re.match(r"^\d+\.\d+\.\d+", harken.__version__)
