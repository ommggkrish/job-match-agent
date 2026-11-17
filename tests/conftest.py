import pytest

from agent import tools


@pytest.fixture(autouse=True)
def use_sample_cv(monkeypatch, tmp_path):
    """Tests always use the sample CV, even if data/my_cv.md exists on this machine."""
    monkeypatch.setattr(tools, "REAL_CV", tmp_path / "does-not-exist.md")
