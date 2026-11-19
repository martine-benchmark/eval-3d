"""Every tool of the supplementary report imports and exposes a main() for `python -m`."""

import importlib

import pytest

MODULES = ["curves", "document", "jpeg", "montage", "photos", "pick_view", "plot_zones", "prf_objects", "tasks", "zones"]


@pytest.mark.parametrize("name", MODULES)
def test_tool_is_importable_and_runnable(name):
    module = importlib.import_module(f"eval3d.figures.supplementary.{name}")
    assert callable(module.main) and f"python -m eval3d.figures.supplementary.{name} " in module.__doc__
