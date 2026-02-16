import importlib
import sys
import types


def _register_legacy_image2layout() -> None:
    if "image2layout" in sys.modules:
        return

    image2layout = types.ModuleType("image2layout")
    train = types.ModuleType("image2layout.train")
    helpers = types.ModuleType("image2layout.train.helpers")
    relationships = importlib.import_module("ralf.train.helpers.relationships")

    helpers.relationships = relationships
    train.helpers = helpers
    image2layout.train = train

    sys.modules["image2layout"] = image2layout
    sys.modules["image2layout.train"] = train
    sys.modules["image2layout.train.helpers"] = helpers
    sys.modules["image2layout.train.helpers.relationships"] = relationships


_register_legacy_image2layout()
