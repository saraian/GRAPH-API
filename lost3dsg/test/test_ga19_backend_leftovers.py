"""GA-19 leftovers in cloud/client.py: an unknown backend name is refused (it used to fall
through to the local stub and read as an empty scene); the managed provider refuses before
paying for a call whose answer it cannot parse; the local stub's health tells the truth."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import numpy as np  # noqa: E402
from cloud.client import (  # noqa: E402
    LocalPerceptionBackend,
    ManagedPerceptionBackend,
    get_perception_backend,
)

try:
    get_perception_backend({"perception": {"backend": "locla"}})
    raise AssertionError("a typo in perception.backend must not build the local stub")
except ValueError as exc:
    assert "locla" in str(exc), exc

assert isinstance(get_perception_backend({"perception": {"backend": "local"}}), LocalPerceptionBackend)
assert isinstance(get_perception_backend({}), LocalPerceptionBackend)

h = LocalPerceptionBackend().health()
assert h == {"reachable": True, "type": "local", "models_loaded": False}, h

sys.modules.pop("fal_client", None)
managed = ManagedPerceptionBackend(provider="fal", api_key="k")
try:
    managed.detect_and_segment(np.zeros((4, 4, 3), np.uint8), ["chair"])
    raise AssertionError("must refuse")
except NotImplementedError as exc:
    assert "parser" in str(exc), exc
assert "fal_client" not in sys.modules, "refused AFTER reaching for the provider client"
print("test_ga19_backend_leftovers: ok")
