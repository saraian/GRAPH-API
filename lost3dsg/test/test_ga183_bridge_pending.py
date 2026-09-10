"""GA-183: a service call that was DISPATCHED and timed out is DispatchedTimeout, and the
bridge answers it with 202 {pending: true}, never 500."""
import json
import os
import sys
from types import SimpleNamespace as NS

os.environ["BRIDGE_SERVICE_TIMEOUT"] = "0.1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import graph_api_bridge as gb  # noqa: E402

client = NS(wait_for_service=lambda timeout_sec: True,
            call_async=lambda req: NS(done=lambda: False, result=lambda: None, exception=lambda: None))
node = NS(cli={"merge": client})
try:
    gb.BridgeNode.call(node, "merge", object())
    raise AssertionError("must raise")
except gb.DispatchedTimeout as exc:
    assert "merge" in str(exc) and isinstance(exc, RuntimeError)
    resp = gb._dispatched_timeout(None, exc)
assert resp.status_code == 202, resp.status_code
body = json.loads(resp.body)
assert body["pending"] is True and body["success"] is False and "merge" in body["message"], body
print("OK GA-183: dispatched timeout -> 202 pending")
