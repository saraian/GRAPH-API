#!/usr/bin/env python3
"""GA-183 instrument: read BOTH logs of a run bundle before saying 'zero merge errors'.
Usage: bridge_merge_check.py <run_dir>   (exit 1 on any /merge 500, bridge Timeout, or service failure)"""
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
bridge = (run / "logs" / "bridge.log").read_text(errors="replace") if (run / "logs" / "bridge.log").exists() else ""
om6 = (run / "logs" / "om6.log").read_text(errors="replace") if (run / "logs" / "om6.log").exists() else ""
n = {
    "merge_200": len(re.findall(r'POST /merge HTTP/1.1" 200', bridge)),
    "merge_202_pending": len(re.findall(r'POST /merge HTTP/1.1" 202', bridge)),
    "merge_500": len(re.findall(r'POST /merge HTTP/1.1" 500', bridge)),
    "bridge_timeouts": len(re.findall(r"Timeout (dopo|in attesa)", bridge)),
    "om6_merge_service_failed": len(re.findall(r"_cb_merge_objects failed", om6)),
}
print(f"{run.name}: " + "  ".join(f"{k}={v}" for k, v in n.items()))
bad = n["merge_500"] + n["bridge_timeouts"] + n["om6_merge_service_failed"]
print("merge errors across bridge.log AND om6.log:", bad)
sys.exit(1 if bad else 0)
