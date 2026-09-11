#!/usr/bin/env bash
# A BASE RUN ON A MACHINE WITH NO SCREEN. Same run as ./run_sim.sh, with the three things that need a
# window turned off. Takes the same arguments:
#
#   ./run_sim_headless.sh                 the scene from the config file
#   ./run_sim_headless.sh hm3d_00861      that scene, this run only
#   ./run_sim_headless.sh --one-storey    a single storey
#
# WHY THIS SCRIPT EXISTS, measured on the Gin lab machine on 2026-09-10: six attempts between
# 12:24 and 12:59 all ended with `rviz2 exited with status -6` -- SIGABRT -- and run_sim.sh
# treats a missing node as fatal. THE GATE HAD ALREADY PASSED. So a headless machine produced a
# gate-passing run that then died on a viewer nobody was watching, and the cause was three steps
# from the symptom: no X server, so any Qt process aborts.
#
# WHAT IT SETS, and nothing else:
#   RVIZ=0        rviz2 is a viewer. It aborts without a display and its death ends the run.
#   FEED_SHOW=0   the preview window. Newer trees detect this themselves -- habitat_feed_host.py
#                 tests for the X SOCKET rather than trusting DISPLAY, because run_sim.sh exports
#                 DISPLAY=:1 whether or not a server is there. On an older tree the variable is
#                 what stops cv2.imshow killing the feed host.
#
# WHAT IT DOES NOT TOUCH: FEED_OVERLAY. The overlay draws the belief's boxes INTO the frame that
# goes to the dashboard over HTTP. It opens no window, so a headless run keeps it and you still see
# the boxes in the browser.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SAY WHAT THIS MACHINE ACTUALLY HAS, rather than assuming the reason you ran this script. A person
# who runs the headless script on a machine WITH a screen should be told they are giving up the
# viewers for nothing.
if [ -n "$(ls /tmp/.X11-unix/ 2>/dev/null)" ]; then
  echo "note: this machine HAS an X display ($(ls /tmp/.X11-unix/ | tr '\n' ' ')), so ./run_sim.sh would work"
  echo "      and would give you rviz and the preview window. Continuing without them."
else
  echo "no X display on this machine — running without rviz or the preview window"
fi

export RVIZ=0
export FEED_SHOW=0
# CALLED THROUGH bash, NOT AS ./run_sim.sh. Git records run_sim.sh as mode 100644 (checked
# 2026-09-11: every entry script in this repository is 100644), so on a fresh clone
# "$HERE/run_sim.sh" fails with "Permission denied". It only looks executable in a working tree
# whose filesystem is permissive, which is why this path has never been exercised from a
# clean clone. Invoking the interpreter removes the dependency on the mode entirely.
exec bash "$HERE/run_sim.sh" "$@"
