"""Read required ROS exits even while startup waits for model/TF readiness."""
import json
import os
from pathlib import Path


def apply_component_settings(components, env):
    """Explain disabled perception and identify the actual camera input."""
    if env.get('TIAGO_BAG_MODE') == '1':
        components['feed']['name'] = 'TIAGO Bag Replay'
    elif env.get('PAL_ROBOT_CONNECTED') == '1':
        components['feed']['name'] = 'TIAGO Robot Cameras'
    if env.get('FOUND_START_PERCEPTION') == '0':
        for key in ('perception', 'object_manager'):
            components[key].update(active=False, enabled=False,
                                   details='Disabled for this run (--no-perception)')
    return all(component['active'] for component in components.values() if component.get('enabled', True))


def pane_exit(line, pending, now, bag_code=None):
    """Wait briefly for tmux to publish a closed pane's numeric exit status."""
    name, dead, code = (line.split('|') + ['', '', ''])[:3]
    if dead != '1':
        pending.pop(name, None)
        return None
    if name in ('rviz', 'monitor'):
        return None
    if name == 'bag' and bag_code is not None and bag_code.strip().isdecimal():
        code = bag_code.strip()
    if not code:
        since = pending.setdefault(name, now)
        if now - since < 2:
            return None
        code = 'status unavailable; possibly terminated by a signal'
    return name, code


def stack_failure(events, launcher_pid):
    path = Path(events)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # The ROS callback may still be appending its last line.
            if event.get('required') and not event.get('during_shutdown') and not event.get('expected'):
                return event
    try:
        os.kill(launcher_pid, 0)
        stat = Path(f'/proc/{launcher_pid}/stat')
        if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
            return None
    except (ProcessLookupError, FileNotFoundError):
        pass
    return {'component': 'ROS2_LAUNCH', 'returncode': 1, 'required': True,
            'during_shutdown': False}
