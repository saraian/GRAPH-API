"""ROS launch wrapper: retain each launch graph and record required child exits."""
import json
import os
from pathlib import Path
import time

from launch import LaunchDescription
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _exit(event, context):
    details = event.action.process_details
    name = details.get('name', 'unknown')
    cmd = details.get('cmd', [])
    playback = list(cmd[:3]) == ['ros2', 'bag', 'play']
    expected = playback and event.returncode == 0
    required = 'rviz' not in name.lower()
    path = Path(os.environ['GRAPHAPI_COMPONENT_EVENTS'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(component=name, returncode=event.returncode,
                                     required=required, expected=expected,
                                     during_shutdown=bool(context.is_shutdown), time=time.time())) + '\n')
    if (required or expected) and not context.is_shutdown:
        return [EmitEvent(event=Shutdown(reason=f'{name} exited ({event.returncode})'))]
    return []


def generate_launch_description():
    return LaunchDescription([
        RegisterEventHandler(OnProcessExit(on_exit=_exit)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.environ['GRAPHAPI_LAUNCH_SOURCE'])),
    ])
