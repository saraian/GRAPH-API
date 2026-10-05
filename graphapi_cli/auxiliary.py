"""Register and supervise auxiliary workflows without changing their CLI payloads."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

from .registry import Registry, owned_containers, process_identity, write_json


def supervise(root, env, mode, factory, *, gpu=False, ports=None, need_domain=True, extra=()):
    env = dict(env)
    registry = Registry(env['WORKSPACE_ROOT'])
    index = env.get('BASELINE_GPU', env.get('GRAPH_API_GPUS', 'device=0').removeprefix('device=')) if gpu else None
    operation, allocated, domain = registry.reserve(mode, gpu=index, ports=ports, need_domain=need_domain, extra=extra)
    env.update(GRAPHAPI_OPERATION_ID=operation, GRAPH_API_RUN_ID=operation,
               GRAPH_API_CONTAINER_NAME='graphapi-' + operation.lower())
    if domain is not None:
        env['ROS_DOMAIN_ID'] = str(domain)
    env.update({key: str(value) for key, value in allocated.items()})
    if index is not None:
        env['BASELINE_GPU'] = index
        env['GRAPH_API_GPUS'] = 'device=' + index
    directory = registry.operations / operation
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(operation_id=operation, mode=mode, state='PREPARING', pid=os.getpid(),
                 process_identity=process_identity(os.getpid()), actor=env.get('GRAPHAPI_RESOURCE_ACTOR'),
                 project=str(root), workspace=env['WORKSPACE_ROOT'], ports=allocated, domain=domain,
                 gpu=index, created=time.time(), log=str(directory / 'launch.log'), bundles=[])
    status = directory / 'status.json'
    write_json(status, state)
    process, stopped = None, False

    def interrupt(_sig, _frame):
        nonlocal stopped
        stopped = True
        state['state'] = 'DRAINING'
        write_json(status, state)
        # Docker clients use --sig-proxy=false. Interrupting the client alone
        # leaves its server alive and keeps this stdout loop blocked forever.
        names = [name for name in owned_containers(operation) if name != state.get('actor')]
        if names:
            subprocess.run(['docker', 'stop', '-t', '10', *names],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        process = subprocess.Popen(factory(env), cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state.update(state='RUNNING', ready=time.time())
        write_json(status, state)
        print('operation: ' + operation, file=sys.stderr, flush=True)
        with open(state['log'], 'ab', buffering=0) as stream:
            for line in iter(process.stdout.readline, b''):
                stream.write(line)
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
        rc = process.wait()
        events = directory / 'component_events.jsonl'
        if events.exists():
            import json
            failures = [json.loads(line) for line in events.read_text().splitlines() if line]
            failures = [e for e in failures if e.get('required') and not e.get('during_shutdown') and not e.get('expected')]
            if failures:
                rc = rc or 1
                state['error'] = f"required component {failures[0]['component']} exited"

        state.update(state='INTERRUPTED' if stopped else ('COMPLETED' if rc == 0 else 'FAILED'),
                     returncode=130 if stopped else rc, finished=time.time())
        write_json(status, state)
        return state['returncode']
    except BaseException as exc:
        state.update(state='FAILED', error=str(exc), returncode=1, finished=time.time())
        write_json(status, state)
        raise
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        names = owned_containers(operation)
        if names:
            subprocess.run(['docker', 'stop', '-t', '30', *names], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not owned_containers(operation):
            registry.release(operation)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
