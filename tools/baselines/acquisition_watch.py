"""Run one shared acquisition with a guard on its own RAM scratch consumption."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .run_pair import write


def main():
    root=Path(sys.argv[1]).resolve()
    command=json.loads((root/'acquisition-command.json').read_text())
    process=subprocess.Popen(command, start_new_session=True)
    status={'pid':process.pid,'started_at':time.time(),'complete':False}
    write(root/'acquisition-watch.json',status)
    while process.poll() is None:
        free=shutil.disk_usage('/dev/shm').free
        if free<3*1024**3:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            status.update(error='Stopped this acquisition before shared RAM scratch exhaustion',
                scratch_free_bytes=free,returncode=process.returncode,finished_at=time.time())
            write(root/'acquisition-watch.json',status)
            return 1
        time.sleep(5)
    status.update(complete=process.returncode==0,returncode=process.returncode,finished_at=time.time())
    write(root/'acquisition-watch.json',status)
    return process.returncode


if __name__=='__main__':
    raise SystemExit(main())
