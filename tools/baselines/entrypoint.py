"""Small entry point imported by each baseline's run_scheduled.py."""
import sys


def main(baseline):
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        if baseline == 'hovsg':
            from .hovsg_run import main as run
            return run(sys.argv[2:])
        if baseline == 'dynamicgsg':
            from .dynamicgsg_run import main as run
            return run(sys.argv[2:])
        from .clio_run import main as run
        return run(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'export-bag':
        from .clio_bag import main as export
        sys.argv.pop(1)
        return export()
    from .runtime import main as acquire
    return acquire()
