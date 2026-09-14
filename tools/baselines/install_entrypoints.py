"""Install only thin import entrypoints into existing baseline checkouts."""
import argparse
from pathlib import Path

TEMPLATE = '''#!/usr/bin/env python3
"""Import shared scheduled-tour/dynamic-dataset integration; implementation lives upstream."""
import os
from pathlib import Path
import sys

integration = Path(os.environ.get("BASELINE_INTEGRATION_ROOT", {root!r})).resolve()
if not (integration / "tools/baselines/entrypoint.py").is_file():
    raise SystemExit(f"Shared baseline integration is missing: {{integration}}")
sys.path.insert(0, str(integration))
from tools.baselines.entrypoint import main

if __name__ == "__main__":
    raise SystemExit(main({baseline!r}))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--integration-root', required=True)
    p.add_argument('--clio-root', required=True)
    p.add_argument('--hovsg-root', required=True)
    a = p.parse_args()
    for baseline, root in (('clio', a.clio_root), ('hovsg', a.hovsg_root)):
        output = Path(root) / 'run_scheduled.py'
        content = TEMPLATE.format(root=str(Path(a.integration_root).resolve()), baseline=baseline)
        if output.exists() and output.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite {output}')
        output.write_text(content)
        print(output)


if __name__ == '__main__':
    main()
