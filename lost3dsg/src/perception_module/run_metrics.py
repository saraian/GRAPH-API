#!/usr/bin/env python3
"""Esegue in sequenza la pipeline di valutazione HM3D.

Esempio:
    python3 run_metrics.py

I valori predefiniti riproducono i tre comandi usati per la scena 00824.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_OUTPUT = HERE.parents[2] / "output"
# Keep metric artifacts next to this script.  In the standard container this
# is /root/exchange/lost3dsg/src/perception_module, which is bind-mounted and
# therefore visible on the host as well.
PROJECT_OUTPUT = HERE


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="ID scena HM3D, per esempio 824 o 00824")
    parser.add_argument("--ground-truth", type=Path,
                        default=None,
                        help="manifest ground truth")
    parser.add_argument("--run-dir", type=Path, default=RUN_OUTPUT,
                        help="directory con gli artefatti del run")
    parser.add_argument("--persistent-perception", type=Path,
                        help="JSON persistente da usare, eventualmente arricchito con CLIP")
    parser.add_argument("--manifest-output", type=Path,
                        default=None,
                        help="manifest valutato da generare")
    parser.add_argument("--metrics-output", type=Path,
                        default=None,
                        help="report JSON delle metriche")
    parser.add_argument("--visualization-output", type=Path,
                        default=None,
                        help="pagina HTML con le visualizzazioni (default: output del progetto)")
    args = parser.parse_args()

    scene = str(args.scene).strip().lower().removeprefix("scene_").lstrip("0") or "0"
    if args.ground_truth is None:
        args.ground_truth = HERE / f"manifest_gt_{int(scene):05d}.json"
    ground_truth = args.ground_truth.resolve()
    run_dir = args.run_dir.resolve()
    if not ground_truth.is_file():
        parser.error(f"manifest ground truth inesistente: {ground_truth}")
    if not run_dir.is_dir():
        parser.error(f"run directory inesistente: {run_dir}")

    scene_name = ground_truth.stem
    if scene_name.startswith("manifest_gt_"):
        scene_name = scene_name.removeprefix("manifest_gt_")
    if not scene_name:
        scene_name = "scene"
    if args.manifest_output is None:
        args.manifest_output = PROJECT_OUTPUT / f"manifest_eval_{scene_name}.json"
    if args.metrics_output is None:
        args.metrics_output = PROJECT_OUTPUT / f"risultati_eval_{scene_name}.json"

    # La cartella output del progetto è quella condivisa insieme al codice.
    # Scrivere in /root/exchange/output non garantisce invece che il file sia
    # visibile sull'host: dipende dalla configurazione dei volumi Docker.
    if args.visualization_output is None:
        args.visualization_output = PROJECT_OUTPUT / f"boxes_{scene_name}.html"

    args.manifest_output = args.manifest_output.resolve()
    args.metrics_output = args.metrics_output.resolve()
    args.visualization_output = args.visualization_output.resolve()
    persistent_perception = (args.persistent_perception.resolve()
                             if args.persistent_perception else None)
    if persistent_perception is not None and not persistent_perception.is_file():
        parser.error(f"JSON persistente inesistente: {persistent_perception}")

    for output in (args.manifest_output, args.metrics_output,
                   args.visualization_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    evaluation_script = HERE / "build_hm3d_eval_manifest.py"
    metrics_script = HERE / "metrics_eval.py"
    visualize_script = HERE / "metrics_eval_visualize.py"

    _run([sys.executable, str(evaluation_script),
          "--ground-truth", str(ground_truth),
          "--run-dir", str(run_dir),
          *(["--persistent-perception", str(persistent_perception)]
            if persistent_perception else []),
          "--output", str(args.manifest_output)])
    _run([sys.executable, str(metrics_script),
          str(args.manifest_output),
          "--output", str(args.metrics_output)])
    _run([sys.executable, str(visualize_script),
          str(args.manifest_output),
          "--output", str(args.visualization_output)])

    if not args.visualization_output.is_file():
        raise RuntimeError(f"visualizzazione non creata: {args.visualization_output}")

    print("\nCreati:")
    print(args.manifest_output)
    print(args.metrics_output)
    print(args.visualization_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
