# Shared single-floor inputs

Selected and locally validated on 2026-09-13. Use each complete schedule and the
same resulting observation recording for Clio, HOV-SG and GRAPH-API.

| FOUND scene | Annotated room regions | Native GT objects | Scan stops | Navmesh area |
|---|---:|---:|---:|---:|
| 00813-svBbv1Pavdk | 10 | 394 | 42 | 75.83 m² |
| 00880-Nfvxx8J5NCo | 12 | 360 | 43 | 69.91 m² |

Evidence: `artifacts/baselines/selected-single-floor-scenes.json`. Complete asset
paths, SHA-256 hashes, resolved dataset configurations and native object GT exports
are under `artifacts/baselines/selected-scenes-20260913-v2/`. Scene 824 in that
extraction directory is an evaluation reference for the existing replay, not a
third selected scene.

For both selected scenes the RGB mesh, semantic GLB, semantic descriptor, navmesh
and full schedule exist and were read. Native Habitat object IDs match every
semantic descriptor ID. Canonical `schedule_batch.scene_storeys` finds one floor,
no minor levels and zero unassigned sample share. Every schedule goal snaps within
0.25 m and every consecutive path, including the last-to-first lap closure, is
connected. Route height ranges are 0.0775–0.1686 m for 813 and exactly 0.1809 m for
880. These are dataset/route checks; no new baseline run has been performed.
Annotated room counts do not prove that every room will be observed during a run.

The distributed dataset config assumes a different asset directory layout. The
verifier imports its stage defaults and writes a separate configuration with
absolute RGB/semantic paths; it does not modify dataset files. Use these resolved
paths when loading GT. A descriptor that loads without its native geometry is
rejected, rather than accepted as usable GT.

Reproduce in the Habitat environment with GPU access and a new output directory:

```bash
python -m tools.baselines.select_scenes \
  --scenes 00813-svBbv1Pavdk 00880-Nfvxx8J5NCo \
  --output artifacts/baselines/selected-scenes-new
```

Remaining before baseline execution: compile scene-specific dynamic scripts using
the shared FOUND compiler, check Gin storage (2.5 GB free at this verification),
record complete tours, and run both unchanged native pipelines on identical inputs.
No multi-floor dependency is needed for these two single-floor scenes.
