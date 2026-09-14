# DynamicGSG external baseline integration

Every run names one source variant. `upstream-pristine` is the clean official
commit. `upstream-pristine-io` adds only the common final-output adapter needed
for offline completion. `upstream-execfix` adds the isolated removal repair to
that common adapter. `delivered-found-fork` preserves the separately delivered
repository for provenance and comparison. `algorithm-experiment` is never a
paper baseline row. The selected checkout is always mounted at `/baseline:ro`.

## Input

`dynamicgsg_export.py` converts a completed shared acquisition to the native
`ReplicaDataset` layout. RGB and depth files are hard links, so the exported
RGB remains the original lossless PNG even though the immutable native reader
selects it through a `frame*.jpg` glob. `traj.txt` converts each Habitat/OpenGL
camera-to-world pose to DynamicGSG's OpenCV camera basis by right-multiplying
`diag(1,-1,-1,1)`. This preserves the Habitat world frame while matching native
positive-Z depth backprojection. `frame_mapping.jsonl` and
`export_manifest.json` bind every exported index to source hashes.
The versioned input contract rejects legacy exports that did not declare this
basis conversion, and verifies the generated trajectory hash before inference.

For a bounded smoke on Gin:

```bash
python3 -m tools.baselines.dynamicgsg_export \
  --recording /dev/shm/graphapi-baselines-824-two-lap-20260913/recording \
  --output /dev/shm/dynamicgsg-824-smoke-input \
  --max-frames 220
```

The transferred checkout lacks every `configs/data/*.yaml` file. The external
runner therefore derives the missing camera/depth YAML from the acquisition.
It derives focal length from the recorded horizontal field of view and keeps
the native millimetre depth scale of 1000.

## Runtime

Build `dynamicgsg-baseline:cu121` with the native checkout as context and
`tools/baselines/dynamicgsg.Dockerfile` as the Dockerfile. The final image has
the compiled CUDA and GroundingDINO packages but no copied baseline source.
`gin.sh dynamicgsg ...` mounts the exact checkout, model files, Hugging Face
cache (OpenCLIP and BERT), Torch cache (LPIPS AlexNet), integration code, and
input read-only. Only a new result directory is writable. Network access is
disabled for measured runs. Exact cache identities are recorded in
`artifacts/baselines/dynamicgsg-prep-20260913/MODELS.json`.

The paper-candidate runner starts from the official
`configs/realsense/dgsg.py` profile. Its detector, vocabulary, dynamic-update
switch, tracking and mapping settings, cadence, thresholds, and iteration counts
remain upstream values. Dataset layout, camera dimensions, pinned model paths,
output paths, W&B, checkpoints, and display behavior are external runtime
adaptations. The dataset scheduler may set `data.frame_begin_update`; the
resolved value is labelled `dataset_scheduler`. A stride override is permitted
only for an explicitly labelled smoke or differential test.

The offline runner disables the optional DAM/Qwen description pass. Those
descriptions are written only during final serialization and do not feed native
tracking, association, Gaussian mapping, or dynamic removal. The execution-fix
source serializes the existing native object records when DAM is disabled; it
does not invent replacement categories. Raw records contain per-detection
`class_id` histories, but those IDs index per-frame detector vocabularies and are
not a stable semantic label. The graph and result therefore identify this mode
as `class_agnostic_geometry_tracking`, set `semantic_accuracy_eligible: false`,
and mark its timing as excluding the official DAM/Qwen post-processing. Do not
report it as a semantic or complete official end-to-end timing result.

Official semantic output requires the exact DAM-3B snapshot plus the upstream
`qwen2.5-vl-72b-instruct` category/caption calls. The published script contains
only the placeholder `YOUR QWEN API KEY` and no endpoint/service recipe. Until
both model/service identities are pinned and exercised, substituting detector
class IDs or another captioner would change the baseline and is prohibited.
`MPLBACKEND=Agg` and Xvfb make the
upstream comparison plot non-interactive without changing numeric processing.

The Gin launcher defaults to the verified cumulative execution-fix checkout,
official profile, and exact patch hash. A dynamic recording must additionally
declare its scheduler-derived effective-frame boundary:

```bash
DYNAMICGSG_DYNAMIC_START_FRAME=50 \
tools/baselines/gin.sh dynamicgsg INPUT_EXPORT OUTPUT
```

The default patch is
`4766e876fbba97c50d57b3c5c4e2c10f48cbda4e038fadf5c4b4eceec96d5deb`:
the removal transaction, raw no-DAM serialization, and `int32` ownership/mask
storage. The latter covers the first-frame mask, later empty/non-empty masks,
and stored Gaussian ownership. It preserves upstream IDs below the old limits
and prevents silent wrap at 128/255. The delivered fork remains available only through an
explicit root, variant, profile, and matching patch policy.

The launcher rejects a dirty checkout unless its binary Git diff matches the
declared hash. `upstream-pristine` rejects any patch. `upstream-execfix` rejects
an empty patch. The output records the variant, revision, patch hash, profile,
and a source label for every resolved configuration leaf.

## Output and replay

A successful run must contain native `params_with_idx.npz`, `objects.pkl.gz`,
non-empty fused object geometry, `dynamicgsg_graph.json`, measured
`execution_timing.json`, and a final `baseline_result.json` with
`complete: true`. The native dataset normalizes trajectories to the first
camera. The output adapter applies the first exported
OpenCV-camera-to-Habitat-world pose exactly once to saved `means3D`. Replay and
native 3D scene exporters accept the normalized baseline name `dynamicgsg`.

This preparation does not itself establish accuracy. Accuracy requires a
completed run and the same declared GT matching policy used for the other
baselines.
