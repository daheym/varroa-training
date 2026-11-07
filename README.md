# Varroa training

A Python 3.12 pipeline for training and evaluating a pretrained YOLO Varroa detector. Training selects CUDA when available and otherwise uses CPU; the original-image annotation and evaluation workflow remains available locally. Point-derived boxes are temporary supervision, not measured mite outlines.

## Setup

Run commands from this repository root. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,export]"
.\.venv\Scripts\python.exe -m pytest
```

If Python is managed by uv, use `uv venv --python 3.12 --seed .venv` for the first command. On Linux use `python3.12 -m venv .venv` and `.venv/bin/python`. Activate the environment to use the shorter `python` and `varroa` commands below. Pretrained weights download on the first run; use a local `.pt` path in `model` for offline runs.

The CPU installation follows [PyTorch's installation guidance](https://pytorch.org/get-started/locally/). Training defaults to `device: auto` (CUDA when available, otherwise CPU), supports explicit devices, and records its resolved settings and software versions. Fixed seeds and deterministic algorithms are used; exact numerical reproducibility still depends on software versions and hardware.

`requirements-cpu-win-py312.txt` pins the tested Windows environment. For the same versions, install its pinned `torch`/`torchvision` from the CPU index first, then run `python -m pip install -r requirements-cpu-win-py312.txt` and `python -m pip install -e ".[dev,export]"`.

## Layout

```text
data/images/         Input JPEG/PNG originals
data/annotations/    One JSON per image
data/processed/      Generated YOLO datasets and tile manifests
data/splits/         Generated-tile split manifests
configs/             JSON configuration files
models/              Optional local pretrained/exported models
runs/                Experiments, metrics, weights and predictions
training/            Canonical train and standalone export entry points
cloud/kaggle/        Thin Kaggle launcher and setup instructions
scripts/             Legacy command and evaluation/inference wrappers
src/dataset/         Validation, geometry, splitting, tiling
src/training/        Training and experiment records
src/evaluation/      Point matching, image counts, leave-one-out
src/inference/       Full-image prediction, NMS, visualization, ONNX
tests/               Synthetic unit and pipeline tests
```

Data and generated models/runs are excluded from Git.

In this workspace, `data/images` and `data/annotations` are Windows directory junctions to the matching folders in `../varroa-labeling/data`. The default config therefore reads the live labeling inputs. `data/processed` and `data/splits` remain local to this training repository. The junctions are local filesystem setup and are not stored in Git.

## Annotation formats and inspection

The sibling repository `../varroa-labeling` was inspected before finalizing the adapter. Its two initial JSON files have 2 and 8 reviewed points. `configs/labeling.json` reads its `data/images` and `data/annotations` directly without modifying them. Match `<image_id>.json` with `<image_id>.png`, **not** the human-readable `filename` or the `.original` upload. The PNG is already EXIF-oriented by the labeling app. Keep the original PNG as input: preprocessing uses physical perspective metadata to normalize it before tiling. Original-space legacy points are converted; canonical points are validated against the transform. See [the physical normalization contract](PERSPECTIVE.md).

The supported labeling document is:

```json
{
  "image_id": "example-id",
  "filename": "original-photo.jpg",
  "original_width": 4000,
  "original_height": 3000,
  "coordinate_space": "original",
  "perspective": null,
  "dataset_version": "1",
  "model_version": null,
  "annotations": [
    {"x": 120.5, "y": 85.0, "id": "point-id", "className": "varroa",
     "source": "human", "confidence": null, "reviewed": true}
  ],
  "rejected_predictions": [],
  "effort": {}
}
```

Only reviewed `annotations` become ground truth; unreviewed entries cause an error. Rejected predictions and effort metadata are not training labels. Perspective metadata defines the canonical raster and physical scale. Unknown document and point fields are rejected to expose schema changes.

For independently collected data, also accepted is `photo.json` alongside `photo.jpg` or `photo.png`:

```json
{"image": "photo.jpg", "width": 4000, "height": 3000, "points": [{"x": 120.5, "y": 85.0}]}
```

`image`, `width` and `height` are optional in this compact format. `points` is required, including an explicit empty list for a negative image. The legacy example above requires explicit `allow_legacy_original: true` and its coordinates refer to EXIF-oriented full-resolution pixels, with the top-left at `(0,0)`, `0 <= x < width`, and `0 <= y < height`. Annotations must cover all visible mites: an omitted mite becomes incorrect background supervision.

Validation checks decoding, dimensions, required fields, finite numeric coordinates, bounds, duplicate points/IDs, missing/orphan annotations, duplicate image stems and identical decoded image content. Similar photographs, bursts or crops are not automatically detected.

## Prepare and train

### Next production training run

Run the production launcher from the repository root:

```powershell
.\scripts\train-production.ps1
```

The script validates the live labeling data, prepares fresh tiles, and trains for up to
50 epochs, stopping after 10 epochs without validation improvement. It chooses the next unused name (`training-production-01`,
`training-production-02`, and so on) and uses that name for the run, processed dataset,
split manifest, and generated config. This prevents `prepare` from colliding with an
existing `data/processed/labeling` directory or `data/splits/labeling.json` file.

Override the epoch count or base config when needed:

```powershell
.\scripts\train-production.ps1 -Epochs 100
.\scripts\train-production.ps1 -Epochs 100 -Patience 15
.\scripts\train-production.ps1 -BaseConfig configs/labeling.json -RunPrefix training-production
```

Generated configs are stored under `runs/.configs/`. `configs/labeling.json` remains the
source configuration and continues to read images and annotations from the sibling
`../varroa-labeling` repository.

### Single-image overfit experiment

For exactly one annotated original, run this command from the repository root (no separate preparation step):

```powershell
.\.venv\Scripts\python.exe -m src.cli --config configs/overfit.json overfit --output runs/single-image
```

If additional images have been added, explicitly select the annotated original by its filename stem:

```powershell
.\.venv\Scripts\python.exe -m src.cli --config configs/overfit.json overfit --image-id cea9932e-3b40-4c8c-ad13-8b01f54fc8a7 --output runs/single-image
```

The selected image and its annotations are validated and captured in the run manifest; other originals are excluded from this experiment.

This uses the linked `data/images` and `data/annotations`, a fresh pretrained `yolo11n.pt`, CPU, 100 epochs, 640-pixel inputs and batch size 2. Edit `configs/overfit.json` to change epochs, input size or pseudo-box size. All positive and negative tiles are retained by this config. Random color and geometric augmentation are disabled for the memorization check. Choose a new output directory for each experiment.

There is **no independent validation set**. The manifest contains one training original and an empty validation split. Ultralytics requires a validation path even with validation disabled, so its internal finalization evaluates training tiles. The experiment uses fixed epochs and `last.pt`, never selects a checkpoint using supposed held-out performance, and computes full-image reconstruction metrics. Both JSON and CLI reports label these as `training_image_reconstruction`; they cannot establish accuracy on unseen images. Tile holdout can use one source when it generates multiple tiles; leave-one-original-out requires at least two originals.

Outputs include:

```text
runs/single-image/config.json                         Resolved run configuration
runs/single-image/dataset/manifest.json               Source snapshot and tile mapping
runs/single-image/experiment/yolo/weights/last.pt      Trained checkpoint
runs/single-image/experiment/evaluation.json          Training-image point/count metrics
runs/single-image/experiment/reconstruction/*.png     Full-image detection visualization
runs/single-image/experiment/reconstruction/*.json    Full-image detections and count
```

Repeat reconstruction with the saved run config:

```powershell
.\.venv\Scripts\python.exe -m src.cli --config runs/single-image/config.json evaluate --weights runs/single-image/experiment/yolo/weights/last.pt --output runs/single-image/experiment/recheck.json
```

### Tile holdout experiment

```powershell
.\scripts\train-production.ps1
```

The same prepared dataset can be trained through the canonical provider-neutral entry point. Its `--output` is a run directory root; it creates a unique `run-<timestamp>` child unless `--run-name` is supplied:

```bash
python training/train.py \
  --config configs/labeling.json \
  --data data/processed/labeling/dataset.yaml \
  --output runs \
  --run-name first
```

For an already labeled YOLO dataset (`images/{train,val}` and `labels/{train,val}`), pass its `dataset.yaml` directly. This route uses Ultralytics validation metrics; the project's generated-tile point metrics require the prepared dataset and manifest above.

```bash
python training/train.py \
  --config training/configs/varroa.yaml \
  --data ./datasets/varroa/dataset.yaml \
  --output ./runs
```

The configuration accepts `device: auto`, `cpu`, `cuda`, `cuda:0`, or a CUDA index. CLI values such as `--device`, `--seed`, `--imgsz`, `--batch-size`, `--epochs`, `--patience`, `--workers`, and `--resume` override the config. `patience` is the number of epochs without validation improvement before early stopping; zero disables it. A resume checkpoint must retain Ultralytics optimizer state, usually an interrupted run's `last.pt`.

For inputs copied into this repository, use `configs/default.json`. Config paths are relative to your working directory. Copy a config and change `processed` and `splits` when preparing a new variant. Preparation refuses existing output data/split files to prevent stale tiles from contaminating a new experiment. Training also requires a new run directory.

Preparation generates the tile windows and labels first, then deterministically assigns tile IDs to train or validation. A source image may therefore contribute tiles to both splits. It writes:

```text
data/processed/labeling/
  dataset.yaml
  manifest.json
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
```

`manifest.json` includes source paths, dimensions, decoded-image and annotation SHA-256 hashes, the annotation points, split membership and the configuration. Every emitted tile records its image/label path, source ID/path, split, top-left `(x,y)`, actual width/height and label count. Local boxes map back by adding `(x,y)`. A source smaller than the tile size remains at its actual size; YOLO handles letterboxing. Edge tiles are anchored to the image boundary, so their overlap may be larger than requested.

| Setting | Default | Meaning |
| --- | --- | --- |
| `box_size` | 32 | Square pseudo-box side length in canonical pixels; clipped at image edges |
| `tile_size` / `overlap` | 640 / 128 | Square tile side and overlap in pixels, `0 <= overlap < tile_size` |
| `negative_fraction` | 0.25 | Seeded fraction of available negative training tiles retained per original, rounded up; 0 excludes, 1 keeps all |
| `val_fraction` / `seed` | 0.25 / 42 | Deterministic generated-tile split, with at least one train and one validation tile |
| `model` | `yolo11n.pt` | Small pretrained detector variant or local checkpoint |
| `imgsz` / `batch` | 640 / 2 | YOLO input size (multiple of 32) and batch size |
| `epochs` / `patience` / `workers` | 20 / 10 / 0 | Maximum epochs, early-stopping patience, and worker count; zero workers is Windows-friendly |
| `confidence` | 0.25 | Inference confidence threshold |
| `merge_iou` | 0.3 | Tile and global non-maximum suppression threshold |
| `match_radius` | 16 | Evaluation center-to-point tolerance in canonical pixels |
| `max_det` | 3000 | Maximum detections per tile; raise for exceptionally dense tiles |

All positive tiles are retained. Intersecting boxes are clipped into every tile they touch, including boxes whose point is just outside a tile. A tile is negative only if no pseudo-box intersects it. Validation retains **all** negative tiles irrespective of `negative_fraction`. Holdout mode requires at least two generated tiles and positive training supervision; it can use one or more source images. Overfit mode still requires exactly one original.

Because neighboring tiles can overlap, a tile-level split may place shared pixels in both train and validation. Its metrics measure performance on held-out tiles, not generalization to a new photograph. Use leave-one-original-image-out when source-level independence is required.

Use a box size representative of mites at your source resolution. Overlap at least as large as the box size helps edge objects appear fully in another tile. NMS can still leave duplicates with low IoU, or merge nearby mites if boxes are too large. Inspect visualizations before interpreting counts.

Each training run stores its resolved `config.yaml`, `data.yaml`, `metadata.json`, `metrics.json`, `requirements-freeze.txt`, and Ultralytics outputs under `yolo/`, including `weights/best.pt`, `weights/last.pt`, CSV logs, and plots. Prepared point-annotation runs also save `dataset_manifest.json` and `evaluation.json`. Metadata records requested/resolved device, versions, seed/config, and relevant hashes. Mosaic/mixup are disabled for the initial small-data baseline. Normal train/validation runs use early stopping and select `weights/best.pt` using validation performance; `last.pt` remains available for resuming. These scores are exploratory validation scores, not independent test results. Overfit and leave-one-original-out protocols intentionally disable early stopping and evaluate `last.pt` after their fixed epoch budget.

For Kaggle setup, the lightweight launcher, GPU instructions, dependency guidance, and artifact locations are in [cloud/kaggle/README.md](cloud/kaggle/README.md). It calls this same training entry point. ONNX export remains a separate step, for example `python training/export.py --checkpoint runs/run-20260913-120000/yolo/weights/best.pt --format onnx`.

## Evaluate and infer

### Interactive mite overview and quality review

Open **Mite overview** in the sibling Varroa Labeling browser interface. It shows
one 96px context crop per annotated mite, plus unmatched prediction crops, with
run, source, result, split and quality filters, adjustable density and enlarged
details. A mite can belong to both train and validation because tiles overlap.
Mixed outcomes retain every tile result. Unmatched predictions can repeat across
overlapping tiles. These are disagreements with annotations, not proof that a label
or prediction is wrong.

New prepared-data training runs automatically save `inspection/manifest.json`,
`inspection/crops/*.jpg`, and a portable `inspection/index.html`. They also predict
training tiles after fitting for inspection. These reconstruction results are stored
in `evaluation.json` under `training_diagnostics` and never enter validation scores.
External YOLO datasets without this project's source manifest do not get this overview.

Build an overview for an older completed run without inference or retraining:

```powershell
.\.venv\Scripts\python.exe -m src.cli inspect --run runs/training-production-04
```

The labeling tab can also build older overviews. Crops without saved predictions
are marked **Not evaluated**. The standalone HTML supports filters, enlargement,
and printing the visible page to PDF. Keep it beside its `crops` directory.

In the labeling tab, **Not a mite — remove label** confirms and deletes just that
annotation from the live labeled data. The next `prepare` uses those current labels:
the removed object has no positive target, and neighboring mites and tiles remain
available. An already completed image review stays complete. A red × in the labeling
overview reflects the current absence of the label; historical crops and run scores
remain unchanged. Legacy quality flags in `data/quality_reviews/` are ignored during
preparation and can be revisited in the labeling app's legacy-flags filter. They are
not automatically converted into label deletions. Previously prepared datasets and
run artifacts are unchanged.

Visualize an existing experiment without retraining or running inference again:

```powershell
.\.venv\Scripts\python.exe -m src.cli visualize --run runs/single-image/experiment
```

The `visualizations` folder contains a full-image overlay, paginated close-ups of discrepancies, an `errors.json` with canonical-pixel coordinates, and training-loss curves. Green boxes are matched predictions, cyan crosses are annotation points, red crosses/circles are unmatched annotations, and orange boxes are unmatched predictions. Matching uses the same one-to-one rule and tolerance as evaluation, with the saved annotation snapshot. The source image hash is checked before rendering. These labels describe agreement with annotations; inspect close-ups for annotation mistakes as well as model errors.

For overfit runs, Ultralytics' epoch validation is disabled: zeros in intermediate validation columns are placeholders, not measured zero accuracy. The plots therefore show training losses, not misleading validation curves. Falling loss and strong reconstruction scores demonstrate learning on the training image, not generalization.

```powershell
python -m src.cli --config configs/labeling.json evaluate --weights runs/first/yolo/weights/best.pt --output runs/first/recheck.json
python -m src.cli --config configs/labeling.json infer --weights runs/first/yolo/weights/best.pt --image ../varroa-labeling/data/images/5102ef6f-7636-4774-b6c9-71a6d0e692a2.png --perspective ../varroa-labeling/data/annotations/5102ef6f-7636-4774-b6c9-71a6d0e692a2.json --output runs/first/prediction
```

Inference first normalizes the selected physical region, visits every canonical tile, maps boxes to canonical coordinates and applies confidence-ordered global IoU NMS. It prints the final mite count and writes `<output>.json` plus a full-resolution `<output>.png` visualization. Output coordinates and visualization use canonical pixels; inverse-transformed original polygons are included separately. A matching weight preprocessing sidecar is required.

Evaluation predicts the validation tiles and matches predicted box centers to the generated tile-label centers within `match_radius`, one-to-one, maximizing valid matches then minimizing distance. Reports contain TP/FP/FN, precision, recall, F1, true/predicted counts, signed counting error (`predicted - true`) and absolute error per tile. Aggregates use micro precision/recall/F1 and count MAE, RMSE and bias. Undefined precision/recall/F1 are reported as zero, including empty/empty cases; counting error is then zero. Leave-one-original-out and overfit runs retain their full-image evaluation protocols.

## Leave-one-original-image-out

```powershell
python -m src.cli --config configs/labeling.json loo --output runs/loo-first
```

Each fold prepares a fresh dataset and initializes a fresh model from `model`. One original is held out, fixed epochs are used, early stopping is disabled, and `last.pt` is evaluated. The outer held-out image is excluded from Ultralytics' training/validation data configuration; any internal finalization metrics on the training set are not held-out results. Every original contributes once to `runs/loo-first/evaluation.json`; partial completion is recorded in `progress.json`. Avoid tuning thresholds/epochs on these same held-out results if you need an unbiased final estimate. A fold with no training positives fails explicitly.

## ONNX

```powershell
python -m src.cli --config configs/labeling.json export --weights runs/first/yolo/weights/best.pt
```

Exports static batch-one float32 ONNX, opset 17, without embedded NMS, next to the checkpoint. The script checks the graph and writes an adjacent JSON describing preprocessing and weights. This follows [Ultralytics ONNX export options](https://docs.ultralytics.com/modes/export/). A future browser client still needs RGB conversion, aspect-preserving letterboxing, normalization, output decoding, filtering, tile mapping and NMS; this repository does not implement the browser client. Output tensor layouts can differ between configured model variants, so inspect the exported graph when integrating.

## First experiment

`configs/smoke.json` uses the sibling dataset, one epoch and 320-pixel input for a quick integration check. Run its `validate`, `prepare`, and `train` commands with a fresh output directory. One epoch on a few source images establishes execution, not detection quality.

For the learning experiment, inspect source annotation completeness and pseudo-box sizes, then run a fixed 20-epoch leave-one-out protocol. Review false positives, missed mites, counts and visualizations alongside metrics, and compare count error against a trivial zero-count baseline. The pretrained COCO model has no Varroa class; transfer learning replaces its detection head. With only two initial images, estimates will be highly variable; collect more independent sources before claiming generalization.

## Physical-scale preprocessing

Default preparation requires per-image physical dimensions and perspective metadata. The legacy JSON examples above describe compatibility inputs, not physically normalized training data. Use the annotation tool to migrate them first. `configs/normalization.json` defines 8 pixels/mm by default, warning below 75% source resolution; `quality_policy: "error"` rejects low-resolution inputs. Board dimensions vary per image. `box_size_mm` overrides `box_size` with a physical pseudo-box size. Tiles remain fixed in pixels; small boards use blank padding. See [PERSPECTIVE.md](PERSPECTIVE.md) for the full reproducible algorithm, migration, quality checks and model/PWA contract.

## Active-learning interchange

For the local continuous cycle, train from a reviewed dataset snapshot here, then use
**Request predictions** in the sibling `varroa-labeling` app. The app invokes this repository's
compatible completed run for inference; after review, export a cumulative ground-truth snapshot
and import it into a new training cycle. Training remains a command run from this repository.
The [active-learning runbook](docs/active-learning.md) covers the interactive cycle and the
separate batch/file exchange workflow. See also the [v1 JSON Schema](schemas/varroa-annotation-v1.schema.json)
and [example files](examples/interchange).
