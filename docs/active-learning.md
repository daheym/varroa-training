# Continuous active learning with varroa-labeling

The normal local cycle combines two integration paths. For interactive work, the labeling
backend discovers completed compatible runs under `../varroa-training/runs` and starts this
repository's `src.cli infer` in its own Python environment when you request predictions for
an image. Reviewed annotations then return to training through a versioned dataset export.
Training itself is still run here; the labeling app does not start training.

Dataset snapshots and batch inference use the **files plus JSON Schema** contract. The two
repositories keep local implementations and duplicate
`schemas/varroa-annotation-v1.schema.json`; update both schema copies together when changing
that exchange contract. The interactive path additionally depends on the `src.cli infer`
arguments and preprocessing sidecar staying compatible with the labeling backend.

## Coordinates and provenance

Original images are the source of truth. Export copies their exact uploaded bytes;
filenames become stable image IDs plus the detected `.jpg` or `.png` extension to avoid
collisions between uploads with the same filename. IDs survive every round trip.
Width, height, points and boxes refer to **EXIF-oriented original-image pixels**:
apply EXIF orientation before measuring or displaying, without downsampling.
The origin is top left; x increases right and y increases down. Points satisfy
`0 <= x < width`, `0 <= y < height`. Bounding boxes use top-left x/y and positive
width/height, and must fit inside the image. Coordinates retain floating-point precision.

`perspective` is optional metadata (`null` when absent). It never changes the exchanged
coordinate space. Existing physical-normalization metadata can be preserved here;
local corrected-image filenames are informational, not portable dependencies.
The annotator converts canonical points through its inverse homography on export and
through its current homography on import. Training owns normalization, tiling and merging.
Prediction centers and bounding-box corners are mapped back after duplicate merging;
a transformed box becomes an axis-aligned enclosing box, clipped to original bounds.
No exchanged coordinates are tile-local or canonical.

Each annotation has `id`, `class: "varroa"`, `x`, `y`, `source`, `confidence`,
`model_version` and `reviewed`. `bbox` is optional or null. Humans have
`source: "human"`, `reviewed: true`, normally null confidence/model version.
New model predictions require `source: "model"`, numeric confidence in [0,1],
a nonempty model version and `reviewed: false`. Accepting or correcting a prediction
sets reviewed to true and preserves source, confidence and model version. Moving a
point invalidates its old bbox, so the corrected annotation has `bbox: null`.
The original box and coordinates remain in prediction history.

Every exchanged JSON document has `schema_version: "1.0"`. This implementation rejects
all other versions, including unsupported major versions, with a clear error. Future
minor versions require explicit support; unknown fields are rejected to catch mistakes.
JSON Schema handles structure and provenance. Pydantic and bundle validation additionally
handle dimensions, finite numbers, coordinate bounds, duplicate IDs and review state.

## Directory layout

```text
varroa-dataset/
  manifest.json
  images/<image-id>.jpg
  annotations/<image-id>.json
  reviews/<image-id>.json       # completed-image statistics, separate from training labels
```

A manifest has schema_version, dataset_version, created_at (RFC 3339), images with
id/filename/annotation_file, plus two deliberately explicit fields:

- `purpose`: `ground_truth`, `unreviewed`, or `predictions`.
- `reviewed_image_ids`: IDs whose entire image has been checked, including empty images.

Ground-truth import requires purpose ground_truth, every manifest image explicitly
reviewed, and every retained annotation reviewed. It rejects incomplete datasets;
it does not filter unreviewed predictions and silently train on the remainder.
A reviewed empty image is a valid negative. A missing annotation file is never a negative.
Image IDs, filenames and annotation filenames must be unique; annotation IDs must be
unique across the bundle. Member paths cannot traverse directories or escape via symlinks.
Only manifest-listed images are part of the dataset.

Prediction directories contain just `manifest.json` and `annotations/`. They reference
images already present in the annotator by stable ID, filename and dimensions. An
unreviewed export includes original images so the training repository can perform inference.
Prediction export skips explicitly completed images and fails if there are none left.

Directories can be copied, shared through linked folders, or ZIPped using an ordinary
archive tool and extracted before import. There is no ZIP extraction endpoint. A linked
folder path is interpreted on the **API server's machine**, not the browser's machine.
Keep export and processed directories separate from application data. Use fresh paths;
exports and predictions refuse to overwrite existing directories. The manifest is written
last; treat it as the publication marker. Do not consume a directory while it is being written.

## Continuous local cycle (recommended)

This is the shortest path when `varroa-training` and `varroa-labeling` are sibling folders on
the same machine. The labeling API server must be able to read the training repository and
its `.venv`. By default it uses `../varroa-training/.venv/Scripts/python.exe` on Windows or
`../varroa-training/.venv/bin/python` on Linux/macOS. The `VARROA_TRAINING_ROOT` and
`VARROA_TRAINING_PYTHON` environment variables override those defaults; restart the API after
changing them.

### Bootstrap the first model

1. In the annotator, physically correct each seed image, add or verify every mite, and click
   **Finish image review (including empty images)**. The standard holdout preparation needs
   at least two distinct source images and positive training supervision.
2. In Dataset exchange, export the completed images as ground truth to a new directory, for
   example `F:/ImkerApp/exports/reviewed-v001`, with a unique dataset version.
3. From the `varroa-training` root, import that immutable snapshot and train into new output
   paths. In Windows PowerShell:

   ```powershell
   $python = ".\.venv\Scripts\python.exe"
   $dataset = "F:\ImkerApp\exports\reviewed-v001"
   & $python scripts\import_dataset.py $dataset --output data\processed\active-001
   & $python scripts\train.py data\processed\active-001\training-config.json --output runs\active-001
   ```

   On Linux/macOS, the equivalent commands from this repository root are:

   ```bash
   ./.venv/bin/python scripts/import_dataset.py /path/to/exports/reviewed-v001 --output data/processed/active-001
   ./.venv/bin/python scripts/train.py data/processed/active-001/training-config.json --output runs/active-001
   ```

   Import validates that every image is explicitly reviewed and every retained
   annotation is reviewed. It creates the tiled dataset, split, manifest and reusable
   `training-config.json`; the trainer records a new run directly in `runs/active-001`.
   Choose fresh `data/processed` and `runs` names for every cycle because these commands do
   not overwrite previous artifacts.

4. Wait for training to finish successfully. The annotator only offers completed runs with
   their metadata, configuration, checkpoint and preprocessing sidecar intact. Keep the run
   under this repository's `runs` directory. In the labeling app, enter Model Review mode,
   click **Refresh model list**, select the new run, and click **Request predictions** on a
   physically corrected image.

### Add reviewed predictions to the next cycle

1. In Model Review mode, inspect the entire image. Accept correct predictions, drag incorrect
   ones to the mite center, delete false detections, and add missed mites manually. Then click
   **Finish image review**. For a genuinely empty image, inspect it and finish the review even
   when there are no markers. Predictions stay unreviewed until this review is completed.
2. In Dataset exchange, export completed images as ground truth to a **new** directory and
   dataset version. Include the earlier reviewed images as well as the newly reviewed images
   so this is a cumulative training snapshot. Accepted and corrected predictions are now
   reviewed annotations; rejected predictions are excluded. Incomplete images are not
   eligible for training export.
3. Import and train the new snapshot with new output names, for example:

   ```powershell
   $python = ".\.venv\Scripts\python.exe"
   $dataset = "F:\ImkerApp\exports\reviewed-v002"
   & $python scripts\import_dataset.py $dataset --output data\processed\active-002
   & $python scripts\train.py data\processed\active-002\training-config.json --output runs\active-002
   ```

   Repeat the review, cumulative export, import and training steps for each cycle. By default,
   each run initializes from the model configured in the imported training config (normally
   `yolo11n.pt`) and learns from the full cumulative snapshot. To deliberately warm-start from
   an earlier detector, set `model` in the new `training-config.json` to that run's
   `yolo/weights/best.pt` before training. Use a new run output either way; `resume` is for
   continuing an interrupted run with its optimizer state, not starting a new active-learning
   cycle.

The model selector only shows runs whose physical-scale settings match the annotator's
normalization settings. If a completed run does not appear, refresh the list and check the
training API path, `.venv`, and that `configs/normalization.json` agrees with the labeling
repository's normalization values (`pixels_per_mm`, `min_source_scale_ratio`, and
`quality_policy`). Each prediction request requires a current physical perspective correction.
The request runs on the local API server and persists candidates in the labeling app; there is
no prediction bundle to export or import for this per-image path. An image accepts only one
prediction batch at a time, so normal cycles request predictions for new images. To deliberately
run a new model on an already-predicted image, first preserve its reviewed dataset export, then
use the annotator's reset action before requesting another batch.

## Batch / portable exchange workflow

Use this route to run predictions over a set of images in one command, or to move datasets
between machines. For ordinary local, per-image review, use the continuous cycle above.

1. **Annotate images.** Upload JPEG/PNG files in the annotator. Annotate the original
   view directly, or configure physical normalization and annotate the corrected view.
   Click **Finish image review (including empty images)** after inspecting all mites.
2. **Export dataset.** Under Dataset exchange, enter a new linked folder such as
   `F:/ImkerApp/exports/varroa-dataset-v1` and version `2026-09-12-001`.
   Click **Export completed images as ground truth**. This selects only explicitly
   completed images; the API rejects any selected incomplete image.
3. **Import into training repo.** From varroa-training, using its installed environment:

   ```bash
   python scripts/import_dataset.py ../exports/varroa-dataset-v1 --output data/processed/v1
   ```

   This validates original images and annotations and generates YOLO tiles/labels only
   under data/processed/v1. It leaves the input directory unchanged. A split record is
   written beside the processed directory and a reusable training-config.json inside it.
   At least two distinct source images are required for the default holdout split.
   For point-only datasets without physical normalization, explicitly append
   `--allow-legacy-original` (also supported by predict_dataset.py). An optional
   `--config configs/custom.json` controls box sizes, tiling and other existing settings.
4. **Train model.**

   ```bash
   python scripts/train.py data/processed/v1/training-config.json --output runs/active-v1
   ```

   Keep trained weights together with their existing preprocessing contract sidecar.
5. **Predict unreviewed images.** Upload/select the next images in the annotator and
   export them to a fresh folder with **Export unreviewed images for prediction**.
   The manifest purpose is unreviewed and its reviewed_image_ids is empty.
6. **Export predictions.** The prediction command runs inference and exports in one step:

   ```bash
   python scripts/predict_dataset.py --dataset ../exports/unreviewed-v1 --model runs/active-v1/yolo/weights/best.pt --model-version active-v1 --config data/processed/v1/training-config.json --output predictions/active-v1
   ```

   Use the actual trained checkpoint path and retain its preprocessing metadata.
   model_version defaults to the checkpoint stem; choose a unique immutable version
   explicitly for repeatable audits. The output contains no duplicate image copies.
7. **Import predictions into annotator.** Enter the prediction directory's absolute
   server path under Dataset exchange and click **Import prediction directory**.
   All members are validated before document changes. Images must already exist with
   matching IDs, filenames and dimensions. No filename-only fuzzy matching is performed.
8. **Review/correct.** Select Review mode. Blue dashed markers are unreviewed model
   predictions; green markers are confirmed/human annotations. Select a marker to see
   confidence. Accept selected, drag to correct, delete to reject, and click to add missed
   mites manually. Finish image review explicitly when done. Statistics appear below
   Dataset exchange and are also available from the API.
9. **Export new dataset version.** Choose a fresh directory and version such as
   `2026-09-12-002`. Export completed images, including previously reviewed images
   for a cumulative training set. Rejected predictions are excluded from labels.
10. **Retrain.** For a cumulative next training set, export all reviewed images as in the
    continuous cycle above, then import into a fresh processed directory and train to a fresh
    run directory. The batch prediction bundle itself is not a training dataset.

    ```bash
    python scripts/import_dataset.py ../exports/varroa-dataset-v2 --output data/processed/v2
    python scripts/train.py data/processed/v2/training-config.json --output runs/active-v2
    ```

## API examples for linked folders

The UI uses these same local API operations. POST JSON to the annotator (default port 8000):

```bash
curl -X POST http://127.0.0.1:8000/api/exchange/export -H "Content-Type: application/json" -d '{"directory":"F:/ImkerApp/exports/v1","dataset_version":"2026-09-12-001","purpose":"ground_truth","image_ids":["IMAGE_UUID"]}'
curl -X POST http://127.0.0.1:8000/api/exchange/import -H "Content-Type: application/json" -d '{"directory":"F:/ImkerApp/varroa-training/predictions/varroa-v3"}'
curl http://127.0.0.1:8000/api/images/IMAGE_UUID/review-statistics
```

Omit image_ids to export all images; ground_truth then requires all images complete.
Set image review through the existing PUT `/api/images/{id}/annotations` document API
with `review_complete: true`; unresolved predictions prevent completion.

## Review history and statistics

The annotator retains the validated original prediction JSON under
`data/prediction_history/<id>.json` and its pre-import local document beside it.
Reviewed exports write `reviews/<id>.json` separately. Counts describe final outcomes,
not mouse-event totals: model_predictions, accepted_unchanged, corrected, rejected,
manually_added, final_reviewed_mite_count. Corrections compare final original coordinates
with the original prediction (1e-7 pixel tolerance for homography round-off), plus bbox.
Undoing a move restores its previous outcome. Manually added counts exclude human points
already present when the batch arrived. Existing effort event counters remain separate.

A second prediction batch for the same image is rejected rather than overwriting its
review history. Normally each cycle predicts new unreviewed images. To intentionally
re-evaluate the same image, first export its reviewed dataset, then use the existing
explicit reset action; reset archives the document and original prediction history
before clearing annotations. Import the new batch afterward. Historical reviewed data
remains in the earlier dataset version; reset does not automatically combine it with
new predictions. The exchange API validates all members first but does not provide a
multi-file database transaction; keep backups if the local disk can fail during writes.

After complete review, prediction precision can use
(accepted_unchanged + corrected) / model_predictions and recall can use
(accepted_unchanged + corrected) / (accepted_unchanged + corrected + manually_added),
with explicit handling for zero denominators. These are review-outcome estimates:
localization-quality metrics need the preserved original predictions and reviewed ground
truth. Effort reduction also needs a defined manual baseline; raw counts alone are not
a measured time saving.

## Compatibility and scope

Existing unversioned local labeling documents load with safe defaults (`review_complete`
is false). Existing legacy training inputs and commands still work. The old single-image
annotation export/candidate endpoint remains available; use Dataset exchange for the
portable v1 contract. New local documents include a schema version, but their internal
canonical representation is distinct from exchange documents.

The current training pipeline derives fixed-size boxes from reviewed point centers,
including when an optional source bbox is present. It preserves the source bbox in JSON;
it does not yet train using variable-size imported boxes. This retains the existing
training semantics and avoids coupling training geometry to the annotator.

Examples are in `examples/interchange/`: annotation.json, prediction.json, manifest.json,
and review.json. They are illustrative individual files, not a full image dataset.

## Tests

Training: `python -m pytest` from varroa-training.
Annotator: `python -m pytest backend/tests` and `npm --prefix frontend test` from varroa-labeling.
Frontend production check: `npm --prefix frontend run build`.
The tests exercise reviewed-state rejection, schema versions, missing images, dimensions,
bounds, duplicate IDs, provenance, tiled and perspective coordinate conversion, preserved
source bytes, empty reviewed images, and prediction/review/export round trips.
