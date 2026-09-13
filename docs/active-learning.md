# File-based active learning (schema 1.0)

The integration contract is **files plus JSON Schema**. Each repository runs independently;
neither imports the other's implementation. The schema is duplicated at
`schemas/varroa-annotation-v1.schema.json`. Each repository has local Pydantic models
and validates incoming and outgoing exchange JSON with that schema. The annotator
frontend mirrors the contract in TypeScript; its Python API performs runtime validation.
Update both schema copies together when changing the contract. No shared package is required.

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

## End-to-end workflow

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
   python scripts/predict_dataset.py --dataset ../exports/unreviewed-v1 --model models/varroa-v3.pt --model-version varroa-v3 --config data/processed/v1/training-config.json --output predictions/varroa-v3
   ```

   Use the actual trained checkpoint path and retain its preprocessing metadata.
   model_version defaults to the checkpoint stem; choose a unique immutable version
   explicitly for repeatable audits. The output contains no duplicate image copies.
7. **Import predictions into annotator.** Enter the prediction directory's absolute
   server path under Dataset exchange and click **Import prediction directory**.
   All members are validated before document changes. Images must already exist with
   matching IDs, filenames and dimensions. No filename-only fuzzy matching is performed.
8. **Review/correct.** Select Review mode. Amber dashed markers are unreviewed model
   predictions; green markers are confirmed/human annotations. Select a marker to see
   confidence. Accept selected, drag to correct, delete to reject, and click to add missed
   mites manually. Finish image review explicitly when done. Statistics appear below
   Dataset exchange and are also available from the API.
9. **Export new dataset version.** Choose a fresh directory and version such as
   `2026-09-12-002`. Export completed images, including previously reviewed images
   for a cumulative training set. Rejected predictions are excluded from labels.
10. **Retrain.**

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
