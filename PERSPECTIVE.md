# Physical normalization contract v1

The invariant is a fixed physical scale, not a fixed board size. A 400 x 600 mm board at 8 px/mm produces 3200 x 4800 pixels; a 300 x 500 mm board produces 2400 x 4000 pixels. No dimensions are inferred from camera aspect ratio, and no board profile is selected implicitly.

## Configuration and metadata

The annotation repository uses `normalization.json` (override path with `VARROA_NORMALIZATION_CONFIG`). Training reads `configs/normalization.json` through `read_config`; experiment JSON can explicitly override scale/quality settings. Keep these settings synchronized. Python `Config` defaults match the shipped JSON. Board profiles are named width/height pairs; custom values are supported by the UI and API. Profiles resolve to explicit per-image physical dimensions at save time.

```json
{
  "pixels_per_mm": 8,
  "min_source_scale_ratio": 0.75,
  "quality_policy": "warn",
  "board_profiles": {
    "400x600": {"physical_width_mm": 400, "physical_height_mm": 600},
    "300x500": {"physical_width_mm": 300, "physical_height_mm": 500}
  }
}
```

The document-wide `coordinate_space` guarantees one space for every retained and rejected annotation. New annotations use `canonical`; legacy documents use `original`. Point-level coordinate-space fields are rejected, preventing contradictory mixed spaces. Metadata lives in `perspective` and records source points, original dimensions, physical dimensions, pixels/mm, canonical/output dimensions, forward/inverse homographies, version, interpolation, quality estimate and warning. The annotation app also records the generated raster filename and selected profile.

## Deterministic geometry / future PWA

1. Decode the complete source and apply EXIF orientation once. Corners and original coordinates refer to that oriented raster. Preserve original uploaded bytes separately.
2. The user supplies semantic board TL, TR, BR, BL, clockwise in downward-positive image Y. The first edge is physical width; the second is height. Rotation of the photographed board is allowed. Do not sort by screen X/Y, automatically rotate, or swap width/height. Validate finite, in-bounds, distinct, convex corners and positive winding; reject crossed, reversed or degenerate input.
3. Compute `W = physical_width_mm * pixels_per_mm`, `H = physical_height_mm * pixels_per_mm`. Require integer results (within 1e-7), at least two pixels per side, and at most 80 million output pixels. Reject fractional results rather than quietly changing the scale.
4. Map board edges to `(0,0), (W,0), (W,H), (0,H)`. The raster samples `[0,W) x [0,H)`. This is deliberately an **edge** convention, not `(W-1,H-1)`, so a physical millimetre maps to exactly the configured scale.
5. Solve eight linear equations in float64 with `H[2,2]=1`. For each `(x,y)->(u,v)`, rows are `[x,y,1,0,0,0,-u*x,-u*y]` and `[0,0,0,x,y,1,-v*x,-v*y]`, right sides `u,v`. Store the 3x3 matrix row-major plus its inverse. Transform points by multiplying `[x,y,1]` and dividing by the third component.
6. Resample directly from the oriented original with OpenCV `warpPerspective`, `INTER_LINEAR`, constant-zero border, exact output `(W,H)`. No intermediate resize. The two Python implementations are mirrored and tested for byte equality when both repositories are checked out. Port these equations, edge convention, and interpolation settings to TypeScript/WASM/OpenCV.js; compare float coordinates with tolerance, not exact equality.
7. Annotate, tile, train and infer in canonical pixels only. Tile windows have fixed pixel stride/size. A board smaller than a tile is padded with black on the right/bottom; labels and returned detections are clipped to valid board pixels. The original exterior is never included. Tile offsets map boxes to canonical space; inverse-project all four box corners to return an `original_polygon`, because an original-space result is generally not an axis-aligned rectangle.

## Source quality

Compute the inverse-transform Jacobian at a deterministic 9 x 9 grid including board edges. The minimum of its smallest singular values, multiplied by target pixels/mm, estimates the weakest directional source pixels/mm. Store `source_effective_pixels_per_mm`; warn or reject if it is below `pixels_per_mm * min_source_scale_ratio`. Default: warn below 75% of target. This is a sampling estimate, not a guarantee about focus, blur, lens distortion, or all possible extrema between grid samples. Upscaling never creates true detail. Physical consistency assumes correct dimensions, corners and a planar surface.

## Migration and compatibility

Existing files are never bulk rewritten automatically. In the annotation tool, open Original, choose physical dimensions, review the four corners and generate correction. Original-space retained/rejected points transform through the new homography. Existing canonical points first map back through the old inverse, then through the new homography. If any point falls outside the new region, reject the operation and preserve the current document. Correct an overly tight region or curate the legacy records explicitly; no points are silently dropped. Successful operations archive the exact previous JSON in `data/history/` and retain previous corrected rasters. Undo history is reset, and stale saves using an earlier perspective are rejected.

Training can convert original-space points when physical perspective metadata is available. Old perspective metadata without physical dimensions is rejected with a migration message, even in legacy mode. Regenerate it in the annotation tool; physical dimensions cannot be recovered from a homography alone. Old datasets with no perspective at all are accessible only with explicit `allow_legacy_original: true`; that compatibility path does not claim normalized physical scale. Default preparation and inference require a physical region. Legacy experiments should be retrained for the new contract.

Canonical annotations are checked against the recomputed transform and output dimensions. A different pixels/mm value is an error, not an implicit resize. `box_size_mm` optionally sets the square pseudo-box side in millimetres; pixels are `box_size_mm * pixels_per_mm`. Otherwise `box_size` is in canonical pixels. Fixed tile dimensions and model `imgsz` are part of the trained preprocessing contract. Geometric scale/perspective augmentation is disabled so training does not randomize this scale.

## Inference and model artifacts

`predict_image(model, original_path, config, perspective=metadata)` uses the same resolve/normalize operation as preparation. Evaluation and visualizations use the saved canonical annotation snapshot. Outputs declare canonical coordinate space, dimensions, perspective metadata and original polygons. `save_prediction` renders the canonical raster.

Each trained weight file receives `<weights-stem>.preprocessing.json`. CLI evaluation/inference and models exposing `ckpt_path` check it against normalization, quality, tiling and model-input settings. Custom detector adapters must enforce the same contract before calling preprocessing. ONNX export includes this contract in its JSON sidecar. A PWA must collect per-photo board dimensions/corners and normalize before the detector; feeding a full uncorrected photo is invalid.

Candidate import into annotation requires `coordinate_space: "canonical"`, `perspective_file` equal to the current document's perspective filename, `model_version`, and annotation points. Detector boxes can be converted to center points in canonical pixels before import. This ties predictions to the displayed transform and prevents stale batches after recorrection.
