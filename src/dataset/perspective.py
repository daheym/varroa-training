"""Physical normalization v1. Mirrored in varroa-training/src/dataset/perspective.py.

Corners are semantic TL, TR, BR, BL (clockwise in image coordinates).
Board edges map to (0,0), (W,0), (W,H), (0,H); pixels sample [0,W) x [0,H).
This edge convention preserves exactly pixels_per_mm, without a W-1 scale error.
"""
import math
import warnings

import cv2
import numpy as np
from PIL import Image


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return []
    result = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if not np.isfinite(result).all() or np.any(np.abs(result[:, 2]) < 1e-12):
        raise ValueError("Invalid homogeneous coordinate")
    return (result[:, :2] / result[:, 2:]).tolist()


def build_transform(source_points, original_width, original_height, physical_width_mm,
                    physical_height_mm, pixels_per_mm=8, min_source_scale_ratio=0.75,
                    quality_policy="warn"):
    values = [physical_width_mm, physical_height_mm, pixels_per_mm]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Physical dimensions and pixels_per_mm must be finite and positive")
    if not 0 < min_source_scale_ratio <= 1 or quality_policy not in ("warn", "error"):
        raise ValueError("Invalid quality policy")
    src = np.asarray(source_points, dtype=np.float64)
    if src.shape != (4, 2) or not np.isfinite(src).all():
        raise ValueError("Exactly four finite source points required")
    if np.any(src < 0) or np.any(src[:, 0] >= original_width) or np.any(src[:, 1] >= original_height):
        raise ValueError("Corners outside image")
    edges = np.roll(src, -1, axis=0) - src
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(cross <= 1e-6) or cv2.contourArea(src.astype(np.float32)) < 4:
        raise ValueError("Corners must be convex clockwise TL, TR, BR, BL; no crossings or duplicates")
    # Reject fractional output pixels rather than silently changing the requested scale.
    dimensions = [physical_width_mm * pixels_per_mm, physical_height_mm * pixels_per_mm]
    if any(abs(v - round(v)) > 1e-7 for v in dimensions):
        raise ValueError("Physical dimensions times pixels_per_mm must be integer pixels")
    w, h = map(round, dimensions)
    if min(w, h) < 2 or w * h > 80_000_000:
        raise ValueError("Canonical image must be between 2 pixels per side and 80 megapixels")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    # Float64 linear solve, portable to TS/WASM. H[2,2] = 1.
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    try:
        matrix = np.append(np.linalg.solve(a, b), 1).reshape(3, 3)
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular perspective transform") from exc
    # Minimum local source px/mm: smallest singular value of inverse Jacobian
    # on a deterministic 9x9 board grid, including edges. Estimate, not an optical MTF.
    effective = float("inf")
    for y in np.linspace(0, h, 9):
        for x in np.linspace(0, w, 9):
            q = inverse @ [x, y, 1]
            jac = (inverse[:2, :2] * q[2] - np.outer(q[:2], inverse[2, :2])) / q[2]**2
            effective = min(effective, float(np.linalg.svd(jac, compute_uv=False)[-1]) * pixels_per_mm)
    low = effective < pixels_per_mm * min_source_scale_ratio
    warning = (f"Insufficient source resolution: estimated {effective:.3f} px/mm; target {pixels_per_mm:g} px/mm. "
               "Upscaling does not recover true detail.") if low else None
    if warning and quality_policy == "error":
        raise ValueError(warning)
    return dict(version=1, coordinate_space="canonical", corner_order="TL,TR,BR,BL",
                source_points=src.tolist(), original_width=original_width, original_height=original_height,
                physical_width_mm=physical_width_mm, physical_height_mm=physical_height_mm,
                pixels_per_mm=pixels_per_mm, canonical_width_px=w, canonical_height_px=h,
                output_width=w, output_height=h, width=w, height=h,
                homography=matrix.tolist(), inverse_homography=inverse.tolist(),
                source_effective_pixels_per_mm=effective, min_source_scale_ratio=min_source_scale_ratio,
                quality_policy=quality_policy, quality_warning=warning,
                interpolation="opencv_INTER_LINEAR", border_mode="constant_zero", pixel_convention="board_edges")


def normalize_image(image, metadata):
    return Image.fromarray(cv2.warpPerspective(np.asarray(image.convert("RGB")),
        np.asarray(metadata["homography"]), (metadata["output_width"], metadata["output_height"]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))


def resolve_metadata(metadata, width, height, pixels_per_mm=8, min_source_scale_ratio=.75, quality_policy="warn"):
    if not metadata:
        raise ValueError("Physical perspective region required")
    required = ("physical_width_mm", "physical_height_mm")
    if any(k not in metadata for k in required):
        raise ValueError("Legacy perspective needs physical board dimensions; regenerate in annotation tool")
    if metadata.get("pixels_per_mm", pixels_per_mm) != pixels_per_mm:
        raise ValueError("pixels_per_mm differs from normalization config; migrate annotations explicitly")
    result = build_transform(metadata.get("source_points", metadata.get("source_corners")), width, height,
        metadata["physical_width_mm"], metadata["physical_height_mm"], pixels_per_mm,
        min_source_scale_ratio, quality_policy)
    if result["quality_warning"]:
        warnings.warn(result["quality_warning"], UserWarning, stacklevel=2)
    return result
