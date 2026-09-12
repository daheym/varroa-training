"""Persist with detector weights; preprocessing is part of the model contract."""
import json
from pathlib import Path


KEYS = ('pixels_per_mm', 'tile_size', 'overlap', 'imgsz', 'min_source_scale_ratio',
        'quality_policy', 'allow_legacy_original')


def preprocessing_contract(config):
    return {'normalization_version': 1, 'pixel_convention': 'board_edges',
            **{key: getattr(config, key) for key in KEYS}}


def check_model_contract(weights, config):
    path = Path(weights).with_suffix('.preprocessing.json')
    if not path.exists():
        if config.allow_legacy_original:
            return
        raise ValueError(f'Missing model preprocessing contract: {path}. Train with physical normalization first.')
    if json.loads(path.read_text('utf-8')) != preprocessing_contract(config):
        raise ValueError('Inference preprocessing differs from trained model contract')
