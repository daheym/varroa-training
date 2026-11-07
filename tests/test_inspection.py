import importlib
import json

from PIL import Image
import pytest

from src.config import Config, write_json
from src.dataset.prepare import prepare
from src.dataset.quality import review_key, intersects_exclusion
from src.dataset.validation import validate_dataset
from src.evaluation.inspection import build_inspection


def fixture_dataset(tmp_path):
    images, annotations = tmp_path / 'images', tmp_path / 'annotations'
    images.mkdir()
    annotations.mkdir()
    for i in range(2):
        Image.new('RGB', (256, 128), color=(i * 30, 100, 50)).save(images / f'{i}.png')
        write_json(annotations / f'{i}.json', {'points': [{'x': 20, 'y': 20}, {'x': 200, 'y': 30}]})
    return Config(images=images, annotations=annotations, allow_legacy_original=True,
                  processed=tmp_path / 'processed', splits=tmp_path / 'splits.json',
                  tile_size=128, overlap=32, negative_fraction=1)


def test_diagnostic_predictions_and_gallery_preserve_validation_metrics(tmp_path, monkeypatch):
    config = fixture_dataset(tmp_path)
    manifest = prepare(config)
    evaluation = importlib.import_module('src.evaluation.evaluate')
    monkeypatch.setattr(evaluation, 'predict_image', lambda *a, **kw: {
        'detections': [{'box': [70, 70, 80, 80], 'confidence': .7}]})
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'dataset_manifest.json', manifest)
    baseline = evaluation.evaluate(None, manifest, config, run / 'baseline.json')
    diagnostics = evaluation.evaluate(None, manifest, config, run / 'evaluation.json', inspect_training=True)
    assert baseline['aggregate'] == diagnostics['aggregate']
    assert len(diagnostics['training_diagnostics']) == len(manifest['splits']['train'])
    assert build_inspection(run).is_file()
    gallery = json.loads((run / 'inspection' / 'manifest.json').read_text())
    annotations = [i for i in gallery['items'] if i['kind'] == 'annotation']
    assert len(annotations) == 4  # No duplicate object crops from overlapping tiles.
    assert all(item['status'] == 'missed' for item in annotations)
    assert any(item['status'] == 'extra' for item in gallery['items'])
    assert any('train' in item['splits'] for item in annotations)
    for item in gallery['items']:
        with Image.open(run / 'inspection' / item['crop']) as image:
            assert image.size == (96, 96)
    # Old runs still work, with unavailable training results explicitly labeled.
    write_json(run / 'evaluation.json', baseline)
    build_inspection(run)
    assert 'not_evaluated' in (run / 'inspection' / 'manifest.json').read_text()
    Image.new('RGB', (256, 128), 'black').save(config.images / '0.png')
    with pytest.raises(ValueError, match='changed since training'):
        build_inspection(run)


def test_legacy_flags_keep_tiles_and_neighboring_mites(tmp_path):
    config = fixture_dataset(tmp_path)
    records = validate_dataset(config.images, config.annotations, config=config)
    record = records[0]
    region = {'x': 20, 'y': 20, 'crop_size': 96, 'reason': 'poor_quality'}
    write_json(tmp_path / 'quality_reviews' / '0.json', {
        'schema_version': 1, 'regions': {review_key(record, 20, 20): region}})
    manifest = prepare(config)
    assert 'quality_exclusions' not in manifest['sources'][0]
    assert any(intersects_exclusion((t['x'], t['y'], t['width'], t['height']), [region])
               for t in manifest['tiles'] if t['source_id'] == '0')
    # Old flags do not change current annotations or remove training context.
    assert [20, 20] in json.loads((config.processed / 'manifest.json').read_text())['sources'][0]['points']
    from src.dataset.quality import load_exclusions
    assert load_exclusions({**record, 'sha256': 'changed'}, config.annotations) == []
    assert load_exclusions({**record, 'points': [(21, 20)]}, config.annotations) == []


def test_deleted_label_becomes_background_without_losing_neighbor(tmp_path):
    config = fixture_dataset(tmp_path)
    # A real mite sits close to the incorrectly labeled object at (20, 20).
    write_json(config.annotations / '0.json', {'points': [{'x': 24, 'y': 20}, {'x': 200, 'y': 30}]})
    manifest = prepare(config)
    source = next(r for r in manifest['sources'] if r['id'] == '0')
    assert [20, 20] not in [list(p) for p in source['points']]
    assert [24, 20] in [list(p) for p in source['points']]
    tiles = [t for t in manifest['tiles'] if t['source_id'] == '0' and t['x'] == 0]
    assert tiles and all(t['positive'] for t in tiles)
    for tile in tiles:
        assert len((config.processed / tile['label']).read_text().splitlines()) == 1
