import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from src.config import Config
from src.dataset.perspective import build_transform, transform_points, normalize_image, resolve_metadata
from src.dataset.prepare import prepare
from src.dataset.validation import validate_dataset
from src.inference.predict import predict_image, save_prediction


@pytest.mark.parametrize('board', [(400,600), (300,500), (600,400)])
@pytest.mark.parametrize('source_scale', [1, 3, 9])
def test_same_physical_point_and_mite_size(board, source_scale):
    bw, bh = board
    # Independent physical-board -> camera projective model, including distance.
    camera = np.array([[source_scale, .12, 20], [.04, source_scale, 30], [.0001, .0002, 1.]])
    corners = transform_points([(0,0), (bw,0), (bw,bh), (0,bh)], camera)
    metadata = build_transform(corners, 10000, 10000, bw, bh, 8)
    physical = [(50,70), (51.5,70), (50,71)]
    source = transform_points(physical, camera)
    canonical = np.array(transform_points(source, metadata['homography']))
    assert np.allclose(canonical, np.array(physical)*8, atol=1e-7)
    assert np.linalg.norm(canonical[1]-canonical[0]) == pytest.approx(12)
    assert np.linalg.norm(canonical[2]-canonical[0]) == pytest.approx(8)
    assert np.allclose(transform_points(canonical, metadata['inverse_homography']), source)
    assert (metadata['output_width'], metadata['output_height']) == (bw*8,bh*8)


@pytest.mark.parametrize('corners', [
    [(0,0),(0,90),(90,90),(90,0)],  # reversed winding
    [(0,0),(90,90),(90,0),(0,90)],  # crossed
    [(0,0),(90,0),(20,20),(0,90)], # concave
    [(0,0)]*4,
    [(0,0),(90,0),(90,90),(float('nan'),90)],
])
def test_reject_corner_order_and_invalid_geometry(corners):
    with pytest.raises(ValueError):
        build_transform(corners, 100, 100, 10, 10)


def test_quality_and_dimensions():
    corners = [(0,0),(80,0),(80,80),(0,80)]
    p = build_transform(corners, 100, 100, 20, 20, 8)
    assert p['source_effective_pixels_per_mm'] == pytest.approx(4)
    assert 'true detail' in p['quality_warning']
    with pytest.raises(ValueError, match='Insufficient'):
        build_transform(corners, 100, 100, 20, 20, 8, quality_policy='error')
    assert build_transform(corners,100,100,20,20,8,min_source_scale_ratio=.4)['quality_warning'] is None
    with pytest.raises(ValueError, match='integer pixels'):
        build_transform(corners,100,100,20.01,20)
    with pytest.raises(ValueError, match='80 megapixels'):
        build_transform(corners,100,100,2000,2000)


def make_dataset(tmp_path, coordinate_space='original'):
    images, annotations = tmp_path/'images', tmp_path/'annotations'
    images.mkdir(); annotations.mkdir()
    # Red exterior must never enter tiles. Black board with a white annotated dot.
    image = np.zeros((100,140,3), dtype=np.uint8)
    image[:] = (255,0,0)
    image[20:81,20:101] = 0
    cv2.circle(image,(60,50),2,(255,255,255),-1)
    Image.fromarray(image).save(images/'board.png')
    p = build_transform([(20,20),(100,20),(100,80),(20,80)],140,100,10,7.5,8)
    point = [60,50] if coordinate_space == 'original' else [40,30]
    (annotations/'board.json').write_text(json.dumps(dict(width=140,height=100,
        points=[dict(x=point[0],y=point[1])],coordinate_space=coordinate_space,perspective=p)))
    return Config(mode='overfit',images=images,annotations=annotations,processed=tmp_path/'tiles',
        splits=tmp_path/'split.json',tile_size=32,overlap=8,negative_fraction=1,box_size_mm=1.5), p


@pytest.mark.parametrize('space',['original','canonical'])
def test_annotation_alignment_and_only_canonical_tiles(tmp_path, space):
    config, p = make_dataset(tmp_path,space)
    records = validate_dataset(config.images,config.annotations,config=config)
    assert np.allclose(records[0]['points'], [(40,30)])
    manifest = prepare(config,records=records)
    assert (manifest['sources'][0]['width'],manifest['sources'][0]['height']) == (80,60)
    saw_center = False
    for tile in manifest['tiles']:
        raster = np.array(Image.open(config.processed/tile['image']))
        assert raster.shape[:2] == (32,32)
        assert not np.any((raster[:,:,0]>0) & (raster[:,:,1]==0))
        assert tile['x']+tile['valid_width'] <= 80
        assert tile['y']+tile['valid_height'] <= 60
        for row in (config.processed/tile['label']).read_text().splitlines():
            _, x,y,w,h = map(float,row.split())
            if w*32 == pytest.approx(12) and h*32 == pytest.approx(12):
                assert (x*32+tile['x'],y*32+tile['y']) == pytest.approx((40,30))
                assert raster[round(y*32),round(x*32)].min() > 240
                saw_center = True
    assert saw_center


def test_inference_canonical_mapping_and_overlay(tmp_path):
    config,p = make_dataset(tmp_path)
    config = config.model_copy(update={'tile_size':96,'overlap':0})
    class Tensor:
        def __init__(self,v): self.v=v
        def cpu(self): return self
        def tolist(self): return self.v
    class Model:
        def predict(self,source,**kwargs):
            assert source.size == (96,96)
            assert source.getpixel((40,30)) == (255,255,255)
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[34,24,46,36],[81,70,90,90]]),conf=Tensor([.9,.8])))]
    result = predict_image(Model(),config.images/'board.png',config,p)
    assert result['coordinate_space'] == 'canonical'
    assert result['count'] == 1
    assert result['detections'][0]['box'] == (34,24,46,36)
    assert np.allclose(result['detections'][0]['original_polygon'],[(54,44),(66,44),(66,56),(54,56)])
    save_prediction(result,tmp_path/'prediction')
    assert Image.open(tmp_path/'prediction.png').size == (80,60)
    with pytest.raises(ValueError,match='required'):
        predict_image(Model(),config.images/'board.png',config)
    with pytest.raises(ValueError,match='differs'):
        predict_image(Model(),config.images/'board.png',config.model_copy(update={'pixels_per_mm':4}),p)


def test_legacy_perspective_never_falls_back_to_original(tmp_path):
    config,p = make_dataset(tmp_path)
    path=config.annotations/'board.json'
    doc=json.loads(path.read_text())
    doc['perspective']={'source_corners':p['source_points']}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='physical board dimensions'):
        validate_dataset(config.images,config.annotations,config=config.model_copy(update={'allow_legacy_original':True}))


def test_annotation_and_training_contract_copies_identical():
    sibling=Path(__file__).resolve().parents[2]/'varroa-labeling'
    if not sibling.exists():
        pytest.skip('Sibling annotation repository not checked out')
    assert (sibling/'backend/app/perspective.py').read_bytes() == (Path(__file__).resolve().parents[1]/'src/dataset/perspective.py').read_bytes()
    assert json.loads((sibling/'normalization.json').read_text()) == json.loads((Path(__file__).resolve().parents[1]/'configs/normalization.json').read_text())


def test_model_contract_rejects_scale_and_tile_mismatch(tmp_path):
    from src.inference.contract import check_model_contract, preprocessing_contract
    weights = tmp_path/'best.pt'
    config = Config()
    with pytest.raises(ValueError, match='Missing'):
        check_model_contract(weights, config)
    weights.with_suffix('.preprocessing.json').write_text(json.dumps(preprocessing_contract(config)))
    check_model_contract(weights, config)
    for change in ({'pixels_per_mm':4}, {'tile_size':320}, {'imgsz':320}):
        with pytest.raises(ValueError, match='differs'):
            check_model_contract(weights, config.model_copy(update=change))


def test_tampered_canonical_transform_and_missing_region_rejected(tmp_path):
    config,p = make_dataset(tmp_path,'canonical')
    path=config.annotations/'board.json'
    doc=json.loads(path.read_text())
    doc['perspective']['homography'][0][2] += 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='transform differs'):
        validate_dataset(config.images,config.annotations,config=config)
    doc['perspective']=None
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='required'):
        validate_dataset(config.images,config.annotations,config=config)
