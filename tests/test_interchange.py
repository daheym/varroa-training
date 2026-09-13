import copy
import json
from pathlib import Path
import pytest
from PIL import Image
from src.interchange import AnnotationFile, Manifest, load_bundle, validate, dump
from src.config import Config
from src.dataset.validation import validate_dataset
from src.dataset.prepare import prepare
from src.inference.interchange import prediction_document, predict_dataset


def bundle(root, purpose="ground_truth", reviewed=True):
    (root/"images").mkdir(parents=True)
    entries=[]
    for i,color in enumerate(("white", "black")):
        identity=f"image_{i}"
        Image.new("RGB", (120,80),color).save(root/"images"/(identity+".png"))
        payload={"schema_version":"1.0", "image":{"id":identity,"filename":identity+".png","width":120,"height":80},
            "perspective":None,"annotations":[{"id":f"ann_{i}","class":"varroa","x":70.125,"y":40.25,
            "source":"human","confidence":None,"model_version":None,"reviewed":True}]}
        dump(root/"annotations"/(identity+".json"),payload,AnnotationFile)
        entries.append({"id":identity,"filename":identity+".png","annotation_file":identity+".json"})
    manifest={"schema_version":"1.0","dataset_version":"test-1","created_at":"2026-09-12T12:00:00Z",
        "purpose":purpose,"reviewed_image_ids":[e["id"] for e in entries] if reviewed else [],"images":entries}
    dump(root/"manifest.json",manifest,Manifest)
    return root


def test_roundtrip_and_derived_labels_preserve_sources(tmp_path):
    root=bundle(tmp_path/"dataset")
    before={p:p.read_bytes() for p in root.rglob("*") if p.is_file()}
    config=Config(images=root/"images",annotations=root/"annotations",processed=tmp_path/"processed",
        splits=tmp_path/"split.json",allow_legacy_original=True,tile_size=64,overlap=16)
    records=validate_dataset(config.images,config.annotations,config=config)
    assert records[0]["points"]==[(70.125,40.25)]
    result=prepare(config,records=records)
    assert result["sources"][0]["points"]==[(70.125,40.25)]
    assert list(config.processed.rglob("*.txt"))
    assert all(p.read_bytes()==data for p,data in before.items())
    _,docs=load_bundle(root,training=True)
    assert docs[0].annotations[0].x==70.125


@pytest.mark.parametrize("change,match",[
    (lambda d:d.update(schema_version="2.0"),"Unsupported schema"),
    (lambda d:d["image"].update(width=121),"dimensions"),
    (lambda d:d["annotations"][0].update(x=120),"outside"),
    (lambda d:d["annotations"].append(copy.deepcopy(d["annotations"][0])),"duplicate"),
    (lambda d:d["annotations"][0].update(source="model",confidence=.8,model_version="v3",reviewed=False),"Unreviewed"),
    (lambda d:d["annotations"][0].update(source="model",confidence=None,model_version="v3"),"Invalid"),
    (lambda d:d["annotations"][0].update(x=float("nan")),"finite"),
])
def test_invalid_contract(tmp_path,change,match):
    root=bundle(tmp_path/"dataset")
    p=root/"annotations/image_0.json"
    doc=json.loads(p.read_text());change(doc);p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match=match):load_bundle(root,training=True)


def test_missing_image_and_unreviewed_empty_image(tmp_path):
    root=bundle(tmp_path/"dataset",purpose="unreviewed",reviewed=False)
    with pytest.raises(ValueError,match="explicitly reviewed"):load_bundle(root,training=True)
    (root/"images/image_0.png").unlink()
    with pytest.raises(FileNotFoundError):load_bundle(root)


def test_prediction_original_coordinates_after_perspective(tmp_path):
    _,docs=load_bundle(bundle(tmp_path/"dataset"))
    result={"perspective":{"inverse_homography":[[2,0,10],[0,3,5],[0,0,1]]},
        "detections":[{"box":[10,10,20,20],"confidence":.87,
            "original_polygon":[[30,35],[50,35],[50,65],[30,65]]}]}
    payload=prediction_document(docs[0],result,"varroa-v3")
    a=validate(payload,AnnotationFile).annotations[0]
    assert (a.x,a.y)==(40,50)
    assert a.bbox.model_dump()=={"x":30,"y":35,"width":20,"height":30}
    assert a.source=="model" and not a.reviewed and a.confidence==.87 and a.model_version=="varroa-v3"


def test_predict_dataset_merged_full_image_coordinates(tmp_path):
    import numpy as np
    class Tensor:
        def __init__(self,value):self.value=value
        def cpu(self):return self
        def tolist(self):return self.value
    class Model:
        def predict(self,**kwargs):
            class Boxes:
                xyxy=Tensor([[2.,4.,12.,14.]])
                conf=Tensor([.8])
            class Result:boxes=Boxes()
            return [Result()]
    root=bundle(tmp_path/"dataset",purpose="unreviewed",reviewed=False)
    output=tmp_path/"predictions"
    predict_dataset(root,Model(),output,Config(allow_legacy_original=True,tile_size=64,overlap=16),"v3")
    manifest,docs=load_bundle(output,images=False)
    assert manifest.purpose=="predictions"
    assert not (output/"images").exists()
    assert any(a.x>48 for a in docs[0].annotations)
    assert all(not a.reviewed for d in docs for a in d.annotations)


def test_manifest_duplicates_and_traversal(tmp_path):
    root=bundle(tmp_path/"dataset")
    p=root/"manifest.json";doc=json.loads(p.read_text())
    doc["images"][1]["id"]=doc["images"][0]["id"]
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="duplicate"):load_bundle(root)
    doc["images"][1]["id"]="image_1"
    doc["images"][0]["annotation_file"]="../outside.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="Invalid"):load_bundle(root)


def test_examples_match_schema():
    from src.interchange import ReviewFile
    root=Path(__file__).resolve().parents[1]/"examples/interchange"
    for name,kind in [("annotation",AnnotationFile),("prediction",AnnotationFile),("manifest",Manifest),("review",ReviewFile)]:
        validate(json.loads((root/(name+".json")).read_text()),kind)


def test_bbox_and_cross_image_duplicate_ids(tmp_path):
    root=bundle(tmp_path/"dataset")
    path=root/"annotations/image_1.json"
    doc=json.loads(path.read_text());doc["annotations"][0]["id"]="ann_0"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="duplicate annotation IDs"):load_bundle(root)
    doc["annotations"][0]["id"]="ann_1"
    doc["annotations"][0]["bbox"]={"x":110,"y":0,"width":20,"height":10}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="bbox outside"):load_bundle(root)


def test_import_cli(tmp_path):
    import subprocess,sys
    root=bundle(tmp_path/"dataset")
    output=tmp_path/"processed"
    result=subprocess.run([sys.executable,"scripts/import_dataset.py",str(root),"--output",str(output),"--allow-legacy-original"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (output/"dataset.yaml").exists()
    config=json.loads((output/"training-config.json").read_text())
    assert Path(config["images"])==root/"images"
    assert config["allow_legacy_original"]
