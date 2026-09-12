from types import SimpleNamespace

from PIL import Image

from src.config import Config
from src.inference.predict import predict_image, save_prediction


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_tiled_inference_maps_and_merges(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (15, 10)).save(path)

    class Model:
        calls = 0

        def predict(self, **kwargs):
            assert kwargs["device"] == "cpu"
            # Same original object at [6,2,9,5], in two overlapping tiles.
            box = [6, 2, 9, 5] if self.calls == 0 else [1, 2, 4, 5]
            self.calls += 1
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=FakeTensor([box]), conf=FakeTensor([.9])))]

    model = Model()
    result = predict_image(model, path, Config(allow_legacy_original=True, tile_size=10, overlap=5))
    assert model.calls == 2 and result["count"] == 1
    assert result["detections"][0]["box"] == (6, 2, 9, 5)
    save_prediction(result, tmp_path / "output")
    assert (tmp_path / "output.json").exists()
    with Image.open(tmp_path / "output.png") as image:
        assert image.size == (15, 10)
