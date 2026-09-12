from pathlib import Path

from src.config import write_json
from src.dataset.prepare import prepare
from src.dataset.validation import validate_dataset
from src.training.train import train


def overfit_image(config, output, image_id=None):
    """Prepare and train one original without inventing an independent validation set."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Experiment directory already exists: {output}. Choose a new output path.")
    config = config.model_copy(update={"mode": "overfit", "processed": output / "dataset",
                                       "splits": output / "split.json"})
    records = validate_dataset(config.images, config.annotations, image_id=image_id)
    prepare(config, records=records)
    write_json(output / "config.json", config.model_dump(mode="json"))
    return train(config, output / "experiment")
