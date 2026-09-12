from pathlib import Path

from src.config import write_json
from src.dataset.prepare import prepare
from src.dataset.validation import validate_dataset
from src.evaluation.metrics import summarize
from src.training.train import train


def leave_one_out(config, output):
    if config.mode != "holdout":
        raise ValueError("Leave-one-out requires holdout mode")
    output = Path(output).resolve()
    records = validate_dataset(config.images, config.annotations, config=config)
    if len(records) < 2:
        raise ValueError("Leave-one-out requires at least two original images")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, record in enumerate(records):
        fold = output / f"fold_{index:03d}"
        fold_config = config.model_copy(update={"processed": fold / "dataset", "splits": fold / "split.json"})
        prepare(fold_config, held_out=record["id"], records=records)
        report = train(fold_config, fold / "experiment", held_out_evaluation=True)
        rows.extend(report["per_image"])
        write_json(output / "progress.json", {"completed_folds": index + 1, "total_folds": len(records)})
    result = {"config": config.model_dump(mode="json"), "per_image": rows, "aggregate": summarize(rows),
              "protocol": "Fresh pretrained weights per fold; fixed epochs; last checkpoint; held-out image evaluated after fitting."}
    write_json(output / "evaluation.json", result)
    return result
