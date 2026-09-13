# Train on Kaggle

Kaggle only supplies the runtime and mounted dataset. Training, checkpoints, and metrics use the same CLI as local runs.

## Setup

1. Create a private Kaggle Dataset. You can upload the repository's existing `images/` plus Varroa point-annotation `annotations/` folders unchanged, or provide a ready-to-train YOLO dataset with `images/train`, `images/val`, `labels/train`, `labels/val`, and a `dataset.yaml`. YOLO label rows use `class x_center y_center width height` with normalized coordinates.
2. Create a Kaggle Notebook and enable a GPU accelerator in Notebook settings.
3. Add the private dataset to the notebook's input data.
4. Make this repository available in the notebook by cloning it, or upload the repository files if it is private and cloning is inconvenient.
5. Install the project from the repository root. Kaggle's preinstalled CUDA-enabled PyTorch is used when it satisfies the project requirements; the CPU-only Windows lock file is not for Kaggle.

```bash
cd /kaggle/working/varroa-training
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -e .
```

If the dataset's Kaggle slug differs from `varroa-dataset`, use its mounted path in `--data`.
The default `yolo11n.pt` pretrained weights download on first use. Enable notebook Internet for that download, or include the checkpoint in a Kaggle Dataset and set `model` to its mounted path in the config.

### Keep the existing point-annotation format

For the repository's JSON annotations, make a small config pointing at the mounted read-only inputs and writable prepared-data paths, then use the regular `prepare` command before training:

```bash
cat > /kaggle/working/varroa-points.yaml <<'YAML'
images: /kaggle/input/varroa-points/images
annotations: /kaggle/input/varroa-points/annotations
processed: /kaggle/working/varroa-prepared
splits: /kaggle/working/varroa-splits.json
device: auto
YAML

python -m src.cli --config /kaggle/working/varroa-points.yaml prepare
python training/train.py \
  --config /kaggle/working/varroa-points.yaml \
  --data /kaggle/working/varroa-prepared/dataset.yaml \
  --output /kaggle/working/runs
```

This path retains the normal source-image validation, tiling, and point-based evaluation. The training entry point itself is still the same one used for a ready-made YOLO dataset.

## Train

Run the regular project entry point directly:

```bash
python training/train.py \
  --config training/configs/varroa.yaml \
  --data /kaggle/input/varroa-dataset/dataset.yaml \
  --output /kaggle/working/runs \
  --device auto
```

Or use the thin helper, which invokes that same command and can check CUDA before starting:

```bash
python cloud/kaggle/train_kaggle.py \
  --data /kaggle/input/varroa-dataset/dataset.yaml \
  --output /kaggle/working/runs \
  --require-cuda
```

Training overrides such as `--epochs 50 --batch-size 8 --seed 7` can be passed to the helper and are forwarded to the normal CLI. The output root contains a unique run folder unless `--run-name` is set, for example `/kaggle/working/runs/run-20260913-120000/`. It contains `config.yaml`, `metadata.json`, `metrics.json`, `requirements-freeze.txt`, the resolved `data.yaml`, and Ultralytics' `yolo/` directory with checkpoints, CSV logs, and plots. The best checkpoint is at `/kaggle/working/runs/<run-name>/yolo/weights/best.pt`; `last.pt` is alongside it. Retrieve the `runs` directory from the notebook output or download the files from `/kaggle/working`.

To export separately after training:

```bash
python training/export.py \
  --checkpoint /kaggle/working/runs/run-20260913-120000/yolo/weights/best.pt \
  --format onnx
```
