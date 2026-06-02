# LightOnOCR fine-tuning for synthetic form image + text pairs

Your current data can be:

```text
DATA_ROOT/
  images/
    form_000000.png
    form_000001.png
  text/
    form_000000.txt
    form_000001.txt
```

The image and text file must have the same stem.

## Install

```bash
pip install -r requirements_lightonocr.txt
```

## Quick training test

```bash
python train_lightonocr_forms.py \
  --data-root /ssd_scratch/shaon/tormented_out \
  --output-dir /ssd_scratch/shaon/lightonocr_form_ft_debug \
  --max-train-samples 24 \
  --max-val-samples 3 \
  --max-test-samples 3 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --longest-edge 900 \
  --max-length 2048 \
  --eval-steps 50 \
  --save-steps 50 \
  --gradient-checkpointing
```

## Train from manifest

If you used `synthetic_form_pipeline.py`, you can train from its manifest:

```bash
python train_lightonocr_forms.py \
  --manifest /path/to/synthetic_forms_out/manifest.jsonl \
  --output-dir lightonocr_form_ft \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --longest-edge 1000 \
  --max-length 2048 \
  --gradient-checkpointing
```

## Memory-saving LoRA option

```bash
python train_lightonocr_forms.py \
  --data-root /path/to/synthetic_forms_out \
  --output-dir lightonocr_form_lora \
  --use-lora \
  --freeze-vision-encoder \
  --gradient-checkpointing \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

## Inference

```bash
python infer_lightonocr_forms.py \
  --model-dir lightonocr_form_ft \
  --image /path/to/image.png
```

or folder:

```bash
python infer_lightonocr_forms.py \
  --model-dir lightonocr_form_ft \
  --image-dir /path/to/test_images \
  --out predictions.jsonl
```
