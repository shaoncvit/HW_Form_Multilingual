#!/usr/bin/env python3
"""
Fine-tune LightOnOCR on image + .txt OCR pairs.

Expected data layout, matching your synthetic form generator:
  DATA_ROOT/
    images/
      form_000000.png
      form_000001.png
    text/
      form_000000.txt
      form_000001.txt

Each .txt file should contain the target OCR text, for example:
  Name: Amit Sharma
  Phone Number: 9876543210
  Address: ...

This script is a structured version of the LightOnOCR fine-tuning notebook:
- loads LightOnOcrProcessor + LightOnOcrForConditionalGeneration
- builds a custom image/text dataset
- formats each sample as chat: user=image, assistant=target text
- masks prompt tokens so loss is calculated only on assistant OCR output
- supports full fine-tuning, freezing, and LoRA
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from jiwer import cer, wer
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# From the official notebook:
# assistant start pattern corresponds to: <|im_end|>\n<|im_start|>assistant\n
DEFAULT_ASSISTANT_START_PATTERN = [151645, 198, 151644, 77091, 198]


@dataclass
class Sample:
    image_path: Path
    text_path: Path
    sample_id: str


class ImageTextOCRDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        text = sample.text_path.read_text(encoding="utf-8").strip()
        return {
            "image": image,
            "text": text,
            "image_path": str(sample.image_path),
            "text_path": str(sample.text_path),
            "sample_id": sample.sample_id,
        }


def collect_samples_from_folders(data_root: Path) -> List[Sample]:
    image_dir = data_root / "images"
    text_dir = data_root / "text"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Missing text folder: {text_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    samples: List[Sample] = []
    missing = 0
    for image_path in image_paths:
        text_path = text_dir / f"{image_path.stem}.txt"
        if not text_path.exists():
            missing += 1
            continue
        if text_path.read_text(encoding="utf-8").strip() == "":
            continue
        samples.append(Sample(image_path=image_path, text_path=text_path, sample_id=image_path.stem))

    if not samples:
        raise RuntimeError(f"No valid image/text pairs found under {data_root}")
    if missing:
        print(f"[warning] {missing} images skipped because matching .txt was missing.")

    return samples


def collect_samples_from_manifest(manifest_path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = Path(row["image_path"])
            text_path = Path(row["text_path"])
            sample_id = row.get("sample_id", image_path.stem)
            if image_path.exists() and text_path.exists() and text_path.read_text(encoding="utf-8").strip():
                samples.append(Sample(image_path=image_path, text_path=text_path, sample_id=sample_id))
            else:
                print(f"[warning] manifest line {line_no} skipped: missing image/text path")
    if not samples:
        raise RuntimeError(f"No valid samples found in manifest: {manifest_path}")
    return samples


def split_samples(
    samples: Sequence[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    max_test_samples: int = 0,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, n - n_val - n_test)

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    if max_train_samples > 0:
        train = train[:max_train_samples]
    if max_val_samples > 0:
        val = val[:max_val_samples]
    if max_test_samples > 0:
        test = test[:max_test_samples]

    return train, val, test


def find_subsequence(sequence: List[int], pattern: List[int]) -> Optional[int]:
    if not pattern:
        return None
    for idx in range(0, len(sequence) - len(pattern) + 1):
        if sequence[idx:idx + len(pattern)] == pattern:
            return idx
    return None


class LightOnOCRCollator:
    def __init__(
        self,
        processor: LightOnOcrProcessor,
        max_length: int = 2048,
        longest_edge: int = 1000,
        assistant_start_pattern: Optional[List[int]] = None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.longest_edge = longest_edge
        self.assistant_start_pattern = assistant_start_pattern or DEFAULT_ASSISTANT_START_PATTERN

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_images = []
        batch_messages = []

        for ex in examples:
            image = ex["image"].convert("RGB")
            target_text = ex["text"].strip()

            batch_images.append(image)
            messages = [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]
            batch_messages.append(messages)

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch_messages
        ]

        inputs = self.processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            size={"longest_edge": self.longest_edge},
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id

        for i in range(len(labels)):
            full_ids = inputs["input_ids"][i].tolist()
            assistant_marker_idx = find_subsequence(full_ids, self.assistant_start_pattern)

            if assistant_marker_idx is None:
                # Safer fallback: train only last 60% tokens if marker cannot be found.
                # This should rarely happen with the official tokenizer/chat template.
                print(f"[warning] assistant marker not found in batch item {i}; masking whole sample.")
                labels[i, :] = -100
            else:
                assistant_content_start = assistant_marker_idx + len(self.assistant_start_pattern)
                labels[i, :] = -100
                for tok_idx in range(assistant_content_start, len(full_ids)):
                    if full_ids[tok_idx] == pad_token_id:
                        break
                    labels[i, tok_idx] = inputs["input_ids"][i, tok_idx]

            labels[i, inputs["input_ids"][i] == pad_token_id] = -100

        inputs["labels"] = labels

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs


def load_model_and_processor(args):
    processor = LightOnOcrProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.freeze_vision_encoder and hasattr(model.model, "vision_encoder"):
        for p in model.model.vision_encoder.parameters():
            p.requires_grad = False
        print("[freeze] vision_encoder frozen")

    if args.freeze_vision_projection and hasattr(model.model, "vision_projection"):
        for p in model.model.vision_projection.parameters():
            p.requires_grad = False
        print("[freeze] vision_projection frozen")

    if args.freeze_language_model and hasattr(model.model, "language_model"):
        for p in model.model.language_model.parameters():
            p.requires_grad = False
        print("[freeze] language_model frozen")

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is not installed. Install with: pip install peft")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


@torch.no_grad()
def run_inference(
    model,
    processor,
    image: Image.Image,
    device: str,
    max_length: int,
    longest_edge: int,
    max_new_tokens: int,
) -> str:
    model.eval()
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[[image.convert("RGB")]],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        size={"longest_edge": longest_edge},
    ).to(device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_len:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generation_evaluate(
    model,
    processor,
    dataset: ImageTextOCRDataset,
    device: str,
    max_length: int,
    longest_edge: int,
    max_new_tokens: int,
    num_samples: int,
) -> Dict[str, float]:
    n = min(num_samples, len(dataset)) if num_samples > 0 else len(dataset)
    predictions = []
    references = []

    print(f"[eval-generation] evaluating {n} samples")
    for idx in tqdm(range(n)):
        item = dataset[idx]
        pred = run_inference(
            model=model,
            processor=processor,
            image=item["image"],
            device=device,
            max_length=max_length,
            longest_edge=longest_edge,
            max_new_tokens=max_new_tokens,
        )
        gt = item["text"].strip()
        predictions.append(pred)
        references.append(gt)

        if idx < 3:
            print("\n--- sample", idx, "---")
            print("PRED:", pred[:500])
            print("GT  :", gt[:500])

    return {
        "cer": cer(references, predictions) * 100,
        "wer": wer(references, predictions) * 100,
        "exact_match": sum(p == g for p, g in zip(predictions, references)) / max(1, len(references)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LightOnOCR on form image + txt pairs.")

    parser.add_argument("--data-root", type=str, default=None, help="Folder containing images/ and text/")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest.jsonl with image_path and text_path")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model-id", type=str, default="lightonai/LightOnOCR-2-1B-base")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--device-map", type=str, default="auto")

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--longest-edge", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=768)

    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=100)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--freeze-vision-projection", action="store_true")
    parser.add_argument("--freeze-language-model", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--skip-generation-eval", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.manifest is None and args.data_root is None:
        raise ValueError("Provide either --data-root or --manifest")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.manifest:
        samples = collect_samples_from_manifest(Path(args.manifest))
    else:
        samples = collect_samples_from_folders(Path(args.data_root))

    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    print(f"[data] total={len(samples)} train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"[data] example image={train_samples[0].image_path}")
    print(f"[data] example text={train_samples[0].text_path}")

    train_ds = ImageTextOCRDataset(train_samples)
    val_ds = ImageTextOCRDataset(val_samples)
    test_ds = ImageTextOCRDataset(test_samples)

    model, processor = load_model_and_processor(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    collator = LightOnOCRCollator(
        processor=processor,
        max_length=args.max_length,
        longest_edge=args.longest_edge,
    )

    use_bf16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if len(val_ds) > 0 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
    )

    print("[train] starting")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("[save] saving model and processor")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if not args.skip_generation_eval and len(test_ds) > 0:
        metrics = generation_evaluate(
            model=model,
            processor=processor,
            dataset=test_ds,
            device=device,
            max_length=args.max_length,
            longest_edge=args.longest_edge,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.max_test_samples,
        )
        metrics_path = Path(args.output_dir) / "generation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("[eval-generation]", metrics)


if __name__ == "__main__":
    main()
