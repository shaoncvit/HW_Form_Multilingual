#!/usr/bin/env python3
"""
Run inference with a fine-tuned LightOnOCR model on one image or a folder of images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@torch.no_grad()
def infer_one(model, processor, image_path: Path, device: str, longest_edge: int, max_length: int, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=[[image]],
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out", default="lightonocr_predictions.jsonl")
    parser.add_argument("--longest-edge", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image is None and args.image_dir is None:
        raise ValueError("Provide --image or --image-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = LightOnOcrProcessor.from_pretrained(args.model_dir)
    processor.tokenizer.padding_side = "left"
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        device_map="auto",
    ).to(device)
    model.eval()

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    with Path(args.out).open("w", encoding="utf-8") as f:
        for image_path in tqdm(image_paths):
            pred = infer_one(
                model=model,
                processor=processor,
                image_path=image_path,
                device=device,
                longest_edge=args.longest_edge,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            row = {"image_path": str(image_path), "prediction": pred}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if len(image_paths) == 1:
                print(pred)


if __name__ == "__main__":
    main()
