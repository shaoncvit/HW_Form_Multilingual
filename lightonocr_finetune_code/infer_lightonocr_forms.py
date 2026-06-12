from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_lora_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def find_images(image: Optional[str], image_dir: Optional[str], recursive: bool = False) -> List[Path]:
    if image is not None:
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return [p]

    if image_dir is None:
        raise ValueError("Provide either --image or --image-dir")

    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    image_paths = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise RuntimeError(f"No image files found in: {root}")
    return image_paths


def load_processor(processor_dir: Optional[str], model_dir: Optional[str], base_model_id: Optional[str]):
    # Priority: explicit processor dir -> model dir -> base model id
    candidates = [processor_dir, model_dir, base_model_id]
    last_error = None
    for c in candidates:
        if not c:
            continue
        try:
            processor = LightOnOcrProcessor.from_pretrained(c)
            processor.tokenizer.padding_side = "left"
            return processor
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not load LightOnOCR processor. Last error: {last_error}")


def load_model(args):
    dtype = torch.bfloat16 if torch.cuda.is_available() and args.dtype == "bf16" else torch.float32

    # Case 1: LoRA adapter directory was explicitly provided.
    if args.adapter_dir:
        if PeftModel is None:
            raise ImportError("peft is required for --adapter-dir. Install with: pip install peft")
        if not args.base_model_id:
            raise ValueError("When using --adapter-dir, also provide --base-model-id")

        base = LightOnOcrForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        if args.merge_lora:
            model = model.merge_and_unload()
        return model

    # Case 2: model-dir itself is a LoRA adapter directory.
    model_dir = Path(args.model_dir)
    if is_lora_adapter(model_dir):
        if PeftModel is None:
            raise ImportError("peft is required for LoRA adapter loading. Install with: pip install peft")
        if not args.base_model_id:
            raise ValueError(
                "--model-dir looks like a LoRA adapter directory. Provide --base-model-id as the original LightOnOCR base model."
            )
        base = LightOnOcrForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
        if args.merge_lora:
            model = model.merge_and_unload()
        return model

    # Case 3: full fine-tuned model directory.
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    return model


@torch.no_grad()
def infer_one(
    model,
    processor,
    image_path: Path,
    longest_edge: int,
    max_length: int,
    max_new_tokens: int,
    prompt_text: Optional[str] = None,
) -> str:
    image = Image.open(image_path).convert("RGB")

    if prompt_text:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    else:
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
    )

    # Put tensors on the model's first parameter device. This works for normal and device_map=auto cases.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = inputs.to(device)
    if "pixel_values" in inputs and torch.cuda.is_available():
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated[0, input_len:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="LightOnOCR inference for one image or folder.")

    # Model loading
    parser.add_argument("--model-dir", default=None, help="Full fine-tuned model dir OR LoRA adapter dir.")
    parser.add_argument("--base-model-id", default=None, help="Base model path/id. Required for LoRA adapter inference.")
    parser.add_argument("--adapter-dir", default=None, help="LoRA adapter dir. Use together with --base-model-id.")
    parser.add_argument("--processor-dir", default=None, help="Optional processor dir. Defaults to model-dir then base-model-id.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA into base model before inference.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not try downloading from Hugging Face.")

    # Input images
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--image-dir", default=None, help="Folder containing images.")
    parser.add_argument("--recursive", action="store_true", help="Search images inside subfolders.")

    # Output
    parser.add_argument("--out-dir", default="lightonocr_pred_texts", help="Folder to save one .txt per image.")
    parser.add_argument("--jsonl-out", default="lightonocr_predictions.jsonl", help="JSONL file for all predictions. Use empty string to disable.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing txt predictions.")

    # Generation settings
    parser.add_argument("--longest-edge", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--prompt", default=None, help="Optional text instruction with the image.")

    # Runtime
    parser.add_argument("--device-map", default="auto", help="Use 'auto' or None-like string 'none'.")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])

    args = parser.parse_args()

    if args.model_dir is None and args.adapter_dir is None:
        raise ValueError("Provide --model-dir for full model OR --adapter-dir with --base-model-id for LoRA.")

    if args.device_map.lower() in {"none", "null", "false"}:
        args.device_map = None

    return args


def main():
    args = parse_args()

    image_paths = find_images(args.image, args.image_dir, recursive=args.recursive)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(
        processor_dir=args.processor_dir,
        model_dir=args.model_dir or args.adapter_dir,
        base_model_id=args.base_model_id,
    )
    model = load_model(args)
    model.eval()

    jsonl_f = None
    if args.jsonl_out:
        jsonl_path = Path(args.jsonl_out)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True) if jsonl_path.parent != Path(".") else None
        jsonl_f = jsonl_path.open("w", encoding="utf-8")

    try:
        for image_path in tqdm(image_paths, desc="infer"):
            txt_path = out_dir / f"{image_path.stem}.txt"
            if txt_path.exists() and not args.overwrite:
                pred = txt_path.read_text(encoding="utf-8").strip()
            else:
                pred = infer_one(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    longest_edge=args.longest_edge,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    prompt_text=args.prompt,
                )
                txt_path.write_text(pred + "\n", encoding="utf-8")

            row = {
                "image_path": str(image_path),
                "prediction_text_path": str(txt_path),
                "prediction": pred,
            }
            if jsonl_f is not None:
                jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl_f.flush()

            if len(image_paths) == 1:
                print(pred)
    finally:
        if jsonl_f is not None:
            jsonl_f.close()


if __name__ == "__main__":
    main()
