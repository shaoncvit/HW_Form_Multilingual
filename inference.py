import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image
from pathlib import Path

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = LightOnOcrForConditionalGeneration.from_pretrained(
    "lightonai/LightOnOCR-2-1B",
    torch_dtype=dtype,
).to(device)

processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")
model.eval()

# -----------------------------
# CHANGE THESE PATHS
# -----------------------------
image_dir = Path("/home/vlm/handwritten_form_digitization/bengali_synth_forms_2/images")
result_dir = Path("/home/vlm/handwritten_form_digitization/bengali_synth_forms_2/result")
result_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

image_paths = sorted([
    p for p in image_dir.iterdir()
    if p.suffix.lower() in image_exts
])

print(f"[INFO] Found {len(image_paths)} images")
print(f"[INFO] Saving results to: {result_dir}")

def run_ocr(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")

    # Resize if image is too large
    max_side = 1600
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)))

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    torch.cuda.empty_cache()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    return output_text.strip()

for idx, image_path in enumerate(image_paths, start=1):
    txt_path = result_dir / f"{image_path.stem}.txt"

    if txt_path.exists():
        print(f"[SKIP] {idx}/{len(image_paths)} already exists: {txt_path.name}")
        continue

    try:
        print(f"[RUN] {idx}/{len(image_paths)} {image_path.name}")
        output_text = run_ocr(image_path)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")

        print(f"[SAVE] {txt_path}")

    except Exception as e:
        error_path = result_dir / f"{image_path.stem}_ERROR.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(e))

        print(f"[ERROR] {image_path.name}: {e}")

print("[DONE] All images processed.")