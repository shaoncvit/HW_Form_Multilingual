import os
import zipfile
from pathlib import Path
from sarvamai import SarvamAI

# =====================================
# CONFIG
# =====================================
LANGUAGES = {
    "bengali": "bn-IN",

}

BASE_INPUT = "/home/vlm/handwritten_form_digitization/bengali_synth_forms_2/images"
BASE_OUTPUT = "/home/vlm/handwritten_form_digitization/sarvam/results"

# os.makedirs(BASE_OUTPUT, exist_ok=True)


SARVAM_API_KEY = "sk_f29zvuql_ZEKxg7wvTsca4yBDbpksuV7z"

# =====================================
# INIT CLIENT
# =====================================
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# =====================================
# RUN
# =====================================
for lang_name, lang_code in LANGUAGES.items():

    print(f"\nRunning Sarvam OCR for {lang_name}")

    input_dir = Path(BASE_INPUT)
    output_dir = Path(BASE_OUTPUT) / lang_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(input_dir.glob("*.*")):
        print(img_path)

        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        print(f"Processing {img_path.name}")

        # 1️⃣ Create job
        job = client.document_intelligence.create_job(
            language=lang_code,
            output_format="html"  # can change to "text" if supported
        )

        # 2️⃣ Upload image
        job.upload_file(str(img_path))

        # 3️⃣ Start processing
        job.start()

        # 4️⃣ Wait for completion
        status = job.wait_until_complete()

        if status.job_state != "COMPLETED":
            print("Job failed:", status.job_state)
            continue

        # 5️⃣ Download output ZIP
        zip_path = output_dir / f"{img_path.stem}.zip"
        job.download_output(str(zip_path))

        # 6️⃣ Extract text from ZIP
        text_output = ""

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

            # Look for HTML or TXT file inside
            for file in zip_ref.namelist():
                if file.endswith(".html") or file.endswith(".txt"):
                    extracted_file = output_dir / file
                    if extracted_file.exists():
                        with open(extracted_file, "r", encoding="utf-8") as f:
                            text_output = f.read()
                        break

        # 7️⃣ Save clean .txt
        txt_output_path = output_dir / f"{img_path.stem}.txt"

        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(text_output.strip())

        # 8️⃣ Cleanup ZIP
        if zip_path.exists():
            zip_path.unlink()

print("\nSarvam OCR completed.")
