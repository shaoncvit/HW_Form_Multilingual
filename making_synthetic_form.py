import os
import cv2
import json
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from augraphy import *

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-2.5-flash-lite"
model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

FONTS_DIR = "fonts/"
PRINTED_FONTS_DIR = os.path.join(FONTS_DIR, "printed")
HANDWRITTEN_FONTS_DIR = os.path.join(FONTS_DIR, "handwritten")
BGS_DIR = "backgrounds/"
OUT_DIR = "outputs/"
FORM_SIZE = (1500, 2100)
BACKGROUND_SIZE = (1650, 2250)
SUPPORTED_EXTENSIONS = (".ttf", ".otf", ".ttc", ".png", ".jpg", ".jpeg", ".bmp")

os.makedirs(OUT_DIR, exist_ok=True)

# List of domains to cycle through
DOMAINS = [
    "Bank Account Opening",
    "Hospital Patient Registration", 
    "School Admission",
    "Railway Ticket Booking",
    "Insurance Claim",
    "Passport Application"
]

def get_random_asset(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Missing directory: {directory}")

    files = [
        f for f in os.listdir(directory)
        if not f.startswith(".") and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not files:
        raise FileNotFoundError(f"Missing assets in {directory}.")
    return os.path.join(directory, random.choice(files))

def get_font_path(preferred_directory, fallback_directory=None):
    """Loads a font from the preferred directory and optionally falls back."""
    try:
        return get_random_asset(preferred_directory)
    except FileNotFoundError:
        if fallback_directory is not None:
            return get_random_asset(fallback_directory)
        raise

def prettify_key(key):
    return str(key).replace("_", " ").replace("-", " ").title()

def pick_first_key(data_json, candidates):
    for key in candidates:
        if key in data_json and str(data_json[key]).strip():
            return key
    return None

def pop_field(data_json, candidates, default_value=""):
    key = pick_first_key(data_json, candidates)
    if key is None:
        label = prettify_key(candidates[0])
        return candidates[0], label, default_value
    value = data_json.pop(key)
    return key, prettify_key(key), str(value)

def draw_box(draw, box, width=2, outline=(40, 40, 40)):
    draw.rectangle(box, outline=outline, width=width)

def draw_field_row(draw, label_font, hw_font, row_box, label, value, key_name, ground_truth_bboxes):
    x1, y1, x2, y2 = row_box
    label_width = int((x2 - x1) * 0.24)
    split_x = x1 + label_width

    draw_box(draw, row_box, width=2)
    draw.line([(split_x, y1), (split_x, y2)], fill=(50, 50, 50), width=2)

    draw.text((x1 + 14, y1 + 10), label, font=label_font, fill="black")

    value = str(value)
    ink_x = split_x + 18
    ink_y = y1 + 8 + random.randint(-2, 4)
    ink_color = random.choice([(18, 18, 18), (10, 10, 95)])
    draw.text((ink_x, ink_y), value, font=hw_font, fill=ink_color)
    text_bbox = draw.textbbox((ink_x, ink_y), value, font=hw_font)

    ground_truth_bboxes.append({
        "key_label": key_name,
        "text_value": value,
        "bbox": [text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]],
    })

def draw_checkbox_row(draw, label_font, hw_font, row_box, label, value, key_name, ground_truth_bboxes):
    x1, y1, x2, y2 = row_box
    draw_box(draw, row_box, width=2)
    draw.text((x1 + 14, y1 + 10), label, font=label_font, fill="black")

    options = ["Male", "Female", "Other"]
    normalized_value = str(value).strip().lower()
    start_x = x1 + 210
    center_y = y1 + (y2 - y1) // 2
    selected_bbox = None

    for option in options:
        box_size = 28
        box_x1 = start_x
        box_y1 = center_y - box_size // 2
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size
        draw_box(draw, (box_x1, box_y1, box_x2, box_y2), width=2)
        draw.text((box_x2 + 10, y1 + 8), option, font=label_font, fill="black")

        if option.lower() == normalized_value:
            mark_x1 = box_x1 + 5
            mark_y1 = box_y1 + 5
            mark_x2 = box_x2 - 5
            mark_y2 = box_y2 - 5
            draw.line([(mark_x1, mark_y1), (mark_x2, mark_y2)], fill=(25, 25, 25), width=3)
            draw.line([(mark_x1, mark_y2), (mark_x2, mark_y1)], fill=(25, 25, 25), width=3)
            selected_bbox = [box_x1, box_y1, box_x2, box_y2]

        start_x += 170

    ground_truth_bboxes.append({
        "key_label": key_name,
        "text_value": str(value),
        "bbox": selected_bbox if selected_bbox else [x1, y1, x1, y1],
    })

def build_section_groups(data_json):
    working = {k: str(v) for k, v in data_json.items()}

    primary_fields = [
        pop_field(working, ["applicant_name", "name", "full_name"], "Ravi Kumar"),
        pop_field(working, ["age", "date_of_birth"], "29"),
        pop_field(working, ["gender", "sex"], "Female"),
    ]

    contact_fields = [
        pop_field(working, ["occupation", "profession"], "Service"),
        pop_field(working, ["mobile_number", "phone_number", "contact_number"], "9876543210"),
        pop_field(working, ["email", "email_id"], "ravi@example.com"),
    ]

    address_fields = [
        pop_field(working, ["address", "address_line_1"], "24 MG Road, Pune"),
        pop_field(working, ["pin_code", "postal_code"], "411001"),
        pop_field(working, ["district", "city"], "Pune"),
        pop_field(working, ["state", "nationality"], "Maharashtra"),
        pop_field(working, ["id_number", "aadhaar_number", "passport_number"], "ID294756"),
    ]

    remaining_fields = [(key, prettify_key(key), str(value)) for key, value in working.items()]
    additional_fields = remaining_fields[:3] if remaining_fields else [
        ("remarks", "Remarks", "Submitted for review"),
        ("status", "Status", "Submitted"),
        ("reference_id", "Reference Id", f"REF{random.randint(1000, 9999)}"),
    ]

    return {
        "primary": primary_fields,
        "contact": contact_fields,
        "address": address_fields,
        "additional": additional_fields,
    }

# ==========================================
# 2. THE MULTI-DOMAIN PROMPT ENGINE
# ==========================================
def get_domain_data(domain):
    """Dynamically prompts Gemini based on the selected domain."""
    fallback_data = {
        "applicant_name": "Ravi Kumar",
        "mobile_number": "9876543210",
        "date_of_birth": "14/08/1996",
        "address": "24 MG Road, Pune",
        "city": "Pune",
        "state": "Maharashtra",
        "pin_code": "411001",
        "id_number": f"{random.randint(10000000, 99999999)}",
        "date": "27/05/2026",
        "status": "submitted",
    }

    if model is None:
        print(f"GEMINI_API_KEY not set. Using local fallback data for {domain}.")
        return fallback_data
    
    # We ask Gemini to give us realistic keys based on the domain
    prompt = f"""
    You are a data generator for synthetic forms. Generate a highly realistic, 
    filled-out {domain} form for an Indian citizen. 
    
    Generate exactly 10 to 12 key-value pairs relevant to a {domain} form.
    For example, a Railway form needs 'train_number' and 'pnr', a Hospital form 
    needs 'blood_group' and 'allergies',Traceback (most recent call last):
  File "/home/vlm/handwritten_form_digitization/making_synthetic_form.py", line 450, in <module>
    main()
  File "/home/vlm/handwritten_form_digitization/making_synthetic_form.py", line 426, in main
    final_img, final_bboxes = apply_realism(clean_img, bboxes)
  File "/home/vlm/handwritten_form_digitization/making_synthetic_form.py", line 372, in apply_realism
    Folding(fold_count=1, fold_noise=0.02, fold_x=None, fold_y=None, p=0.22),
TypeError: __init__() got an unexpected keyword argument 'fold_y' etc.
    
    Rules:
    1. Output strictly valid JSON. 
    2. Do NOT use nested JSON objects (keep it flat).
    3. Do NOT include markdown formatting like ```json.
    """
    
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"Gemini API Error for {domain}: {e}")
        # Fallback to prevent crash if API fails
        return fallback_data

# ==========================================
# 3. DYNAMIC LAYOUT & DEGRADATION (From previous step)
# ==========================================
def generate_dynamic_layout(data_json, title):
    """Draws a structured form similar to scanned registration sheets."""
    width, height = FORM_SIZE
    img = Image.new('RGB', (width, height), color=(252, 250, 244))
    draw = ImageDraw.Draw(img)
    
    printed_font_path = get_font_path(PRINTED_FONTS_DIR, FONTS_DIR)
    handwritten_font_path = get_font_path(HANDWRITTEN_FONTS_DIR, FONTS_DIR)
    title_font = ImageFont.truetype(printed_font_path, 36)
    subhead_font = ImageFont.truetype(printed_font_path, 20)
    label_font = ImageFont.truetype(printed_font_path, 23)
    section_font = ImageFont.truetype(printed_font_path, 27)
    hw_font = ImageFont.truetype(handwritten_font_path, 34)

    margin = 75
    content_left = margin
    content_right = width - margin
    ground_truth_bboxes = []

    form_no = random.randint(1000, 9999)
    date_value = f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/2026"
    subtitle = random.choice(["Application Desk", "Member Services", "Records Department"])
    section_groups = build_section_groups(data_json)

    draw.text((content_left, 70), f"{title} Form", font=title_font, fill="black")
    draw.text((content_left, 118), subtitle, font=subhead_font, fill="black")
    draw.text((content_right - 220, 76), f"Form No: {form_no}", font=subhead_font, fill="black")
    draw.text((content_right - 220, 116), f"Date: {date_value}", font=subhead_font, fill="black")
    draw.line([(content_left, 155), (content_right, 155)], fill=(45, 45, 45), width=2)

    current_y = 180

    section_specs = [
        ("Member Information", section_groups["primary"] + section_groups["contact"]),
        ("Address and Identity", section_groups["address"]),
        ("Additional Information", section_groups["additional"]),
    ]

    for section_title, fields in section_specs:
        section_header_box = (content_left, current_y, content_right, current_y + 44)
        draw_box(draw, section_header_box, width=2)
        draw.text((content_left + 12, current_y + 6), section_title, font=section_font, fill="black")
        current_y += 56

        for key_name, label, value in fields:
            row_height = 58 if len(value) < 34 else 74
            row_box = (content_left, current_y, content_right, current_y + row_height)
            if key_name in {"gender", "sex"}:
                draw_checkbox_row(draw, label_font, hw_font, row_box, label, value, key_name, ground_truth_bboxes)
            else:
                draw_field_row(draw, label_font, hw_font, row_box, label, value, key_name, ground_truth_bboxes)
            current_y += row_height

        current_y += 20

    declaration_y = height - 250
    draw.text(
        (content_left, declaration_y),
        "I confirm that the above information is accurate to the best of my knowledge.",
        font=subhead_font,
        fill="black",
    )

    sig_top = declaration_y + 42
    box_gap = 22
    sig_box_w = (content_right - content_left - box_gap * 2) // 3
    signature_box = (content_left, sig_top, content_left + sig_box_w, sig_top + 82)
    date_box = (content_left + sig_box_w + box_gap, sig_top, content_left + sig_box_w * 2 + box_gap, sig_top + 82)
    office_box = (content_right - sig_box_w, sig_top, content_right, sig_top + 82)

    for box, label in [
        (signature_box, "Signature"),
        (date_box, "Date"),
        (office_box, "For Office Use Only"),
    ]:
        draw_box(draw, box, width=2)
        draw.text((box[0] + 12, box[1] - 30), label, font=subhead_font, fill="black")

    signature_text = "Signature"
    sig_x = signature_box[0] + 18
    sig_y = signature_box[1] + 18
    draw.text((sig_x, sig_y), signature_text, font=hw_font, fill=(15, 15, 15))
    sig_bbox = draw.textbbox((sig_x, sig_y), signature_text, font=hw_font)
    ground_truth_bboxes.append({
        "key_label": "signature",
        "text_value": signature_text,
        "bbox": [sig_bbox[0], sig_bbox[1], sig_bbox[2], sig_bbox[3]],
    })

    date_x = date_box[0] + 18
    date_y = date_box[1] + 18
    draw.text((date_x, date_y), date_value, font=hw_font, fill=(20, 20, 20))
    date_bbox = draw.textbbox((date_x, date_y), date_value, font=hw_font)
    ground_truth_bboxes.append({
        "key_label": "form_date",
        "text_value": date_value,
        "bbox": [date_bbox[0], date_bbox[1], date_bbox[2], date_bbox[3]],
    })

    return img, ground_truth_bboxes

def transform_bboxes(bboxes, scale_x, scale_y, x_offset, y_offset):
    """Maps clean-form bounding boxes into the final saved image coordinates."""
    transformed = []
    for field in bboxes:
        x1, y1, x2, y2 = field["bbox"]
        transformed.append({
            **field,
            "bbox": [
                int(round(x1 * scale_x + x_offset)),
                int(round(y1 * scale_y + y_offset)),
                int(round(x2 * scale_x + x_offset)),
                int(round(y2 * scale_y + y_offset)),
            ],
        })
    return transformed

def ensure_color_image(image):
    """Normalizes grayscale/BGRA images to standard 3-channel BGR."""
    if image is None:
        raise ValueError("Received an empty image during realism processing.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def extract_augraphy_image(result):
    """Accepts either a raw ndarray or tuple/list-like Augraphy result."""
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (list, tuple)) and result:
        return result[0]
    raise ValueError(f"Unexpected Augraphy output type: {type(result)}")

def apply_realism(pil_img, bboxes):
    """Adds gentle scan/photo realism without overpowering the document."""
    form_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    bg_path = get_random_asset(BGS_DIR)
    
    pipeline = AugraphyPipeline(
        paper_phase=[
            Folding(fold_count=1, p=0.22),
            NoiseTexturize(sigma_range=(2, 4), turbulence_range=(2, 3), p=0.2),
        ],
        post_phase=[
            LightingGradient(
                light_position=None,
                direction=None,
                max_brightness=220,
                min_brightness=190,
                mode="gaussian",
                transparency=None,
                p=0.16,
            ),
        ],
    )
    
    augmented_result = pipeline(form_cv)
    augmented_form = ensure_color_image(extract_augraphy_image(augmented_result))
    bg_raw = cv2.imread(bg_path)
    if bg_raw is None:
        raise ValueError(f"Could not read background image: {bg_path}")
    bg_img = ensure_color_image(cv2.resize(bg_raw, BACKGROUND_SIZE))
    bg_img = cv2.GaussianBlur(bg_img, (0, 0), 6)
    bg_img = cv2.addWeighted(bg_img, 0.18, np.full_like(bg_img, 246), 0.82, 0)
    
    form_h, form_w = augmented_form.shape[:2]
    new_w, new_h = int(form_w * 0.92), int(form_h * 0.92)
    augmented_form_resized = ensure_color_image(cv2.resize(augmented_form, (new_w, new_h)))
    
    x_offset = (bg_img.shape[1] - new_w) // 2
    y_offset = (bg_img.shape[0] - new_h) // 2
    shadow = np.zeros_like(bg_img)
    shadow[y_offset + 12:y_offset + new_h + 12, x_offset + 12:x_offset + new_w + 12] = 25
    shadow = cv2.GaussianBlur(shadow, (0, 0), 9)
    bg_img = cv2.subtract(bg_img, shadow)
    bg_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = augmented_form_resized
    
    scale_x = new_w / form_w
    scale_y = new_h / form_h
    transformed_bboxes = transform_bboxes(bboxes, scale_x, scale_y, x_offset, y_offset)

    return bg_img, transformed_bboxes

def main():
    NUM_FORMS_TO_GENERATE = 10  # Change this to 1000 for your pilot
    
    print(f"Starting generation of {NUM_FORMS_TO_GENERATE} multi-domain forms...")
    
    for i in range(NUM_FORMS_TO_GENERATE):
        current_domain = random.choice(DOMAINS)
        print(f"[{i+1}/{NUM_FORMS_TO_GENERATE}] Generating: {current_domain}")
        
        fake_data = get_domain_data(current_domain)
        clean_img, bboxes = generate_dynamic_layout(fake_data, title=current_domain)
        final_img, final_bboxes = apply_realism(clean_img, bboxes)
        
        filename_base = f"{current_domain.replace(' ', '_').lower()}_{i:04d}"
        cv2.imwrite(os.path.join(OUT_DIR, f"{filename_base}.jpg"), final_img)
        
        with open(os.path.join(OUT_DIR, f"{filename_base}.json"), 'w') as f:
            json.dump(
                {
                    "image": f"{filename_base}.jpg",
                    "domain": current_domain,
                    "fields": final_bboxes,
                },
                f,
                indent=4,
            )
            
        time.sleep(2)
        
    print("Batch generation complete!")

# ==========================================
# 4. THE BATCH GENERATOR 
# ==========================================
if __name__ == "__main__":
    main()
