import argparse
import json
import os
import random
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from augraphy import AugraphyPipeline, Folding, LightingGradient, NoiseTexturize


FONTS_DIR = "fonts"
PRINTED_FONTS_DIR = os.path.join(FONTS_DIR, "printed")
HANDWRITTEN_FONTS_DIR = os.path.join(FONTS_DIR, "handwritten")
BGS_DIR = "backgrounds"
DEFAULT_OUT_DIR = "variation_outputs_2"
FORM_SIZE = (1500, 2100)
BACKGROUND_SIZE = (1650, 2250)
SUPPORTED_EXTENSIONS = (".ttf", ".otf", ".ttc", ".png", ".jpg", ".jpeg", ".bmp")

DOMAINS = [
    "Bank Account Opening",
    "Hospital Patient Registration",
    "School Admission",
    "Railway Ticket Booking",
    "Insurance Claim",
    "Passport Application",
]

FIRST_NAMES = ["Anjali", "Ravi", "Priya", "Karan", "Meera", "Aisha", "Vikram", "Sneha"]
LAST_NAMES = ["Sharma", "Verma", "Singh", "Nair", "Das", "Reddy", "Patel", "Khan"]
CITIES = ["Pune", "Kolkata", "Hyderabad", "Bhopal", "Jaipur", "Lucknow", "Surat", "Nagpur"]
STATES = ["Maharashtra", "West Bengal", "Telangana", "Madhya Pradesh", "Rajasthan", "Gujarat"]
STREETS = ["MG Road", "Lake View Road", "Park Street", "Station Road", "Civil Lines", "Temple Street"]
OCCUPATIONS = ["Teacher", "Service", "Student", "Clerk", "Designer", "Nurse", "Engineer", "Trader"]
INTERESTS = ["Workshop Access", "Scholarship Request", "Premium Service", "Document Renewal", "Weekend Classes"]
REMARKS = ["Submitted for review", "Urgent processing requested", "Documents attached", "Verified at desk"]
TRAINS = ["12245", "12951", "17604", "22817", "12724"]
BLOOD_GROUPS = ["A+", "B+", "O+", "AB+", "O-"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate many layout and augmentation variations without Gemini.")
    parser.add_argument("--num-forms", type=int, default=100, help="Number of forms to generate.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save generated images and JSON.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for repeatability.")
    return parser.parse_args()


def get_random_asset(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Missing directory: {directory}")
    files = [
        name for name in os.listdir(directory)
        if not name.startswith(".") and name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not files:
        raise FileNotFoundError(f"No usable assets found in {directory}")
    return os.path.join(directory, random.choice(files))


def get_font_path(preferred_directory, fallback_directory=None):
    try:
        return get_random_asset(preferred_directory)
    except FileNotFoundError:
        if fallback_directory is None:
            raise
        return get_random_asset(fallback_directory)


def ensure_color_image(image):
    if image is None:
        raise ValueError("Received an empty image.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def extract_augraphy_image(result):
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (list, tuple)) and result:
        return result[0]
    raise ValueError(f"Unexpected Augraphy output type: {type(result)}")


def prettify_key(key):
    return str(key).replace("_", " ").replace("-", " ").title()


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_phone():
    return "9" + "".join(str(random.randint(0, 9)) for _ in range(9))


def random_date(year=2026):
    return f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{year}"


def random_address():
    return f"Plot {random.randint(10, 999)}, {random.choice(STREETS)}, {random.choice(CITIES)}"


def random_email(name):
    handle = name.lower().replace(" ", ".")
    domain = random.choice(["example.com", "mailbox.in", "civic.org", "records.net"])
    return f"{handle}@{domain}"


def random_id(prefix):
    return f"{prefix}{random.randint(1000, 9999)}{random.choice(['A', 'B', 'C', 'X'])}"


def build_domain_data(domain):
    name = random_name()
    base = {
        "applicant_name": name,
        "age": str(random.randint(18, 62)),
        "gender": random.choice(["Male", "Female", "Other"]),
        "occupation": random.choice(OCCUPATIONS),
        "mobile_number": random_phone(),
        "email": random_email(name),
        "address": random_address(),
        "pin_code": str(random.randint(100000, 999999)),
        "city": random.choice(CITIES),
        "state": random.choice(STATES),
        "date": random_date(),
        "remarks": random.choice(REMARKS),
    }

    domain_specific = {
        "Bank Account Opening": {
            "account_type": random.choice(["Savings", "Current", "Salary"]),
            "branch_code": f"BR{random.randint(100,999)}",
            "id_number": random_id("PAN"),
        },
        "Hospital Patient Registration": {
            "patient_id": random_id("PT"),
            "blood_group": random.choice(BLOOD_GROUPS),
            "allergies": random.choice(["None", "Dust", "Penicillin", "Peanuts"]),
        },
        "School Admission": {
            "class_applied": random.choice(["Class 5", "Class 7", "Class 9", "Class 11"]),
            "guardian_name": random_name(),
            "previous_school": random.choice(["Green Valley", "St. Thomas", "City Public", "Bright Future"]),
        },
        "Railway Ticket Booking": {
            "train_number": random.choice(TRAINS),
            "pnr": str(random.randint(1000000000, 9999999999)),
            "travel_class": random.choice(["SL", "3A", "2A", "CC"]),
        },
        "Insurance Claim": {
            "claim_number": random_id("CL"),
            "policy_number": random_id("PL"),
            "incident_date": random_date(2025),
        },
        "Passport Application": {
            "passport_type": random.choice(["Fresh", "Reissue", "Tatkal"]),
            "nationality": "Indian",
            "id_number": random_id("PID"),
        },
    }

    merged = deepcopy(base)
    merged.update(domain_specific[domain])
    merged["signature"] = "Signature"
    return merged


def draw_box(draw, box, width=2, outline=(40, 40, 40)):
    draw.rectangle(box, outline=outline, width=width)


def add_text_bbox(draw, text, position, font, fill, bboxes, key_name):
    draw.text(position, text, font=font, fill=fill)
    bbox = draw.textbbox(position, text, font=font)
    bboxes.append({
        "key_label": key_name,
        "text_value": str(text),
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
    })


def draw_row_field(draw, row_box, label, value, key_name, label_font, hw_font, bboxes, split_ratio=0.26):
    x1, y1, x2, y2 = row_box
    split_x = x1 + int((x2 - x1) * split_ratio)
    draw_box(draw, row_box)
    draw.line([(split_x, y1), (split_x, y2)], fill=(55, 55, 55), width=2)
    draw.text((x1 + 12, y1 + 10), label, font=label_font, fill="black")
    value_x = split_x + 18
    value_y = y1 + 8 + random.randint(-2, 4)
    add_text_bbox(draw, str(value), (value_x, value_y), hw_font, random.choice([(10, 10, 10), (12, 12, 90)]), bboxes, key_name)


def draw_inline_field(draw, x, y, label, value, key_name, label_font, hw_font, bboxes, width):
    draw.text((x, y), f"{label}:", font=label_font, fill="black")
    line_y = y + 34
    label_width = max(180, int(len(label) * 10))
    start_x = x + label_width
    end_x = x + width
    draw.line([(start_x, line_y), (end_x, line_y)], fill=(110, 110, 110), width=2)
    value_x = start_x + 14
    value_y = y - 8 + random.randint(-2, 4)
    add_text_bbox(draw, str(value), (value_x, value_y), hw_font, random.choice([(15, 15, 15), (15, 15, 95)]), bboxes, key_name)


def draw_checkbox_group(draw, x, y, label, value, key_name, label_font, bboxes):
    draw.text((x, y), f"{label}:", font=label_font, fill="black")
    options = ["Male", "Female", "Other"]
    current_x = x + 170
    selected_bbox = [x, y, x, y]
    for option in options:
        box = (current_x, y + 4, current_x + 26, y + 30)
        draw_box(draw, box)
        draw.text((current_x + 36, y - 2), option, font=label_font, fill="black")
        if option.lower() == str(value).lower():
            draw.line([(box[0] + 5, box[1] + 5), (box[2] - 5, box[3] - 5)], fill=(20, 20, 20), width=3)
            draw.line([(box[0] + 5, box[3] - 5), (box[2] - 5, box[1] + 5)], fill=(20, 20, 20), width=3)
            selected_bbox = [box[0], box[1], box[2], box[3]]
        current_x += 165
    bboxes.append({
        "key_label": key_name,
        "text_value": str(value),
        "bbox": selected_bbox,
    })


def draw_stamp(draw, x, y, text, color):
    radius = random.randint(42, 60)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.ellipse((x - radius + 10, y - radius + 10, x + radius - 10, y + radius - 10), outline=color, width=1)
    draw.text((x - radius + 12, y - 10), text, fill=color)


def layout_sectioned_registration(draw, data, fonts, bboxes, domain):
    width, height = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left = 75
    right = width - 75

    draw.text((left, 70), f"{domain} Form", font=title_font, fill="black")
    draw.text((left, 118), random.choice(["Application Desk", "Records Section", "Member Services"]), font=subhead_font, fill="black")
    draw.text((right - 230, 78), f"Form No: {random.randint(1000, 9999)}", font=subhead_font, fill="black")
    draw.text((right - 230, 118), f"Date: {data['date']}", font=subhead_font, fill="black")
    draw.line([(left, 155), (right, 155)], fill=(40, 40, 40), width=2)

    groups = [
        ("Primary Information", ["applicant_name", "age", "gender", "occupation", "mobile_number", "email"]),
        ("Address and Identity", ["address", "pin_code", "city", "state"]),
        ("Domain Details", [key for key in data if key not in {"applicant_name", "age", "gender", "occupation", "mobile_number", "email", "address", "pin_code", "city", "state", "date", "signature"}][:4]),
    ]

    current_y = 180
    for section_title, keys in groups:
        draw_box(draw, (left, current_y, right, current_y + 44))
        draw.text((left + 12, current_y + 6), section_title, font=section_font, fill="black")
        current_y += 56
        for key in keys:
            if key not in data:
                continue
            row_h = 60 if len(str(data[key])) < 34 else 78
            row_box = (left, current_y, right, current_y + row_h)
            if key == "gender":
                draw_box(draw, row_box)
                draw_checkbox_group(draw, left + 14, current_y + 10, prettify_key(key), data[key], key, label_font, bboxes)
            else:
                draw_row_field(draw, row_box, prettify_key(key), data[key], key, label_font, hw_font, bboxes)
            current_y += row_h
        current_y += 18

    footer_y = height - 210
    draw.text((left, footer_y), "I confirm that the above information is accurate to the best of my knowledge.", font=subhead_font, fill="black")
    sig_box = (left, footer_y + 42, left + 340, footer_y + 120)
    date_box = (left + 400, footer_y + 42, left + 720, footer_y + 120)
    draw_box(draw, sig_box)
    draw_box(draw, date_box)
    draw.text((sig_box[0], sig_box[1] - 28), "Signature", font=subhead_font, fill="black")
    draw.text((date_box[0], date_box[1] - 28), "Date", font=subhead_font, fill="black")
    add_text_bbox(draw, "Signature", (sig_box[0] + 18, sig_box[1] + 18), hw_font, (20, 20, 20), bboxes, "signature")
    add_text_bbox(draw, data["date"], (date_box[0] + 18, date_box[1] + 18), hw_font, (20, 20, 20), bboxes, "form_date")


def layout_two_column(draw, data, fonts, bboxes, domain):
    width, _ = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left_margin = 80
    column_gap = 70
    column_width = (width - left_margin * 2 - column_gap) // 2
    x_positions = [left_margin, left_margin + column_width + column_gap]

    draw.text((left_margin, 70), f"{domain}", font=title_font, fill="black")
    draw.text((left_margin, 118), "General Application Sheet", font=subhead_font, fill="black")
    draw.text((width - 320, 80), f"Ref: REF-{random.randint(10000, 99999)}", font=subhead_font, fill="black")
    draw.line([(left_margin, 155), (width - left_margin, 155)], fill=(40, 40, 40), width=2)

    ordered_keys = [key for key in data if key not in {"signature"}]
    midpoint = (len(ordered_keys) + 1) // 2
    columns = [ordered_keys[:midpoint], ordered_keys[midpoint:]]

    for column_index, keys in enumerate(columns):
        current_y = 200
        x = x_positions[column_index]
        for key in keys:
            if key == "gender":
                draw_box(draw, (x, current_y, x + column_width, current_y + 56))
                draw_checkbox_group(draw, x + 10, current_y + 10, "Gender", data[key], key, label_font, bboxes)
                current_y += 74
                continue
            draw_inline_field(draw, x, current_y, prettify_key(key), data[key], key, label_font, hw_font, bboxes, column_width)
            current_y += 82 if len(str(data[key])) < 30 else 104


def layout_grid_table(draw, data, fonts, bboxes, domain):
    width, height = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left = 70
    right = width - 70
    top = 70

    draw.text((left, top), f"{domain} Record Form", font=title_font, fill="black")
    draw.text((right - 280, top + 10), f"Date: {data['date']}", font=subhead_font, fill="black")

    table_top = 170
    draw_box(draw, (left, table_top, right, height - 220))
    draw.line([(left + 320, table_top), (left + 320, height - 220)], fill=(55, 55, 55), width=2)
    draw.text((left + 16, table_top + 10), "Field", font=section_font, fill="black")
    draw.text((left + 340, table_top + 10), "Information", font=section_font, fill="black")

    keys = [key for key in data if key not in {"signature"}]
    row_top = table_top + 56
    row_height = 96
    for key in keys[:14]:
        draw.line([(left, row_top), (right, row_top)], fill=(70, 70, 70), width=2)
        draw.text((left + 14, row_top + 14), prettify_key(key), font=label_font, fill="black")
        if key == "gender":
            draw_checkbox_group(draw, left + 340, row_top + 14, "", data[key], key, label_font, bboxes)
        else:
            add_text_bbox(draw, str(data[key]), (left + 340, row_top + 10), hw_font, (15, 15, 15), bboxes, key)
        row_top += row_height
        if row_top > height - 340:
            break

    sig_y = height - 190
    draw.text((left, sig_y - 28), "Signature", font=subhead_font, fill="black")
    draw_box(draw, (left, sig_y, left + 320, sig_y + 74))
    add_text_bbox(draw, "Signature", (left + 18, sig_y + 16), hw_font, (20, 20, 20), bboxes, "signature")


def layout_bands_and_cards(draw, data, fonts, bboxes, domain):
    width, height = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left = 75
    right = width - 75

    draw_box(draw, (left, 70, right, 160), width=2)
    draw.text((left + 18, 88), f"{domain} Intake Form", font=title_font, fill="black")
    draw.text((left + 18, 128), random.choice(["Front Office Copy", "Desk Register", "Intake Counter"]), font=subhead_font, fill="black")
    draw.text((right - 240, 95), f"Doc ID {random.randint(1000, 9999)}", font=subhead_font, fill="black")
    draw.text((right - 240, 126), data["date"], font=subhead_font, fill="black")

    band_y = 200
    sections = [
        ("Applicant Card", ["applicant_name", "age", "gender", "occupation"]),
        ("Contact Card", ["mobile_number", "email", "address"]),
        ("Reference Card", [key for key in data if key not in {"applicant_name", "age", "gender", "occupation", "mobile_number", "email", "address", "date", "signature"}][:4]),
    ]

    for title, keys in sections:
        draw_box(draw, (left, band_y, right, band_y + 44))
        draw.text((left + 12, band_y + 6), title, font=section_font, fill="black")
        band_y += 60
        card_height = 180 if len(keys) <= 3 else 230
        card_box = (left, band_y, right, band_y + card_height)
        draw_box(draw, card_box)
        row_y = band_y + 14
        for key in keys:
            if key not in data:
                continue
            if key == "gender":
                draw_checkbox_group(draw, left + 16, row_y, "Gender", data[key], key, label_font, bboxes)
                row_y += 52
            else:
                draw_inline_field(draw, left + 16, row_y, prettify_key(key), data[key], key, label_font, hw_font, bboxes, right - left - 40)
                row_y += 54 if len(str(data[key])) < 28 else 82
        band_y += card_height + 22
        if band_y > height - 300:
            break

    add_text_bbox(draw, "Signature", (left + 40, height - 140), hw_font, (20, 20, 20), bboxes, "signature")


def layout_office_blocks(draw, data, fonts, bboxes, domain):
    width, height = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left = 70
    right = width - 70

    draw.text((left, 70), f"{domain} Office Record", font=title_font, fill="black")
    draw.text((left, 118), "Internal Processing Sheet", font=subhead_font, fill="black")
    draw.text((right - 260, 82), f"Desk: {random.randint(1, 12)}", font=subhead_font, fill="black")
    draw.text((right - 260, 118), f"Date: {data['date']}", font=subhead_font, fill="black")

    left_col = (left, 190, left + 610, height - 260)
    right_col = (left + 660, 190, right, height - 260)
    for box, title in [(left_col, "Applicant Snapshot"), (right_col, "Processing Notes")]:
        draw_box(draw, box)
        draw_box(draw, (box[0], box[1], box[2], box[1] + 42))
        draw.text((box[0] + 12, box[1] + 6), title, font=section_font, fill="black")

    left_keys = ["applicant_name", "gender", "age", "occupation", "mobile_number", "email"]
    right_keys = ["address", "city", "state", "pin_code"] + [
        key for key in data if key not in {"applicant_name", "gender", "age", "occupation", "mobile_number", "email", "address", "city", "state", "pin_code", "date", "signature"}][:3]

    current_y = left_col[1] + 60
    for key in left_keys:
        if key == "gender":
            draw_box(draw, (left_col[0] + 14, current_y, left_col[2] - 14, current_y + 56))
            draw_checkbox_group(draw, left_col[0] + 28, current_y + 10, "Gender", data[key], key, label_font, bboxes)
            current_y += 74
        else:
            draw_row_field(draw, (left_col[0] + 14, current_y, left_col[2] - 14, current_y + 62), prettify_key(key), data[key], key, label_font, hw_font, bboxes)
            current_y += 78

    current_y = right_col[1] + 60
    for key in right_keys:
        if key not in data:
            continue
        row_h = 72 if len(str(data[key])) > 30 else 60
        draw_row_field(draw, (right_col[0] + 14, current_y, right_col[2] - 14, current_y + row_h), prettify_key(key), data[key], key, label_font, hw_font, bboxes)
        current_y += row_h + 14

    footer_y = height - 190
    draw_box(draw, (left, footer_y, right, footer_y + 100))
    draw.text((left + 14, footer_y + 8), "Verifier Signature", font=subhead_font, fill="black")
    add_text_bbox(draw, "Signature", (left + 18, footer_y + 40), hw_font, (20, 20, 20), bboxes, "signature")


def layout_zigzag_panels(draw, data, fonts, bboxes, domain):
    width, height = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left = 80
    right = width - 80

    draw.text((left, 70), f"{domain} Intake Register", font=title_font, fill="black")
    draw.text((left, 120), random.choice(["Walk-in Application", "Counter Submission", "Document Intake"]), font=subhead_font, fill="black")
    draw.line([(left, 158), (right, 158)], fill=(50, 50, 50), width=2)

    panels = [
        (left, 200, right - 180, 420, "Identity Panel", ["applicant_name", "age", "gender", "occupation"]),
        (left + 180, 455, right, 705, "Contact Panel", ["mobile_number", "email", "address"]),
        (left, 740, right - 140, 1010, "Reference Panel", ["city", "state", "pin_code"]),
        (left + 140, 1045, right, 1355, "Domain Panel", [key for key in data if key not in {"applicant_name", "age", "gender", "occupation", "mobile_number", "email", "address", "city", "state", "pin_code", "date", "signature"}][:4]),
    ]

    for x1, y1, x2, y2, title, keys in panels:
        draw_box(draw, (x1, y1, x2, y2))
        draw_box(draw, (x1, y1, x2, y1 + 40))
        draw.text((x1 + 12, y1 + 6), title, font=section_font, fill="black")
        row_y = y1 + 56
        for key in keys:
            if key not in data:
                continue
            if key == "gender":
                draw_checkbox_group(draw, x1 + 14, row_y, "Gender", data[key], key, label_font, bboxes)
                row_y += 52
            else:
                draw_inline_field(draw, x1 + 14, row_y, prettify_key(key), data[key], key, label_font, hw_font, bboxes, x2 - x1 - 28)
                row_y += 56 if len(str(data[key])) < 28 else 84

    add_text_bbox(draw, "Signature", (left + 30, height - 160), hw_font, (18, 18, 18), bboxes, "signature")
    draw.text((left, height - 196), "Applicant Sign-Off", font=subhead_font, fill="black")


def layout_compact_receipt(draw, data, fonts, bboxes, domain):
    width, height = FORM_SIZE
    title_font, subhead_font, label_font, section_font, hw_font = fonts
    left = 110
    right = width - 110

    draw_box(draw, (left, 70, right, height - 110))
    draw.text((left + 18, 88), f"{domain} Receipt Form", font=title_font, fill="black")
    draw.text((right - 250, 96), f"No. {random.randint(10000, 99999)}", font=subhead_font, fill="black")
    draw.text((left + 18, 132), "Compact Office Slip", font=subhead_font, fill="black")
    draw.line([(left + 18, 164), (right - 18, 164)], fill=(55, 55, 55), width=2)

    keys = [key for key in data if key not in {"signature"}]
    y = 190
    for key in keys[:12]:
        if key == "gender":
            draw_checkbox_group(draw, left + 16, y, "Gender", data[key], key, label_font, bboxes)
            y += 52
        else:
            draw_inline_field(draw, left + 16, y, prettify_key(key), data[key], key, label_font, hw_font, bboxes, right - left - 32)
            y += 56 if len(str(data[key])) < 26 else 84

    draw.line([(left + 18, height - 230), (right - 18, height - 230)], fill=(80, 80, 80), width=1)
    draw.text((left + 18, height - 212), "Applicant Signature", font=subhead_font, fill="black")
    add_text_bbox(draw, "Signature", (left + 24, height - 175), hw_font, (15, 15, 15), bboxes, "signature")


LAYOUTS = {
    "sectioned_registration": layout_sectioned_registration,
    "two_column_sheet": layout_two_column,
    "grid_table_form": layout_grid_table,
    "bands_and_cards": layout_bands_and_cards,
    "office_blocks": layout_office_blocks,
    "zigzag_panels": layout_zigzag_panels,
    "compact_receipt": layout_compact_receipt,
}


def build_augraphy_pipeline(profile_name):
    paper_phase = []
    post_phase = []

    if profile_name in {"folded_scan", "balanced_scan"}:
        paper_phase.append(Folding(fold_count=1, p=0.18))
    if profile_name in {"balanced_scan", "noisy_scan"}:
        paper_phase.append(NoiseTexturize(p=0.18 if profile_name == "balanced_scan" else 0.28))
    if profile_name in {"photo_shadow", "balanced_scan"}:
        post_phase.append(LightingGradient(p=0.12 if profile_name == "balanced_scan" else 0.2))

    return AugraphyPipeline(paper_phase=paper_phase, post_phase=post_phase)


def add_speckles(image, amount):
    output = image.copy()
    count = int(output.shape[0] * output.shape[1] * amount)
    for _ in range(count):
        x = random.randint(0, output.shape[1] - 1)
        y = random.randint(0, output.shape[0] - 1)
        shade = random.randint(180, 245)
        output[y, x] = (shade, shade, shade)
    return output


def rotate_with_background(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def warp_perspective(image, strength):
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dx = int(w * strength)
    dy = int(h * strength)
    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), random.randint(0, dy)],
        [random.randint(0, dx), h - 1 - random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), h - 1 - random.randint(0, dy)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def add_crumple_texture(image, strength):
    h, w = image.shape[:2]
    noise = np.random.normal(0, strength, (h, w)).astype(np.float32)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=31, sigmaY=31)
    texture = np.repeat(blur[:, :, np.newaxis], 3, axis=2)
    output = image.astype(np.float32) - texture
    return np.clip(output, 0, 255).astype(np.uint8)


def add_fold_lines(image, count):
    output = image.copy()
    h, w = output.shape[:2]
    for _ in range(count):
        if random.random() < 0.5:
            x = random.randint(int(w * 0.15), int(w * 0.85))
            cv2.line(output, (x, 0), (x, h), (185, 185, 185), 2)
            cv2.line(output, (x + 2, 0), (x + 2, h), (120, 120, 120), 1)
        else:
            y = random.randint(int(h * 0.15), int(h * 0.85))
            cv2.line(output, (0, y), (w, y), (185, 185, 185), 2)
            cv2.line(output, (0, y + 2), (w, y + 2), (120, 120, 120), 1)
    return output


def add_edge_shadow(image, side, strength):
    output = image.astype(np.float32)
    h, w = output.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    if side in {"left", "right"}:
        gradient = np.linspace(strength, 0, int(w * 0.24), dtype=np.float32)
        if side == "left":
            mask[:, :gradient.shape[0]] -= gradient
        else:
            mask[:, -gradient.shape[0]:] -= gradient[::-1]
    else:
        gradient = np.linspace(strength, 0, int(h * 0.24), dtype=np.float32)
        if side == "top":
            mask[:gradient.shape[0], :] -= gradient[:, None]
        else:
            mask[-gradient.shape[0]:, :] -= gradient[::-1][:, None]
    output *= np.clip(mask[:, :, None], 0.6, 1.0)
    return np.clip(output, 0, 255).astype(np.uint8)


def add_corner_fade(image):
    output = image.astype(np.float32)
    h, w = output.shape[:2]
    overlay = np.zeros((h, w), dtype=np.float32)
    cx = random.choice([0, w - 1])
    cy = random.choice([0, h - 1])
    for y in range(h):
        for x in range(w):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            overlay[y, x] = max(0.0, 1.0 - dist / (min(w, h) * 0.42))
    output += overlay[:, :, None] * random.uniform(18, 34)
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_profile_effects(image, profile_name):
    output = image.copy()
    if profile_name == "clean_flatbed":
        output = cv2.GaussianBlur(output, (0, 0), 0.3)
    elif profile_name == "balanced_scan":
        output = add_speckles(output, 0.0007)
        output = rotate_with_background(output, random.uniform(-0.8, 0.8))
    elif profile_name == "folded_scan":
        output = add_speckles(output, 0.0011)
        output = add_fold_lines(output, random.randint(1, 2))
        output = rotate_with_background(output, random.uniform(-1.4, 1.4))
    elif profile_name == "photo_shadow":
        output = rotate_with_background(output, random.uniform(-2.0, 2.0))
    elif profile_name == "noisy_scan":
        output = add_speckles(output, 0.0018)
        output = cv2.GaussianBlur(output, (0, 0), 0.45)
        output = rotate_with_background(output, random.uniform(-1.0, 1.0))
    elif profile_name == "crumpled_copy":
        output = add_crumple_texture(output, random.uniform(10.0, 18.0))
        output = add_fold_lines(output, random.randint(2, 4))
        output = rotate_with_background(output, random.uniform(-2.2, 2.2))
    elif profile_name == "perspective_photo":
        output = warp_perspective(output, random.uniform(0.015, 0.05))
        output = add_edge_shadow(output, random.choice(["left", "right", "top", "bottom"]), random.uniform(0.12, 0.22))
        output = rotate_with_background(output, random.uniform(-2.5, 2.5))
    elif profile_name == "archive_worn":
        output = add_speckles(output, 0.0022)
        output = add_crumple_texture(output, random.uniform(5.0, 10.0))
        output = add_corner_fade(output)
        output = cv2.GaussianBlur(output, (0, 0), 0.55)
    elif profile_name == "xerox_distorted":
        output = warp_perspective(output, random.uniform(0.008, 0.028))
        output = add_speckles(output, 0.0026)
        output = cv2.GaussianBlur(output, (0, 0), 0.8)
        output = rotate_with_background(output, random.uniform(-1.6, 1.6))
    return output


def transform_bboxes(bboxes, scale_x, scale_y, x_offset, y_offset):
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


def apply_realism(pil_img, bboxes, profile_name):
    form_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    pipeline = build_augraphy_pipeline(profile_name)
    augmented_result = pipeline(form_cv)
    augmented_form = ensure_color_image(extract_augraphy_image(augmented_result))
    augmented_form = apply_profile_effects(augmented_form, profile_name)

    bg_raw = cv2.imread(get_random_asset(BGS_DIR))
    bg_img = ensure_color_image(cv2.resize(bg_raw, BACKGROUND_SIZE))
    bg_img = cv2.GaussianBlur(bg_img, (0, 0), 5)
    bg_img = cv2.addWeighted(bg_img, 0.14, np.full_like(bg_img, 247), 0.86, 0)

    form_h, form_w = augmented_form.shape[:2]
    scale = random.uniform(0.90, 0.965)
    new_w, new_h = int(form_w * scale), int(form_h * scale)
    augmented_form_resized = cv2.resize(augmented_form, (new_w, new_h))
    augmented_form_resized = ensure_color_image(augmented_form_resized)

    x_offset = (bg_img.shape[1] - new_w) // 2 + random.randint(-18, 18)
    y_offset = (bg_img.shape[0] - new_h) // 2 + random.randint(-18, 18)
    x_offset = max(0, min(x_offset, bg_img.shape[1] - new_w))
    y_offset = max(0, min(y_offset, bg_img.shape[0] - new_h))

    shadow = np.zeros_like(bg_img)
    shadow[y_offset + 10:y_offset + new_h + 10, x_offset + 10:x_offset + new_w + 10] = 18
    shadow = cv2.GaussianBlur(shadow, (0, 0), 10)
    bg_img = cv2.subtract(bg_img, shadow)
    bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = augmented_form_resized

    if profile_name in {"photo_shadow", "perspective_photo", "crumpled_copy"}:
        bg_img = add_edge_shadow(bg_img, random.choice(["left", "right", "top", "bottom"]), random.uniform(0.08, 0.18))
    if profile_name in {"archive_worn", "crumpled_copy"} and random.random() < 0.35:
        pil_bg = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_bg)
        draw_stamp(draw, random.randint(200, bg_img.shape[1] - 200), random.randint(220, bg_img.shape[0] - 220), "RECEIVED", (80, 70, 155))
        bg_img = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)

    scale_x = new_w / form_w
    scale_y = new_h / form_h
    final_bboxes = transform_bboxes(bboxes, scale_x, scale_y, x_offset, y_offset)
    return bg_img, final_bboxes


def render_layout(data, domain, layout_name):
    image = Image.new("RGB", FORM_SIZE, color=(252, 250, 244))
    draw = ImageDraw.Draw(image)

    printed_font_path = get_font_path(PRINTED_FONTS_DIR, FONTS_DIR)
    handwritten_font_path = get_font_path(HANDWRITTEN_FONTS_DIR, FONTS_DIR)
    fonts = (
        ImageFont.truetype(printed_font_path, 36),
        ImageFont.truetype(printed_font_path, 20),
        ImageFont.truetype(printed_font_path, 24),
        ImageFont.truetype(printed_font_path, 28),
        ImageFont.truetype(handwritten_font_path, 34),
    )

    bboxes = []
    LAYOUTS[layout_name](draw, data, fonts, bboxes, domain)
    return image, bboxes


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")

    layout_names = list(LAYOUTS.keys())
    profile_names = [
        "clean_flatbed",
        "balanced_scan",
        "folded_scan",
        "photo_shadow",
        "noisy_scan",
        "crumpled_copy",
        "perspective_photo",
        "archive_worn",
        "xerox_distorted",
    ]

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for index in range(args.num_forms):
            domain = random.choice(DOMAINS)
            layout_name = random.choice(layout_names)
            profile_name = random.choice(profile_names)
            data = build_domain_data(domain)
            clean_img, bboxes = render_layout(data, domain, layout_name)
            final_img, final_bboxes = apply_realism(clean_img, bboxes, profile_name)

            filename_base = f"{domain.replace(' ', '_').lower()}_{layout_name}_{index:04d}"
            image_name = f"{filename_base}.jpg"
            image_path = os.path.join(args.out_dir, image_name)
            meta_path = os.path.join(args.out_dir, f"{filename_base}.json")

            cv2.imwrite(image_path, final_img)

            metadata = {
                "image": image_name,
                "domain": domain,
                "layout_style": layout_name,
                "augmentation_profile": profile_name,
                "fields": final_bboxes,
                "source_data": data,
            }
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)

            manifest_file.write(json.dumps({
                "image": image_name,
                "domain": domain,
                "layout_style": layout_name,
                "augmentation_profile": profile_name,
            }) + "\n")

            print(f"[{index + 1}/{args.num_forms}] {image_name} | layout={layout_name} | aug={profile_name}")


if __name__ == "__main__":
    main()
