#!/usr/bin/env python3
"""
Generate EMPTY template + FILLED reference form pairs for handwriting collection.

For each sample this creates:
  - clean empty template image
  - clean filled reference image
  - annotation JSON with label bbox, writing-region bbox, and filled-value bbox
  - simple JSON ground truth
  - text key-value ground truth
  - optional merged PDFs per language

No augmentation is applied.

Expected font structure, strict by default:
  assets/fonts/bengali/printed/*.ttf
  assets/fonts/bengali/handwritten/*.ttf
  assets/fonts/hindi/printed/*.ttf
  assets/fonts/hindi/handwritten/*.ttf

Example:
  export GEMINI_API_KEY="your_key"
  python generate_collection_templates_v2.py \
    --languages bengali hindi \
    --num-per-language 50 \
    --out-dir /ssd_scratch/shaon/form_collection_bn_hi \
    --assets-dir assets \
    --use-gemini \
    --make-pdf
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:  # pragma: no cover
    genai = None

FORM_SIZE = (1500, 2100)  # W, H
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
SUPPORTED_LANGUAGES = {"bengali", "hindi", "english"}

INK_COLORS = {
    "black": (18, 18, 18),
    "blue": (20, 35, 130),
    "dark_blue": (10, 25, 95),
}

LABELS: Dict[str, Dict[str, str]] = {
    "english": {
        "name": "Name", "father_name": "Father's Name", "mother_name": "Mother's Name",
        "guardian_name": "Guardian Name", "age": "Age", "gender": "Gender", "dob": "Date of Birth",
        "phone": "Phone Number", "email": "Email", "address": "Address", "pin_code": "PIN Code",
        "city": "City", "district": "District", "state": "State", "nationality": "Nationality",
        "occupation": "Occupation", "blood_group": "Blood Group", "allergy": "Allergy",
        "remarks": "Remarks", "account_no": "Account No.", "pan": "PAN", "amount": "Amount",
        "date": "Date", "course": "Course", "roll_no": "Roll No.", "registration_no": "Registration No.",
        "signature": "Signature", "marital_status": "Marital Status", "membership_type": "Membership Type",
        "branch": "Branch", "policy_no": "Policy No.", "claim_no": "Claim No.", "passport_type": "Passport Type",
        "exam_name": "Exam Name", "year": "Year", "board": "Board", "marks": "Marks",
    },
    "bengali": {
        "name": "নাম", "father_name": "পিতার নাম", "mother_name": "মাতার নাম", "guardian_name": "অভিভাবকের নাম",
        "age": "বয়স", "gender": "লিঙ্গ", "dob": "জন্ম তারিখ", "phone": "ফোন নম্বর", "email": "ইমেইল",
        "address": "ঠিকানা", "pin_code": "পিন কোড", "city": "শহর", "district": "জেলা", "state": "রাজ্য",
        "nationality": "জাতীয়তা", "occupation": "পেশা", "blood_group": "রক্তের গ্রুপ", "allergy": "অ্যালার্জি",
        "remarks": "মন্তব্য", "account_no": "অ্যাকাউন্ট নম্বর", "pan": "প্যান", "amount": "টাকার পরিমাণ",
        "date": "তারিখ", "course": "কোর্স", "roll_no": "রোল নম্বর", "registration_no": "রেজিস্ট্রেশন নম্বর",
        "signature": "স্বাক্ষর", "marital_status": "বৈবাহিক অবস্থা", "membership_type": "সদস্যতার ধরন",
        "branch": "শাখা", "policy_no": "পলিসি নম্বর", "claim_no": "ক্লেইম নম্বর", "passport_type": "পাসপোর্টের ধরন",
        "exam_name": "পরীক্ষার নাম", "year": "বছর", "board": "বোর্ড", "marks": "নম্বর",
    },
    "hindi": {
        "name": "नाम", "father_name": "पिता का नाम", "mother_name": "माता का नाम", "guardian_name": "अभिभावक का नाम",
        "age": "आयु", "gender": "लिंग", "dob": "जन्म तिथि", "phone": "फोन नंबर", "email": "ईमेल",
        "address": "पता", "pin_code": "पिन कोड", "city": "शहर", "district": "जिला", "state": "राज्य",
        "nationality": "राष्ट्रीयता", "occupation": "पेशा", "blood_group": "रक्त समूह", "allergy": "एलर्जी",
        "remarks": "टिप्पणी", "account_no": "खाता संख्या", "pan": "पैन", "amount": "राशि",
        "date": "दिनांक", "course": "कोर्स", "roll_no": "रोल नंबर", "registration_no": "पंजीकरण संख्या",
        "signature": "हस्ताक्षर", "marital_status": "वैवाहिक स्थिति", "membership_type": "सदस्यता प्रकार",
        "branch": "शाखा", "policy_no": "पॉलिसी नंबर", "claim_no": "क्लेम नंबर", "passport_type": "पासपोर्ट प्रकार",
        "exam_name": "परीक्षा का नाम", "year": "वर्ष", "board": "बोर्ड", "marks": "अंक",
    },
}

FAMILY_TITLES = {
    "english": {
        "hospital": "Hospital Registration Form", "bank": "Bank Account Opening Form", "school": "School Admission Form",
        "university": "University Examination Form", "membership": "Membership Form", "tax": "Income Tax Declaration Form",
        "insurance": "Insurance Claim Form", "passport": "Passport Application Form",
    },
    "bengali": {
        "hospital": "হাসপাতাল নিবন্ধন ফর্ম", "bank": "ব্যাংক অ্যাকাউন্ট খোলার ফর্ম", "school": "বিদ্যালয় ভর্তি ফর্ম",
        "university": "বিশ্ববিদ্যালয় পরীক্ষার ফর্ম", "membership": "সদস্যতা ফর্ম", "tax": "আয়কর ঘোষণা ফর্ম",
        "insurance": "বীমা দাবি ফর্ম", "passport": "পাসপোর্ট আবেদন ফর্ম",
    },
    "hindi": {
        "hospital": "अस्पताल पंजीकरण फॉर्म", "bank": "बैंक खाता खोलने का फॉर्म", "school": "विद्यालय प्रवेश फॉर्म",
        "university": "विश्वविद्यालय परीक्षा फॉर्म", "membership": "सदस्यता फॉर्म", "tax": "आयकर घोषणा फॉर्म",
        "insurance": "बीमा दावा फॉर्म", "passport": "पासपोर्ट आवेदन फॉर्म",
    },
}

FAMILY_FIELDS = {
    "hospital": ["name", "father_name", "mother_name", "age", "gender", "dob", "phone", "email", "address", "pin_code", "district", "blood_group", "allergy", "remarks"],
    "bank": ["name", "father_name", "dob", "gender", "phone", "email", "address", "pin_code", "pan", "account_no", "branch", "occupation", "amount", "signature"],
    "school": ["name", "guardian_name", "father_name", "mother_name", "dob", "gender", "phone", "email", "address", "pin_code", "course", "remarks"],
    "university": ["name", "registration_no", "roll_no", "course", "exam_name", "year", "phone", "email", "address", "date", "remarks", "signature"],
    "membership": ["name", "dob", "gender", "phone", "email", "address", "city", "nationality", "membership_type", "occupation", "date", "signature"],
    "tax": ["name", "father_name", "dob", "pan", "phone", "email", "address", "pin_code", "amount", "occupation", "date", "signature"],
    "insurance": ["name", "policy_no", "claim_no", "dob", "gender", "phone", "email", "address", "amount", "date", "remarks", "signature"],
    "passport": ["name", "father_name", "mother_name", "dob", "gender", "phone", "email", "address", "pin_code", "district", "state", "passport_type", "signature"],
}

DEFAULT_FAMILIES = list(FAMILY_FIELDS.keys())
OPTION_KEYS = {"gender", "marital_status", "membership_type", "passport_type"}

LANG_DIGITS = {
    "english": "0123456789",
    "bengali": "০১২৩৪৫৬৭৮৯",
    "hindi": "०१२३४५६७८९",
}

LOCAL_VALUES = {
    "bengali": {
        "first_names": ["অরিন্দম", "সৌরভ", "অমর্ত্য", "পারমিতা", "দীপায়ন", "সুব্রত", "ঋতুপর্ণা", "মধুমিতা", "কৌশিক", "স্নেহা"],
        "last_names": ["সেন", "দাস", "রায়", "মণ্ডল", "চক্রবর্তী", "সেনগুপ্ত", "ভট্টাচার্য", "মুখার্জি", "চ্যাটার্জি"],
        "cities": ["কলকাতা", "হাওড়া", "শিলিগুড়ি", "দুর্গাপুর", "বর্ধমান", "বহরমপুর", "কৃষ্ণনগর", "আসানসোল"],
        "states": ["পশ্চিমবঙ্গ", "অসম", "ওড়িশা", "ত্রিপুরা", "বিহার"],
        "roads": ["লেক রোড", "স্টেশন রোড", "কলেজ স্ট্রিট", "রবীন্দ্র সরণি", "নেতাজি সুভাষ রোড", "বালিগঞ্জ রোড"],
        "occupations": ["শিক্ষক", "ছাত্র", "গবেষক", "ব্যাংক কর্মী", "নার্স", "কর্মচারী", "ব্যবসায়ী", "প্রকৌশলী"],
        "gender": ["পুরুষ", "মহিলা", "অন্যান্য"],
        "marital_status": ["অবিবাহিত", "বিবাহিত"],
        "membership_type": ["সাধারণ", "গোল্ড", "প্লাটিনাম"],
        "passport_type": ["নতুন", "পুনঃইস্যু", "তৎকাল"],
        "blood": ["এ+", "বি+", "ও+", "এবি+", "ও-"],
        "allergy": ["নেই", "ধুলো", "বাদাম", "পেনিসিলিন"],
        "remarks": ["নথি সংযুক্ত", "দ্রুত প্রক্রিয়াকরণের অনুরোধ", "যাচাই করা হয়েছে", "পর্যালোচনার জন্য জমা"],
        "courses": ["বাংলা অনার্স", "ইতিহাস স্নাতকোত্তর", "কম্পিউটার বিজ্ঞান", "দ্বাদশ শ্রেণি", "সার্টিফিকেট কোর্স"],
        "boards": ["পশ্চিমবঙ্গ বোর্ড", "রাজ্য বোর্ড", "বিশ্ববিদ্যালয়", "উচ্চ মাধ্যমিক শিক্ষা সংসদ"],
        "branches": ["কলকাতা প্রধান শাখা", "পার্ক স্ট্রিট শাখা", "গড়িয়াহাট শাখা", "সল্টলেক শাখা"],
        "email_text": "ইমেইল নেই",
    },
    "hindi": {
        "first_names": ["अमित", "राहुल", "अरिंदम", "कविता", "स्नेहा", "प्रिया", "दीपक", "सौरभ", "मधुरिमा", "अनन्या"],
        "last_names": ["शर्मा", "सिंह", "वर्मा", "दास", "राय", "मंडल", "चक्रवर्ती", "गुप्ता", "नायर"],
        "cities": ["दिल्ली", "कोलकाता", "पटना", "लखनऊ", "भोपाल", "जयपुर", "वाराणसी", "रांची"],
        "states": ["दिल्ली", "उत्तर प्रदेश", "बिहार", "पश्चिम बंगाल", "मध्य प्रदेश", "राजस्थान"],
        "roads": ["स्टेशन रोड", "लेक रोड", "एम जी रोड", "कॉलेज स्ट्रीट", "नेताजी मार्ग", "मुख्य सड़क"],
        "occupations": ["शिक्षक", "छात्र", "शोधकर्ता", "बैंक कर्मचारी", "नर्स", "कर्मचारी", "व्यवसायी", "अभियंता"],
        "gender": ["पुरुष", "महिला", "अन्य"],
        "marital_status": ["अविवाहित", "विवाहित"],
        "membership_type": ["सामान्य", "गोल्ड", "प्लैटिनम"],
        "passport_type": ["नया", "पुनः जारी", "तत्काल"],
        "blood": ["ए+", "बी+", "ओ+", "एबी+", "ओ-"],
        "allergy": ["नहीं", "धूल", "मूंगफली", "पेनिसिलिन"],
        "remarks": ["दस्तावेज़ संलग्न", "शीघ्र प्रक्रिया का अनुरोध", "सत्यापित", "समीक्षा के लिए जमा"],
        "courses": ["हिंदी ऑनर्स", "इतिहास स्नातकोत्तर", "कंप्यूटर विज्ञान", "कक्षा बारह", "प्रमाणपत्र पाठ्यक्रम"],
        "boards": ["राज्य बोर्ड", "विश्वविद्यालय", "माध्यमिक शिक्षा बोर्ड", "उच्च माध्यमिक बोर्ड"],
        "branches": ["दिल्ली मुख्य शाखा", "पटना शाखा", "लखनऊ शाखा", "कोलकाता शाखा"],
        "email_text": "ईमेल नहीं",
    },
}


@dataclass
class FieldBox:
    key: str
    label: str
    value: str
    style: str
    label_bbox: Optional[List[int]]
    value_region_bbox: Optional[List[int]]
    value_bbox: Optional[List[int]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "style": self.style,
            "label_bbox": self.label_bbox,
            "value_region_bbox": self.value_region_bbox,
            "value_bbox": self.value_bbox,
        }


@dataclass
class FontPack:
    title: ImageFont.FreeTypeFont
    subtitle: ImageFont.FreeTypeFont
    label: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    value_font: ImageFont.FreeTypeFont
    value_font_small: ImageFont.FreeTypeFont


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files(directory: Path, extensions: Iterable[str]) -> List[Path]:
    if not directory.exists():
        return []
    exts = {e.lower() for e in extensions}
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")])


def choose_asset(directory: Path, extensions: Iterable[str]) -> Optional[Path]:
    files = list_files(directory, extensions)
    return random.choice(files) if files else None


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int]) -> List[int]:
    text = str(text)
    draw.text(xy, text, font=font, fill=fill)
    b = draw.textbbox(xy, text, font=font)
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]


def rectangle(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], width: int = 2, fill: Tuple[int, int, int] = (60, 60, 60)) -> None:
    draw.rectangle(box, outline=fill, width=width)


def line(draw: ImageDraw.ImageDraw, pts: Sequence[Tuple[int, int]], width: int = 2, fill: Tuple[int, int, int] = (70, 70, 70)) -> None:
    draw.line(pts, fill=fill, width=width)


def now_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def label_for(language: str, key: str) -> str:
    return LABELS.get(language, LABELS["english"]).get(key, key.replace("_", " ").title())


def title_for(language: str, family: str) -> str:
    return FAMILY_TITLES.get(language, FAMILY_TITLES["english"]).get(family, f"{family.title()} Form")


def localize_digits(s: str, language: str) -> str:
    digits = LANG_DIGITS.get(language, LANG_DIGITS["english"])
    return str(s).translate(str.maketrans("0123456789", digits))


def rand_digits(n: int, language: str = "english") -> str:
    raw = "".join(str(random.randint(0, 9)) for _ in range(n))
    return localize_digits(raw, language)


def random_date(language: str = "english", start_year: int = 1975, end_year: int = 2005) -> str:
    raw = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(start_year, end_year)}"
    return localize_digits(raw, language)


def fake_name(language: str) -> str:
    if language in LOCAL_VALUES:
        data = LOCAL_VALUES[language]
        return f"{random.choice(data['first_names'])} {random.choice(data['last_names'])}"
    return random.choice(["Amit Sharma", "Paromita Sen", "Dipayan Das", "Sourav Dutta"])


def fake_value(language: str, key: str, name: str) -> str:
    if language in LOCAL_VALUES:
        data = LOCAL_VALUES[language]
        city = random.choice(data["cities"])
        road = random.choice(data["roads"])
        house = localize_digits(f"{random.randint(1, 99)}/{random.randint(1, 9)}", language)
        values = {
            "name": name,
            "father_name": fake_name(language),
            "mother_name": fake_name(language),
            "guardian_name": fake_name(language),
            "age": localize_digits(str(random.randint(18, 65)), language),
            "gender": random.choice(data["gender"]),
            "dob": random_date(language, 1975, 2005),
            "phone": rand_digits(10, language),
            "email": data["email_text"],
            "address": f"{house}, {road}, {city}",
            "pin_code": rand_digits(6, language),
            "city": city,
            "district": city,
            "state": random.choice(data["states"]),
            "nationality": "ভারতীয়" if language == "bengali" else "भारतीय",
            "occupation": random.choice(data["occupations"]),
            "blood_group": random.choice(data["blood"]),
            "allergy": random.choice(data["allergy"]),
            "remarks": random.choice(data["remarks"]),
            "account_no": rand_digits(11, language),
            "pan": "প্যান নম্বর নেই" if language == "bengali" else "पैन संख्या नहीं",
            "amount": localize_digits(str(random.randint(5000, 250000)), language),
            "date": random_date(language, 2024, 2026),
            "course": random.choice(data["courses"]),
            "roll_no": rand_digits(3, language),
            "registration_no": rand_digits(8, language),
            "signature": name,
            "marital_status": random.choice(data["marital_status"]),
            "membership_type": random.choice(data["membership_type"]),
            "branch": random.choice(data["branches"]),
            "policy_no": rand_digits(8, language),
            "claim_no": rand_digits(8, language),
            "passport_type": random.choice(data["passport_type"]),
            "exam_name": random.choice(data["courses"]),
            "year": localize_digits(str(random.randint(2017, 2026)), language),
            "board": random.choice(data["boards"]),
            "marks": localize_digits(str(random.randint(280, 495)), language),
        }
        return values.get(key, f"{label_for(language, key)} {rand_digits(3, language)}")

    city = random.choice(["Kolkata", "Hyderabad", "Delhi", "Pune", "Chennai"])
    values = {
        "name": name,
        "father_name": fake_name(language),
        "mother_name": fake_name(language),
        "guardian_name": fake_name(language),
        "age": str(random.randint(18, 65)),
        "gender": random.choice(["Male", "Female", "Other"]),
        "dob": random_date(language),
        "phone": rand_digits(10, language),
        "email": "email not provided",
        "address": f"{random.randint(1, 99)}/{random.randint(1, 9)}, Lake Road, {city}",
        "pin_code": rand_digits(6, language),
        "city": city,
        "district": city,
        "state": random.choice(["West Bengal", "Delhi", "Telangana"]),
        "nationality": "Indian",
        "occupation": random.choice(["Student", "Teacher", "Engineer"]),
        "remarks": random.choice(["Documents attached", "Submitted for review"]),
        "signature": name,
    }
    return values.get(key, f"{key}_{random.randint(100, 999)}")


def fallback_record(language: str, keys: List[str]) -> Dict[str, str]:
    name = fake_name(language)
    return {k: fake_value(language, k, name) for k in keys}


def has_english_letters(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(value)))


def options_for(language: str, key: str) -> List[str]:
    if language in LOCAL_VALUES:
        data = LOCAL_VALUES[language]
        if key in {"gender", "marital_status", "membership_type", "passport_type"}:
            return data[key]
    return {
        "gender": ["Male", "Female", "Other"],
        "marital_status": ["Single", "Married"],
        "membership_type": ["Regular", "Gold", "Platinum"],
        "passport_type": ["Fresh", "Reissue", "Tatkal"],
    }.get(key, ["Yes", "No", "NA"])


def enforce_language_record(language: str, record: Dict[str, str], keys: List[str], source: str) -> Dict[str, str]:
    if language not in {"bengali", "hindi"}:
        return record
    fixed = dict(record)
    replacement_name = fake_name(language)
    changed = []
    for key in keys:
        value = str(fixed.get(key, "")).strip()
        opts = options_for(language, key)
        if key in OPTION_KEYS and value not in opts:
            fixed[key] = random.choice(opts)
            changed.append(key)
            continue
        if not value or has_english_letters(value):
            fixed[key] = fake_value(language, key, replacement_name)
            changed.append(key)
        else:
            fixed[key] = localize_digits(value, language)
    if changed:
        print(f"[language-fix] source={source} language={language} replaced: {', '.join(changed)}")
    return fixed


class FontManager:
    def __init__(self, assets_dir: Path, language: str, filled_value_style: str = "handwritten", strict_fonts: bool = True):
        self.assets_dir = assets_dir
        self.language = language
        self.filled_value_style = filled_value_style
        self.strict_fonts = strict_fonts

    def _pick_font(self, kind: str) -> Path:
        language_dir = self.assets_dir / "fonts" / self.language / kind
        p = choose_asset(language_dir, FONT_EXTENSIONS)
        if p:
            return p
        if self.strict_fonts:
            raise FileNotFoundError(
                f"No {kind} font found for language={self.language}. Expected font inside: {language_dir}"
            )
        candidates = [
            self.assets_dir / "fonts" / kind,
            self.assets_dir / "fonts" / "fallback",
            self.assets_dir / "fonts",
            Path("/usr/share/fonts/truetype/noto"),
            Path("/usr/share/fonts/truetype/dejavu"),
        ]
        for directory in candidates:
            p = choose_asset(directory, FONT_EXTENSIONS)
            if p:
                return p
        raise FileNotFoundError(f"No font found for kind={kind}")

    def load(self) -> FontPack:
        printed = self._pick_font("printed")
        value_kind = "handwritten" if self.filled_value_style == "handwritten" else "printed"
        value_font = self._pick_font(value_kind)
        print(f"[font] language={self.language} printed={printed}")
        print(f"[font] language={self.language} value_style={self.filled_value_style} value_font={value_font}")
        return FontPack(
            title=ImageFont.truetype(str(printed), 38),
            subtitle=ImageFont.truetype(str(printed), 25),
            label=ImageFont.truetype(str(printed), 25),
            small=ImageFont.truetype(str(printed), 18),
            value_font=ImageFont.truetype(str(value_font), 34),
            value_font_small=ImageFont.truetype(str(value_font), 28),
        )


class GeminiRateLimiter:
    def __init__(self, state_path: Path, rpm: int = 10, rpd: int = 1500):
        self.state_path = state_path
        self.rpm = int(rpm)
        self.rpd = int(rpd)
        self.state = load_json(state_path, {"date": now_utc_date(), "day_count": 0, "minute_times": []})
        self._rollover_if_needed()

    def _rollover_if_needed(self) -> None:
        today = now_utc_date()
        if self.state.get("date") != today:
            self.state = {"date": today, "day_count": 0, "minute_times": []}
            atomic_write_json(self.state_path, self.state)

    def wait_for_slot(self) -> None:
        while True:
            self._rollover_if_needed()
            now = time.time()
            minute_times = [t for t in self.state.get("minute_times", []) if now - float(t) < 60.0]
            self.state["minute_times"] = minute_times
            day_count = int(self.state.get("day_count", 0))
            if day_count >= self.rpd:
                now_dt = datetime.now(timezone.utc)
                next_midnight = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc).timestamp() + 86400
                sleep_s = max(60, int(next_midnight - now_dt.timestamp()) + 5)
                print(f"[rate-limit] RPD limit reached ({self.rpd}). Sleeping {sleep_s}s until next UTC day.")
                time.sleep(sleep_s)
                continue
            if len(minute_times) >= self.rpm:
                oldest = min(minute_times)
                sleep_s = max(1.0, 60.0 - (now - oldest) + 0.5)
                print(f"[rate-limit] RPM limit reached ({self.rpm}). Sleeping {sleep_s:.1f}s.")
                time.sleep(sleep_s)
                continue
            self.state["minute_times"].append(now)
            self.state["day_count"] = day_count + 1
            atomic_write_json(self.state_path, self.state)
            return


class GeminiContentGenerator:
    def __init__(self, api_key: str, model_name: str, limiter: GeminiRateLimiter, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled and bool(api_key) and genai is not None
        self.limiter = limiter
        self.model = None
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        elif enabled:
            print("[gemini] NOT ACTIVE: missing GEMINI_API_KEY or google-generativeai package. Fallback data will be used.")

    def generate(self, language: str, family: str, keys: List[str]) -> Optional[Dict[str, str]]:
        if not self.enabled or self.model is None:
            return None
        labels = {k: label_for(language, k) for k in keys}
        if language == "bengali":
            language_rule = "All values must be in Bengali script. Do not use English words like Kolkata, Lake Road, Student, Documents attached. Use Bengali digits. For email, use 'ইমেইল নেই'."
        elif language == "hindi":
            language_rule = "All values must be in Devanagari Hindi script. Do not use English words like Kolkata, Lake Road, Student, Documents attached. Use Devanagari digits. For email, use 'ईमेल नहीं'."
        else:
            language_rule = "Values may be in English."
        prompt = f"""
Generate fake but realistic form-filling data for a handwriting collection reference form.
Language: {language}
Family: {family}
Return STRICT JSON only, no markdown.
Required keys exactly:
{json.dumps(keys, ensure_ascii=False)}
Field labels:
{json.dumps(labels, ensure_ascii=False, indent=2)}
Rules:
- Fake data only; do not use real personal data.
- {language_rule}
- Keep the exact keys; do not add extra keys.
""".strip()
        print(f"[gemini] request language={language} family={family} model={self.model_name}")
        self.limiter.wait_for_slot()
        try:
            response = self.model.generate_content(prompt)
            text = getattr(response, "text", "").replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Gemini did not return a JSON object")
            return {k: str(data.get(k, "")).strip() for k in keys}
        except Exception as e:
            print(f"[warning] Gemini failed: {e}. Using fallback.")
            return None


class PairRenderer:
    def __init__(self, language: str, fonts: FontPack, ink: Tuple[int, int, int], fill_values: bool):
        self.language = language
        self.fonts = fonts
        self.ink = ink
        self.fill_values = fill_values
        self.fields: List[FieldBox] = []

    def label(self, key: str) -> str:
        return label_for(self.language, key)

    def add_field(self, key: str, value: str, style: str, label_bbox: Optional[List[int]], value_region_bbox: Optional[List[int]], value_bbox: Optional[List[int]]) -> None:
        self.fields.append(FieldBox(key, self.label(key), str(value), style, label_bbox, value_region_bbox, value_bbox))

    def draw_value(self, draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], value: str, small: bool = False) -> Optional[List[int]]:
        if not self.fill_values:
            return None
        font = self.fonts.value_font_small if small or len(str(value)) > 28 else self.fonts.value_font
        x1, y1, _, _ = box
        return draw_text(draw, (x1 + 12, y1 + 6), str(value), font, self.ink)

    def inline_field(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str, width: int, label_width: int = 260, line_height: int = 46) -> int:
        label_bbox = draw_text(draw, (x, y), f"{self.label(key)}:", self.fonts.label, (25, 25, 25))
        start = x + label_width
        baseline = y + line_height
        line(draw, [(start, baseline), (x + width, baseline)], width=2)
        region = [start + 4, y - 3, x + width - 4, baseline - 4]
        value_bbox = self.draw_value(draw, tuple(region), value)
        self.add_field(key, value, "inline_line", label_bbox, region, value_bbox)
        return baseline + 22

    def top_box_field(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str, width: int, box_h: int = 58) -> int:
        label_bbox = draw_text(draw, (x, y), self.label(key), self.fonts.label, (25, 25, 25))
        box = (x, y + 36, x + width, y + 36 + box_h)
        rectangle(draw, box, width=2)
        region = [box[0] + 4, box[1] + 4, box[2] - 4, box[3] - 4]
        value_bbox = self.draw_value(draw, tuple(region), value, small=(box_h < 58))
        self.add_field(key, value, "top_box", label_bbox, region, value_bbox)
        return box[3] + 20

    def checkbox_field(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str, options: Sequence[str]) -> int:
        label_bbox = draw_text(draw, (x, y), f"{self.label(key)}:", self.fonts.label, (25, 25, 25))
        current_x = x + max(160, len(self.label(key)) * 14 + 40)
        selected_region = None
        value_norm = str(value).strip()
        for opt in options:
            b = (current_x, y + 5, current_x + 28, y + 33)
            rectangle(draw, b, width=2)
            draw_text(draw, (current_x + 38, y), opt, self.fonts.label, (25, 25, 25))
            if self.fill_values and (opt == value_norm or opt in value_norm):
                line(draw, [(b[0] + 3, b[1] + 15), (b[0] + 12, b[3] - 3), (b[2] + 4, b[1] + 4)], width=3, fill=self.ink)
                selected_region = [b[0], b[1], b[2], b[3]]
            current_x += 190 if self.language in {"bengali", "hindi"} else 180
        self.add_field(key, value, "checkbox", label_bbox, selected_region, selected_region if self.fill_values else None)
        return y + 58


def draw_header(draw: ImageDraw.ImageDraw, renderer: PairRenderer, family: str) -> int:
    w, _ = FORM_SIZE
    title = title_for(renderer.language, family)
    bbox = draw.textbbox((0, 0), title, font=renderer.fonts.title)
    tw = bbox[2] - bbox[0]
    draw_text(draw, ((w - tw) // 2, 70), title, renderer.fonts.title, (20, 20, 20))
    line(draw, [(100, 140), (w - 100, 140)], width=2)
    return 180


def render_simple_form(language: str, family: str, record: Dict[str, str], fonts: FontPack, fill_values: bool, ink: Tuple[int, int, int]) -> Tuple[Image.Image, List[FieldBox]]:
    img = Image.new("RGB", FORM_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    renderer = PairRenderer(language, fonts, ink, fill_values)
    w, h = FORM_SIZE
    left, right = 110, w - 110
    y = draw_header(draw, renderer, family)
    layout = random.choice([
        "single_column", "two_column", "table_block",
        "declaration_letter", "portal_grid", "receipt_style",
        "kyc_grid", "handwriting_sample", "pre_enrollment"
    ])
    keys = FAMILY_FIELDS[family]

    if layout == "declaration_letter":
        # A letter/declaration form: heading + paragraph + small table + footer/signature.
        heading = title_for(language, family)
        draw_text(draw, (left, y), heading, fonts.subtitle, (20, 20, 20))
        y += 50
        para_map = {
            "bengali": "আমি ঘোষণা করছি যে উপরোক্ত তথ্য আমার জ্ঞান অনুযায়ী সঠিক। প্রয়োজনীয় নথি সংযুক্ত করা হয়েছে এবং যাচাইয়ের জন্য জমা দেওয়া হলো।",
            "hindi": "मैं घोषणा करता/करती हूँ कि ऊपर दी गई जानकारी मेरे ज्ञान के अनुसार सही है। आवश्यक दस्तावेज़ संलग्न हैं और सत्यापन के लिए जमा किए गए हैं।",
            "english": "I declare that the information provided above is correct to the best of my knowledge and the required documents are attached for verification.",
        }
        words = para_map.get(language, para_map["english"]).split()
        line_txt = ""
        for word in words:
            if len(line_txt + " " + word) > 82:
                draw_text(draw, (left, y), line_txt.strip(), fonts.small, (25, 25, 25))
                y += 34
                line_txt = word
            else:
                line_txt += " " + word
        if line_txt.strip():
            draw_text(draw, (left, y), line_txt.strip(), fonts.small, (25, 25, 25))
            y += 48
        for key in keys[:8]:
            y = renderer.inline_field(draw, left, y, key, record.get(key, ""), right - left, label_width=330)
            if y > h - 420:
                break
        y += 30
        table = (left, y, right, min(y + 360, h - 260))
        rectangle(draw, table, width=2)
        split = left + 360
        line(draw, [(split, table[1]), (split, table[3])], width=2)
        row_h = 70
        yy = table[1]
        for key in keys[8:13]:
            if yy + row_h > table[3]:
                break
            line(draw, [(left, yy + row_h), (right, yy + row_h)], width=1)
            label_bbox = draw_text(draw, (left + 14, yy + 18), renderer.label(key), fonts.label, (25, 25, 25))
            region = [split + 6, yy + 5, right - 6, yy + row_h - 5]
            value_bbox = renderer.draw_value(draw, tuple(region), record.get(key, ""), small=True)
            renderer.add_field(key, record.get(key, ""), "declaration_table", label_bbox, region, value_bbox)
            yy += row_h
        renderer.inline_field(draw, left, h - 190, "date", record.get("date", ""), 520, label_width=140)
        renderer.inline_field(draw, right - 540, h - 190, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    elif layout == "portal_grid":
        # Admission/portal style: many small character boxes and checkbox rows.
        draw_text(draw, (left, y), title_for(language, family), fonts.subtitle, (20, 20, 20))
        draw_text(draw, (right - 260, y), "PHOTO", fonts.small, (80, 80, 80))
        rectangle(draw, (right - 260, y + 35, right - 80, y + 215), width=2)
        y += 250
        for key in [k for k in keys if k not in OPTION_KEYS][:10]:
            label_bbox = draw_text(draw, (left, y + 10), renderer.label(key), fonts.label, (25, 25, 25))
            box_x = left + 350
            n_boxes = 26 if key in {"name", "address", "email"} else 16
            cell_w = min(38, (right - box_x) // n_boxes)
            for j in range(n_boxes):
                rectangle(draw, (box_x + j * cell_w, y, box_x + (j + 1) * cell_w, y + 44), width=1)
            region = [box_x + 4, y + 2, box_x + n_boxes * cell_w - 4, y + 42]
            value_bbox = renderer.draw_value(draw, tuple(region), record.get(key, ""), small=True)
            renderer.add_field(key, record.get(key, ""), "portal_char_boxes", label_bbox, region, value_bbox)
            y += 62
            if y > h - 360:
                break
        for key in [k for k in keys if k in OPTION_KEYS][:3]:
            y = renderer.checkbox_field(draw, left, y, key, record.get(key, ""), options_for(language, key))
        renderer.inline_field(draw, left, h - 190, "date", record.get("date", ""), 520, label_width=140)
        renderer.inline_field(draw, right - 540, h - 190, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    elif layout == "receipt_style":
        # Medical bill/receipt style: compact top receipt fields plus amount/payment lines.
        draw_text(draw, (left + 360, y), "MEDICAL BILL RECEIPT" if language == "english" else title_for(language, family), fonts.subtitle, (20, 20, 20))
        y += 70
        renderer.inline_field(draw, right - 520, y, "date", record.get("date", ""), 500, label_width=120)
        y += 25
        receipt_keys = ["name", "phone", "address", "amount", "remarks", "occupation", "branch", "policy_no", "claim_no"]
        for key in receipt_keys:
            if key in keys or key in record:
                y = renderer.inline_field(draw, left, y, key, record.get(key, fake_value(language, key, record.get("name", ""))), right - left, label_width=390)
            if y > h - 360:
                break
        y += 45
        draw_text(draw, (left, y), "OFFICIAL RECEIPT", fonts.subtitle, (25, 25, 25))
        y += 65
        for key in ["date", "amount", "remarks"]:
            y = renderer.top_box_field(draw, left, y, key, record.get(key, fake_value(language, key, record.get("name", ""))), right - left, box_h=70)
        renderer.inline_field(draw, right - 540, h - 190, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    elif layout == "kyc_grid":
        # KYC/bank style: dense sections, boxes, and checkbox options.
        draw_text(draw, (left + 300, y), "KYC DETAILS UPDATION" if language == "english" else title_for(language, family), fonts.subtitle, (20, 20, 20))
        rectangle(draw, (right - 230, y - 20, right, y + 120), width=2)
        y += 150
        section_no = 1
        for section_title, section_keys in [
            ("Personal Details", ["name", "father_name", "mother_name", "dob", "gender", "marital_status"]),
            ("Contact Details", ["phone", "email", "address", "pin_code", "city", "state"]),
            ("Account Details", ["account_no", "pan", "branch", "occupation", "amount", "date"]),
        ]:
            draw_text(draw, (left, y), f"{section_no}. {section_title}", fonts.subtitle, (20, 20, 20))
            y += 50
            for key in section_keys:
                if key in OPTION_KEYS:
                    y = renderer.checkbox_field(draw, left + 20, y, key, record.get(key, fake_value(language, key, record.get("name", ""))), options_for(language, key))
                else:
                    y = renderer.top_box_field(draw, left + 20, y, key, record.get(key, fake_value(language, key, record.get("name", ""))), right - left - 40, box_h=46)
                if y > h - 280:
                    break
            section_no += 1
            if y > h - 280:
                break
        renderer.inline_field(draw, right - 540, h - 190, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    elif layout == "handwriting_sample":
        # Handwriting collection page: numbers, short words, and one paragraph copy block.
        draw_text(draw, (left + 360, y), "HANDWRITING SAMPLE FORM" if language == "english" else title_for(language, family), fonts.subtitle, (20, 20, 20))
        y += 80
        for key in ["name", "date", "city", "state", "pin_code"]:
            y = renderer.inline_field(draw, left, y, key, record.get(key, fake_value(language, key, record.get("name", ""))), right - left, label_width=180)
        y += 40
        for row in range(5):
            x = left
            for col in range(4):
                key = f"sample_number_{row}_{col}"
                val = rand_digits(random.randint(2, 6), language)
                rectangle(draw, (x, y, x + 250, y + 70), width=2)
                region = [x + 8, y + 4, x + 242, y + 66]
                value_bbox = renderer.draw_value(draw, tuple(region), val, small=False)
                renderer.add_field(key, val, "number_sample_box", None, region, value_bbox)
                x += 300
            y += 92
        para = fake_value(language, "remarks", record.get("name", "")) + " " + fake_value(language, "address", record.get("name", ""))
        draw_text(draw, (left, y), "Copy the following text:" if language == "english" else renderer.label("remarks"), fonts.small, (25, 25, 25))
        y += 35
        rectangle(draw, (left, y, right, y + 260), width=2)
        region = [left + 10, y + 10, right - 10, y + 250]
        value_bbox = renderer.draw_value(draw, tuple(region), para, small=True)
        renderer.add_field("paragraph_sample", para, "paragraph_sample", None, region, value_bbox)

    elif layout == "pre_enrollment":
        # Pre-enrollment form: horizontal rules with a declaration footer.
        draw_text(draw, (left, y), "PRE-ENROLMENT FORM" if language == "english" else title_for(language, family), fonts.subtitle, (20, 20, 20))
        draw_text(draw, (right - 260, y), str(random.randint(1990, 2026)), fonts.subtitle, (20, 20, 20))
        y += 80
        for key in ["name", "father_name", "mother_name", "dob", "nationality", "address", "course", "occupation", "remarks"]:
            y = renderer.inline_field(draw, left, y, key, record.get(key, fake_value(language, key, record.get("name", ""))), right - left, label_width=310)
            if y > h - 430:
                break
        y += 30
        declaration = {
            "bengali": "আমি ঘোষণা করছি যে এই ফর্মে দেওয়া তথ্য সঠিক এবং প্রয়োজনীয় নথি জমা দেওয়া হয়েছে।",
            "hindi": "मैं घोषणा करता/करती हूँ कि इस फॉर्म में दी गई जानकारी सही है और आवश्यक दस्तावेज़ जमा किए गए हैं।",
            "english": "I hereby declare that the information given in this form is true and all required documents are submitted.",
        }.get(language, "I hereby declare that the information is true.")
        rectangle(draw, (left, y, right, y + 190), width=2)
        region = [left + 12, y + 12, right - 12, y + 178]
        value_bbox = renderer.draw_value(draw, tuple(region), declaration, small=True)
        renderer.add_field("declaration_text", declaration, "declaration_box", None, region, value_bbox)
        renderer.inline_field(draw, left, h - 190, "date", record.get("date", ""), 520, label_width=140)
        renderer.inline_field(draw, right - 540, h - 190, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    elif layout == "single_column":
        for key in keys:
            if key in OPTION_KEYS:
                y = renderer.checkbox_field(draw, left, y, key, record.get(key, ""), options_for(language, key))
            elif key in {"address", "remarks"}:
                y = renderer.top_box_field(draw, left, y, key, record.get(key, ""), right - left, box_h=90)
            else:
                y = renderer.inline_field(draw, left, y, key, record.get(key, ""), right - left, label_width=320)
            if y > h - 220:
                break
        renderer.inline_field(draw, left, h - 180, "date", record.get("date", ""), 520, label_width=140)
        renderer.inline_field(draw, right - 540, h - 180, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    elif layout == "two_column":
        col_gap = 55
        col_w = (right - left - col_gap) // 2
        for col_idx in range(2):
            x = left + col_idx * (col_w + col_gap)
            yy = y
            for key in keys[col_idx::2]:
                if key in OPTION_KEYS:
                    yy = renderer.checkbox_field(draw, x, yy, key, record.get(key, ""), options_for(language, key))
                else:
                    yy = renderer.top_box_field(draw, x, yy, key, record.get(key, ""), col_w, box_h=58)
                if yy > h - 260:
                    break
        renderer.inline_field(draw, left, h - 180, "date", record.get("date", ""), 520, label_width=140)
        renderer.inline_field(draw, right - 540, h - 180, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    else:
        intro_map = {"bengali": "আবেদনকারীর বিবরণ", "hindi": "आवेदक का विवरण", "english": "Applicant Details"}
        draw_text(draw, (left, y), intro_map.get(language, "Applicant Details"), fonts.subtitle, (25, 25, 25))
        y += 55
        row_h = 78
        split_x = left + 390
        for key in keys[:18]:
            box = (left, y, right, y + row_h)
            rectangle(draw, box, width=2)
            line(draw, [(split_x, y), (split_x, y + row_h)], width=2)
            label_bbox = draw_text(draw, (left + 15, y + 20), renderer.label(key), fonts.label, (25, 25, 25))
            region = [split_x + 4, y + 4, right - 4, y + row_h - 4]
            value_bbox = renderer.draw_value(draw, tuple(region), record.get(key, ""), small=True)
            renderer.add_field(key, record.get(key, ""), "table_row", label_bbox, region, value_bbox)
            y += row_h
            if y > h - 250:
                break
        renderer.inline_field(draw, left, h - 180, "date", record.get("date", ""), 520, label_width=140)
        renderer.inline_field(draw, right - 540, h - 180, "signature", record.get("signature", record.get("name", "")), 540, label_width=190)

    return img, renderer.fields


def record_to_text(language: str, record: Dict[str, str], keys: List[str]) -> str:
    return "\n".join(f"{label_for(language, k)}: {record.get(k, '')}" for k in keys)


def save_sample(out_root: Path, language: str, sample_id: str, family: str, empty_img: Image.Image, filled_img: Image.Image, fields: List[FieldBox], record: Dict[str, str], keys: List[str], layout_seed: int, source: str) -> None:
    lang_root = out_root / language
    empty_dir = lang_root / "empty_templates"
    filled_dir = lang_root / "filled_references"
    anno_dir = lang_root / "annotations"
    json_dir = lang_root / "json"
    text_dir = lang_root / "text"
    for d in [empty_dir, filled_dir, anno_dir, json_dir, text_dir]:
        safe_mkdir(d)

    empty_path = empty_dir / f"{sample_id}_empty.png"
    filled_path = filled_dir / f"{sample_id}_filled.png"
    anno_path = anno_dir / f"{sample_id}.json"
    json_path = json_dir / f"{sample_id}.json"
    text_path = text_dir / f"{sample_id}.txt"

    empty_img.save(empty_path)
    filled_img.save(filled_path)

    gt_text = record_to_text(language, record, keys)
    text_path.write_text(gt_text, encoding="utf-8")
    simple_json = {
        "sample_id": sample_id,
        "family": family,
        "language": language,
        "title": title_for(language, family),
        "source": source,
        "fields": {label_for(language, k): record.get(k, "") for k in keys},
    }
    json_path.write_text(json.dumps(simple_json, ensure_ascii=False, indent=2), encoding="utf-8")

    anno = {
        "sample_id": sample_id,
        "family": family,
        "language": language,
        "title": title_for(language, family),
        "source": source,
        "layout_seed": layout_seed,
        "empty_template_path": str(empty_path),
        "filled_reference_path": str(filled_path),
        "text_path": str(text_path),
        "json_path": str(json_path),
        "fields": [f.as_dict() for f in fields],
        "note": "value_region_bbox indicates where the writer should enter the value on the empty template; value_bbox is the bbox of the rendered value on the filled reference.",
    }
    anno_path.write_text(json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = lang_root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps({
            "sample_id": sample_id,
            "language": language,
            "family": family,
            "source": source,
            "empty_template_path": str(empty_path),
            "filled_reference_path": str(filled_path),
            "annotation_path": str(anno_path),
            "json_path": str(json_path),
            "text_path": str(text_path),
        }, ensure_ascii=False) + "\n")


def save_images_as_pdf(image_paths: List[Path], pdf_path: Path, dpi: int = 200) -> None:
    if not image_paths:
        return
    safe_mkdir(pdf_path.parent)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    images[0].save(pdf_path, "PDF", save_all=True, append_images=images[1:], resolution=dpi)
    for img in images:
        img.close()
    print(f"[pdf] saved {pdf_path} pages={len(image_paths)}")


def build_language_pdfs(out_root: Path, language: str, dpi: int = 200) -> None:
    lang_root = out_root / language
    manifest_path = lang_root / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"[pdf] no manifest: {manifest_path}")
        return
    rows = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid in seen:
                continue
            seen.add(sid)
            rows.append(row)

    empty_paths = [Path(r["empty_template_path"]) for r in rows if Path(r["empty_template_path"]).exists()]
    filled_paths = [Path(r["filled_reference_path"]) for r in rows if Path(r["filled_reference_path"]).exists()]
    paired_paths: List[Path] = []
    for r in rows:
        ep = Path(r["empty_template_path"])
        fp = Path(r["filled_reference_path"])
        if ep.exists():
            paired_paths.append(ep)
        if fp.exists():
            paired_paths.append(fp)

    pdf_dir = lang_root / "pdf"
    save_images_as_pdf(empty_paths, pdf_dir / f"{language}_empty_templates_merged.pdf", dpi=dpi)
    save_images_as_pdf(filled_paths, pdf_dir / f"{language}_filled_references_merged.pdf", dpi=dpi)
    save_images_as_pdf(paired_paths, pdf_dir / f"{language}_empty_filled_pairs_merged.pdf", dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate empty+filled form pairs for handwriting collection.")
    parser.add_argument("--languages", nargs="+", default=["bengali", "hindi"], choices=sorted(SUPPORTED_LANGUAGES))
    parser.add_argument("--num-per-language", type=int, default=50)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES, choices=DEFAULT_FAMILIES)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ink", choices=["black", "blue", "dark_blue", "random"], default="black")
    parser.add_argument("--filled-value-style", choices=["printed", "handwritten"], default="printed", help="Style for the reference filled form values.")
    parser.add_argument("--use-gemini", action="store_true")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--rpd", type=int, default=1500)
    parser.add_argument("--allow-font-fallback", action="store_true", help="Allow generic/system fonts. Default requires assets/fonts/<language>/<kind>.")
    parser.add_argument("--make-pdf", action="store_true", help="Create merged PDF files after generation.")
    parser.add_argument("--pdf-dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_root = Path(args.out_dir)
    safe_mkdir(out_root)

    state_path = out_root / "gemini_rate_state.json"
    limiter = GeminiRateLimiter(state_path=state_path, rpm=args.rpm, rpd=args.rpd)
    generator = GeminiContentGenerator(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        model_name=args.gemini_model,
        limiter=limiter,
        enabled=args.use_gemini,
    )
    print(f"[gemini] requested={args.use_gemini} active={generator.enabled} model={args.gemini_model} rpm={args.rpm} rpd={args.rpd}")

    for language in args.languages:
        fonts = FontManager(
            Path(args.assets_dir),
            language,
            filled_value_style=args.filled_value_style,
            strict_fonts=not args.allow_font_fallback,
        ).load()

        done_file = out_root / language / "done_ids.txt"
        safe_mkdir(done_file.parent)
        done_ids = set(done_file.read_text(encoding="utf-8").splitlines()) if done_file.exists() else set()

        for i in range(args.num_per_language):
            sample_id = f"{language[:2]}_collection_{i:04d}"
            if sample_id in done_ids:
                print(f"[skip] {sample_id}")
                continue

            family = random.choice(args.families)
            keys = FAMILY_FIELDS[family]
            gemini_record = generator.generate(language, family, keys)
            source = "gemini" if gemini_record is not None else "fallback"
            record = gemini_record or fallback_record(language, keys)
            record = enforce_language_record(language, record, keys, source=source)

            if "signature" in keys and not record.get("signature"):
                record["signature"] = record.get("name", "")
            if "date" in keys and not record.get("date"):
                record["date"] = random_date(language, 2024, 2026)

            ink = random.choice(list(INK_COLORS.values())) if args.ink == "random" else INK_COLORS[args.ink]

            layout_seed = random.randint(0, 10**9)
            random.seed(layout_seed)
            empty_img, _empty_fields = render_simple_form(language, family, record, fonts, fill_values=False, ink=ink)
            random.seed(layout_seed)
            filled_img, fields = render_simple_form(language, family, record, fonts, fill_values=True, ink=ink)
            random.seed(args.seed + i + len(done_ids))

            save_sample(out_root, language, sample_id, family, empty_img, filled_img, fields, record, keys, layout_seed, source)
            with done_file.open("a", encoding="utf-8") as df:
                df.write(sample_id + "\n")
            print(f"[saved] {sample_id} language={language} family={family} source={source}")

        if args.make_pdf:
            build_language_pdfs(out_root, language, dpi=args.pdf_dpi)

    print(f"[done] output: {out_root}")


if __name__ == "__main__":
    main()
