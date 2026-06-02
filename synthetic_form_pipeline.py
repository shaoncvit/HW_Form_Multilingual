from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:  # pragma: no cover
    genai = None

try:
    from augraphy import AugraphyPipeline, Folding, LightingGradient, NoiseTexturize
except Exception:  # Augraphy is useful but optional
    AugraphyPipeline = None
    Folding = None
    LightingGradient = None
    NoiseTexturize = None


# -----------------------------
# Constants
# -----------------------------

FORM_SIZE = (1500, 2100)        # W, H clean page canvas
BACKGROUND_SIZE = (1700, 2300)  # W, H final photo canvas
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

SUPPORTED_LANGUAGES = {
    "english", "bengali", "hindi", "tamil", "telugu", "malayalam", "odia"
}

DEFAULT_FAMILIES = [
    "hospital", "prescription", "bank", "school", "university", "membership", "tax", "insurance", "passport"
]

INK_COLORS = {
    "black": (18, 18, 18),
    "blue": (20, 35, 130),
    "dark_blue": (10, 25, 95),
}


# -----------------------------
# Labels / field schemas
# -----------------------------

LABELS: Dict[str, Dict[str, str]] = {
    "english": {
        "name": "Name", "father_name": "Father's Name", "mother_name": "Mother's Name",
        "age": "Age", "gender": "Gender", "dob": "Date of Birth", "phone": "Phone Number",
        "email": "Email", "address": "Address", "pin_code": "PIN Code", "city": "City",
        "district": "District", "state": "State", "nationality": "Nationality", "occupation": "Occupation",
        "blood_group": "Blood Group", "allergy": "Allergy", "remarks": "Remarks", "account_no": "Account No.",
        "pan": "PAN", "amount": "Amount", "date": "Date", "course": "Course", "roll_no": "Roll No.",
        "registration_no": "Registration No.", "signature": "Signature", "marital_status": "Marital Status",
        "membership_type": "Membership Type", "branch": "Branch", "policy_no": "Policy No.",
        "claim_no": "Claim No.", "passport_type": "Passport Type", "guardian_name": "Guardian Name",
        "exam_name": "Exam Name", "year": "Year", "board": "Board", "marks": "Marks",
        "patient_id": "Patient ID", "bed_no": "Bed No.", "doctor_name": "Doctor Name", "nurse_name": "Nurse Name",
        "diagnosis": "Provisional Diagnosis", "investigation_orders": "Investigation Orders",
        "plan_of_care": "Plan of Care", "chief_complaints": "Chief Complaints",
        "history_present_illness": "History of Present Illness", "past_history": "Past History",
        "procedure": "Procedure", "time": "Time", "bp": "BP", "hr": "HR", "rr": "RR",
        "spo2": "SPO2", "temperature": "Temperature", "grbs": "GRBS", "review_systems": "Review of Systems",
        "drug_name": "Drug Name", "dose": "Dose", "route": "Route", "frequency": "Frequency",
        "relation": "Relation to Patient",
    },
    "bengali": {
        "name": "নাম", "father_name": "পিতার নাম", "mother_name": "মাতার নাম",
        "age": "বয়স", "gender": "লিঙ্গ", "dob": "জন্ম তারিখ", "phone": "ফোন নম্বর",
        "email": "ইমেইল", "address": "ঠিকানা", "pin_code": "পিন কোড", "city": "শহর",
        "district": "জেলা", "state": "রাজ্য", "nationality": "জাতীয়তা", "occupation": "পেশা",
        "blood_group": "রক্তের গ্রুপ", "allergy": "অ্যালার্জি", "remarks": "মন্তব্য", "account_no": "অ্যাকাউন্ট নম্বর",
        "pan": "প্যান", "amount": "টাকার পরিমাণ", "date": "তারিখ", "course": "কোর্স", "roll_no": "রোল নম্বর",
        "registration_no": "রেজিস্ট্রেশন নম্বর", "signature": "স্বাক্ষর", "marital_status": "বৈবাহিক অবস্থা",
        "membership_type": "সদস্যতার ধরন", "branch": "শাখা", "policy_no": "পলিসি নম্বর", "claim_no": "ক্লেইম নম্বর",
        "passport_type": "পাসপোর্টের ধরন", "guardian_name": "অভিভাবকের নাম",
        "exam_name": "পরীক্ষার নাম", "year": "বছর", "board": "বোর্ড", "marks": "নম্বর",
        "patient_id": "রোগী আইডি", "bed_no": "বেড নম্বর", "doctor_name": "ডাক্তারের নাম", "nurse_name": "নার্সের নাম",
        "diagnosis": "প্রাথমিক রোগ নির্ণয়", "investigation_orders": "পরীক্ষার নির্দেশ", "plan_of_care": "চিকিৎসার পরিকল্পনা",
        "chief_complaints": "প্রধান উপসর্গ", "history_present_illness": "বর্তমান অসুস্থতার ইতিহাস", "past_history": "পূর্ববর্তী ইতিহাস",
        "procedure": "প্রক্রিয়া", "time": "সময়", "bp": "রক্তচাপ", "hr": "হার্ট রেট", "rr": "শ্বাস-প্রশ্বাস", "spo2": "এসপিও২",
        "temperature": "তাপমাত্রা", "grbs": "জিআরবিএস", "review_systems": "সিস্টেম রিভিউ", "drug_name": "ওষুধের নাম",
        "dose": "ডোজ", "route": "রুট", "frequency": "ফ্রিকোয়েন্সি", "relation": "রোগীর সঙ্গে সম্পর্ক",
    },
    "hindi": {
        "name": "नाम", "father_name": "पिता का नाम", "mother_name": "माता का नाम",
        "age": "आयु", "gender": "लिंग", "dob": "जन्म तिथि", "phone": "फोन नंबर",
        "email": "ईमेल", "address": "पता", "pin_code": "पिन कोड", "city": "शहर",
        "district": "जिला", "state": "राज्य", "nationality": "राष्ट्रीयता", "occupation": "पेशा",
        "blood_group": "रक्त समूह", "allergy": "एलर्जी", "remarks": "टिप्पणी", "account_no": "खाता संख्या",
        "pan": "पैन", "amount": "राशि", "date": "दिनांक", "course": "कोर्स", "roll_no": "रोल नंबर",
        "registration_no": "पंजीकरण संख्या", "signature": "हस्ताक्षर", "marital_status": "वैवाहिक स्थिति",
        "membership_type": "सदस्यता प्रकार", "branch": "शाखा", "policy_no": "पॉलिसी नंबर", "claim_no": "क्लेम नंबर",
        "passport_type": "पासपोर्ट प्रकार", "guardian_name": "अभिभावक का नाम",
        "exam_name": "परीक्षा का नाम", "year": "वर्ष", "board": "बोर्ड", "marks": "अंक",
        "patient_id": "रोगी आईडी", "bed_no": "बेड नंबर", "doctor_name": "डॉक्टर का नाम", "nurse_name": "नर्स का नाम",
        "diagnosis": "प्रारंभिक निदान", "investigation_orders": "जांच निर्देश", "plan_of_care": "उपचार योजना",
        "chief_complaints": "मुख्य शिकायतें", "history_present_illness": "वर्तमान बीमारी का इतिहास", "past_history": "पूर्व इतिहास",
        "procedure": "प्रक्रिया", "time": "समय", "bp": "बीपी", "hr": "एचआर", "rr": "आरआर", "spo2": "एसपीओ2",
        "temperature": "तापमान", "grbs": "जीआरबीएस", "review_systems": "सिस्टम समीक्षा", "drug_name": "दवा का नाम",
        "dose": "डोज़", "route": "रूट", "frequency": "आवृत्ति", "relation": "रोगी से संबंध",
    },
    "tamil": {
        "name": "பெயர்", "father_name": "தந்தையின் பெயர்", "mother_name": "தாயின் பெயர்",
        "age": "வயது", "gender": "பாலினம்", "dob": "பிறந்த தேதி", "phone": "தொலைபேசி எண்",
        "email": "மின்னஞ்சல்", "address": "முகவரி", "pin_code": "அஞ்சல் குறியீடு", "city": "நகரம்",
        "district": "மாவட்டம்", "state": "மாநிலம்", "nationality": "தேசியம்", "occupation": "தொழில்",
        "blood_group": "இரத்த வகை", "allergy": "ஒவ்வாமை", "remarks": "குறிப்பு", "account_no": "கணக்கு எண்",
        "date": "தேதி", "course": "பாடநெறி", "roll_no": "ரோல் எண்", "signature": "கையொப்பம்",
        "membership_type": "உறுப்பினர் வகை", "branch": "கிளை", "policy_no": "பாலிசி எண்", "claim_no": "கோரிக்கை எண்",
    },
    "telugu": {
        "name": "పేరు", "father_name": "తండ్రి పేరు", "mother_name": "తల్లి పేరు",
        "age": "వయస్సు", "gender": "లింగం", "dob": "పుట్టిన తేదీ", "phone": "ఫోన్ నంబర్",
        "email": "ఇమెయిల్", "address": "చిరునామా", "pin_code": "పిన్ కోడ్", "city": "నగరం",
        "district": "జిల్లా", "state": "రాష్ట్రం", "nationality": "జాతీయత", "occupation": "వృత్తి",
        "blood_group": "రక్త గ్రూప్", "allergy": "అలర్జీ", "remarks": "గమనిక", "account_no": "ఖాతా సంఖ్య",
        "date": "తేదీ", "course": "కోర్సు", "roll_no": "రోల్ నంబర్", "signature": "సంతకం",
        "membership_type": "సభ్యత్వ రకం", "branch": "శాఖ", "policy_no": "పాలసీ నంబర్", "claim_no": "క్లెయిమ్ నంబర్",
    },
    "malayalam": {
        "name": "പേര്", "father_name": "പിതാവിന്റെ പേര്", "mother_name": "മാതാവിന്റെ പേര്",
        "age": "വയസ്", "gender": "ലിംഗം", "dob": "ജനന തീയതി", "phone": "ഫോൺ നമ്പർ",
        "email": "ഇമെയിൽ", "address": "വിലാസം", "pin_code": "പിൻ കോഡ്", "city": "നഗരം",
        "district": "ജില്ല", "state": "സംസ്ഥാനം", "nationality": "ദേശീയത", "occupation": "തൊഴിൽ",
        "blood_group": "രക്ത ഗ്രൂപ്പ്", "allergy": "അലർജി", "remarks": "കുറിപ്പ്", "account_no": "അക്കൗണ്ട് നമ്പർ",
        "date": "തീയതി", "course": "കോഴ്സ്", "roll_no": "റോൾ നമ്പർ", "signature": "ഒപ്പ്",
        "membership_type": "അംഗത്വ തരം", "branch": "ശാഖ", "policy_no": "പോളിസി നമ്പർ", "claim_no": "ക്ലെയിം നമ്പർ",
    },
    "odia": {
        "name": "ନାମ", "father_name": "ପିତାଙ୍କ ନାମ", "mother_name": "ମାତାଙ୍କ ନାମ",
        "age": "ବୟସ", "gender": "ଲିଙ୍ଗ", "dob": "ଜନ୍ମ ତାରିଖ", "phone": "ଫୋନ ନମ୍ବର",
        "email": "ଇମେଲ", "address": "ଠିକଣା", "pin_code": "ପିନ କୋଡ", "city": "ସହର",
        "district": "ଜିଲ୍ଲା", "state": "ରାଜ୍ୟ", "nationality": "ଜାତୀୟତା", "occupation": "ବୃତ୍ତି",
        "date": "ତାରିଖ", "signature": "ସ୍ୱାକ୍ଷର",
    },
}

FAMILY_TITLES = {
    "english": {
        "hospital": "Hospital Registration Form", "prescription": "Hospital Prescription / Case Sheet", "bank": "Bank Account Opening Form",
        "school": "School Admission Form", "university": "University Examination Form",
        "membership": "Membership Form", "tax": "Income Tax Declaration Form",
        "insurance": "Insurance Claim Form", "passport": "Passport Application Form",
    },
    "bengali": {
        "hospital": "হাসপাতাল নিবন্ধন ফর্ম", "prescription": "হাসপাতালের প্রেসক্রিপশন / কেস শীট", "bank": "ব্যাংক অ্যাকাউন্ট খোলার ফর্ম",
        "school": "বিদ্যালয় ভর্তি ফর্ম", "university": "বিশ্ববিদ্যালয় পরীক্ষার ফর্ম",
        "membership": "সদস্যতা ফর্ম", "tax": "আয়কর ঘোষণা ফর্ম",
        "insurance": "বীমা দাবি ফর্ম", "passport": "পাসপোর্ট আবেদন ফর্ম",
    },
    "hindi": {
        "hospital": "अस्पताल पंजीकरण फॉर्म", "prescription": "अस्पताल प्रिस्क्रिप्शन / केस शीट", "bank": "बैंक खाता खोलने का फॉर्म",
        "school": "विद्यालय प्रवेश फॉर्म", "university": "विश्वविद्यालय परीक्षा फॉर्म",
        "membership": "सदस्यता फॉर्म", "tax": "आयकर घोषणा फॉर्म",
        "insurance": "बीमा दावा फॉर्म", "passport": "पासपोर्ट आवेदन फॉर्म",
    },
}

FAMILY_FIELDS = {
    "hospital": ["name", "father_name", "mother_name", "age", "gender", "dob", "phone", "email", "address", "pin_code", "district", "blood_group", "allergy", "remarks"],
    "prescription": ["name", "patient_id", "bed_no", "age", "gender", "date", "time", "doctor_name", "nurse_name", "chief_complaints", "history_present_illness", "past_history", "bp", "hr", "rr", "spo2", "temperature", "grbs", "diagnosis", "investigation_orders", "plan_of_care", "drug_name", "dose", "route", "frequency", "allergy", "relation", "signature"],
    "bank": ["name", "father_name", "dob", "gender", "phone", "email", "address", "pin_code", "pan", "account_no", "branch", "occupation", "amount", "signature"],
    "school": ["name", "guardian_name", "father_name", "mother_name", "dob", "gender", "phone", "email", "address", "pin_code", "course", "remarks"],
    "university": ["name", "registration_no", "roll_no", "course", "exam_name", "year", "phone", "email", "address", "date", "remarks", "signature"],
    "membership": ["name", "dob", "gender", "phone", "email", "address", "city", "nationality", "membership_type", "occupation", "date", "signature"],
    "tax": ["name", "father_name", "dob", "pan", "phone", "email", "address", "pin_code", "amount", "occupation", "date", "signature"],
    "insurance": ["name", "policy_no", "claim_no", "dob", "gender", "phone", "email", "address", "amount", "date", "remarks", "signature"],
    "passport": ["name", "father_name", "mother_name", "dob", "gender", "phone", "email", "address", "pin_code", "district", "state", "passport_type", "signature"],
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FieldBox:
    key: str
    label: str
    value: str
    style: str
    label_bbox: Optional[List[int]]
    value_bbox: Optional[List[int]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "style": self.style,
            "label_bbox": self.label_bbox,
            "value_bbox": self.value_bbox,
        }


@dataclass
class FontPack:
    title: ImageFont.FreeTypeFont
    subtitle: ImageFont.FreeTypeFont
    label: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    handwritten: ImageFont.FreeTypeFont
    handwritten_small: ImageFont.FreeTypeFont


# -----------------------------
# Utility functions
# -----------------------------

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


def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


def text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont) -> List[int]:
    b = draw.textbbox(xy, str(text), font=font)
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int]) -> List[int]:
    draw.text(xy, str(text), font=font, fill=fill)
    return text_bbox(draw, xy, str(text), font)


def rectangle(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], width: int = 2, fill: Tuple[int, int, int] = (50, 50, 50)) -> None:
    draw.rectangle(box, outline=fill, width=width)


def line(draw: ImageDraw.ImageDraw, pts: Sequence[Tuple[int, int]], width: int = 2, fill: Tuple[int, int, int] = (60, 60, 60)) -> None:
    draw.line(pts, fill=fill, width=width)


def normalize_bbox(b: Optional[Sequence[float]], width: int, height: int) -> Optional[List[int]]:
    if b is None:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
    x1 = clamp(x1, 0, width - 1)
    x2 = clamp(x2, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    y2 = clamp(y2, 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def transform_points(points: np.ndarray, matrix: np.ndarray, perspective: bool) -> np.ndarray:
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    if perspective:
        out = cv2.perspectiveTransform(pts, matrix)
    else:
        out = cv2.transform(pts, matrix)
    return out.reshape(-1, 2)


def transform_bbox(bbox: Optional[List[int]], matrix: np.ndarray, perspective: bool, width: int, height: int) -> Optional[List[int]]:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    out = transform_points(corners, matrix, perspective)
    min_x, min_y = out.min(axis=0)
    max_x, max_y = out.max(axis=0)
    return normalize_bbox([min_x, min_y, max_x, max_y], width, height)


def transform_field_boxes(fields: List[FieldBox], matrix: np.ndarray, perspective: bool, width: int, height: int) -> List[FieldBox]:
    output = []
    for f in fields:
        output.append(FieldBox(
            key=f.key,
            label=f.label,
            value=f.value,
            style=f.style,
            label_bbox=transform_bbox(f.label_bbox, matrix, perspective, width, height),
            value_bbox=transform_bbox(f.value_bbox, matrix, perspective, width, height),
        ))
    return output


def language_labels(language: str) -> Dict[str, str]:
    if language in LABELS:
        return LABELS[language]
    return LABELS["english"]


def label_for(language: str, key: str) -> str:
    labels = language_labels(language)
    return labels.get(key, key.replace("_", " ").title())


def title_for(language: str, family: str) -> str:
    if language in FAMILY_TITLES and family in FAMILY_TITLES[language]:
        return FAMILY_TITLES[language][family]
    if "english" in FAMILY_TITLES and family in FAMILY_TITLES["english"]:
        return FAMILY_TITLES["english"][family]
    return f"{family.title()} Form"


# -----------------------------
# Rate limiter
# -----------------------------

class GeminiRateLimiter:
    """Persistent RPM/RPD limiter. Safe for stop/resume in one process."""

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
                # Sleep until UTC midnight + a small buffer. Useful for long runs.
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


# -----------------------------
# Gemini content generation
# -----------------------------

class GeminiContentGenerator:
    def __init__(self, api_key: str, model_name: str, limiter: GeminiRateLimiter, enabled: bool = True):
        self.enabled = enabled and bool(api_key) and genai is not None
        self.model_name = model_name
        self.limiter = limiter
        self.model = None
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        elif enabled:
            print("[warning] Gemini disabled: missing GEMINI_API_KEY or google-generativeai package. Using local fallback.")

    def generate(self, language: str, family: str, required_keys: List[str]) -> Optional[Dict[str, str]]:
        if not self.enabled or self.model is None:
            return None

        labels = {k: label_for(language, k) for k in required_keys}
        prompt = f"""
You generate realistic but fake synthetic form data for OCR research.
Language: {language}
Form family: {family}
Return STRICT JSON only, no markdown.

Required keys exactly:
{json.dumps(required_keys, ensure_ascii=False)}

For each key, generate a realistic value in {language}. Keep emails/phone numbers syntactically valid.
Do not use real personal information. Use fake names, fake addresses, fake IDs.
The output must be a flat JSON object where keys are exactly the required keys.
Field label reference:
{json.dumps(labels, ensure_ascii=False, indent=2)}
""".strip()

        self.limiter.wait_for_slot()
        try:
            response = self.model.generate_content(prompt)
            text = getattr(response, "text", "")
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Gemini response is not a JSON object")
            return {k: str(data.get(k, "")).strip() for k in required_keys}
        except Exception as e:
            print(f"[warning] Gemini failed for {family}/{language}: {e}. Using fallback.")
            return None


# -----------------------------
# Fallback synthetic content
# -----------------------------

EN_FIRST = ["Animesh", "Paromita", "Dipayan", "Subhendu", "Joseph", "Aritra", "Madhurima", "Ritwik", "Sneha", "Ravi"]
EN_LAST = ["Chatterjee", "Sengupta", "Das", "Mondal", "Fernandes", "Sharma", "Roy", "Nair", "Reddy", "Khan"]
CITIES = ["Kolkata", "Hyderabad", "Delhi", "Pune", "Chennai", "Bengaluru", "Guwahati", "Bhubaneswar"]
STATES = ["West Bengal", "Telangana", "Delhi", "Maharashtra", "Tamil Nadu", "Karnataka", "Assam", "Odisha"]
OCCUPATIONS = ["Student", "Engineer", "Teacher", "Clerk", "Designer", "Research Assistant", "Nurse", "Accountant"]
BLOOD = ["A+", "B+", "O+", "AB+", "O-"]
GENDERS = ["Male", "Female", "Other"]

LOCAL_TRANSLITERATED_NAMES = {
    "bengali": ["অমর্ত্য সেন", "পারমিতা সেনগুপ্ত", "দীপায়ন দাস", "সুব্রত মণ্ডল"],
    "hindi": ["अमित शर्मा", "कविता वर्मा", "राहुल सिंह", "स्नेहा नायर"],
    "tamil": ["அருண் குமார்", "மீனா ரவி", "காவ்யா நாயர்", "விக்ரம் ராஜ்"],
    "telugu": ["రవి కుమార్", "స్నేహ రెడ్డి", "అనిల్ రావు", "కావ్య నాయుడు"],
    "malayalam": ["അനിൽ കുമാർ", "മീന നായർ", "രവി വർമ്മ", "കാവ്യ ദാസ്"],
    "odia": ["ଅମିତ ଦାସ", "ସ୍ନେହା ପାତ୍ର", "ରବି ନାୟକ", "କବିତା ମହାନ୍ତି"],
}


def rand_digits(n: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def random_date(start_year: int = 1975, end_year: int = 2005) -> str:
    return f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(start_year, end_year)}"


def fake_name(language: str) -> str:
    if language == "english":
        return f"{random.choice(EN_FIRST)} {random.choice(EN_LAST)}"
    return random.choice(LOCAL_TRANSLITERATED_NAMES.get(language, LOCAL_TRANSLITERATED_NAMES["hindi"]))


def email_from_name(name: str) -> str:
    ascii_name = re.sub(r"[^a-zA-Z ]", "", name).strip().lower().replace(" ", ".")
    if not ascii_name:
        ascii_name = random.choice(["applicant", "student", "member"])
    return f"{ascii_name}{random.randint(10,99)}@gmail.com"


def fake_value(language: str, key: str, name: str) -> str:
    city = random.choice(CITIES)
    state = random.choice(STATES)
    mapping = {
        "name": name,
        "father_name": fake_name(language),
        "mother_name": fake_name(language),
        "guardian_name": fake_name(language),
        "age": str(random.randint(18, 65)),
        "gender": random.choice(GENDERS),
        "dob": random_date(),
        "phone": "9" + rand_digits(9),
        "email": email_from_name(name),
        "address": f"{random.randint(1, 99)}/{random.randint(1, 9)}, {random.choice(['Park Street', 'MG Road', 'Lake Road', 'Station Road'])}, {city}",
        "pin_code": str(random.randint(700001, 799999)),
        "city": city,
        "district": city,
        "state": state,
        "nationality": "Indian",
        "occupation": random.choice(OCCUPATIONS),
        "blood_group": random.choice(BLOOD),
        "allergy": random.choice(["None", "Dust", "Penicillin", "Peanuts"]),
        "remarks": random.choice(["Documents attached", "Urgent processing requested", "Verified at desk", "Submitted for review"]),
        "account_no": rand_digits(11),
        "pan": "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) + rand_digits(4) + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "amount": str(random.randint(5000, 250000)),
        "date": random_date(2024, 2026),
        "course": random.choice(["B.A. English", "M.A. History", "B.Tech CSE", "Class XII", "Certificate Course"]),
        "roll_no": str(random.randint(1, 999)),
        "registration_no": f"JU{random.randint(100000,999999)}/{random.randint(2020,2026)}",
        "signature": name,
        "marital_status": random.choice(["Single", "Married"]),
        "membership_type": random.choice(["Regular", "Gold", "Platinum"]),
        "branch": random.choice(["Kolkata Main", "Hyderabad Central", "Park Street", "Gachibowli"]),
        "policy_no": f"PL{random.randint(100000,999999)}",
        "claim_no": f"CL{random.randint(100000,999999)}",
        "passport_type": random.choice(["Fresh", "Reissue", "Tatkal"]),
        "exam_name": random.choice(["B.A. Final Year", "Semester Examination", "Corrected Mark Sheet", "Grade Card Revision"]),
        "year": str(random.randint(2017, 2026)),
        "board": random.choice(["W.B.B.S.E", "CBSE", "State Board", "University"]),
        "marks": str(random.randint(280, 495)),
        "patient_id": f"UHID-{random.randint(100, 9999)}",
        "bed_no": str(random.randint(1, 25)),
        "doctor_name": random.choice(["Dr. Praveen", "Dr. S. Rao", "Dr. Meera", "Dr. A. Reddy"]),
        "nurse_name": random.choice(["Sister Raji", "Nurse Kavya", "Bro. Ashok", "Sister Priyanka"]),
        "diagnosis": random.choice(["Viral fever", "Hypovolemic shock", "Hyponatremia", "Bronchitis", "Acute gastritis"]),
        "investigation_orders": random.choice(["CBC, RFT, LFT, ECG", "Chest X-ray, CRP, Electrolytes", "Troponin-I, LFT, RBS", "Urine routine, CBC, ECG"]),
        "plan_of_care": random.choice(["Medical management", "Observation and IV fluids", "Antibiotics and supportive care", "Further evaluation advised"]),
        "chief_complaints": random.choice(["Fever and cough for 3 days", "Shortness of breath since morning", "Abdominal pain and nausea", "Weakness and dizziness"]),
        "history_present_illness": random.choice(["Symptoms gradually worsening over 2 days", "Patient reports intermittent fever and chills", "Complaints started after travel", "Known diabetic with recent deterioration"]),
        "past_history": random.choice(["Diabetes", "Hypertension", "Asthma", "No major illness"]),
        "procedure": random.choice(["Intubation", "Central Line", "Nebulization", "Wound Dressing"]),
        "time": random.choice(["6:20 AM", "7:40 AM", "8:15 PM", "11:05 AM"]),
        "bp": random.choice(["120/80", "90/60", "130/90", "110/70"]),
        "hr": str(random.randint(68, 112)),
        "rr": str(random.randint(16, 30)),
        "spo2": str(random.randint(88, 99)),
        "temperature": random.choice(["98.4 F", "99.8 F", "100.2 F", "101.0 F"]),
        "grbs": str(random.randint(80, 180)),
        "review_systems": random.choice(["CVS: S1S2; RS: Clear", "CNS: NAD; PA: Soft", "RS: B/L AE+; CVS: NAD"]),
        "drug_name": random.choice(["Pantop 40 mg", "Paracetamol 650", "Cefixime 200 mg", "Ondansetron 4 mg"]),
        "dose": random.choice(["1 tab", "2 ml", "40 mg", "500 mg"]),
        "route": random.choice(["IV", "PO", "IM"]),
        "frequency": random.choice(["BD", "TDS", "OD", "SOS"]),
        "relation": random.choice(["Husband", "Wife", "Father", "Mother", "Brother"]),
    }
    return mapping.get(key, f"{key}_{random.randint(100,999)}")


def fallback_record(language: str, family: str, keys: List[str]) -> Dict[str, str]:
    name = fake_name(language)
    return {k: fake_value(language, k, name) for k in keys}


# -----------------------------
# Fonts
# -----------------------------

class FontManager:
    def __init__(self, assets_dir: Path, language: str):
        self.assets_dir = assets_dir
        self.language = language

    def _pick_font(self, kind: str) -> Path:
        candidates = [
            self.assets_dir / "fonts" / self.language / kind,
            self.assets_dir / "fonts" / kind,
            self.assets_dir / "fonts" / "fallback",
            self.assets_dir / "fonts",
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype/noto"),
        ]
        for directory in candidates:
            p = choose_asset(directory, FONT_EXTENSIONS)
            if p:
                return p
        raise FileNotFoundError(
            "No font found. Put .ttf/.otf files in assets/fonts/<language>/printed and handwritten."
        )

    def load(self) -> FontPack:
        printed = self._pick_font("printed")
        handwritten = self._pick_font("handwritten")
        return FontPack(
            title=ImageFont.truetype(str(printed), 38),
            subtitle=ImageFont.truetype(str(printed), 23),
            label=ImageFont.truetype(str(printed), 24),
            small=ImageFont.truetype(str(printed), 18),
            handwritten=ImageFont.truetype(str(handwritten), 34),
            handwritten_small=ImageFont.truetype(str(handwritten), 28),
        )


# -----------------------------
# Layout rendering primitives
# -----------------------------

class LayoutRenderer:
    def __init__(self, language: str, fonts: FontPack, ink_color: Tuple[int, int, int], option_mark_style: str = "tick"):
        self.language = language
        self.fonts = fonts
        self.ink = ink_color
        self.option_mark_style = option_mark_style
        self.fields: List[FieldBox] = []

    def label(self, key: str) -> str:
        return label_for(self.language, key)

    def add_field(self, key: str, value: str, style: str, label_bbox: Optional[List[int]], value_bbox: Optional[List[int]]) -> None:
        self.fields.append(FieldBox(key=key, label=self.label(key), value=str(value), style=style,
                                    label_bbox=label_bbox, value_bbox=value_bbox))

    def _measure_text(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, List[int]]:
        tmp = Image.new("RGB", (10, 10), (255, 255, 255))
        d = ImageDraw.Draw(tmp)
        b = d.textbbox((0, 0), str(text), font=font)
        return b[2] - b[0], b[3] - b[1], [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

    def _draw_rotated_text(self, draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
                           font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int], angle: float) -> List[int]:
        base_img = getattr(draw, "_image", None)
        if base_img is None:
            return draw_text(draw, (x, y), str(text), font, fill)

        tw, th, raw_bbox = self._measure_text(text, font)
        pad = 18
        patch = Image.new("RGBA", (max(1, tw + pad * 2), max(1, th + pad * 2)), (0, 0, 0, 0))
        pd = ImageDraw.Draw(patch)
        rgba_fill = (fill[0], fill[1], fill[2], 255)
        pd.text((pad - raw_bbox[0], pad - raw_bbox[1]), str(text), font=font, fill=rgba_fill)
        rotated = patch.rotate(angle, expand=True, resample=Image.BICUBIC)
        base_img.paste(rotated, (int(x), int(y)), rotated)
        bbox = rotated.getbbox()
        if bbox is None:
            return [int(x), int(y), int(x + tw), int(y + th)]
        return [int(x + bbox[0]), int(y + bbox[1]), int(x + bbox[2]), int(y + bbox[3])]

    def _draw_handwritten_in_region(self, draw: ImageDraw.ImageDraw, region: Tuple[int, int, int, int], text: str,
                                    font: ImageFont.FreeTypeFont, angle_range: Tuple[float, float] = (-5.0, 5.0),
                                    center_bias: float = 0.22, loose_vertical: bool = True) -> List[int]:
        x1, y1, x2, y2 = region
        text = str(text)
        tw, th, _ = self._measure_text(text, font)
        region_w = max(1, x2 - x1)
        region_h = max(1, y2 - y1)

        # horizontal placement: sometimes starts after some space, sometimes nearer middle
        can_center = region_w > tw + 24
        if can_center and random.random() < center_bias:
            x = x1 + max(0, (region_w - tw) // 2) + random.randint(-12, 12)
        else:
            left_pad = random.randint(6, max(8, min(28, region_w // 8)))
            if random.random() < 0.45:
                left_pad += random.randint(8, max(10, min(65, region_w // 4)))
            x = x1 + left_pad

        # keep within region but allow a tiny natural overshoot sometimes
        x = min(max(x, x1 - 3), max(x1 - 3, x2 - tw + 4))

        # vertical placement: not always top-aligned, often middle-ish or slightly tilted
        if loose_vertical and region_h > th + 8 and random.random() < 0.55:
            y = y1 + max(0, (region_h - th) // 2) + random.randint(-6, 6)
        else:
            max_pad = max(0, region_h - th - 2)
            y = y1 + random.randint(-2, max(0, min(max_pad, 10)))
        y = min(max(y, y1 - 4), max(y1 - 4, y2 - th + 4))

        angle = random.uniform(angle_range[0], angle_range[1])
        return self._draw_rotated_text(draw, x, y, text, font, self.ink, angle)

    def draw_inline(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str, width: int,
                    label_width: Optional[int] = None, line_height: int = 40) -> int:
        label = self.label(key)
        label_bbox = draw_text(draw, (x, y), f"{label}:", self.fonts.label, (25, 25, 25))
        start = x + (label_width if label_width else max(150, min(330, len(label) * 13 + 40)))
        baseline = y + line_height
        line(draw, [(start, baseline), (x + width, baseline)], width=2, fill=(90, 90, 90))
        write_region = (start + 6, y - 8, x + width - 4, baseline - 6)
        value_bbox = self._draw_handwritten_in_region(draw, write_region, str(value), self.fonts.handwritten, angle_range=(-4.5, 4.5), center_bias=0.10, loose_vertical=True)
        self.add_field(key, value, "inline_line", label_bbox, value_bbox)
        return max(baseline + 16, value_bbox[3] + 12)

    def draw_box_row(self, draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], key: str, value: str,
                     split_ratio: float = 0.28) -> None:
        x1, y1, x2, y2 = box
        split_x = x1 + int((x2 - x1) * split_ratio)
        rectangle(draw, box, width=2)
        line(draw, [(split_x, y1), (split_x, y2)], width=2)
        label_bbox = draw_text(draw, (x1 + 12, y1 + 10), self.label(key), self.fonts.label, (20, 20, 20))
        font = self.fonts.handwritten_small if len(str(value)) > 36 else self.fonts.handwritten
        write_region = (split_x + 10, y1 + 5, x2 - 8, y2 - 6)
        value_bbox = self._draw_handwritten_in_region(draw, write_region, str(value), font, angle_range=(-5.0, 5.0), center_bias=0.18, loose_vertical=True)
        self.add_field(key, value, "box_row", label_bbox, value_bbox)

    def draw_top_box(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str, width: int,
                     box_h: int = 58) -> int:
        label_bbox = draw_text(draw, (x, y), self.label(key), self.fonts.label, (20, 20, 20))
        box = (x, y + 34, x + width, y + 34 + box_h)
        rectangle(draw, box, width=2)
        font = self.fonts.handwritten_small if len(str(value)) > 34 else self.fonts.handwritten
        write_region = (box[0] + 8, box[1] + 4, box[2] - 8, box[3] - 6)
        short_value = len(str(value)) <= 10
        value_bbox = self._draw_handwritten_in_region(draw, write_region, str(value), font, angle_range=(-5.5, 5.5), center_bias=0.32 if short_value else 0.16, loose_vertical=True)
        self.add_field(key, value, "top_box", label_bbox, value_bbox)
        return box[3] + 18

    def _mark_option(self, draw: ImageDraw.ImageDraw, b: Tuple[int, int, int, int]) -> None:
        if self.option_mark_style == "circle":
            cx = (b[0] + b[2]) // 2 + random.randint(-3, 3)
            cy = (b[1] + b[3]) // 2 + random.randint(-3, 3)
            r = max(6, min(b[2] - b[0], b[3] - b[1]) // 2 - 3 + random.randint(-1, 4))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=self.ink, width=random.choice([2, 3]))
        else:
            pts = [
                (b[0] + random.randint(1, 8), b[1] + random.randint(10, 18)),
                (b[0] + random.randint(8, 16), b[3] + random.randint(-6, 5)),
                (b[2] + random.randint(-5, 7), b[1] + random.randint(1, 8)),
            ]
            line(draw, pts, width=random.choice([3, 4]), fill=self.ink)

    def draw_checkbox(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str,
                      options: Sequence[str]) -> int:
        label_bbox = draw_text(draw, (x, y), f"{self.label(key)}:", self.fonts.label, (20, 20, 20))
        selected_bbox: Optional[List[int]] = None
        current_x = x + max(150, len(self.label(key)) * 14 + 35)
        value_norm = str(value).strip().lower()
        for opt in options:
            b = (current_x, y + 4, current_x + 28, y + 32)
            rectangle(draw, b, width=2)
            draw_text(draw, (current_x + 38, y - 1), opt, self.fonts.label, (20, 20, 20))
            if opt.lower() == value_norm:
                self._mark_option(draw, b)
                selected_bbox = [b[0], b[1], b[2], b[3]]
            current_x += 165 if self.language == "english" else 190
        self.add_field(key, value, "checkbox", label_bbox, selected_bbox)
        return y + 52

    def draw_signature_line(self, draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str, width: int) -> int:
        line(draw, [(x, y + 44), (x + width, y + 44)], width=2)
        write_region = (x + 8, y - 8, x + width - 10, y + 36)
        value_bbox = self._draw_handwritten_in_region(draw, write_region, str(value), self.fonts.handwritten, angle_range=(-6.0, 6.0), center_bias=0.05, loose_vertical=False)
        label_bbox = draw_text(draw, (x + max(20, width // 3), y + 52), self.label(key), self.fonts.small, (25, 25, 25))
        self.add_field(key, value, "signature_line", label_bbox, value_bbox)
        return y + 86


# -----------------------------
# Layouts inspired by the uploaded examples
# -----------------------------

def layout_ta_reporting(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, h = FORM_SIZE
    left, right = 220, w - 150
    center = w // 2
    draw_text(draw, (360, 70), "International Institute of Information Technology, Hyderabad", renderer.fonts.subtitle, (20, 20, 20))
    draw_text(draw, (610, 105), "(Deemed University)", renderer.fonts.small, (20, 20, 20))
    draw_text(draw, (470, 165), title_for(renderer.language, family), renderer.fonts.subtitle, (10, 10, 10))
    draw_text(draw, (900, 160), f"Roll No: {record.get('roll_no', rand_digits(8))}", renderer.fonts.label, (20, 20, 20))

    y = 230
    for key in ["name", "email", "course", "date"]:
        if key in record:
            y = renderer.draw_inline(draw, left, y, key, record[key], right - left, label_width=280)
            y += 12
    y += 10
    if "account_no" in record:
        renderer.draw_box_row(draw, (left + 390, y, right - 250, y + 68), "account_no", record["account_no"], split_ratio=0.02)
        draw_text(draw, (left, y + 16), "SB/Other bank Personal SB Account No:", renderer.fonts.small, (25, 25, 25))
    y += 115
    renderer.draw_checkbox(draw, left, y, "membership_type", record.get("membership_type", "Full"), ["Quarter", "Half", "Full"])
    y += 80
    renderer.draw_checkbox(draw, left, y, "remarks", record.get("remarks", "No"), ["YES", "NO"])
    y += 110
    if "signature" in record:
        renderer.draw_signature_line(draw, right - 520, y, "signature", record["signature"], 430)
    y = h - 430
    for _ in range(3):
        renderer.draw_inline(draw, left + 60, y, "date", record.get("date", random_date(2024, 2026)), 450, label_width=120)
        y += 160


def layout_membership(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, _ = FORM_SIZE
    left, right = 210, w - 210
    header = (left - 80, 90, right + 80, 230)
    rectangle(draw, header, width=2, fill=(100, 100, 100))
    draw.rectangle(header, fill=(165, 165, 165), outline=(90, 90, 90), width=2)
    draw_text(draw, (left + 230, 115), title_for(renderer.language, family), renderer.fonts.title, (250, 250, 250))
    draw_text(draw, (left + 270, 165), "Personal Information", renderer.fonts.subtitle, (250, 250, 250))

    y = 285
    for key in ["name", "city", "dob", "address", "nationality", "gender", "email", "phone"]:
        if key not in record:
            continue
        y = renderer.draw_top_box(draw, left, y, key, record[key], right - left, box_h=52)
    y += 30
    draw_text(draw, (left - 20, y), "Type Of Membership", renderer.fonts.subtitle, (20, 20, 20))
    y += 50
    renderer.draw_checkbox(draw, left - 20, y, "membership_type", record.get("membership_type", "Platinum"), ["Regular", "Gold", "Platinum"])
    y += 95
    draw_text(draw, (left - 20, y), "Term & Condition", renderer.fonts.subtitle, (20, 20, 20))
    for txt in ["This membership is valid", "Membership includes the facilities available", "Membership is not automatic"]:
        y += 34
        draw_text(draw, (left - 20, y), txt, renderer.fonts.small, (25, 25, 25))
    line(draw, [(left - 20, y + 125), (right, y + 125)], width=2)


def layout_university_table(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, h = FORM_SIZE
    left, right = 130, w - 130
    draw_text(draw, (500, 70), "OFFICE OF THE CONTROLLER OF EXAMINATIONS", renderer.fonts.subtitle, (20, 20, 20))
    draw_text(draw, (565, 103), "UNIVERSITY RECORD SECTION", renderer.fonts.small, (20, 20, 20))
    draw_text(draw, (430, 170), title_for(renderer.language, family), renderer.fonts.subtitle, (20, 20, 20))
    draw_text(draw, (left, 245), "Respected Sir,", renderer.fonts.small, (20, 20, 20))
    draw_text(draw, (left, 280), "I request to apply for corrected / revised Grade Card(s) / Mark-sheet(s) as described below.", renderer.fonts.small, (20, 20, 20))

    table = (left, 350, right, 980)
    rectangle(draw, table, width=2)
    col_x = [left, left + 90, left + 560, left + 810, left + 1160, right]
    for x in col_x[1:-1]:
        line(draw, [(x, table[1]), (x, table[3])], width=2)
    header_h = 90
    line(draw, [(left, table[1] + header_h), (right, table[1] + header_h)], width=2)
    headers = ["Sl. No.", "Name of Examination", "Session / Year", "Reason for Correction", "Note"]
    for i, text in enumerate(headers):
        draw_text(draw, (col_x[i] + 10, table[1] + 20), text, renderer.fonts.small, (20, 20, 20))
    row_h = 85
    keys = ["registration_no", "date", "exam_name", "remarks"]
    y = table[1] + header_h
    for idx, key in enumerate(keys, start=1):
        line(draw, [(left, y + row_h), (right, y + row_h)], width=1)
        draw_text(draw, (left + 18, y + 18), f"{idx:02d}.", renderer.fonts.small, (20, 20, 20))
        value = record.get(key, fake_value(renderer.language, key, record.get("name", "Applicant")))
        label_bbox = None
        value_bbox = draw_text(draw, (col_x[1] + 14, y + 12), value, renderer.fonts.handwritten_small, renderer.ink)
        renderer.add_field(key, value, "table_cell", label_bbox, value_bbox)
        y += row_h

    y = 1040
    draw_text(draw, (left, y), "My particulars are given below:", renderer.fonts.small, (20, 20, 20))
    y += 35
    rows = [["name", "roll_no"], ["registration_no", "email"], ["phone", "course"]]
    for pair in rows:
        x = left
        for key in pair:
            renderer.draw_box_row(draw, (x, y, x + (right - left)//2, y + 58), key, record.get(key, ""), split_ratio=0.34)
            x += (right - left)//2
        y += 58
    y += 110
    renderer.draw_signature_line(draw, right - 410, y, "signature", record.get("signature", record.get("name", "Signature")), 350)
    draw_text(draw, (left, h - 220), "Note: The application must be signed by the candidate. Office copy remains for record.", renderer.fonts.small, (25, 25, 25))


def layout_tax_grid(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, h = FORM_SIZE
    left, right = 80, w - 80
    draw_text(draw, (590, 60), "INCOME-TAX RULES, 1962", renderer.fonts.subtitle, (15, 15, 15))
    draw_text(draw, (505, 100), title_for(renderer.language, family), renderer.fonts.small, (15, 15, 15))

    y = 160
    rows = [
        ("name", 90), ("father_name", 90), ("dob", 68), ("address", 90), ("city", 68), ("pin_code", 68),
        ("phone", 68), ("amount", 68), ("date", 68), ("pan", 68), ("occupation", 68), ("remarks", 120)
    ]
    serial = 1
    for key, row_h in rows:
        box = (left, y, right, y + row_h)
        rectangle(draw, box, width=2)
        line(draw, [(left + 60, y), (left + 60, y + row_h)], width=2)
        line(draw, [(left + 360, y), (left + 360, y + row_h)], width=2)
        draw_text(draw, (left + 18, y + 16), str(serial), renderer.fonts.small, (20, 20, 20))
        label_bbox = draw_text(draw, (left + 75, y + 14), renderer.label(key), renderer.fonts.small, (20, 20, 20))
        value = record.get(key, "")
        value_bbox = draw_text(draw, (left + 380, y + 10), value, renderer.fonts.handwritten_small, renderer.ink)
        renderer.add_field(key, value, "pan_grid_cell", label_bbox, value_bbox)
        y += row_h
        serial += 1
        if y > h - 300:
            break
    y += 40
    renderer.draw_checkbox(draw, left, y, "membership_type", record.get("membership_type", "Card"), ["Cash", "Cheque", "Card", "Online"])
    y += 120
    draw_text(draw, (left, y), "Verification", renderer.fonts.subtitle, (15, 15, 15))
    y += 45
    renderer.draw_signature_line(draw, left, y, "signature", record.get("signature", record.get("name", "Signature")), 430)
    renderer.draw_inline(draw, left + 640, y, "date", record.get("date", random_date(2024, 2026)), 430, label_width=120)


def layout_admission_with_marks(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, h = FORM_SIZE
    left, right = 100, w - 100
    draw_text(draw, (475, 65), "UNIVERSITY OF JAMMU, JAMMU", renderer.fonts.title, (20, 20, 20))
    draw_text(draw, (350, 125), "Application for Admission / Certificate Course", renderer.fonts.subtitle, (20, 20, 20))

    y = 210
    for i, key in enumerate(["name", "father_name", "address", "phone", "gender", "dob", "course", "roll_no"], start=1):
        draw_text(draw, (left, y), f"{i}.", renderer.fonts.label, (20, 20, 20))
        if key == "gender":
            renderer.draw_checkbox(draw, left + 60, y, key, record.get(key, "Male"), ["Male", "Female", "Other"])
            y += 60
        else:
            y = renderer.draw_inline(draw, left + 60, y, key, record.get(key, ""), right - left - 60, label_width=260)
            y += 8

    table_top = y + 20
    table = (left, table_top, right, min(table_top + 430, h - 320))
    rectangle(draw, table, width=2)
    col = [left, left + 210, left + 370, left + 570, left + 750, left + 920, left + 1080, right]
    for x in col[1:-1]:
        line(draw, [(x, table[1]), (x, table[3])], width=2)
    line(draw, [(left, table[1] + 70), (right, table[1] + 70)], width=2)
    headers = ["Exam", "Year", "Board", "Marks", "Total", "%", "Subjects"]
    for i, hd in enumerate(headers):
        draw_text(draw, (col[i] + 10, table[1] + 22), hd, renderer.fonts.small, (20, 20, 20))
    yrow = table[1] + 70
    for row_idx, exam in enumerate(["10th", "12th"], start=1):
        line(draw, [(left, yrow + 70), (right, yrow + 70)], width=1)
        values = [exam, str(random.randint(2016, 2023)), record.get("board", "Board"), record.get("marks", "420"), "500", str(random.randint(45, 88)), random.choice(["Arts", "Science", "Commerce"])]
        for i, val in enumerate(values):
            vb = draw_text(draw, (col[i] + 10, yrow + 12), val, renderer.fonts.handwritten_small, renderer.ink)
            renderer.add_field(f"education_{row_idx}_{i}", val, "marks_table", None, vb)
        yrow += 70
    renderer.draw_signature_line(draw, right - 470, h - 250, "signature", record.get("signature", record.get("name", "Signature")), 380)


def layout_dense_office_form(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, h = FORM_SIZE
    left, right = 90, w - 90
    draw_text(draw, (left, 55), title_for(renderer.language, family).upper(), renderer.fonts.title, (20, 20, 20))
    draw_text(draw, (right - 260, 70), f"No. {random.randint(10000, 99999)}", renderer.fonts.subtitle, (20, 20, 20))
    line(draw, [(left, 130), (right, 130)], width=2)
    y = 170
    keys = list(record.keys())
    # Two compact columns, useful for forms with many fields.
    col_w = (right - left - 60) // 2
    for col_idx in range(2):
        x = left + col_idx * (col_w + 60)
        yy = y
        subset = keys[col_idx::2]
        for key in subset[:10]:
            if key == "signature":
                continue
            if key == "gender":
                renderer.draw_checkbox(draw, x, yy, key, record[key], ["Male", "Female", "Other"])
                yy += 65
            else:
                yy = renderer.draw_top_box(draw, x, yy, key, record[key], col_w, box_h=48)
            if yy > h - 300:
                break
    renderer.draw_signature_line(draw, left, h - 210, "signature", record.get("signature", record.get("name", "Signature")), 430)
    renderer.draw_signature_line(draw, right - 460, h - 210, "date", record.get("date", random_date(2024, 2026)), 430)


def layout_prescription_sheet(draw: ImageDraw.ImageDraw, renderer: LayoutRenderer, record: Dict[str, str], family: str) -> None:
    w, h = FORM_SIZE
    left, right = 80, w - 80

    draw_text(draw, (left + 20, 52), "CITY CARE HOSPITAL", renderer.fonts.title, (20, 20, 20))
    draw_text(draw, (right - 300, 62), "24x7 Emergency & Critical Care", renderer.fonts.small, (30, 30, 30))
    draw_text(draw, (right - 280, 88), "Kadapa / Hyderabad Unit", renderer.fonts.small, (30, 30, 30))
    line(draw, [(left, 128), (right, 128)], width=2)

    # Patient strip
    strip = (left, 145, right, 195)
    rectangle(draw, strip, width=2)
    cols = [left, left + 560, left + 700, left + 840, left + 1040, left + 1210, right]
    for x in cols[1:-1]:
        line(draw, [(x, strip[1]), (x, strip[3])], width=1)
    header_keys = [("name", 10), ("age", 575), ("gender", 715), ("patient_id", 855), ("bed_no", 1055)]
    for key, x in header_keys:
        draw_text(draw, (x, 160), renderer.label(key), renderer.fonts.small, (20, 20, 20))
        vb = draw_text(draw, (x + 85, 152), record.get(key, ""), renderer.fonts.handwritten_small, renderer.ink)
        renderer.add_field(key, record.get(key, ""), "patient_strip", None, vb)

    title = title_for(renderer.language, family)
    draw_text(draw, (left + 250, 230), title, renderer.fonts.subtitle, (20, 20, 20))

    # Assessment row
    y = 285
    y = renderer.draw_inline(draw, left, y, "date", record.get("date", ""), 280, label_width=90)
    renderer.draw_inline(draw, left + 350, 285, "time", record.get("time", ""), 240, label_width=90)
    renderer.draw_inline(draw, left + 700, 285, "doctor_name", record.get("doctor_name", ""), right - (left + 700), label_width=150)
    y += 20
    y = renderer.draw_inline(draw, left, y, "nurse_name", record.get("nurse_name", ""), 520, label_width=130)

    # Vitals + review systems
    vitals_box = (left, y + 25, left + 820, y + 400)
    review_box = (left + 850, y + 25, right, y + 400)
    rectangle(draw, vitals_box, width=2)
    rectangle(draw, review_box, width=2)
    draw_text(draw, (vitals_box[0] + 10, vitals_box[1] + 8), "Vitals", renderer.fonts.subtitle, (20, 20, 20))
    draw_text(draw, (review_box[0] + 10, review_box[1] + 8), "Review of Systems", renderer.fonts.subtitle, (20, 20, 20))
    inner_top = vitals_box[1] + 48
    v_cols = [vitals_box[0], vitals_box[0] + 180, vitals_box[0] + 360, vitals_box[0] + 540, vitals_box[0] + 820]
    for x in v_cols[1:-1]:
        line(draw, [(x, inner_top), (x, vitals_box[3])], width=1)
    line(draw, [(vitals_box[0], inner_top), (vitals_box[2], inner_top)], width=1)
    vitals = [("bp", record.get("bp", "")), ("hr", record.get("hr", "")), ("rr", record.get("rr", "")), ("spo2", record.get("spo2", "")), ("temperature", record.get("temperature", "")), ("grbs", record.get("grbs", ""))]
    row_y = inner_top + 12
    for idx, (key, val) in enumerate(vitals):
        row_bottom = row_y + 46
        line(draw, [(vitals_box[0], row_bottom), (vitals_box[2], row_bottom)], width=1)
        draw_text(draw, (vitals_box[0] + 14, row_y + 8), renderer.label(key), renderer.fonts.small, (20, 20, 20))
        vb = draw_text(draw, (vitals_box[0] + 210, row_y + 2), val, renderer.fonts.handwritten_small, renderer.ink)
        renderer.add_field(key, val, "vitals_cell", None, vb)
        row_y += 46
    rv = record.get("review_systems", renderer.label("review_systems"))
    rvb = draw_text(draw, (review_box[0] + 14, review_box[1] + 60), rv, renderer.fonts.handwritten_small, renderer.ink)
    renderer.add_field("review_systems", rv, "review_box", None, rvb)

    y = vitals_box[3] + 30
    y = renderer.draw_top_box(draw, left, y, "chief_complaints", record.get("chief_complaints", ""), right - left, box_h=82)
    y = renderer.draw_top_box(draw, left, y, "history_present_illness", record.get("history_present_illness", ""), right - left, box_h=92)
    y = renderer.draw_top_box(draw, left, y, "past_history", record.get("past_history", ""), right - left, box_h=66)

    # Procedure block with options
    draw_text(draw, (left, y + 6), "Procedure Details", renderer.fonts.subtitle, (20, 20, 20))
    proc_y = y + 44
    procedure_value = record.get("procedure", "Intubation")
    renderer.draw_checkbox(draw, left, proc_y, "procedure", procedure_value, ["Intubation", "Central Line", "ICD", "Others"])
    proc_y += 70
    renderer.draw_checkbox(draw, left, proc_y, "allergy", record.get("allergy", "None"), ["None", "Dust", "Penicillin", "Peanuts"])

    # Diagnosis and orders
    y = proc_y + 80
    diag_h = 90
    renderer.draw_box_row(draw, (left, y, right, y + diag_h), "diagnosis", record.get("diagnosis", ""), split_ratio=0.25)
    y += diag_h + 12
    renderer.draw_box_row(draw, (left, y, right, y + diag_h), "investigation_orders", record.get("investigation_orders", ""), split_ratio=0.25)
    y += diag_h + 12
    renderer.draw_box_row(draw, (left, y, right, y + diag_h), "plan_of_care", record.get("plan_of_care", ""), split_ratio=0.25)

    # Drug order table
    y += 115
    table = (left, y, right, min(h - 255, y + 300))
    rectangle(draw, table, width=2)
    col = [left, left + 90, left + 680, left + 860, left + 1010, left + 1180, right]
    for x in col[1:-1]:
        line(draw, [(x, table[1]), (x, table[3])], width=1)
    headers = ["S.No.", renderer.label("drug_name"), renderer.label("dose"), renderer.label("route"), renderer.label("frequency"), renderer.label("signature")]
    line(draw, [(left, table[1] + 46), (right, table[1] + 46)], width=1)
    for i, hd in enumerate(headers):
        draw_text(draw, (col[i] + 8, table[1] + 12), hd, renderer.fonts.small, (20, 20, 20))
    row_top = table[1] + 46
    for row_idx in range(1, 4):
        row_h = 64
        line(draw, [(left, row_top + row_h), (right, row_top + row_h)], width=1)
        draw_text(draw, (col[0] + 16, row_top + 18), str(row_idx), renderer.fonts.small, (20, 20, 20))
        drug_val = record.get("drug_name", "") if row_idx == 1 else random.choice(["Cefixime 200 mg", "Pantop 40 mg", "IV Fluids", "Ondansetron 4 mg"])
        dose_val = record.get("dose", "") if row_idx == 1 else random.choice(["1 tab", "500 mg", "2 ml"])
        route_val = record.get("route", "") if row_idx == 1 else random.choice(["PO", "IV", "IM"])
        freq_val = record.get("frequency", "") if row_idx == 1 else random.choice(["BD", "TDS", "OD"])
        vals = [("drug_name", drug_val), ("dose", dose_val), ("route", route_val), ("frequency", freq_val), ("signature", record.get("doctor_name", ""))]
        for c_idx, (k, val) in enumerate(vals, start=1):
            vb = draw_text(draw, (col[c_idx] + 8, row_top + 12), val, renderer.fonts.handwritten_small, renderer.ink)
            renderer.add_field(f"{k}_{row_idx}", val, "drug_table", None, vb)
        row_top += row_h

    # Footer sign lines
    renderer.draw_inline(draw, left, h - 180, "relation", record.get("relation", ""), 420, label_width=170)
    renderer.draw_signature_line(draw, right - 460, h - 215, "signature", record.get("signature", record.get("name", "Signature")), 390)


LAYOUT_FUNCTIONS = {
    "ta_reporting": layout_ta_reporting,
    "membership": layout_membership,
    "university_table": layout_university_table,
    "tax_grid": layout_tax_grid,
    "admission_marks": layout_admission_with_marks,
    "dense_office": layout_dense_office_form,
    "prescription_sheet": layout_prescription_sheet,
}


# -----------------------------
# Rendering / augmentation
# -----------------------------

def render_clean_form(language: str, family: str, record: Dict[str, str], fonts: FontPack,
                      layout_name: str, ink_color: Tuple[int, int, int], option_mark_style: str = "tick") -> Tuple[Image.Image, List[FieldBox]]:
    img = Image.new("RGB", FORM_SIZE, color=(252, 250, 244))
    draw = ImageDraw.Draw(img)
    renderer = LayoutRenderer(language, fonts, ink_color, option_mark_style=option_mark_style)
    layout_fn = LAYOUT_FUNCTIONS[layout_name]
    layout_fn(draw, renderer, record, family)
    return img, renderer.fields


def apply_augraphy_if_available(image_bgr: np.ndarray, profile: str) -> np.ndarray:
    if AugraphyPipeline is None:
        return image_bgr
    paper_phase = []
    post_phase = []
    try:
        if profile in {"balanced_scan", "folded_scan", "archive_worn"} and Folding is not None:
            # Keep only stable args because Augraphy versions differ.
            paper_phase.append(Folding(fold_count=1, p=0.18))
        if profile in {"balanced_scan", "noisy_scan", "archive_worn"} and NoiseTexturize is not None:
            paper_phase.append(NoiseTexturize(p=0.18))
        if profile in {"photo", "perspective_photo"} and LightingGradient is not None:
            post_phase.append(LightingGradient(p=0.14))
        pipe = AugraphyPipeline(paper_phase=paper_phase, post_phase=post_phase)
        result = pipe(image_bgr)
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], np.ndarray):
            return result[0]
    except Exception as e:
        print(f"[warning] Augraphy skipped: {e}")
    return image_bgr


def add_speckles(image: np.ndarray, amount: float) -> np.ndarray:
    out = image.copy()
    count = int(out.shape[0] * out.shape[1] * amount)
    if count <= 0:
        return out
    ys = np.random.randint(0, out.shape[0], count)
    xs = np.random.randint(0, out.shape[1], count)
    vals = np.random.randint(170, 245, count)
    out[ys, xs] = np.stack([vals, vals, vals], axis=1)
    return out


def add_fold_lines(image: np.ndarray, count: int = 1) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(count):
        if random.random() < 0.5:
            x = random.randint(int(0.15 * w), int(0.85 * w))
            cv2.line(out, (x, 0), (x, h), (190, 190, 190), 2)
            cv2.line(out, (x + 2, 0), (x + 2, h), (130, 130, 130), 1)
        else:
            y = random.randint(int(0.15 * h), int(0.85 * h))
            cv2.line(out, (0, y), (w, y), (190, 190, 190), 2)
            cv2.line(out, (0, y + 2), (w, y + 2), (130, 130, 130), 1)
    return out


def add_texture(image: np.ndarray, strength: float = 8.0) -> np.ndarray:
    h, w = image.shape[:2]
    noise = np.random.normal(0, strength, (h, w)).astype(np.float32)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=31, sigmaY=31)
    out = image.astype(np.float32) - blur[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)

def add_water_stains(image: np.ndarray, count: int = 4) -> np.ndarray:
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    overlay = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(count):
        cx = random.randint(int(0.08 * w), int(0.92 * w))
        cy = random.randint(int(0.08 * h), int(0.92 * h))
        rx = random.randint(int(0.04 * w), int(0.12 * w))
        ry = random.randint(int(0.03 * h), int(0.10 * h))
        dist = ((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2
        ring = np.exp(-dist * random.uniform(1.6, 3.2))
        inner = np.exp(-dist * random.uniform(4.5, 6.5))
        overlay += np.clip(ring - 0.7 * inner, 0, 1) * random.uniform(20, 48)
    stain = np.stack([overlay * 0.70, overlay * 0.55, overlay * 0.28], axis=2)
    out = out - stain
    return np.clip(out, 0, 255).astype(np.uint8)


def add_vertical_crease(image: np.ndarray, x_ratio: float = 0.5) -> np.ndarray:
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx = int(w * x_ratio) + random.randint(-20, 20)
    band = max(10, w // 120)
    xs = np.arange(w, dtype=np.float32)
    profile = np.exp(-((xs - cx) ** 2) / (2 * (band ** 2)))
    dark = profile * random.uniform(18, 32)
    light = np.roll(profile, random.randint(4, 9)) * random.uniform(8, 18)
    out -= dark[None, :, None]
    out += light[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize_wood_background(size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    base = np.array([68, 45, 28], dtype=np.float32)
    noise = np.random.normal(0, 14, (height, width, 1)).astype(np.float32)
    bg[:] = base
    bg = bg.astype(np.float32) + noise
    # wood grain lines
    for _ in range(max(80, width // 14)):
        x = random.randint(0, width - 1)
        thickness = random.randint(1, 3)
        color_shift = random.randint(-18, 18)
        cv2.line(bg, (x, 0), (x + random.randint(-25, 25), height), (68 + color_shift, 45 + color_shift//2, 28), thickness)
    # scratches
    for _ in range(max(40, width // 20)):
        x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
        x2, y2 = x1 + random.randint(-120, 120), y1 + random.randint(-120, 120)
        shade = random.randint(95, 155)
        cv2.line(bg, (x1, y1), (x2, y2), (shade, shade, shade), 1)
    return np.clip(bg, 0, 255).astype(np.uint8)


def add_edge_wear(image: np.ndarray, severity: float = 1.0) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    tear_color = (230, 224, 210)
    # side nicks
    for _ in range(int(6 * severity)):
        side = random.choice(["left", "right", "top", "bottom"])
        radius = random.randint(12, 35)
        if side in {"left", "right"}:
            cy = random.randint(radius, h - radius)
            cx = 0 if side == "left" else w - 1
            cv2.circle(out, (cx, cy), radius, tear_color, -1)
        else:
            cx = random.randint(radius, w - radius)
            cy = 0 if side == "top" else h - 1
            cv2.circle(out, (cx, cy), radius, tear_color, -1)
    # corner wear
    for (cx, cy) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if random.random() < 0.8:
            radius = random.randint(22, 70)
            cv2.circle(out, (cx, cy), radius, tear_color, -1)
    return out


def add_corner_curl(image: np.ndarray, corner: str = "tl") -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    curl_w = random.randint(max(80, w // 10), max(140, w // 5))
    curl_h = random.randint(max(80, h // 12), max(160, h // 5))
    overlay = out.copy()
    shadow = out.copy()
    if corner == "tl":
        pts = np.array([[0, 0], [curl_w, 0], [0, curl_h]], np.int32)
        fold_pts = np.array([[0, 0], [curl_w, 0], [int(curl_w * 0.28), int(curl_h * 0.72)]], np.int32)
    elif corner == "tr":
        pts = np.array([[w - 1, 0], [w - 1 - curl_w, 0], [w - 1, curl_h]], np.int32)
        fold_pts = np.array([[w - 1, 0], [w - 1 - curl_w, 0], [w - 1 - int(curl_w * 0.28), int(curl_h * 0.72)]], np.int32)
    elif corner == "bl":
        pts = np.array([[0, h - 1], [curl_w, h - 1], [0, h - 1 - curl_h]], np.int32)
        fold_pts = np.array([[0, h - 1], [curl_w, h - 1], [int(curl_w * 0.28), h - 1 - int(curl_h * 0.72)]], np.int32)
    else:
        pts = np.array([[w - 1, h - 1], [w - 1 - curl_w, h - 1], [w - 1, h - 1 - curl_h]], np.int32)
        fold_pts = np.array([[w - 1, h - 1], [w - 1 - curl_w, h - 1], [w - 1 - int(curl_w * 0.28), h - 1 - int(curl_h * 0.72)]], np.int32)

    cv2.fillConvexPoly(overlay, pts, (212, 202, 188))
    cv2.fillConvexPoly(shadow, pts, (125, 110, 92))
    out = cv2.addWeighted(out, 0.82, overlay, 0.18, 0)
    # add shadow just beside fold
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, fold_pts, 255)
    blur = cv2.GaussianBlur(mask, (0, 0), 9)
    out = out.astype(np.float32)
    out -= blur[:, :, None] * 0.08
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.polylines(out, [pts], True, (120, 110, 98), 2)
    return out

def apply_paper_tint(image: np.ndarray, warm_strength: float = 0.12, vignette_strength: float = 0.10) -> np.ndarray:
    out = image.astype(np.float32)
    h, w = out.shape[:2]
    paper_tone = np.zeros_like(out, dtype=np.float32)
    paper_tone[:, :] = np.array([225, 232, 240], dtype=np.float32)  # BGR warm paper tone
    out = out * (1.0 - warm_strength) + paper_tone * warm_strength

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xx - cx) / max(1.0, w / 2.0)) ** 2 + ((yy - cy) / max(1.0, h / 2.0)) ** 2)
    dist = np.clip(dist, 0, 1.5)
    mask = 1.0 - vignette_strength * np.clip(dist - 0.25, 0, 1.0)
    out = out * mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_profile_effects(image_bgr: np.ndarray, profile: str) -> np.ndarray:
    out = apply_augraphy_if_available(image_bgr, profile)
    out = out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if profile == "clean_flatbed":
        out = cv2.GaussianBlur(out, (0, 0), 0.25)
    elif profile == "balanced_scan":
        out = add_speckles(out, 0.0006)
    elif profile == "folded_scan":
        out = add_speckles(out, 0.001)
        out = add_fold_lines(out, random.randint(1, 2))
    elif profile == "noisy_scan":
        out = add_speckles(out, 0.0018)
        out = cv2.GaussianBlur(out, (0, 0), 0.55)
    elif profile == "archive_worn":
        out = add_texture(out, random.uniform(6, 12))
        out = add_speckles(out, 0.002)
        out = add_fold_lines(out, random.randint(1, 3))
    elif profile == "xerox":
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out = cv2.GaussianBlur(out, (0, 0), 0.8)
        out = add_speckles(out, 0.0025)
    elif profile == "tormented":
        out = add_texture(out, random.uniform(10, 16))
        out = add_speckles(out, 0.0022)
        out = add_fold_lines(out, random.randint(2, 4))
        out = add_vertical_crease(out, x_ratio=random.uniform(0.40, 0.60))
        out = add_water_stains(out, count=random.randint(4, 7))
    return out


def affine_rotate(image: np.ndarray, fields: List[FieldBox], max_degrees: float) -> Tuple[np.ndarray, List[FieldBox]]:
    angle = random.uniform(-max_degrees, max_degrees)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, transform_field_boxes(fields, matrix, perspective=False, width=w, height=h)


def perspective_warp(image: np.ndarray, fields: List[FieldBox], strength: float) -> Tuple[np.ndarray, List[FieldBox]]:
    h, w = image.shape[:2]
    dx, dy = int(w * strength), int(h * strength)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), h - 1 - random.randint(0, dy)],
        [random.randint(0, dx), h - 1 - random.randint(0, dy)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, transform_field_boxes(fields, matrix, perspective=True, width=w, height=h)


def paste_on_background(form_bgr: np.ndarray, fields: List[FieldBox], assets_dir: Path, use_backgrounds: bool, profile: Optional[str] = None) -> Tuple[np.ndarray, List[FieldBox]]:
    bg_path = choose_asset(assets_dir / "backgrounds", IMAGE_EXTENSIONS) if use_backgrounds else None
    if profile == "tormented" and (bg_path is None or random.random() < 0.65):
        bg = synthesize_wood_background(BACKGROUND_SIZE)
    elif bg_path is not None:
        bg = cv2.imread(str(bg_path))
        if bg is None:
            bg = np.full((BACKGROUND_SIZE[1], BACKGROUND_SIZE[0], 3), 247, dtype=np.uint8)
        else:
            bg = cv2.resize(bg, BACKGROUND_SIZE)
            bg = cv2.GaussianBlur(bg, (0, 0), 4)
            bg = cv2.addWeighted(bg, 0.14, np.full_like(bg, 247), 0.86, 0)
    else:
        bg = np.full((BACKGROUND_SIZE[1], BACKGROUND_SIZE[0], 3), 247, dtype=np.uint8)

    form_h, form_w = form_bgr.shape[:2]
    scale = random.uniform(0.90, 0.965)
    new_w, new_h = int(form_w * scale), int(form_h * scale)
    resized = cv2.resize(form_bgr, (new_w, new_h))
    x_offset = (bg.shape[1] - new_w) // 2 + random.randint(-30, 30)
    y_offset = (bg.shape[0] - new_h) // 2 + random.randint(-30, 30)
    x_offset = clamp(x_offset, 0, bg.shape[1] - new_w)
    y_offset = clamp(y_offset, 0, bg.shape[0] - new_h)

    # soft paper shadow
    shadow = np.zeros_like(bg)
    shadow_strength = 22 if profile != "tormented" else 34
    shadow_offset = 10 if profile != "tormented" else 14
    shadow_blur = 12 if profile != "tormented" else 16
    shadow[y_offset + shadow_offset:y_offset + new_h + shadow_offset, x_offset + shadow_offset:x_offset + new_w + shadow_offset] = shadow_strength
    shadow = cv2.GaussianBlur(shadow, (0, 0), shadow_blur)
    bg = cv2.subtract(bg, shadow)
    bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    matrix = np.array([[scale, 0.0, x_offset], [0.0, scale, y_offset]], dtype=np.float32)
    final_fields = transform_field_boxes(fields, matrix, perspective=False, width=bg.shape[1], height=bg.shape[0])
    return bg, final_fields


def boxes_overlap(a: Sequence[int], b: Sequence[int], pad: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + pad < bx1 or bx2 + pad < ax1 or ay2 + pad < by1 or by2 + pad < ay1)


def add_stamp_optional(final_bgr: np.ndarray, fields: Optional[List[FieldBox]] = None, probability: float = 0.20) -> np.ndarray:
    if random.random() > probability:
        return final_bgr

    protected_boxes: List[List[int]] = []
    if fields:
        for f in fields:
            if f.label_bbox is not None:
                protected_boxes.append(f.label_bbox)
            if f.value_bbox is not None:
                protected_boxes.append(f.value_bbox)

    rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    color = random.choice([(80, 60, 150), (120, 50, 50), (60, 90, 120)])
    r = random.randint(50, 75)

    chosen = None
    for _ in range(80):
        x = random.randint(190, pil.width - 250)
        y = random.randint(170, pil.height - 240)
        stamp_box = [x - r - 18, y - r - 18, x + r + 18, y + r + 18]
        if all(not boxes_overlap(stamp_box, pb, pad=14) for pb in protected_boxes):
            chosen = (x, y)
            break

    if chosen is None:
        # If no safe blank area is found, skip the stamp entirely instead of covering text.
        return final_bgr

    x, y = chosen
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    draw.ellipse((x - r + 10, y - r + 10, x + r - 10, y + r - 10), outline=color, width=1)
    draw.text((x - r + 10, y - 10), random.choice(["RECEIVED", "VERIFIED", "OFFICE"]), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def apply_realism(clean_img: Image.Image, fields: List[FieldBox], assets_dir: Path, profile: str,
                  use_backgrounds: bool = True) -> Tuple[np.ndarray, List[FieldBox]]:
    form_bgr = cv2.cvtColor(np.array(clean_img), cv2.COLOR_RGB2BGR)
    if profile in {"perspective_photo", "photo", "tormented"}:
        form_bgr, fields = perspective_warp(form_bgr, fields, random.uniform(0.012, 0.045) if profile != "tormented" else random.uniform(0.020, 0.055))
        form_bgr, fields = affine_rotate(form_bgr, fields, max_degrees=2.5 if profile != "tormented" else 3.8)
    else:
        form_bgr, fields = affine_rotate(form_bgr, fields, max_degrees=1.4)
    form_bgr = apply_profile_effects(form_bgr, profile)

    if profile == "tormented":
        # Apply aging/corner damage directly on the form page before pasting, so the effect is visible on the paper itself.
        form_bgr = apply_paper_tint(form_bgr, warm_strength=random.uniform(0.12, 0.20), vignette_strength=random.uniform(0.08, 0.16))
        form_bgr = add_edge_wear(form_bgr, severity=random.uniform(0.9, 1.5))
        if random.random() < 0.92:
            form_bgr = add_corner_curl(form_bgr, corner=random.choice(["tl", "tr", "bl", "br"]))

    if profile == "archive_worn":
        form_bgr = apply_paper_tint(form_bgr, warm_strength=random.uniform(0.06, 0.12), vignette_strength=random.uniform(0.04, 0.08))

    final_bgr, final_fields = paste_on_background(form_bgr, fields, assets_dir, use_backgrounds=use_backgrounds, profile=profile)
    if profile in {"archive_worn", "xerox"}:
        final_bgr = add_stamp_optional(final_bgr, fields=final_fields, probability=0.22)
    if profile == "tormented":
        final_bgr = cv2.GaussianBlur(final_bgr, (0, 0), 0.25)
    return final_bgr, final_fields


# -----------------------------
# Output writers
# -----------------------------

def record_to_flat_json(sample_id: str, record_id: str, family: str, language: str, title: str,
                        fields: Dict[str, str]) -> Dict[str, Any]:
    label_map = {label_for(language, k): v for k, v in fields.items() if k != "signature"}
    return {
        "sample_id": sample_id,
        "record_id": record_id,
        "language": language,
        "family": family,
        f"title_{language}": title,
        "fields": label_map,
    }


def record_to_text(language: str, fields: Dict[str, str]) -> str:
    lines = []
    for k, v in fields.items():
        if k == "signature":
            continue
        lines.append(f"{label_for(language, k)}: {v}")
    return "\n".join(lines) + "\n"


def annotation_json(sample_id: str, record_id: str, family: str, language: str, title: str,
                    variant: str, image_path: Path, fields: List[FieldBox]) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "record_id": record_id,
        "language": language,
        "family": family,
        f"title_{language}": title,
        "variant": variant,
        "image_path": str(image_path),
        "fields": [f.as_dict() for f in fields],
    }


def validate_record(fields: Dict[str, str]) -> Dict[str, Any]:
    checks = {}
    if "phone" in fields:
        checks["phone"] = bool(re.fullmatch(r"[6-9]\d{9}", fields["phone"]))
    if "email" in fields:
        checks["email"] = bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", fields["email"]))
    if "dob" in fields:
        checks["dob"] = bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", fields["dob"]))
    if "pin_code" in fields:
        checks["pin_code"] = bool(re.fullmatch(r"\d{6}", fields["pin_code"]))
    valid_count = sum(1 for v in checks.values() if v)
    return {
        "checks": checks,
        "pass_ratio": round(valid_count / max(1, len(checks)), 4),
    }


# -----------------------------
# Main generator
# -----------------------------

def build_keys_for_family(family: str, max_fields: Optional[int] = None) -> List[str]:
    keys = list(FAMILY_FIELDS.get(family, FAMILY_FIELDS["hospital"]))
    if max_fields is not None and max_fields > 0:
        # keep signature if present
        sig = [k for k in keys if k == "signature"]
        non = [k for k in keys if k != "signature"][:max_fields]
        keys = non + sig
    return keys


def choose_layout_for_family(family: str, requested: str) -> str:
    if requested != "random":
        return requested
    family_bias = {
        "membership": ["membership", "dense_office"],
        "university": ["university_table", "admission_marks", "ta_reporting"],
        "tax": ["tax_grid", "dense_office"],
        "bank": ["tax_grid", "dense_office", "university_table"],
        "school": ["admission_marks", "dense_office"],
        "hospital": ["prescription_sheet", "dense_office", "membership", "university_table"],
        "prescription": ["prescription_sheet", "dense_office"],
    }
    choices = family_bias.get(family, list(LAYOUT_FUNCTIONS.keys()))
    return random.choice(choices)


def existing_count(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    count = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic multilingual filled form images with GT and bboxes.")
    parser.add_argument("--language", default="english", choices=sorted(SUPPORTED_LANGUAGES))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=None, help="Default: continue after manifest length.")
    parser.add_argument("--out-dir", default="synthetic_forms_out")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES, choices=DEFAULT_FAMILIES)
    parser.add_argument("--layout", default="random", choices=["random"] + sorted(LAYOUT_FUNCTIONS.keys()))
    parser.add_argument("--profile", default="random", choices=["random", "clean_flatbed", "balanced_scan", "folded_scan", "photo", "perspective_photo", "noisy_scan", "archive_worn", "xerox", "tormented"])
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini for field values. Otherwise fallback only.")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--rpd", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-fields", type=int, default=0, help="0 means use all family fields.")
    parser.add_argument("--no-backgrounds", action="store_true", help="Save page on plain background instead of tabletop/background images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    assets_dir = Path(args.assets_dir)
    image_dir = out_dir / "images"
    json_dir = out_dir / "json"
    text_dir = out_dir / "text"
    ann_dir = out_dir / "annotations"
    for p in [out_dir, image_dir, json_dir, text_dir, ann_dir]:
        safe_mkdir(p)

    manifest_path = out_dir / "manifest.jsonl"
    state_path = out_dir / "gemini_rate_state.json"
    start_index = args.start_index if args.start_index is not None else existing_count(manifest_path)

    fonts = FontManager(assets_dir, args.language).load()
    limiter = GeminiRateLimiter(state_path=state_path, rpm=args.rpm, rpd=args.rpd)
    gemini = GeminiContentGenerator(
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        model_name=args.model,
        limiter=limiter,
        enabled=args.use_gemini,
    )

    profiles = ["clean_flatbed", "balanced_scan", "folded_scan", "photo", "perspective_photo", "noisy_scan", "archive_worn", "xerox", "tormented"]

    print(f"[start] language={args.language} samples={args.num_samples} out={out_dir}")
    print(f"[resume] start_index={start_index}; Gemini={'on' if gemini.enabled else 'off'}; rpm={args.rpm}; rpd={args.rpd}")

    with manifest_path.open("a", encoding="utf-8") as mf:
        for local_i in range(args.num_samples):
            idx = start_index + local_i
            sample_id = f"form_{idx:06d}"
            record_id = f"rec_{idx + 1:06d}"
            family = random.choice(args.families)
            keys = build_keys_for_family(family, max_fields=args.max_fields if args.max_fields > 0 else None)

            record = gemini.generate(args.language, family, keys)
            if record is None:
                record = fallback_record(args.language, family, keys)
            # ensure no missing values
            fallback = fallback_record(args.language, family, keys)
            record = {k: str(record.get(k) or fallback[k]) for k in keys}
            if "signature" in keys and not record.get("signature"):
                record["signature"] = record.get("name", "Signature")

            title = title_for(args.language, family)
            layout_name = choose_layout_for_family(family, args.layout)
            profile = random.choice(profiles) if args.profile == "random" else args.profile
            ink_name, ink_color = random.choice(list(INK_COLORS.items()))
            option_mark_style = random.choices(["tick", "circle"], weights=[0.88, 0.12], k=1)[0]

            clean_img, clean_fields = render_clean_form(args.language, family, record, fonts, layout_name, ink_color, option_mark_style=option_mark_style)
            final_bgr, final_fields = apply_realism(clean_img, clean_fields, assets_dir, profile, use_backgrounds=not args.no_backgrounds)

            image_path = image_dir / f"{sample_id}.png"
            compact_json_path = json_dir / f"{sample_id}.json"
            text_path = text_dir / f"{sample_id}.txt"
            ann_path = ann_dir / f"{sample_id}.json"

            cv2.imwrite(str(image_path), final_bgr)
            compact = record_to_flat_json(sample_id, record_id, family, args.language, title, record)
            compact["validation"] = validate_record(record)
            ann = annotation_json(sample_id, record_id, family, args.language, title, layout_name, image_path, final_fields)
            ann["augmentation_profile"] = profile
            ann["ink_color"] = ink_name
            ann["option_mark_style"] = option_mark_style

            compact_json_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
            text_path.write_text(record_to_text(args.language, record), encoding="utf-8")
            ann_path.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")

            mf.write(json.dumps({
                "sample_id": sample_id,
                "record_id": record_id,
                "language": args.language,
                "family": family,
                "layout": layout_name,
                "augmentation_profile": profile,
                "option_mark_style": option_mark_style,
                "image_path": str(image_path),
                "json_path": str(compact_json_path),
                "text_path": str(text_path),
                "annotation_path": str(ann_path),
            }, ensure_ascii=False) + "\n")
            mf.flush()

            print(f"[{local_i + 1:04d}/{args.num_samples:04d}] {sample_id} | {args.language} | {family} | {layout_name} | {profile} | ink={ink_name}")

    print("[done] Generation complete.")


if __name__ == "__main__":
    main()
