"""
Donut-based Document Understanding Extractor.

Uses the pre-trained Donut model (naver-clova-ix/donut-base-finetuned-cord-v2)
for OCR-free structured field extraction from receipt/invoice images.

Falls back to the heuristic Tesseract-based extractor if Donut is unavailable
or fails on a specific image.
"""

import re
import json
from PIL import Image

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Lazy-loaded globals to avoid import cost at module level
_processor = None
_model = None
_device = None
_donut_available = None

# The CORD-v2 fine-tuned Donut model is ideal for receipts
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"


def _ensure_donut_loaded():
    """Lazy-load Donut model and processor on first use."""
    global _processor, _model, _device, _donut_available

    if _donut_available is not None:
        return _donut_available

    if not _TORCH_AVAILABLE:
        print("[DonutExtractor] torch not installed. Using heuristic fallback.")
        _donut_available = False
        return False

    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DonutExtractor] Loading model '{MODEL_NAME}' on {_device}...")

        _processor = DonutProcessor.from_pretrained(MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()

        _donut_available = True
        print("[DonutExtractor] Model loaded successfully.")
    except Exception as e:
        print(f"[DonutExtractor] Could not load Donut model: {e}")
        print("[DonutExtractor] Will fall back to heuristic extractor.")
        _donut_available = False

    return _donut_available


def _run_donut_inference(image: Image.Image) -> dict:
    """
    Run the Donut model on a single PIL image and return the raw
    parsed token output as a Python dict.
    """
    global _processor, _model, _device

    # The CORD-v2 task prompt
    task_prompt = "<s_cord-v2>"

    # Prepare pixel values
    pixel_values = _processor(image, return_tensors="pt").pixel_values.to(_device)

    # Prepare decoder input ids from the task prompt
    decoder_input_ids = _processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(_device)

    # Generate
    with torch.no_grad():
        outputs = _model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=_model.decoder.config.max_position_embeddings,
            pad_token_id=_processor.tokenizer.pad_token_id,
            eos_token_id=_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,          # greedy for speed
            bad_words_ids=[[_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    # Decode
    sequence = _processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(_processor.tokenizer.eos_token, "").replace(
        _processor.tokenizer.pad_token, ""
    )
    # Remove the task prompt token
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    # The Donut token2json helper converts the XML-like tokens to a dict
    try:
        parsed = _processor.token2json(sequence)
    except Exception:
        parsed = {}

    return parsed


def _extract_vendor_from_donut(parsed: dict) -> str | None:
    """Pull vendor/store name from the Donut CORD output."""
    # CORD schema nests under "menu", "sub_total", "total", etc.
    # The store name is not a standard CORD field, but sometimes it
    # appears under custom keys. We check a few common locations.
    for key in ("store_name", "company", "vendor", "nm"):
        if key in parsed:
            val = parsed[key]
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, dict):
                    for sub_key in ("nm", "name", "value"):
                        if sub_key in first:
                            return str(first[sub_key]).strip()
                return str(first).strip()
    return None


def _extract_date_from_donut(parsed: dict) -> str | None:
    """Pull the transaction date from Donut output."""
    for key in ("date", "transaction_date", "dt"):
        if key in parsed:
            val = parsed[key]
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list) and val:
                return str(val[0]).strip()
    return None


def _extract_total_from_donut(parsed: dict) -> str | None:
    """
    Pull the total amount from Donut output.
    CORD schema puts totals under "total" → "total_price" or "total_etc".
    """
    total_section = parsed.get("total", parsed.get("total_price"))

    if isinstance(total_section, str):
        # Direct string value
        cleaned = re.sub(r"[^\d.]", "", total_section)
        if cleaned:
            try:
                return f"{float(cleaned):.2f}"
            except ValueError:
                pass

    if isinstance(total_section, list):
        # Walk list looking for the total_price entry
        for item in total_section:
            if isinstance(item, dict):
                for key in ("total_price", "price", "total_etc", "unitprice"):
                    if key in item:
                        cleaned = re.sub(r"[^\d.]", "", str(item[key]))
                        if cleaned:
                            try:
                                return f"{float(cleaned):.2f}"
                            except ValueError:
                                continue

    if isinstance(total_section, dict):
        for key in ("total_price", "price", "total_etc"):
            if key in total_section:
                cleaned = re.sub(r"[^\d.]", "", str(total_section[key]))
                if cleaned:
                    try:
                        return f"{float(cleaned):.2f}"
                    except ValueError:
                        continue

    # Last resort: scan entire parsed dict for anything that looks like a total
    flat = json.dumps(parsed)
    total_match = re.search(r'"total_price"\s*:\s*"([^"]+)"', flat)
    if total_match:
        cleaned = re.sub(r"[^\d.]", "", total_match.group(1))
        if cleaned:
            try:
                return f"{float(cleaned):.2f}"
            except ValueError:
                pass

    return None


def extract_fields_donut(image_path: str) -> tuple:
    """
    Primary extraction entry point using the Donut model.

    Returns:
        (vendor, date, total, raw_parsed_dict)
    """
    available = _ensure_donut_loaded()

    if not available:
        # Fall back to heuristic extractor
        from src.extractors.improved_extraction import extract_fields_ultra
        return extract_fields_ultra(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
        parsed = _run_donut_inference(image)

        vendor = _extract_vendor_from_donut(parsed)
        date = _extract_date_from_donut(parsed)
        total = _extract_total_from_donut(parsed)

        return vendor, date, total, parsed
    except Exception as e:
        print(f"[DonutExtractor] Inference failed on {image_path}: {e}")
        # Fall back to heuristic
        from src.extractors.improved_extraction import extract_fields_ultra
        return extract_fields_ultra(image_path)
