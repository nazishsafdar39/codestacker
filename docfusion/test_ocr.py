import pytesseract
from PIL import Image
import re

# If needed:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_total(text):
    """Find total amount"""
    patterns = [
        r'total[:\s]+\$?(\d+\.?\d*)',
        r'amount[:\s]+\$?(\d+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1)
    return None

def extract_date(text):
    """Find date"""
    match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', text)
    return match.group(1) if match else None

def extract_vendor(text):
    """Get first line as vendor"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[0] if lines else None

# Load and OCR
img = Image.open('test_receipt.jpg')
text = pytesseract.image_to_string(img)

# Extract fields
vendor = extract_vendor(text)
date = extract_date(text)
total = extract_total(text)

# Display
print("="*50)
print("EXTRACTED FIELDS:")
print("="*50)
print(f"Vendor: {vendor}")
print(f"Date: {date}")
print(f"Total: ${total}")
print("="*50)

# Show raw OCR for debugging
print("\nRAW OCR TEXT:")
print(text)