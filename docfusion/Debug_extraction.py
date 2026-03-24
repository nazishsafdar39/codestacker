import pytesseract
from PIL import Image
import re
import os

def extract_fields_debug(img_path):
    """Extract fields AND show what OCR actually sees"""
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    
    # Vendor: first line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    vendor = lines[0] if lines else None
    
    # Date - try multiple patterns
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4})',      # 01/15/2024
        r'(\d{4}-\d{2}-\d{2})',             # 2024-01-15
        r'(\d{2}-\d{2}-\d{4})',             # 15-01-2024
        r'(\d{1,2}\.\d{1,2}\.\d{2,4})',    # 01.15.2024
        r'(\d{1,2}\s+\w+\s+\d{4})',        # 15 Jan 2024
    ]
    date = None
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date = match.group(1)
            break
    
    # Total - try multiple patterns
    total_patterns = [
        r'total[:\s]+\$?(\d+\.?\d*)',       # total: $10.50
        r'TOTAL[:\s]+\$?(\d+\.?\d*)',       # TOTAL: 10.50
        r'Total[:\s]+\$?(\d+\.?\d*)',       # Total: 10.50
        r'amount[:\s]+\$?(\d+\.?\d*)',      # amount: 10.50
        r'sum[:\s]+\$?(\d+\.?\d*)',         # sum: 10.50
        r'grand\s*total[:\s]+\$?(\d+\.?\d*)',  # grand total: 10.50
        r'balance[:\s]+\$?(\d+\.?\d*)',     # balance: 10.50
        r'[\$](\d+\.\d{2})\s*$',            # $10.50 at end of line
    ]
    total = None
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            total = match.group(1)
            break
    
    return vendor, date, total, text

# Test on 5 images from SROIE
img_dir = 'SROIE2019/train/img'
image_files = os.listdir(img_dir)[:5]

print(f"Testing on {len(image_files)} images...\n")
print("="*70)

success_vendor = 0
success_date = 0
success_total = 0

for img_file in image_files:
    img_path = os.path.join(img_dir, img_file)
    vendor, date, total, ocr_text = extract_fields_debug(img_path)
    
    print(f"\n📄 {img_file}")
    print("-" * 70)
    print(f"✅ Vendor: {vendor}")
    print(f"📅 Date: {date}")
    print(f"💰 Total: {total}")
    
    # Show the actual OCR text so you can see what's there
    print("\n📝 RAW OCR TEXT (first 500 chars):")
    print("-" * 70)
    print(ocr_text[:500])
    print("-" * 70)
    
    if vendor: success_vendor += 1
    if date: success_date += 1
    if total: success_total += 1

print("\n" + "="*70)
print(f"📊 SUCCESS RATES:")
print(f"  Vendor: {success_vendor}/{len(image_files)} ({success_vendor/len(image_files)*100:.0f}%)")
print(f"  Date: {success_date}/{len(image_files)} ({success_date/len(image_files)*100:.0f}%)")
print(f"  Total: {success_total}/{len(image_files)} ({success_total/len(image_files)*100:.0f}%)")
print("="*70)