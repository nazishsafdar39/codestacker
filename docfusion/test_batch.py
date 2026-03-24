import pytesseract
from PIL import Image
import re
import os

def extract_fields(img_path):
    """Extract vendor, date, total from image"""
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    
    # Vendor: first line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    vendor = lines[0] if lines else None
    
    # Date
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', text)
    date = date_match.group(1) if date_match else None
    
    # Total
    total_match = re.search(r'total[:\s]+\$?(\d+\.?\d*)', text.lower())
    total = total_match.group(1) if total_match else None
    
    return vendor, date, total

# Test on 5 images from SROIE
img_dir = 'SROIE2019/train/img'
image_files = os.listdir(img_dir)[:5]

print(f"Testing on {len(image_files)} images...\n")

success_vendor = 0
success_date = 0
success_total = 0

for img_file in image_files:
    img_path = os.path.join(img_dir, img_file)
    vendor, date, total = extract_fields(img_path)
    
    print(f"{img_file}:")
    print(f"  Vendor: {vendor}")
    print(f"  Date: {date}")
    print(f"  Total: {total}")
    print()
    
    if vendor: success_vendor += 1
    if date: success_date += 1
    if total: success_total += 1

print("="*50)
print(f"Success Rates:")
print(f"  Vendor: {success_vendor}/{len(image_files)} ({success_vendor/len(image_files)*100:.0f}%)")
print(f"  Date: {success_date}/{len(image_files)} ({success_date/len(image_files)*100:.0f}%)")
print(f"  Total: {success_total}/{len(image_files)} ({success_total/len(image_files)*100:.0f}%)")