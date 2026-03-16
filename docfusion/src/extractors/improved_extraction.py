import pytesseract
from PIL import Image
import re
import os

def extract_vendor_smart(text):
    """
    Smart vendor extraction that filters out common noise
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    if not lines:
        return None
    
    # Noise patterns to skip
    noise_patterns = [
        r'^[\d\s\-\/\.\,]+$',           # Only numbers and punctuation
        r'^[A-Z]\s*\d+',                 # Single letter + numbers (like "A 04025")
        r'^\d{4,}$',                     # Long number sequences
        r'^[\*\#\@\!\%\=\-]+$',          # Special characters only
        r'^receipt',                     # Generic terms
        r'^tax\s*invoice',
        r'^invoice',
        r'^\w{1,2}$',                    # Very short (1-2 chars)
        r'^company\s+reg',               # "Company Reg No."
        r'^gst\s+reg',                   # "GST Reg No."
        r'^lot\s+\d+',                   # "Lot 3..."
        r'^tel\s*:',                     # "Tel : 03-..."
        r'^no\.\s*\d+',                  # "No.50 , JALAN..."
        r'^off\s+jalan',                 # "OFF JALAN..."
        r'^\d+\s+jalan',                 # "19, JALAN..."
    ]
    
    # Company indicators (bonus points for these)
    company_indicators = [
        'sdn bhd', 'sdn. bhd.', 'sdn.bhd.',
        'pte ltd', 'pte. ltd.', 'pvt ltd',
        'limited', 'ltd', 'llc', 'inc', 'corp',
        'corporation', 'company', 'co.',
        'enterprise', 'restaurant', 'cafe', 'restoran',
        'store', 'mart', 'shop', 'kedai',
        'bakeries', 'bakery', 'hardware', 'handicraft',
        'tailoring', 'milk',
    ]
    
    candidates = []
    
    # Check first 10 lines (vendor usually near top)
    for i, line in enumerate(lines[:10]):
        # Skip if too short
        if len(line) < 5:
            continue
        
        # Skip if matches noise patterns
        is_noise = False
        for pattern in noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_noise = True
                break
        
        if is_noise:
            continue
        
        # Skip if mostly numbers (>60% digits)
        digit_ratio = sum(c.isdigit() for c in line) / len(line)
        if digit_ratio > 0.6:
            continue
        
        # Skip if mostly special characters
        special_ratio = sum(not c.isalnum() and not c.isspace() for c in line) / len(line)
        if special_ratio > 0.5:
            continue
        
        # Calculate score for this line
        score = 0
        
        # Bonus: earlier lines are more likely to be vendor
        score += (10 - i) * 2
        
        # Bonus: longer lines (but not too long)
        if 10 <= len(line) <= 60:
            score += 10
        
        # Bonus: has company indicators
        line_lower = line.lower()
        for indicator in company_indicators:
            if indicator in line_lower:
                score += 50  # Big bonus!
                break
        
        # Bonus: has capital letters (company names often capitalized)
        if any(c.isupper() for c in line):
            score += 5
        
        # Bonus: has alphabetic content
        alpha_ratio = sum(c.isalpha() for c in line) / len(line)
        score += alpha_ratio * 10
        
        candidates.append((score, line, i))
    
    if not candidates:
        # Fallback: return first non-empty line
        return lines[0] if lines else None
    
    # Sort by score and return best candidate
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_vendor = candidates[0][1]
    
    # Clean up vendor name
    best_vendor = re.sub(r'^[\W_]+|[\W_]+$', '', best_vendor)
    
    return best_vendor


def extract_date_comprehensive(text):
    """
    ULTRA-COMPREHENSIVE date extraction with all possible patterns
    """
    # Try many different date formats
    date_patterns = [
        # Standard formats with slashes
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',      # 09/04/2018
        r'\b(\d{1,2}/\d{1,2}/\d{2})\b',      # 04/05/18
        
        # Dash formats
        r'\b(\d{4}-\d{2}-\d{2})\b',           # 2018-04-09
        r'\b(\d{2}-\d{2}-\d{4})\b',           # 09-04-2018
        r'\b(\d{2}-\d{2}-\d{2})\b',           # 09-04-18
        r'\b(\d{1,2}-\d{1,2}-\d{4})\b',      # 9-4-2018
        
        # Dot formats
        r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b',    # 09.04.2018
        r'\b(\d{1,2}\.\d{1,2}\.\d{2})\b',    # 09.04.18
        
        # Space formats
        r'\b(\d{1,2}\s+\d{1,2}\s+\d{4})\b',  # 09 04 2018
        
        # Text month formats
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
        
        # With labels (case insensitive)
        r'date[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})',
        r'date[:\s]+(\d{2}-\d{2}-\d{2,4})',
        r'date[:\s]+(\d{2}\.\d{2}\.\d{2,4})',
        
        # DD: format
        r'DD[:\s]+(\d{1,2}/\d{1,2}/\d{4})',
        r'DD[:\s]+(\d{1,2}-\d{1,2}-\d{4})',
        
        # Printed Date format
        r'printed\s+date[:\s]+(\d{2}-\d{2}-\d{2})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Validate it looks like a date (has reasonable values)
            # Skip if it looks like a price (e.g., "12.90")
            if '.' in date_str and date_str.count('.') == 1:
                parts = date_str.split('.')
                if len(parts) == 2 and all(len(p) <= 2 for p in parts):
                    continue  # Likely a price like 12.90
            return date_str
    
    return None


def extract_total_comprehensive(text):
    """
    ULTRA-COMPREHENSIVE total extraction - catches EVERYTHING
    """
    # Try comprehensive total patterns
    total_patterns = [
        # Standard "TOTAL:" patterns
        (r'TOTAL\s*:\s*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'TOTAL: X.XX'),
        (r'total\s*:\s*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'total: X.XX'),
        (r'Total\s*:\s*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'Total: X.XX'),
        
        # TOTAL without colon
        (r'TOTAL\s+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'TOTAL X.XX'),
        (r'total\s+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'total X.XX'),
        
        # With different keywords
        (r'grand\s*total[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'grand total'),
        (r'net\s*total[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'net total'),
        (r'final\s*total[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'final total'),
        
        # "Total Payable:" pattern (from Image 4)
        (r'total\s+payable[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'total payable'),
        
        # "Total Amt Rounded" pattern (from Image 3)
        (r'total\s+amt\s+rounded[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'total amt rounded'),
        
        # "Net Total Rounded (MYR)" pattern (from Image 2)
        (r'net\s+total\s+rounded\s*\(MYR\)\s*:\s*(\d+[.,]\d{2})', 'net total rounded MYR'),
        (r'net\s+total\s+rounded\s*\([^)]+\)\s*:\s*(\d+[.,]\d{2})', 'net total rounded'),
        
        # Amount patterns
        (r'amount[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'amount'),
        (r'total\s+amount[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'total amount'),
        
        # Sum/Balance patterns
        (r'sum[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'sum'),
        (r'balance[:\s]+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'balance'),
        
        # Cash patterns (sometimes total is listed as "Cash")
        (r'cash\s*:\s*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'cash'),
        (r'cash\s+(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'cash no colon'),
        
        # Payment patterns
        (r'payment\s*:\s*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'payment'),
        
        # Just currency symbol + number
        (r'(?:RM|rm)\s*(\d+\.\d{2})\s*$', 'RM XX.XX at end'),
        (r'\$\s*(\d+\.\d{2})\s*$', '$ XX.XX at end'),
        
        # Line starting with TOTAL (more flexible spacing)
        (r'^TOTAL[:\s]*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'TOTAL at line start'),
        
        # Multiline pattern: TOTAL on one line, number on next
        (r'TOTAL\s*[:\s]*\n\s*(?:RM|rm|\$)?\s*(\d+[.,]\d{2})', 'TOTAL multiline'),
    ]
    
    total_candidates = []
    
    # Try each pattern
    for pattern, pattern_name in total_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            amount_str = match.group(1)
            # Clean up: replace comma with period
            amount_str = amount_str.replace(',', '.')
            try:
                amount = float(amount_str)
                # Reasonable range for receipts (0.01 to 1,000,000)
                if 0.01 <= amount <= 1000000:
                    # Priority: patterns with "total" keyword get higher priority
                    priority = 1
                    if 'total' in pattern_name.lower():
                        priority = 0  # Higher priority (sort will put this first)
                    
                    total_candidates.append((priority, amount, pattern_name))
            except:
                continue
    
    if not total_candidates:
        return None
    
    # Sort by priority (0 first), then by amount (largest first)
    total_candidates.sort(key=lambda x: (x[0], -x[1]))
    
    # Return the highest priority, largest amount
    best_total = total_candidates[0][1]
    
    return f"{best_total:.2f}"


def extract_fields_ultra(img_path):
    """
    ULTRA-COMPREHENSIVE extraction with all patterns
    """
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    
    # Extract fields using comprehensive functions
    vendor = extract_vendor_smart(text)
    date = extract_date_comprehensive(text)
    total = extract_total_comprehensive(text)
    
    return vendor, date, total, text


# ==== TESTING ====
if __name__ == "__main__":
    img_dir = 'SROIE2019/train/img'
    
    if not os.path.exists(img_dir):
        print(f"❌ Directory not found: {img_dir}")
        print("Please update the img_dir path to your dataset location.")
        exit(1)
    
    image_files = os.listdir(img_dir)[:5]  # Test on 20 images
    
    print(f"Testing ULTRA-COMPREHENSIVE extraction on {len(image_files)} images...\n")
    print("="*70)
    
    success_vendor = 0
    success_date = 0
    success_total = 0
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        
        try:
            vendor, date, total, ocr_text = extract_fields_ultra(img_path)
            
            print(f"\n📄 {img_file}")
            print("-" * 70)
            print(f"✅ Vendor: {vendor}")
            print(f"📅 Date: {date}")
            print(f"💰 Total: ${total}" if total else "💰 Total: None")
            
            # Show first few lines of OCR for debugging
            lines = [l.strip() for l in ocr_text.split('\n') if l.strip()][:5]
            print(f"\n📝 First 5 OCR lines:")
            for i, line in enumerate(lines, 1):
                marker = " ← VENDOR" if vendor and vendor in line else ""
                print(f"   {i}. {line[:60]}{marker}")
            
            if vendor: success_vendor += 1
            if date: success_date += 1
            if total: success_total += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {img_file}: {e}")
    
    print("\n" + "="*70)
    print(f"📊 ULTRA-COMPREHENSIVE EXTRACTION SUCCESS RATES:")
    print(f"  Vendor: {success_vendor}/{len(image_files)} ({success_vendor/len(image_files)*100:.0f}%)")
    print(f"  Date: {success_date}/{len(image_files)} ({success_date/len(image_files)*100:.0f}%)")
    print(f"  Total: {success_total}/{len(image_files)} ({success_total/len(image_files)*100:.0f}%)")
    print("="*70)
    
    print("\n💡 EXPECTED IMPROVEMENTS:")
    print(f"  ✅ Now catches: 'Total Payable', 'Total Amt Rounded', 'Net Total Rounded (MYR)'")
    print(f"  ✅ Now catches: dates in format 'DD-MM-YY', 'DD-MM-YYYY'")
    print(f"  ✅ Better vendor detection with more noise filters")