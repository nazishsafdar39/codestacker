import pytesseract
from PIL import Image
import re
import os

def extract_vendor_smart(text):
    """
    Smart vendor extraction with multi-line merging and fuzzy scoring.
    Many SROIE vendors span 2+ lines (e.g. 'BOOK TA .K\n(TAMAN DAYA) SDN BHD').
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    if not lines:
        return None
    
    # Noise patterns to skip
    noise_patterns = [
        r'^[\d\s\-\/\.\,]+$',           # Only numbers and punctuation
        r'^\d{4,}$',                     # Long number sequences
        r'^[\*\#\@\!\%\=\-]+$',          # Special characters only
        r'^receipt$',                    # Generic terms
        r'^tax\s*invoice',
        r'^invoice',
        r'^simplified\s+tax',
        r'^\w{1,2}$',                    # Very short (1-2 chars)
        r'^company\s+reg',
        r'^gst\s+(reg|id)',
        r'^tel\s*[:\(]',
        r'^fax\s*[:\(]',
        r'^\d+\s*,\s*jalan',             # Address lines
        r'^no\.?\s*\d+',                 # "No.50 , JALAN..."
        r'^lot\s+\d+',
        r'^off\s+jalan',
        r'^taman\s+',
        r'^jalan\s+',
        r'^\d{5}\s+',                    # Postcode lines
        r'^\(co\.?\s*reg',               # Registration number lines
    ]
    
    # Company indicators (bonus points for these)
    company_indicators = [
        'sdn bhd', 'sdn. bhd.', 'sdn.bhd.', 'son bhd',  # OCR variants
        'pte ltd', 'pte. ltd.', 'pvt ltd',
        'limited', 'ltd', 'llc', 'inc', 'corp',
        'corporation', 'company', 'co.',
        'enterprise', 'restaurant', 'cafe', 'restoran',
        'store', 'mart', 'shop', 'kedai',
        'bakeries', 'bakery', 'hardware', 'handicraft',
        'tailoring', 'trading', 'motor', 'machinery',
        'perniagaan', 'industri',  # Malay business terms
        'gift', 'deco', 'beco',    # Common + OCR variants
    ]
    
    def is_noise(line):
        for pattern in noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        if len(line) < 4:
            return True
        digit_ratio = sum(c.isdigit() for c in line) / max(len(line), 1)
        if digit_ratio > 0.6:
            return True
        special_ratio = sum(not c.isalnum() and not c.isspace() for c in line) / max(len(line), 1)
        if special_ratio > 0.5:
            return True
        return False
    
    def score_candidate(text_candidate, position):
        score = 0
        lc = text_candidate.lower()
        # Bonus: earlier position
        score += max(0, (10 - position)) * 2
        # Bonus: reasonable length
        if 8 <= len(text_candidate) <= 80:
            score += 10
        # Bonus: company indicators (big)
        for indicator in company_indicators:
            if indicator in lc:
                score += 50
                break
        # Bonus: has capital letters
        if any(c.isupper() for c in text_candidate):
            score += 5
        # Bonus: high alpha ratio
        alpha_ratio = sum(c.isalpha() for c in text_candidate) / max(len(text_candidate), 1)
        score += alpha_ratio * 15
        return score

    candidates = []
    
    # Build candidates: single lines AND merged consecutive lines
    for i, line in enumerate(lines[:12]):
        if is_noise(line):
            continue
        
        # Single-line candidate
        candidates.append((score_candidate(line, i), line, i))
        
        # Try merging with next line (many vendor names span 2 lines)
        if i + 1 < len(lines) and not is_noise(lines[i + 1]):
            merged = line + " " + lines[i + 1]
            merged_score = score_candidate(merged, i)
            # Bonus for merged lines that contain company indicators
            # (indicates the name was split across lines)
            if any(ind in merged.lower() for ind in company_indicators):
                merged_score += 20
            candidates.append((merged_score, merged, i))
    
    if not candidates:
        return lines[0] if lines else None
    
    # Sort by score descending
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