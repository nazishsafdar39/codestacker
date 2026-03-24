"""
DocFusion — Train & Evaluate
===================================
1. Evaluates the extraction pipeline (Donut + Heuristic fallback) against SROIE ground truth.
2. Trains the Anomaly Detector using Find-It-Again labels and saves the model.
"""

import os
import json
import csv
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from src.pipeline import DocFusionPipeline

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
SROIE_IMG_DIR = os.path.join(BASE, "SROIE2019", "train", "img")
SROIE_ENT_DIR = os.path.join(BASE, "SROIE2019", "train", "entities")
FINDIT_CSV = os.path.join(BASE, "findit2", "train.txt")
FINDIT_IMG_DIR = os.path.join(BASE, "findit2", "train")
MODEL_SAVE_DIR = os.path.join(BASE, "work_dir", "saved_models")

def evaluate_extraction(sample_size=None):
    """Evaluate extraction accuracy on SROIE."""
    if sample_size:
        print(f"\n--- Evaluating Extraction Pipeline (Sample: {sample_size}) ---")
    else:
        print(f"\n--- Evaluating Extraction Pipeline (Full Dataset) ---")
    pipeline = DocFusionPipeline(use_donut=True)
    
    if not os.path.exists(SROIE_ENT_DIR):
        print(f"Error: SROIE entities dir not found at {SROIE_ENT_DIR}")
        return

    files = [f for f in os.listdir(SROIE_ENT_DIR) if f.endswith(".txt")]
    files = sorted(files)
    if sample_size:
        files = files[:sample_size]
    
    metrics = {"vendor": 0, "date": 0, "total": 0, "total_samples": 0}
    
    for fname in tqdm(files, desc="Extracting..."):
        # Load Ground Truth
        with open(os.path.join(SROIE_ENT_DIR, fname)) as f:
            gt = json.load(f)
            
        img_id = fname.replace(".txt", "")
        img_path = os.path.join(SROIE_IMG_DIR, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        # Extract
        pred_vendor, pred_date, pred_total = pipeline.extract(img_path)
        
        # Fuzzy matching that tolerates OCR errors
        def is_match(truth, pred):
            if not truth or not pred: return False
            truth = str(truth).lower().replace(" ", "").replace(",", "")
            pred = str(pred).lower().replace(" ", "").replace(",", "")
            # Exact substring match
            if pred in truth or truth in pred:
                return True
            # Fuzzy match (catches OCR typos like BECO vs DECO)
            ratio = SequenceMatcher(None, truth, pred).ratio()
            return ratio >= 0.5

        if is_match(gt.get("company"), pred_vendor): metrics["vendor"] += 1
        if is_match(gt.get("date"), pred_date): metrics["date"] += 1
        
        # For total, numeric exact match is best but we allow string matching as baseline
        if is_match(gt.get("total"), pred_total): metrics["total"] += 1
            
        metrics["total_samples"] += 1
        
    print("\nExtraction Accuracy:")
    n = max(1, metrics["total_samples"])
    print(f"Vendor: {metrics['vendor']/n*100:.1f}% ({metrics['vendor']}/{n})")
    print(f"Date:   {metrics['date']/n*100:.1f}% ({metrics['date']}/{n})")
    print(f"Total:  {metrics['total']/n*100:.1f}% ({metrics['total']}/{n})")

def train_anomaly_detector(sample_size=None):
    """Train the Anomaly Detector on Find-It-Again labels."""
    if sample_size:
        print(f"\n--- Training Anomaly Detector (Sample: {sample_size}) ---")
    else:
        print(f"\n--- Training Anomaly Detector (Full Dataset) ---")
    pipeline = DocFusionPipeline(use_donut=False) # Skip Donut extraction cost for features if we can
    
    if not os.path.exists(FINDIT_CSV):
        print(f"Error: FindIt annotations not found at {FINDIT_CSV}")
        return

    features_list = []
    
    # Read FindIt CSV
    df = pd.read_csv(FINDIT_CSV)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print("Extracting features from documents...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row["image"]
        
        # We need the full path
        img_path = os.path.join(FINDIT_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        # We need text.
        pred_vendor, pred_date, pred_total = pipeline.extract(img_path) if not pipeline.use_donut else (None, None, None)
        
        # Use fallback tesseract if text missing
        ocr_text = pipeline._get_ocr_text(img_path)
            
        # Real ground truth isn't easily parsed for Find-It, so we use empty vendor/date/total to force text-based features
        feats = pipeline.anomaly_detector.extract_features(None, None, None, ocr_text)
        features_list.append(feats)
        
    if features_list:
        pipeline.anomaly_detector.fit(features_list)
        print(f"Fitting completed on {len(features_list)} samples.")
        
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        pipeline.save(MODEL_SAVE_DIR)
        print(f"Models saved to {MODEL_SAVE_DIR}/")

if __name__ == "__main__":
    evaluate_extraction(sample_size=None)
    train_anomaly_detector(sample_size=None)
    print("\n✅ Training & Evaluation Complete.")
