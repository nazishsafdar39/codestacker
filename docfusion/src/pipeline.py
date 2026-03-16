import os
import json
import joblib
from functools import lru_cache

from src.extractors.donut_extractor import extract_fields_donut
from src.extractors.improved_extraction import extract_fields_ultra
from src.anomaly.anomaly_detector import AnomalyDetector


class DocFusionPipeline:
    """
    End-to-end pipeline that:
      1. Extracts structured fields (vendor, date, total) from receipt images
         using a Donut VDU model (with Tesseract fallback).
      2. Detects anomalous / forged documents using an IsolationForest
         trained on extracted feature vectors.

    Performance notes:
      - OCR text is cached per image path to avoid redundant Tesseract calls.
      - Donut model is lazy-loaded (see donut_extractor.py).
    """

    def __init__(self, use_donut: bool = True):
        self.use_donut = use_donut
        self.anomaly_detector = AnomalyDetector()
        self._ocr_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_dir: str, model_save_path: str):
        train_info_path = os.path.join(train_dir, "train.jsonl")
        if not os.path.exists(train_info_path):
            print(f"[Pipeline] Skipping training – {train_info_path} not found.")
            self.save(model_save_path)
            return

        features_list = []
        with open(train_info_path, "r") as f:
            for line in f:
                record = json.loads(line)
                doc_id = record["id"]
                vendor = record.get("vendor")
                date = record.get("date")
                total = record.get("total")

                image_path = self._resolve_image_path(
                    os.path.join(train_dir, "images"), doc_id
                )
                ocr_text = self._get_ocr_text(image_path)

                features = self.anomaly_detector.extract_features(
                    vendor, date, total, ocr_text
                )
                features_list.append(features)

        if features_list:
            self.anomaly_detector.fit(features_list)
            print(f"[Pipeline] Anomaly detector fitted on {len(features_list)} samples.")

        self.save(model_save_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_save_path: str):
        os.makedirs(model_save_path, exist_ok=True)
        try:
            joblib.dump(
                self.anomaly_detector.scaler,
                os.path.join(model_save_path, "scaler.pkl"),
            )
            joblib.dump(
                self.anomaly_detector.iso_forest,
                os.path.join(model_save_path, "iso_forest.pkl"),
            )
        except Exception as e:
            print(f"[Pipeline] Warning – could not save model state: {e}")

    def load(self, model_dir: str):
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        iso_forest_path = os.path.join(model_dir, "iso_forest.pkl")
        if os.path.exists(scaler_path) and os.path.exists(iso_forest_path):
            self.anomaly_detector.scaler = joblib.load(scaler_path)
            self.anomaly_detector.iso_forest = joblib.load(iso_forest_path)
            self.anomaly_detector._is_fitted = True
            print("[Pipeline] Model artefacts loaded.")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract(self, image_path: str):
        """
        Extract (vendor, date, total).
        Strategy: Donut first → Tesseract heuristic fallback.
        """
        if self.use_donut:
            try:
                vendor, date, total, _extra = extract_fields_donut(image_path)
                if vendor or date or total:
                    return vendor, date, total
            except Exception as e:
                print(f"[Pipeline] Donut extraction failed: {e}")

        try:
            vendor, date, total, _text = extract_fields_ultra(image_path)
            # Cache the OCR text from this call
            self._ocr_cache[image_path] = _text
            return vendor, date, total
        except Exception as e:
            print(f"[Pipeline] Heuristic extraction failed: {e}")
            return None, None, None

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------
    def predict_anomaly(self, vendor, date, total, image_path: str) -> int:
        """Return 1 (forged) or 0 (genuine)."""
        try:
            # Use cached OCR text if available (avoids running Tesseract twice)
            ocr_text = self._ocr_cache.get(image_path) or self._get_ocr_text(image_path)

            features = self.anomaly_detector.extract_features(
                vendor, date, total, ocr_text
            )

            if not vendor and not date and not total:
                return 1

            return self.anomaly_detector.predict(features)
        except Exception as e:
            print(f"[Pipeline] Anomaly prediction error on {image_path}: {e}")
            return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_image_path(images_dir: str, doc_id: str) -> str:
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = os.path.join(images_dir, f"{doc_id}{ext}")
            if os.path.exists(candidate):
                return candidate
        return os.path.join(images_dir, f"{doc_id}.jpg")

    def _get_ocr_text(self, image_path: str) -> str:
        """Run Tesseract OCR with caching."""
        if image_path in self._ocr_cache:
            return self._ocr_cache[image_path]
        if not os.path.exists(image_path):
            return ""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            self._ocr_cache[image_path] = text
            return text
        except Exception:
            return ""

    def clear_cache(self):
        """Clear the OCR text cache to free memory."""
        self._ocr_cache.clear()
