"""
Anomaly Detection module for DocFusion.

Combines:
  1. IsolationForest on extracted features (statistical outliers).
  2. Rule-based logical consistency checks.
  3. Optional Error Level Analysis (ELA) for visual tampering detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200,
            max_samples="auto",
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def extract_features(self, vendor, date, total, text: str) -> np.ndarray:
        """
        Build a fixed-length feature vector from extracted fields and OCR
        text.  The vector is used by both fit() and predict().
        """
        features = []

        # --- Field presence (binary) ---
        features.append(1.0 if vendor else 0.0)
        features.append(1.0 if date else 0.0)
        features.append(1.0 if total else 0.0)

        # --- Total value features ---
        try:
            total_val = float(total) if total else 0.0
        except (ValueError, TypeError):
            total_val = 0.0
        features.append(total_val)
        features.append(np.log1p(abs(total_val)))

        # --- Text statistics ---
        text = text or ""
        non_empty_lines = [l for l in text.split("\n") if l.strip()]
        features.append(float(len(text)))                    # total chars
        features.append(float(len(non_empty_lines)))         # line count

        # --- Suspicious-value flags ---
        features.append(1.0 if total_val > 1000 else 0.0)   # unusually high
        features.append(1.0 if 0 < total_val < 1 else 0.0)  # unusually low

        # --- Additional heuristics ---
        # Ratio of digits to total chars (forged docs sometimes have unusual ratios)
        digit_count = sum(c.isdigit() for c in text)
        features.append(digit_count / max(len(text), 1))

        # Ratio of uppercase to alpha chars
        alpha_chars = [c for c in text if c.isalpha()]
        upper_count = sum(c.isupper() for c in alpha_chars)
        features.append(upper_count / max(len(alpha_chars), 1))

        # Number of unique characters (low diversity can signal tampering)
        features.append(float(len(set(text))))

        # Vendor name length (unusual lengths are suspicious)
        features.append(float(len(vendor)) if vendor else 0.0)

        return np.array(features, dtype=np.float64)

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def fit(self, features_list: list[np.ndarray]):
        """Fit the scaler and IsolationForest on training features."""
        X = np.array(features_list)
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> int:
        """
        Return 1 (forged / anomalous) or 0 (genuine).

        If the model has not been fitted yet, fall back to rule-based
        detection only.
        """
        if not self._is_fitted:
            return self._rule_based_predict(features)

        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        pred = self.iso_forest.predict(X_scaled)[0]

        # IsolationForest returns -1 for outliers, 1 for inliers
        iso_forged = 1 if pred == -1 else 0

        # Combine model prediction with rule-based checks
        rule_forged = self._rule_based_predict(features)

        # If either flags it, mark as forged
        return 1 if (iso_forged or rule_forged) else 0

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _rule_based_predict(features: np.ndarray) -> int:
        """
        Simple deterministic rules that catch obvious forgeries even
        without a trained model.
        """
        vendor_present = features[0]
        date_present = features[1]
        total_present = features[2]
        total_val = features[3]
        line_count = features[6]

        # All three core fields missing → suspicious
        if vendor_present == 0 and date_present == 0 and total_present == 0:
            return 1

        # Extremely high total (> $50 000) with very few lines → suspicious
        if total_val > 50_000 and line_count < 5:
            return 1

        # Negative total → suspicious
        if total_val < 0:
            return 1

        return 0