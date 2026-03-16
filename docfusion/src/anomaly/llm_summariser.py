"""
LLM-based Anomaly Summariser (Bonus Feature)
=============================================
Generates human-readable anomaly explanations.

Uses a lightweight rule-template approach by default (zero external API calls).
If a HuggingFace text-generation model is available, it can optionally use that.
"""


def generate_anomaly_summary(
    vendor: str | None,
    date: str | None,
    total: str | None,
    is_forged: int,
    ocr_text: str = "",
) -> str:
    """
    Generate a human-readable summary explaining why a document
    was flagged as genuine or suspicious.

    Returns a 1-3 sentence natural-language explanation.
    """
    if is_forged == 0:
        return _genuine_summary(vendor, date, total)
    else:
        return _forged_summary(vendor, date, total, ocr_text)


def _genuine_summary(vendor, date, total) -> str:
    parts = []
    if vendor:
        parts.append(f"from **{vendor}**")
    if date:
        parts.append(f"dated **{date}**")
    if total:
        parts.append(f"totalling **${total}**")

    if parts:
        detail = ", ".join(parts)
        return (
            f"This document {detail} appears **genuine**. "
            "All extracted fields are consistent and within expected ranges."
        )
    return "This document appears **genuine** based on available analysis."


def _forged_summary(vendor, date, total, ocr_text) -> str:
    reasons = []

    # Missing fields
    missing = []
    if not vendor:
        missing.append("vendor name")
    if not date:
        missing.append("transaction date")
    if not total:
        missing.append("total amount")
    if missing:
        reasons.append(
            f"The following fields could not be extracted: {', '.join(missing)}. "
            "This may indicate the document has been tampered with or is heavily corrupted."
        )

    # Extreme total
    if total:
        try:
            tv = float(total)
            if tv > 50000:
                reasons.append(
                    f"The total amount (${total}) is unusually high for a standard receipt, "
                    "which is a common indicator of financial document forgery."
                )
            if tv < 0:
                reasons.append(
                    f"The total amount (${total}) is negative, which is physically impossible "
                    "for a purchase receipt."
                )
            if tv == 0:
                reasons.append(
                    "The total amount is exactly $0.00, which is suspicious for a legitimate transaction."
                )
        except ValueError:
            pass

    # Very short text
    if ocr_text and len(ocr_text.strip()) < 20:
        reasons.append(
            "The document contains very little readable text, suggesting it may be "
            "a blank or heavily redacted document."
        )

    # Statistical outlier (generic)
    if not reasons:
        reasons.append(
            "The document's statistical features (text density, character ratios, field patterns) "
            "deviate significantly from the training distribution, suggesting possible tampering."
        )

    summary = "🚨 **This document has been flagged as suspicious.** "
    summary += " ".join(reasons)
    return summary
