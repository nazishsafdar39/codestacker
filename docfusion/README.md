# DocFusion: Operation Intelligent Documents

An end-to-end intelligent document processing pipeline that extracts structured information from scanned receipts/invoices and detects forged documents.

---

## Architecture Overview

```
                    ┌─────────────┐
                    │  Image In   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Donut VDU │  (OCR-free deep learning)
                    │   Extractor │
                    └──────┬──────┘
                           │ fallback
                    ┌──────▼──────┐
                    │  Tesseract  │  (heuristic regex extraction)
                    │  + Regex    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
       ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
       │ vendor  │   │  date   │   │  total  │
       └────┬────┘   └────┬────┘   └────┬────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────▼──────┐
                    │  Anomaly    │  IsolationForest + Rules
                    │  Detector   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ predictions │  JSONL output
                    │   .jsonl    │
                    └─────────────┘
```

## Project Structure

```
docfusion/
├── solution.py                      # 🏁 Harness entry point (DocFusionSolution)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Containerised deployment
├── check_submission.py              # Local validation checker
│
├── src/
│   ├── pipeline.py                  # Orchestrates extraction + anomaly detection
│   ├── extractors/
│   │   ├── donut_extractor.py       # Donut VDU model (primary)
│   │   └── improved_extraction.py   # Tesseract + regex (fallback)
│   └── anomaly/
│       └── anomaly_detector.py      # IsolationForest + rule-based detection
│
├── ui/
│   └── app.py                       # Streamlit dashboard
│
├── notebooks/
│   └── 01_eda.py                    # Level 1 EDA (run or convert to .ipynb)
│
├── SROIE2019/                       # Dataset A
├── cord-v2-data/                    # Dataset C
└── findit2/                         # Dataset B (anomaly labels)
```

## Quick Start

### 1. Install Dependencies
```bash
cd docfusion
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets
```

### 2. Run Local Validation
```bash
python check_submission.py --submission .
```
Expected output:
```
✅ DocFusionSolution instantiated
✅ Train completed.
✅ Predict completed.
✅ Output format valid.
```

### 3. Run EDA
```bash
python notebooks/01_eda.py
# Or convert to Jupyter: pip install jupytext && jupytext --to notebook notebooks/01_eda.py
```

### 4. Launch Web UI
```bash
streamlit run ui/app.py
```
Upload a receipt image → see extracted fields + anomaly status.

### 5. Docker
```bash
docker build -t docfusion .
docker run -p 8501:8501 docfusion
```

## How It Works

### Information Extraction (Level 2)
- **Primary:** [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) — an OCR-free Document Understanding Transformer fine-tuned on CORD-v2. Takes a raw image and directly outputs structured vendor/date/total.
- **Fallback:** Tesseract OCR + comprehensive regex patterns for vendor, date, and total extraction with noise filtering and scoring.

### Anomaly Detection (Level 3)
Combines two approaches:
1. **IsolationForest** trained on 13 extracted features (field presence, total magnitude, text statistics, character-level ratios).
2. **Rule-based checks** — all fields missing, extreme totals, negative values, impossibly short documents.

### `solution.py` Interface (Level 4)
```python
class DocFusionSolution:
    def train(self, train_dir, work_dir) -> str:
        # Reads train.jsonl, fits anomaly detector, saves to work_dir
        
    def predict(self, model_dir, data_dir, out_path) -> None:
        # Loads model, runs extraction + anomaly on each test image,
        # writes predictions.jsonl
```

**Output format:**
```json
{"id": "t001", "vendor": "ACME Corp", "date": "2024-01-01", "total": "10.00", "is_forged": 0}
```

## Approach & Design Decisions

| Decision | Rationale |
|---|---|
| Donut over LayoutLM | OCR-free = no Tesseract dependency chain, simpler pipeline, fewer failure modes |
| IsolationForest over supervised | Find-It-Again has limited labeled forged samples; unsupervised anomaly detection generalises better |
| Lazy model loading | Donut (~750MB) only loaded on first inference call — fast startup, low memory when not needed |
| Dual extraction strategy | Donut for quality, Tesseract+regex as fallback ensures we never return empty for simple receipts |

## Technologies Used

- **Python 3.12+**
- **PyTorch** — deep learning backend
- **HuggingFace Transformers** — Donut model
- **Tesseract OCR** — fallback text extraction
- **scikit-learn** — IsolationForest, StandardScaler
- **Streamlit** — web dashboard
- **Docker** — containerisation
