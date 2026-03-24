FROM python:3.13-slim

# Install system dependencies (Tesseract OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    scikit-learn Pillow pytesseract pandas numpy \
    streamlit joblib tqdm

# Copy source code (datasets excluded via .dockerignore)
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Health check for Render
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
