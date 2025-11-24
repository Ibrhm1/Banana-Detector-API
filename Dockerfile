# ============================================================
# BANANA RIPENESS CLASSIFICATION API
# Optimized Dockerfile (stable, small, fast)
# ============================================================

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# ------------------------------------------------------------
# Install system dependencies for OpenCV & scientific packages
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install NUMPY FIRST → to avoid sklearn / opencv install issues
# ------------------------------------------------------------
RUN pip install --no-cache-dir numpy==1.26.4

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure uploads folder exists
RUN mkdir -p uploads

# Expose API port
EXPOSE 5000

# Healthcheck to ensure container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/')" || exit 1

# ------------------------------------------------------------
# Run with Gunicorn (Production mode)
# ------------------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "banana_api:app"]
