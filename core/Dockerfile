FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch CPU-ONLY First (Critical optimization)
# This prevents downloading 2GB+ of CUDA libraries.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install Requirements (ignoring lines that might force GPU)
COPY core/ai_service/requirements.txt .
# Filter out torch/cuda lines from requirements to avoid conflict
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/cuda/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY core/ai_service/ .
COPY core/ai_service/demo_logic.py .

# 5. Expose Port
EXPOSE 8000

# 6. Run
CMD ["python", "main.py"]
