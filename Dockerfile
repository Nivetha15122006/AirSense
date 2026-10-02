FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, data, and trained models
COPY . .

# Hugging Face Spaces defaults to port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "run.py"]
