# Use lightweight official Python runtime base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python package dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, config, and models
COPY config.py .
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create output directory
RUN mkdir -p outputs

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit service
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
