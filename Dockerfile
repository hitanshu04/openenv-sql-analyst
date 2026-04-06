# OpenEnv SQL Analyst Environment
# Base: python:3.10-slim for minimal memory footprint (<8GB RAM limit)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies WITH UV added for the hotfix
RUN pip install --no-cache-dir -r requirements.txt uv

# Copy application code
COPY . .

# Expose the OpenEnv serving port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Replaced deprecated 'openenv serve' with the command the runtime error requested
CMD ["uv", "run", "--project", ".", "server", "--port", "7860"]