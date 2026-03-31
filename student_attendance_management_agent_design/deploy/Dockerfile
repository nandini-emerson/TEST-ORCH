

FROM python:3.11-slim

# System dependencies for building and runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Default environment variables (can be overridden at runtime)
ENV PORT=8000

# Startup command
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

