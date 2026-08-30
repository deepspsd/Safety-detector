# Multi-stage Dockerfile for CCTV AI Monitoring Platform

# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (OpenCV, FFmpeg, Tesseract, libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Source
COPY backend/ ./backend/
COPY config/ ./config/

# Copy Frontend Build into backend static dist directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

WORKDIR /app/backend

# Create runtime upload and storage directories
RUN mkdir -p uploads/evidence uploads/snapshots debug/crops face_data

EXPOSE 8000

ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
