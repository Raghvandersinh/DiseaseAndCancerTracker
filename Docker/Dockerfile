# syntax=docker/dockerfile:1

# --- STAGE 1: Build ---
FROM python:3.13-slim AS builder 

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /DiseaseAndCancerTracker

COPY requirements.txt . 
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Production ---
FROM python:3.13-slim

# Create a non-root user for security
RUN useradd -m -r appuser && \
    mkdir /DiseaseAndCancerTracker && \
    chown -R appuser /DiseaseAndCancerTracker

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set the working directory
WORKDIR /DiseaseAndCancerTracker

# Copy your application code
COPY --chown=appuser:appuser . .

# Switch to the non-root user
USER appuser

# Expose the port
EXPOSE 8000

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "Frontend.wsgi:application"]