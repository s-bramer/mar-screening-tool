FROM python:3.11-slim

# System libraries required by geopandas / pyproj / fiona
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev gdal-bin \
        libgeos-dev \
        libproj-dev \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as non-root user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python deps first (cached layer unless requirements.txt changes)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY --chown=user . .

USER user

EXPOSE 7860

CMD ["panel", "serve", "dashboard/app.py", \
     "--address", "0.0.0.0", \
     "--port", "7860", \
     "--allow-websocket-origin=*"]
