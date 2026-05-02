# VoxelSim Backend Dockerfile
# Runs Python simulation + Node.js proxy for WebSocket bridge

# Stage 1: Build Rust Python extension
# Use nightly for edition 2024 support
FROM rustlang/rust:nightly-bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    pkg-config \
    libssl-dev \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install maturin (builds Rust → Python extensions)
# Use --break-system-packages since we're in a clean container
RUN pip3 install --no-cache-dir --break-system-packages maturin

# Set working directory
WORKDIR /build

# Copy entire project (needed for workspace dependencies)
COPY . .

# Build the Python extension in release mode
# Maturin builds wheel in voxelsim-py/target/wheels/ by default
WORKDIR /build/voxelsim-py
RUN maturin build --release && \
    ls -la target/wheels/ || echo "No target/wheels, checking alternatives..." && \
    find /build -name "*.whl" 2>/dev/null || true

# Stage 2: Runtime image (smaller, no Rust compiler)
FROM python:3.11-slim-bookworm

# Install Node.js for proxy server
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built Python wheel from builder stage
# Try multiple possible locations
COPY --from=builder /build/voxelsim-py/target/wheels/*.whl /tmp/

# Install the voxelsim Python package
RUN pip3 install --no-cache-dir --break-system-packages /tmp/*.whl && rm /tmp/*.whl

# Install Python dependencies for simulation
RUN pip3 install --no-cache-dir --break-system-packages pynput

# Copy proxy server
COPY proxy/ /app/proxy/
RUN cd /app/proxy && npm install --production

# Copy Python simulation script
COPY python/ /app/python/

# Copy startup script
COPY docker-entrypoint.sh /app/

# Expose WebSocket ports (browser connects to these)
EXPOSE 9080 9081

# Run proxy + Python simulation
CMD ["/bin/bash", "/app/docker-entrypoint.sh"]
