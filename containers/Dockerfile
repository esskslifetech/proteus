# Proteus - Multi-stage Docker build for reproducible containers
# 
# Build: docker build -t proteus:latest -f containers/Dockerfile .
# Run: docker run --rm -it proteus:latest proteus --help

# ========================================================================
# Stage 1: Rust builder
# ========================================================================
FROM rust:1.80-slim AS rust-builder

WORKDIR /proteus

# Install system dependencies for Rust crates
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo workspace
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build Rust workspace (cached dependencies)
RUN cargo build --workspace --release

# ========================================================================
# Stage 2: C++ builder
# ========================================================================
FROM ubuntu:22.04 AS cpp-builder

WORKDIR /proteus

# Install C++ build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    libeigen3-dev \
    libsundials-dev \
    libhighs-dev \
    python3-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy C++ source
COPY cpp ./cpp

# Build C++ components
RUN cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build cpp/build -j$(nproc)

# ========================================================================
# Stage 3: Python runtime
# ========================================================================
FROM python:3.11-slim AS runtime

WORKDIR /proteus

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libsundials6 \
    libhighs1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts from builders
COPY --from=rust-builder /proteus/target/release/libproteus*.so /usr/local/lib/
COPY --from=cpp-builder /proteus/cpp/build/lib/libproteus*.so /usr/local/lib/

# Copy Python package
COPY python ./python

# Install Python package with maturin
RUN pip install --no-cache-dir maturin && \
    cd python && maturin develop --release

# Set up environment
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/proteus/python:$PYTHONPATH
ENV PROTEUS_DATA_DIR=/data

# Create data directory
RUN mkdir -p /data /output

# Default entrypoint
ENTRYPOINT ["proteus"]
CMD ["--help"]

# ========================================================================
# Stage 4: Development image (optional)
# ========================================================================
FROM runtime AS dev

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    hypothesis \
    ruff \
    mypy \
    pre-commit \
    jupyter \
    ipykernel

# Install Rust dev tools
RUN cargo install cargo-watch cargo-nextest cargo-audit cargo-deny

WORKDIR /proteus