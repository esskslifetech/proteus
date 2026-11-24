//! V7: Reproducibility — content-addressed inputs, pinned versions, manifests

use crate::error::ValidationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// Use byte array for serialization instead of blake3::Hash
type HashBytes = [u8; 32];

use blake3;
use hex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub timestamp_ns: i64,
    pub proteus_version: String,
    pub git_commit: Option<String>,
    pub rust_version: String,
    pub python_version: String,
    pub dependencies: DependencyManifest,
    pub inputs: InputManifest,
    pub outputs: OutputManifest,
    pub environment: EnvironmentManifest,
    pub validation: ValidationManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyManifest {
    pub rust_crates: HashMap<String, String>,
    pub python_packages: HashMap<String, String>,
    pub system_libraries: HashMap<String, String>,
    pub container_image: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputManifest {
    pub reference_genome: ContentRef,
    pub variant_files: Vec<ContentRef>,
    pub annotation_databases: Vec<ContentRef>,
    pub model_weights: Vec<ContentRef>,
    pub config: ContentRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRef {
    pub name: String,
    pub path: PathBuf,
    pub hash: HashBytes,
    pub size: u64,
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputManifest {
    pub artifacts: Vec<ContentRef>,
    pub reports: Vec<ContentRef>,
    pub logs: Vec<ContentRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentManifest {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub memory_gb: f64,
    pub gpu: Option<String>,
    pub container_runtime: Option<String>,
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationManifest {
    pub v1_schema: bool,
    pub v2_semantic: bool,
    pub v3_physical: bool,
    pub v4_triangulation: bool,
    pub v5_determinism: bool,
    pub v6_scientific: bool,
    pub v7_reproducibility: bool,
    pub all_passed: bool,
}

/// Content-addressed store for reproducible artifact management
pub struct ContentStore {
    root: PathBuf,
}

impl ContentStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn put(&self, name: &str, data: &[u8]) -> Result<ContentRef, ValidationError> {
        let hash = blake3::hash(data);
        let hash_bytes: HashBytes = *hash.as_bytes();
        let hash_str = hex::encode(hash_bytes);
        let dir = self.root.join(&hash_str[0..2]).join(&hash_str[2..4]);
        std::fs::create_dir_all(&dir).map_err(ValidationError::Io)?;
        let path = dir.join(&hash_str[4..]);
        std::fs::write(&path, data).map_err(ValidationError::Io)?;

        Ok(ContentRef {
            name: name.to_string(),
            path: path.strip_prefix(&self.root).unwrap().to_path_buf(),
            hash: hash_bytes,
            size: data.len() as u64,
            uri: None,
        })
    }

    pub fn get(&self, hash: &HashBytes) -> Result<Vec<u8>, ValidationError> {
        let hash_str = hex::encode(hash);
        let path = self
            .root
            .join(&hash_str[0..2])
            .join(&hash_str[2..4])
            .join(&hash_str[4..]);
        std::fs::read(path).map_err(ValidationError::Io)
    }

    pub fn verify(&self, hash: &HashBytes) -> Result<bool, ValidationError> {
        let hash_str = hex::encode(hash);
        let path = self
            .root
            .join(&hash_str[0..2])
            .join(&hash_str[2..4])
            .join(&hash_str[4..]);
        if !path.exists() {
            return Ok(false);
        }
        let data = std::fs::read(&path).map_err(ValidationError::Io)?;
        Ok(blake3::hash(&data).as_bytes() == hash)
    }
}

/// Nix flake output for hermetic builds
pub fn generate_nix_flake() -> String {
    r#"
{
  description = "Proteus - Human Genome Simulation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        rust-toolchain = pkgs.rust-bin.stable.latest.default;

        proteus = pkgs.mkShell {
          name = "proteus-dev";
          buildInputs = with pkgs; [
            rust-toolchain
            python311
            python311Packages.maturin
            python311Packages.pip
            cmake
            eigen
            sundials
            highs
            git
          ];
          RUST_BACKTRACE = 1;
        };
      in
      {
        devShells.default = proteus;
        packages.default = pkgs.writeText "proteus-version.txt" (builtins.substring 0 7 (builtins.readFile ./Cargo.lock));
      }
    );
}
"#.to_string()
}

/// Dockerfile for reproducible containers
pub fn generate_dockerfile() -> String {
    r#"
# Build stage
FROM rust:1.80-slim as builder
WORKDIR /proteus
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY cpp ./cpp
COPY python ./python
COPY schemas ./schemas

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    libeigen3-dev \
    libsundials-dev \
    libhighs-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build Rust workspace
RUN cargo build --release --workspace

# Build C++ components
WORKDIR /proteus/cpp/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Python stage
FROM python:3.11-slim
WORKDIR /proteus

# Copy built artifacts
COPY --from=builder /proteus/target/release/libproteus*.so /usr/local/lib/
COPY --from=builder /proteus/cpp/build/lib/libproteus*.so /usr/local/lib/
COPY --from=builder /proteus/python ./python

# Install Python dependencies
RUN pip install --no-cache-dir maturin && \
    cd python && maturin develop --release

# Set up runtime
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/proteus/python:$PYTHONPATH

ENTRYPOINT ["proteus"]
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_content_store() {
        let temp_dir = std::env::temp_dir().join("proteus_test");
        let _ = fs::create_dir_all(&temp_dir);
        let store = ContentStore::new(temp_dir.clone());

        let data = b"test content";
        let ref1 = store.put("test", data).unwrap();
        let ref2 = store.put("test", data).unwrap();

        // Same content should produce same hash
        assert_eq!(ref1.hash, ref2.hash);
        assert_eq!(ref1.path, ref2.path);

        // Verify retrieval
        let retrieved = store.get(&ref1.hash).unwrap();
        assert_eq!(retrieved, data);

        // Verify integrity
        assert!(store.verify(&ref1.hash).unwrap());
    }
}
