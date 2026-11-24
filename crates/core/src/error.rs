//! Error types for proteus-core

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid coordinate: contig={contig}, pos={pos}, len={len}")]
    InvalidCoordinate { contig: String, pos: i64, len: i64 },

    #[error("Allele mismatch: expected {expected}, found {found} at {contig}:{pos}")]
    AlleleMismatch { contig: String, pos: i64, expected: String, found: String },

    #[error("Immutable reference genome cannot be modified")]
    ImmutableReference,

    #[error("Revert target index {target} out of bounds (max {max})")]
    RevertOutOfBounds { target: usize, max: usize },

    #[error("Content hash mismatch: expected {expected}, computed {computed}")]
    HashMismatch { expected: String, computed: String },
}

pub type Result<T> = std::result::Result<T, CoreError>;