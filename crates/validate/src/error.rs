//! Error types for proteus-validate

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Schema validation error: {0}")]
    Schema(String),

    #[error("Semantic validation error: {0}")]
    Semantic(String),

    #[error("Physical constraint error: {0}")]
    Physical(String),

    #[error("Triangulation error: {0}")]
    Triangulation(String),

    #[error("Determinism check failed: {0}")]
    Determinism(String),

    #[error("Scientific validity error: {0}")]
    Scientific(String),

    #[error("Reproducibility check failed: {0}")]
    Reproducibility(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ValidationError>;
