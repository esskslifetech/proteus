//! V2: Semantic validation — biological invariants

use serde::{Deserialize, Serialize};

// Re-export core types for convenience
pub use proteus_core::{ReferenceGenome, VariantEvent};

#[derive(Debug, Clone, Default)]
pub struct SemanticValidator;

impl SemanticValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate variant against reference genome
    pub fn validate_variant(
        &self,
        variant: &VariantEvent,
        reference: &ReferenceGenome,
    ) -> ValidationResult {
        // Check contig exists
        let contig = match reference.contig(&variant.contig) {
            Some(c) => c,
            None => {
                return ValidationResult::Fail {
                    check: "contig_exists".to_string(),
                    message: format!("contig '{}' not found in reference", variant.contig),
                }
            }
        };

        // Check coordinate bounds
        if variant.start < 1 || variant.end > contig.length as i64 {
            return ValidationResult::Fail {
                check: "coordinate_bounds".to_string(),
                message: format!(
                    "variant {} out of bounds: {}-{} vs contig length {}",
                    variant.id, variant.start, variant.end, contig.length
                ),
            };
        }

        // Check IUPAC alphabet
        if !is_iupac_dna(&variant.ref_allele) || !is_iupac_dna(&variant.alt_allele) {
            return ValidationResult::Fail {
                check: "iupac_alphabet".to_string(),
                message: format!(
                    "non-IUPAC characters in alleles: ref={}, alt={}",
                    variant.ref_allele, variant.alt_allele
                ),
            };
        }

        ValidationResult::Pass {
            check: "semantic".to_string(),
        }
    }
}

fn is_iupac_dna(seq: &str) -> bool {
    seq.bytes().all(|b| {
        matches!(
            b,
            b'A' | b'C'
                | b'G'
                | b'T'
                | b'N'
                | b'R'
                | b'Y'
                | b'S'
                | b'W'
                | b'K'
                | b'M'
                | b'B'
                | b'D'
                | b'H'
                | b'V'
                | b'a'
                | b'c'
                | b'g'
                | b't'
                | b'n'
                | b'r'
                | b'y'
                | b's'
                | b'w'
                | b'k'
                | b'm'
                | b'b'
                | b'd'
                | b'h'
                | b'v'
        )
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    Pass { check: String },
    Fail { check: String, message: String },
}

impl ValidationResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, ValidationResult::Pass { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_core::{ReferenceGenomeBuilder, VariantEvent};
    use std::sync::Arc;

    #[test]
    fn test_semantic_validator_pass() {
        let reference = Arc::new(
            ReferenceGenomeBuilder::new("test".to_string())
                .add_contig("chr1".to_string(), b"ATCGATCG".to_vec())
                .build(),
        );
        let validator = SemanticValidator::new();
        let variant = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            1,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let result = validator.validate_variant(&variant, &reference);
        assert!(result.is_pass());
    }

    #[test]
    fn test_semantic_validator_fail_contig() {
        let reference = Arc::new(
            ReferenceGenomeBuilder::new("test".to_string())
                .add_contig("chr1".to_string(), b"ATCGATCG".to_vec())
                .build(),
        );
        let validator = SemanticValidator::new();
        let variant = VariantEvent::snv(
            "v1".to_string(),
            "chrX".to_string(),
            1,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let result = validator.validate_variant(&variant, &reference);
        assert!(!result.is_pass());
    }

    #[test]
    fn test_iupac_check() {
        assert!(is_iupac_dna("ACGT"));
        assert!(is_iupac_dna("ACGTNRYSWKMBDHV"));
        assert!(!is_iupac_dna("ACGTX"));
        assert!(!is_iupac_dna("ACGT "));
    }
}
