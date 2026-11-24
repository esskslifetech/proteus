//! Variant representation and parsing

use crate::error::{CoreError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Variant type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VariantType {
    Snv,
    Mnv,
    Insertion,
    Deletion,
    Delins,
    Inversion,
    Duplication,
    Translocation,
    Cnv,
    StrExpansion,
    RnaEdit,
    SpliceSite,
}

impl fmt::Display for VariantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VariantType::Snv => write!(f, "SNV"),
            VariantType::Mnv => write!(f, "MNV"),
            VariantType::Insertion => write!(f, "INS"),
            VariantType::Deletion => write!(f, "DEL"),
            VariantType::Delins => write!(f, "DELINS"),
            VariantType::Inversion => write!(f, "INV"),
            VariantType::Duplication => write!(f, "DUP"),
            VariantType::Translocation => write!(f, "TRA"),
            VariantType::Cnv => write!(f, "CNV"),
            VariantType::StrExpansion => write!(f, "STR_EXP"),
            VariantType::RnaEdit => write!(f, "RNA_EDIT"),
            VariantType::SpliceSite => write!(f, "SPLICE"),
        }
    }
}

/// A single variant event in the event-sourced genome state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantEvent {
    /// Unique event identifier (UUID v7 or similar)
    pub id: String,
    /// Variant type
    pub ty: VariantType,
    /// Contig/chromosome name
    pub contig: String,
    /// Start position (1-based, inclusive)
    pub start: i64,
    /// End position (1-based, inclusive). For SNV: start == end
    pub end: i64,
    /// Reference allele
    pub ref_allele: String,
    /// Alternate allele
    pub alt_allele: String,
    /// Arbitrary annotations
    #[serde(default)]
    pub annotations: std::collections::HashMap<String, String>,
    /// Event timestamp in nanoseconds since epoch
    pub timestamp_ns: i64,
    /// Source of this event: "user", "simulation", "revert"
    pub source: String,
}

impl VariantEvent {
    /// Create a new SNV event
    pub fn snv(
        id: String,
        contig: String,
        pos: i64,
        ref_allele: String,
        alt_allele: String,
        source: String,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        Self {
            id,
            ty: VariantType::Snv,
            contig,
            start: pos,
            end: pos,
            ref_allele,
            alt_allele,
            annotations: Default::default(),
            timestamp_ns: now,
            source,
        }
    }

    /// Create a new insertion event
    pub fn insertion(
        id: String,
        contig: String,
        pos: i64,
        inserted_seq: String,
        source: String,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        Self {
            id,
            ty: VariantType::Insertion,
            contig,
            start: pos,
            end: pos,
            ref_allele: String::new(),
            alt_allele: inserted_seq,
            annotations: Default::default(),
            timestamp_ns: now,
            source,
        }
    }

    /// Create a new deletion event
    pub fn deletion(
        id: String,
        contig: String,
        start: i64,
        end: i64,
        deleted_seq: String,
        source: String,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        Self {
            id,
            ty: VariantType::Deletion,
            contig,
            start,
            end,
            ref_allele: deleted_seq,
            alt_allele: String::new(),
            annotations: Default::default(),
            timestamp_ns: now,
            source,
        }
    }

    /// Get the length change this variant introduces
    pub fn length_delta(&self) -> i64 {
        (self.alt_allele.len() as i64) - (self.ref_allele.len() as i64)
    }

    /// Check if this variant affects the given position
    pub fn covers(&self, contig: &str, pos: i64) -> bool {
        self.contig == contig && pos >= self.start && pos <= self.end
    }

    /// Validate the variant against a reference sequence
    pub fn validate(&self, ref_seq: &[u8]) -> Result<()> {
        if self.contig.is_empty() {
            return Err(CoreError::InvalidCoordinate {
                contig: self.contig.clone(),
                pos: self.start,
                len: ref_seq.len() as i64,
            });
        }

        if self.start < 1 || self.end < 1 {
            return Err(CoreError::InvalidCoordinate {
                contig: self.contig.clone(),
                pos: self.start,
                len: ref_seq.len() as i64,
            });
        }

        if self.start > self.end {
            return Err(CoreError::InvalidCoordinate {
                contig: self.contig.clone(),
                pos: self.start,
                len: ref_seq.len() as i64,
            });
        }

        if self.end as usize > ref_seq.len() {
            return Err(CoreError::InvalidCoordinate {
                contig: self.contig.clone(),
                pos: self.end,
                len: ref_seq.len() as i64,
            });
        }

        // For non-insertions, check reference allele matches
        if self.ty != VariantType::Insertion && !self.ref_allele.is_empty() {
            let ref_slice = &ref_seq[(self.start - 1) as usize..self.end as usize];
            let ref_str = std::str::from_utf8(ref_slice).map_err(|_| CoreError::AlleleMismatch {
                contig: self.contig.clone(),
                pos: self.start,
                expected: self.ref_allele.clone(),
                found: "invalid UTF-8".to_string(),
            })?;

            if ref_str != self.ref_allele {
                return Err(CoreError::AlleleMismatch {
                    contig: self.contig.clone(),
                    pos: self.start,
                    expected: self.ref_allele.clone(),
                    found: ref_str.to_string(),
                });
            }
        }

        Ok(())
    }
}

/// VCF record representation for interop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcfRecord {
    pub chrom: String,
    pub pos: i64,
    pub id: Option<String>,
    pub ref_allele: String,
    pub alt_alleles: Vec<String>,
    pub qual: Option<f64>,
    pub filter: Vec<String>,
    pub info: std::collections::HashMap<String, String>,
    pub format: Vec<String>,
    pub samples: Vec<std::collections::HashMap<String, String>>,
}

impl VcfRecord {
    /// Convert to VariantEvent (takes first ALT allele)
    pub fn to_variant_event(&self, source: String) -> VariantEvent {
        let alt = self.alt_alleles.first().cloned().unwrap_or_default();
        let id = self.id.clone().unwrap_or_else(|| format!("{}:{}", self.chrom, self.pos));
        let ty = classify_variant(&self.ref_allele, &alt);
        VariantEvent {
            id,
            ty,
            contig: self.chrom.clone(),
            start: self.pos,
            end: self.pos + self.ref_allele.len() as i64 - 1,
            ref_allele: self.ref_allele.clone(),
            alt_allele: alt,
            annotations: self.info.clone(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            source,
        }
    }
}

/// Classify a variant based on REF/ALT alleles
pub fn classify_variant(ref_allele: &str, alt_allele: &str) -> VariantType {
    match (ref_allele.is_empty(), alt_allele.is_empty()) {
        // Both non-empty
        (false, false) if ref_allele.len() == 1 && alt_allele.len() == 1 => VariantType::Snv,
        (false, false) if ref_allele.len() == alt_allele.len() => VariantType::Mnv,
        (false, false) if alt_allele.len() > ref_allele.len() && alt_allele.starts_with(ref_allele) => VariantType::Insertion,
        (false, false) if alt_allele.len() < ref_allele.len() && ref_allele.starts_with(alt_allele) => VariantType::Deletion,
        (false, false) => VariantType::Delins, // Different lengths, not a simple prefix extension
        // One empty
        (true, false) => VariantType::Insertion,
        (false, true) => VariantType::Deletion,
        // Both empty (shouldn't happen)
        (true, true) => VariantType::Snv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snv_classification() {
        assert_eq!(classify_variant("A", "T"), VariantType::Snv);
        assert_eq!(classify_variant("C", "G"), VariantType::Snv);
    }

    #[test]
    fn test_insertion_classification() {
        assert_eq!(classify_variant("", "A"), VariantType::Insertion);
        assert_eq!(classify_variant("A", "AT"), VariantType::Insertion);
    }

    #[test]
    fn test_deletion_classification() {
        assert_eq!(classify_variant("A", ""), VariantType::Deletion);
        assert_eq!(classify_variant("AT", ""), VariantType::Deletion);
    }

    #[test]
    fn test_mnv_classification() {
        assert_eq!(classify_variant("AT", "CG"), VariantType::Mnv);
    }

    #[test]
    fn test_delins_classification() {
        assert_eq!(classify_variant("AT", "C"), VariantType::Delins);
        assert_eq!(classify_variant("A", "CG"), VariantType::Delins);
    }

    #[test]
    fn test_variant_event_length_delta() {
        let snv = VariantEvent::snv(
            "test".to_string(),
            "chr1".to_string(),
            100,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        assert_eq!(snv.length_delta(), 0);

        let ins = VariantEvent::insertion(
            "test".to_string(),
            "chr1".to_string(),
            100,
            "ATCG".to_string(),
            "test".to_string(),
        );
        assert_eq!(ins.length_delta(), 4);

        let del = VariantEvent::deletion(
            "test".to_string(),
            "chr1".to_string(),
            100,
            103,
            "ATCG".to_string(),
            "test".to_string(),
        );
        assert_eq!(del.length_delta(), -4);
    }

    #[test]
    fn test_validate_snv() {
        let event = VariantEvent::snv(
            "test".to_string(),
            "chr1".to_string(),
            1,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let ref_seq = b"ATCGATCG";
        assert!(event.validate(ref_seq).is_ok());
    }

    #[test]
    fn test_validate_allele_mismatch() {
        let event = VariantEvent::snv(
            "test".to_string(),
            "chr1".to_string(),
            1,
            "G".to_string(), // wrong ref allele
            "T".to_string(),
            "test".to_string(),
        );
        let ref_seq = b"ATCGATCG";
        assert!(event.validate(ref_seq).is_err());
    }
}