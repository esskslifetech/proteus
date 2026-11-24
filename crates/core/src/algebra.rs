//! Variant algebra: coordinate transformation, composition, normalization

use crate::error::{CoreError, Result};
use crate::variant::{VariantEvent, VariantType};
use std::collections::HashMap;

/// Transform variant coordinates after another variant is applied
pub fn transform_coordinates(variant: &VariantEvent, prior: &VariantEvent) -> VariantEvent {
    if variant.contig != prior.contig {
        return variant.clone();
    }

    let mut transformed = variant.clone();
    let delta = prior.length_delta();
    let prior_end = prior.end;

    // If variant is entirely before prior variant, no change
    if variant.end < prior.start {
        return transformed;
    }

    // If variant overlaps with prior, we need complex handling
    // For now, adjust coordinates based on length delta
    if variant.start > prior_end {
        // Variant is after prior variant
        transformed.start += delta;
        transformed.end += delta;
    } else if variant.start >= prior.start && variant.end <= prior_end {
        // Variant is contained within prior variant
        // This is complex - for now mark as needing review
        transformed
            .annotations
            .insert("transform_status".to_string(), "contained".to_string());
    } else {
        // Overlapping but not contained
        transformed
            .annotations
            .insert("transform_status".to_string(), "overlapping".to_string());
        transformed.start += delta;
        transformed.end += delta;
    }

    transformed
}

/// Compose multiple variants into a minimal equivalent set
pub fn compose_variants(variants: &[VariantEvent]) -> Vec<VariantEvent> {
    if variants.is_empty() {
        return Vec::new();
    }

    // Group by contig
    let mut by_contig: HashMap<String, Vec<&VariantEvent>> = HashMap::new();
    for v in variants {
        by_contig.entry(v.contig.clone()).or_default().push(v);
    }

    let mut composed = Vec::new();

    for (_contig, mut vars) in by_contig {
        // Sort by position
        vars.sort_by_key(|v| (v.start, v.end));

        // Merge adjacent/overlapping variants of same type
        let mut merged: Vec<VariantEvent> = Vec::new();
        for var in vars {
            if let Some(last) = merged.last_mut() {
                if can_merge(last, var) {
                    *last = merge_variants(last, var);
                    continue;
                }
            }
            merged.push((*var).clone());
        }

        composed.extend(merged);
    }

    composed
}

fn can_merge(a: &VariantEvent, b: &VariantEvent) -> bool {
    a.contig == b.contig
        && a.ty == b.ty
        && a.source == b.source
        // Adjacent or overlapping
        && b.start <= a.end + 1
}

fn merge_variants(a: &VariantEvent, b: &VariantEvent) -> VariantEvent {
    let mut merged = a.clone();
    merged.end = b.end.max(a.end);
    merged.ref_allele = format!(
        "{}{}",
        a.ref_allele,
        &b.ref_allele[a.end as usize - a.start as usize..]
    );
    merged.alt_allele = format!(
        "{}{}",
        a.alt_allele,
        &b.alt_allele[a.end as usize - a.start as usize..]
    );
    merged.id = format!("{}_merged_{}", a.id, b.id);
    merged
}

/// Normalize a variant to its leftmost representation (for VCF compatibility)
pub fn normalize_variant(
    variant: &VariantEvent,
    ref_genome: &crate::reference::ReferenceGenome,
) -> Result<VariantEvent> {
    // For SNVs, already normalized
    if variant.ty == VariantType::Snv {
        return Ok(variant.clone());
    }

    // For indels, left-align
    let contig =
        ref_genome
            .contig(&variant.contig)
            .ok_or_else(|| CoreError::InvalidCoordinate {
                contig: variant.contig.clone(),
                pos: variant.start,
                len: 0,
            })?;

    let mut normalized = variant.clone();
    let mut pos = variant.start as usize;

    // Left-align: while we can shift left and the sequence matches
    while pos > 1 {
        let prev_base = contig.sequence[pos - 2];
        let ref_bytes = variant.ref_allele.as_bytes();
        let alt_bytes = variant.alt_allele.as_bytes();

        // Check if we can shift left
        let can_shift = match variant.ty {
            VariantType::Deletion => !ref_bytes.is_empty() && ref_bytes[0] == prev_base,
            VariantType::Insertion => {
                !alt_bytes.is_empty() && alt_bytes[alt_bytes.len() - 1] == prev_base
            }
            _ => false,
        };

        if !can_shift {
            break;
        }

        pos -= 1;
    }

    if pos != variant.start as usize {
        normalized.start = pos as i64;
        normalized.end = (pos as i64 + variant.ref_allele.len() as i64 - 1).max(pos as i64);
        normalized
            .annotations
            .insert("normalized".to_string(), "true".to_string());
    }

    Ok(normalized)
}

/// Check if two variants are equivalent (same effect on sequence)
pub fn variants_equivalent(
    a: &VariantEvent,
    b: &VariantEvent,
    ref_genome: &crate::reference::ReferenceGenome,
) -> Result<bool> {
    let norm_a = normalize_variant(a, ref_genome)?;
    let norm_b = normalize_variant(b, ref_genome)?;

    Ok(norm_a.contig == norm_b.contig
        && norm_a.start == norm_b.start
        && norm_a.end == norm_b.end
        && norm_a.ref_allele == norm_b.ref_allele
        && norm_a.alt_allele == norm_b.alt_allele)
}

/// Project variant onto transcript coordinates
pub fn project_to_transcript(
    variant: &VariantEvent,
    transcript: &TranscriptModel,
) -> Option<TranscriptVariant> {
    // Simplified - would need full exon/intron mapping
    if variant.contig != transcript.contig {
        return None;
    }

    // Find overlapping exons
    for exon in &transcript.exons {
        if variant.start <= exon.end && variant.end >= exon.start {
            let cds_start = transcript.cds_start?;
            let cds_end = transcript.cds_end?;

            let rel_start = variant.start - cds_start + 1;
            let rel_end = variant.end - cds_start + 1;

            return Some(TranscriptVariant {
                variant_id: variant.id.clone(),
                transcript_id: transcript.id.clone(),
                cds_start: rel_start.max(1),
                cds_end: rel_end.min(cds_end - cds_start + 1),
                exon_index: transcript.exons.iter().position(|e| e == exon)?,
            });
        }
    }

    None
}

/// Transcript model for coordinate projection
#[derive(Debug, Clone)]
pub struct TranscriptModel {
    pub id: String,
    pub gene_id: String,
    pub contig: String,
    pub strand: Strand,
    pub exons: Vec<Exon>,
    pub cds_start: Option<i64>,
    pub cds_end: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Exon {
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Clone)]
pub struct TranscriptVariant {
    pub variant_id: String,
    pub transcript_id: String,
    pub cds_start: i64,
    pub cds_end: i64,
    pub exon_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ReferenceGenomeBuilder;
    use crate::variant::VariantEvent;
    use std::sync::Arc;

    #[test]
    fn test_compose_adjacent_snvs() {
        let _ref_genome = Arc::new(
            ReferenceGenomeBuilder::new("test".to_string())
                .add_contig("chr1".to_string(), b"ATCGATCG".to_vec())
                .build(),
        );

        let v1 = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            2,
            "T".to_string(),
            "C".to_string(),
            "test".to_string(),
        );
        let v2 = VariantEvent::snv(
            "v2".to_string(),
            "chr1".to_string(),
            3,
            "C".to_string(),
            "G".to_string(),
            "test".to_string(),
        );

        let composed = compose_variants(&[v1, v2]);
        assert_eq!(composed.len(), 1); // Should merge into MNV
    }

    #[test]
    fn test_transform_coordinates_after_insertion() {
        let prior = VariantEvent::insertion(
            "prior".to_string(),
            "chr1".to_string(),
            5,
            "AAA".to_string(),
            "test".to_string(),
        );
        let variant = VariantEvent::snv(
            "var".to_string(),
            "chr1".to_string(),
            10,
            "G".to_string(),
            "T".to_string(),
            "test".to_string(),
        );

        let transformed = transform_coordinates(&variant, &prior);
        assert_eq!(transformed.start, 13); // 10 + 3
        assert_eq!(transformed.end, 13);
    }

    #[test]
    fn test_transform_coordinates_before_insertion() {
        let prior = VariantEvent::insertion(
            "prior".to_string(),
            "chr1".to_string(),
            10,
            "AAA".to_string(),
            "test".to_string(),
        );
        let variant = VariantEvent::snv(
            "var".to_string(),
            "chr1".to_string(),
            5,
            "G".to_string(),
            "T".to_string(),
            "test".to_string(),
        );

        let transformed = transform_coordinates(&variant, &prior);
        assert_eq!(transformed.start, 5); // Unchanged
        assert_eq!(transformed.end, 5);
    }

    #[test]
    fn test_normalize_deletion() {
        let ref_genome = Arc::new(
            ReferenceGenomeBuilder::new("test".to_string())
                .add_contig("chr1".to_string(), b"AAAAATTTTT".to_vec())
                .build(),
        );

        // Deletion of "AAA" at position 3-5 (0-indexed 2-4)
        // Can be left-aligned to position 1-3
        let variant = VariantEvent::deletion(
            "v1".to_string(),
            "chr1".to_string(),
            3,
            5,
            "AAA".to_string(),
            "test".to_string(),
        );

        let normalized = normalize_variant(&variant, &ref_genome).unwrap();
        assert_eq!(normalized.start, 1);
        assert_eq!(normalized.end, 3);
    }
}
