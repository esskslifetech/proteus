//! Genome state with event-sourced mutation and exact revert

use crate::error::{CoreError, Result};
use crate::reference::ReferenceGenome;
use crate::variant::VariantEvent;
use blake3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Genome state: immutable reference + append-only variant event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeState {
    /// Reference genome (content-addressed, never mutated)
    #[serde(skip)]
    reference: Arc<ReferenceGenome>,
    /// Reference genome hash for serialization
    reference_hash: Vec<u8>,
    /// Append-only log of variant events
    overlays: Vec<VariantEvent>,
    /// Current head index (for revert/truncation)
    head: usize,
}

impl GenomeState {
    /// Create a new genome state from a reference genome
    pub fn new(reference: Arc<ReferenceGenome>) -> Self {
        let reference_hash = reference.hash().as_bytes().to_vec();
        Self {
            reference,
            reference_hash,
            overlays: Vec::new(),
            head: 0,
        }
    }

    /// Create from a reference and existing event log
    pub fn from_log(reference: Arc<ReferenceGenome>, overlays: Vec<VariantEvent>) -> Self {
        let head = overlays.len();
        let reference_hash = reference.hash().as_bytes().to_vec();
        Self {
            reference,
            reference_hash,
            overlays,
            head,
        }
    }

    /// Get the reference genome
    pub fn reference(&self) -> &Arc<ReferenceGenome> {
        &self.reference
    }

    /// Get all applied variant events (up to head)
    pub fn applied_variants(&self) -> &[VariantEvent] {
        &self.overlays[..self.head]
    }

    /// Get the full event log (including reverted events)
    pub fn full_log(&self) -> &[VariantEvent] {
        &self.overlays
    }

    /// Current number of applied variants
    pub fn variant_count(&self) -> usize {
        self.head
    }

    /// Apply a variant event, returning a new GenomeState (functional update)
    pub fn apply(&self, event: VariantEvent) -> Result<Self> {
        // Validate against reference
        if let Some(contig) = self.reference.contig(&event.contig) {
            event.validate(&contig.sequence)?;
        }

        let mut new_state = self.clone();
        new_state.overlays.push(event);
        new_state.head = new_state.overlays.len();
        Ok(new_state)
    }

    /// Apply multiple variants at once
    pub fn apply_batch(&self, events: Vec<VariantEvent>) -> Result<Self> {
        let mut new_state = self.clone();
        for event in events {
            if let Some(contig) = self.reference.contig(&event.contig) {
                event.validate(&contig.sequence)?;
            }
            new_state.overlays.push(event);
        }
        new_state.head = new_state.overlays.len();
        Ok(new_state)
    }

    /// Revert to a previous state by truncating the event log
    /// Returns a new GenomeState with head at `target` (exact bit-identical revert)
    pub fn revert(&self, target: usize) -> Result<Self> {
        if target > self.head {
            return Err(CoreError::RevertOutOfBounds {
                target,
                max: self.head,
            });
        }

        let mut new_state = self.clone();
        new_state.head = target;
        Ok(new_state)
    }

    /// Revert all variants (return to pristine reference)
    pub fn revert_all(&self) -> Self {
        let mut new_state = self.clone();
        new_state.head = 0;
        new_state
    }

    /// Revert the last N variants
    pub fn revert_last(&self, n: usize) -> Result<Self> {
        if n > self.head {
            return Err(CoreError::RevertOutOfBounds {
                target: self.head.saturating_sub(n),
                max: self.head,
            });
        }
        self.revert(self.head - n)
    }

    /// Get the sequence for a contig with all applied variants
    /// This is O(variants) not O(genome) - the key to "instant" mutation
    pub fn get_sequence(&self, contig: &str) -> Result<Vec<u8>> {
        let ref_contig =
            self.reference
                .contig(contig)
                .ok_or_else(|| CoreError::InvalidCoordinate {
                    contig: contig.to_string(),
                    pos: 0,
                    len: 0,
                })?;

        let mut sequence = ref_contig.sequence.clone();

        // Apply variants in order (reverse for correct position adjustment)
        for event in self.applied_variants().iter().rev() {
            if event.contig != contig {
                continue;
            }

            let start = (event.start - 1) as usize;
            let end = event.end as usize;

            match event.ty {
                crate::variant::VariantType::Snv | crate::variant::VariantType::Mnv => {
                    // Replace ref with alt
                    if start < sequence.len() && end <= sequence.len() {
                        sequence.splice(start..end, event.alt_allele.bytes());
                    }
                }
                crate::variant::VariantType::Insertion => {
                    // Insert at position (start == end for insertions)
                    if start <= sequence.len() {
                        sequence.splice(start..start, event.alt_allele.bytes());
                    }
                }
                crate::variant::VariantType::Deletion => {
                    // Delete range
                    if start < sequence.len() && end <= sequence.len() {
                        sequence.drain(start..end);
                    }
                }
                crate::variant::VariantType::Delins => {
                    // Delete then insert
                    if start < sequence.len() && end <= sequence.len() {
                        sequence.splice(start..end, event.alt_allele.bytes());
                    }
                }
                _ => {
                    // Other types not yet implemented in sequence reconstruction
                }
            }
        }

        Ok(sequence)
    }

    /// Diff two genome states (compares event logs)
    pub fn diff(&self, other: &Self) -> LayerDiff {
        let mut diff = LayerDiff::default();

        // Compare reference hashes
        if self.reference_hash != other.reference_hash {
            diff.reference_changed = true;
        }

        // Compare applied variants
        let self_vars: HashMap<_, _> = self
            .applied_variants()
            .iter()
            .map(|v| (v.id.clone(), v))
            .collect();
        let other_vars: HashMap<_, _> = other
            .applied_variants()
            .iter()
            .map(|v| (v.id.clone(), v))
            .collect();

        // Added variants (in other but not self)
        for (id, var) in &other_vars {
            if !self_vars.contains_key(id) {
                diff.added_variants.push((*var).clone());
            }
        }

        // Removed variants (in self but not other)
        for (id, var) in &self_vars {
            if !other_vars.contains_key(id) {
                diff.removed_variants.push((*var).clone());
            }
        }

        // Changed variants (same ID, different content)
        for (id, var) in &self_vars {
            if let Some(other_var) = other_vars.get(id) {
                if var != other_var {
                    diff.changed_variants
                        .push(((*var).clone(), (*other_var).clone()));
                }
            }
        }

        diff
    }

    /// Compute content hash of this state.
    ///
    /// The hash covers the immutable reference plus the canonical content of
    /// every applied event (id, type, locus, alleles). Wall-clock metadata
    /// (`timestamp_ns`) and free-form `annotations` are deliberately excluded so
    /// that replaying the same event log — or applying equivalent variants in a
    /// different process — yields the same content address. This is what makes
    /// the event-sourced revert and V5/V7 determinism checks meaningful.
    pub fn hash(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.reference_hash);
        for event in self.applied_variants() {
            let canonical = format!(
                "{}\x1f{}\x1f{}\x1f{}\x1f{}\x1f{}\x1f{}\n",
                event.id,
                event.ty,
                event.contig,
                event.start,
                event.end,
                event.ref_allele,
                event.alt_allele,
            );
            hasher.update(canonical.as_bytes());
        }
        hasher.finalize()
    }
}

/// Diff between two genome states
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerDiff {
    pub reference_changed: bool,
    pub added_variants: Vec<VariantEvent>,
    pub removed_variants: Vec<VariantEvent>,
    pub changed_variants: Vec<(VariantEvent, VariantEvent)>,
}

impl LayerDiff {
    pub fn is_empty(&self) -> bool {
        !self.reference_changed
            && self.added_variants.is_empty()
            && self.removed_variants.is_empty()
            && self.changed_variants.is_empty()
    }

    pub fn total_changes(&self) -> usize {
        self.added_variants.len() + self.removed_variants.len() + self.changed_variants.len()
    }
}

/// State transition for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_hash: Vec<u8>,
    pub to_hash: Vec<u8>,
    pub applied: Vec<VariantEvent>,
    pub reverted: Vec<VariantEvent>,
    pub timestamp_ns: i64,
}

impl GenomeState {
    /// Create a transition record from self to new state
    pub fn transition_to(&self, new_state: &Self) -> StateTransition {
        let self_vars: HashMap<_, _> = self
            .applied_variants()
            .iter()
            .map(|v| (v.id.clone(), v))
            .collect();
        let new_vars: HashMap<_, _> = new_state
            .applied_variants()
            .iter()
            .map(|v| (v.id.clone(), v))
            .collect();

        let mut applied = Vec::new();
        let mut reverted = Vec::new();

        for (id, var) in &new_vars {
            if !self_vars.contains_key(id) {
                applied.push((*var).clone());
            }
        }

        for (id, var) in &self_vars {
            if !new_vars.contains_key(id) {
                reverted.push((*var).clone());
            }
        }

        StateTransition {
            from_hash: self.hash().as_bytes().to_vec(),
            to_hash: new_state.hash().as_bytes().to_vec(),
            applied,
            reverted,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ReferenceGenomeBuilder;
    use crate::variant::VariantEvent;

    fn test_genome() -> GenomeState {
        let reference = Arc::new(
            ReferenceGenomeBuilder::new("test".to_string())
                .add_contig("chr1".to_string(), b"ATCGATCGATCG".to_vec())
                .build(),
        );
        GenomeState::new(reference)
    }

    #[test]
    fn test_apply_snv() {
        let state = test_genome();
        let event = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let new_state = state.apply(event).unwrap();
        assert_eq!(new_state.variant_count(), 1);

        let seq = new_state.get_sequence("chr1").unwrap();
        assert_eq!(seq[4], b'T'); // position 5 (0-indexed 4) should be T
    }

    #[test]
    fn test_apply_insertion() {
        let state = test_genome();
        let event = VariantEvent::insertion(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "GGG".to_string(),
            "test".to_string(),
        );
        let new_state = state.apply(event).unwrap();

        let seq = new_state.get_sequence("chr1").unwrap();
        // Original: ATCGATCGATCG (12bp)
        // After insertion at pos 5: ATCG + GGG + ATCGATCG = 15bp
        assert_eq!(seq.len(), 15);
        assert_eq!(&seq[4..7], b"GGG");
    }

    #[test]
    fn test_apply_deletion() {
        let state = test_genome();
        let event = VariantEvent::deletion(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            7,
            "ATC".to_string(),
            "test".to_string(),
        );
        let new_state = state.apply(event).unwrap();

        let seq = new_state.get_sequence("chr1").unwrap();
        // Original: ATCGATCGATCG (12bp)
        // Delete positions 5-7 (ATC): ATCG + GATCG = 9bp
        assert_eq!(seq.len(), 9);
    }

    #[test]
    fn test_revert_exact() {
        let state = test_genome();
        let event = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let mutated = state.apply(event).unwrap();
        let reverted = mutated.revert(0).unwrap();

        // Reverted state should be bit-identical to original
        assert_eq!(reverted.hash(), state.hash());
        assert_eq!(reverted.variant_count(), 0);
        assert_eq!(
            reverted.get_sequence("chr1").unwrap(),
            state.get_sequence("chr1").unwrap()
        );
    }

    #[test]
    fn test_revert_partial() {
        let state = test_genome();
        let event1 = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let event2 = VariantEvent::snv(
            "v2".to_string(),
            "chr1".to_string(),
            8,
            "G".to_string(),
            "C".to_string(),
            "test".to_string(),
        );
        let mutated = state.apply(event1).unwrap().apply(event2).unwrap();
        assert_eq!(mutated.variant_count(), 2);

        // Revert last 1
        let reverted = mutated.revert_last(1).unwrap();
        assert_eq!(reverted.variant_count(), 1);
        assert_eq!(reverted.applied_variants()[0].id, "v1");
    }

    #[test]
    fn test_diff() {
        let state = test_genome();
        let event = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let mutated = state.apply(event).unwrap();

        let diff = state.diff(&mutated);
        assert_eq!(diff.added_variants.len(), 1);
        assert_eq!(diff.added_variants[0].id, "v1");
        assert_eq!(diff.removed_variants.len(), 0);

        let diff2 = mutated.diff(&state);
        assert_eq!(diff2.removed_variants.len(), 1);
        assert_eq!(diff2.removed_variants[0].id, "v1");
    }

    #[test]
    fn test_transition() {
        let state = test_genome();
        let event = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        let mutated = state.apply(event).unwrap();

        let transition = state.transition_to(&mutated);
        assert_eq!(transition.applied.len(), 1);
        assert_eq!(transition.reverted.len(), 0);

        let transition_back = mutated.transition_to(&state);
        assert_eq!(transition_back.reverted.len(), 1);
        assert_eq!(transition_back.applied.len(), 0);
    }

    #[test]
    fn test_batch_apply() {
        let state = test_genome();
        let events = vec![
            VariantEvent::snv(
                "v1".to_string(),
                "chr1".to_string(),
                5,
                "A".to_string(),
                "T".to_string(),
                "test".to_string(),
            ),
            VariantEvent::snv(
                "v2".to_string(),
                "chr1".to_string(),
                8,
                "G".to_string(),
                "C".to_string(),
                "test".to_string(),
            ),
        ];
        let mutated = state.apply_batch(events).unwrap();
        assert_eq!(mutated.variant_count(), 2);
    }

    #[test]
    fn test_hash_is_content_addressed_not_timestamp_addressed() {
        let state = test_genome();

        let mut a = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "T".to_string(),
            "test".to_string(),
        );
        a.timestamp_ns = 1;

        let mut b = a.clone();
        b.timestamp_ns = 999_999; // same content, different wall-clock metadata

        let state_a = state.apply(a).unwrap();
        let state_b = state.apply(b).unwrap();
        assert_eq!(state_a.hash(), state_b.hash());

        // Different alleles must produce a different content address.
        let c = VariantEvent::snv(
            "v1".to_string(),
            "chr1".to_string(),
            5,
            "A".to_string(),
            "G".to_string(),
            "test".to_string(),
        );
        assert_ne!(state_a.hash(), state.apply(c).unwrap().hash());

        // Replaying the same log from baseline must reproduce the mutated hash.
        let replay = state
            .apply_batch(state_a.applied_variants().to_vec())
            .unwrap();
        assert_eq!(replay.hash(), state_a.hash());
    }
}
