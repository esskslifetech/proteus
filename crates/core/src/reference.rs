//! Reference genome representation — immutable, content-addressed

use crate::error::{CoreError, Result};
use blake3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// A single contig in the reference genome
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contig {
    pub name: String,
    pub length: usize,
    #[serde(with = "serde_bytes")]
    pub sequence: Vec<u8>,
    #[serde(skip)]
    sequence_hash: Option<blake3::Hash>,
}

impl Contig {
    /// Create a new contig, computing its content hash
    pub fn new(name: String, sequence: Vec<u8>) -> Self {
        let length = sequence.len();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&sequence);
        let sequence_hash = Some(hasher.finalize());
        Self {
            name,
            length,
            sequence,
            sequence_hash,
        }
    }

    /// Get the content hash of this contig's sequence
    pub fn hash(&self) -> blake3::Hash {
        self.sequence_hash.unwrap_or_else(|| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&self.sequence);
            hasher.finalize()
        })
    }

    /// Get a slice of the sequence (1-based inclusive coordinates)
    pub fn slice(&self, start: usize, end: usize) -> Result<&[u8]> {
        if start < 1 || end < start || end > self.length {
            return Err(CoreError::InvalidCoordinate {
                contig: self.name.clone(),
                pos: start as i64,
                len: self.length as i64,
            });
        }
        Ok(&self.sequence[start - 1..end])
    }

    /// Get the sequence as a string (lossy conversion for display)
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.sequence).unwrap_or("<invalid UTF-8>")
    }
}

/// Content-addressed, immutable reference genome
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceGenome {
    pub assembly: String,
    #[serde(skip)]
    contigs: HashMap<String, Arc<Contig>>,
    #[serde(skip)]
    contig_order: Vec<String>,
    #[serde(with = "serde_bytes")]
    pub genome_hash: Vec<u8>,  // blake3 hash of concatenated contig hashes
}

impl ReferenceGenome {
    /// Create a new reference genome from contigs
    pub fn new(assembly: String, contigs: Vec<Contig>) -> Self {
        let mut contig_map = HashMap::new();
        let mut contig_order = Vec::new();
        let mut hasher = blake3::Hasher::new();

        for contig in contigs {
            let name = contig.name.clone();
            hasher.update(contig.hash().as_bytes());
            contig_map.insert(name.clone(), Arc::new(contig));
            contig_order.push(name);
        }

        let genome_hash = hasher.finalize().as_bytes().to_vec();

        Self {
            assembly,
            contigs: contig_map,
            contig_order,
            genome_hash,
        }
    }

    /// Get a contig by name
    pub fn contig(&self, name: &str) -> Option<&Arc<Contig>> {
        self.contigs.get(name)
    }

    /// Get contig names in order
    pub fn contig_names(&self) -> &[String] {
        &self.contig_order
    }

    /// Get total genome length
    pub fn total_length(&self) -> usize {
        self.contigs.values().map(|c| c.length).sum()
    }

    /// Get the content hash of the entire genome
    pub fn hash(&self) -> blake3::Hash {
        blake3::Hash::from_bytes(
            self.genome_hash.as_slice().try_into().expect("genome_hash is 32 bytes"),
        )
    }

    /// Verify the genome hash matches the contigs
    pub fn verify(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        for name in &self.contig_order {
            if let Some(contig) = self.contigs.get(name) {
                hasher.update(contig.hash().as_bytes());
            } else {
                return false;
            }
        }
        hasher.finalize().as_bytes() == self.genome_hash.as_slice()
    }

    /// Get sequence slice from any contig (1-based inclusive)
    pub fn slice(&self, contig: &str, start: usize, end: usize) -> Result<&[u8]> {
        self.contig(contig)
            .ok_or_else(|| CoreError::InvalidCoordinate {
                contig: contig.to_string(),
                pos: start as i64,
                len: 0,
            })?
            .slice(start, end)
    }
}

/// Builder for ReferenceGenome from FASTA
pub struct ReferenceGenomeBuilder {
    assembly: String,
    contigs: Vec<Contig>,
}

impl ReferenceGenomeBuilder {
    pub fn new(assembly: String) -> Self {
        Self {
            assembly,
            contigs: Vec::new(),
        }
    }

    pub fn add_contig(mut self, name: String, sequence: Vec<u8>) -> Self {
        self.contigs.push(Contig::new(name, sequence));
        self
    }

    pub fn build(self) -> ReferenceGenome {
        ReferenceGenome::new(self.assembly, self.contigs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contig_creation() {
        let contig = Contig::new("chr1".to_string(), b"ATCGATCG".to_vec());
        assert_eq!(contig.name, "chr1");
        assert_eq!(contig.length, 8);
        assert_eq!(contig.slice(1, 4).unwrap(), b"ATCG");
    }

    #[test]
    fn test_contig_slice_bounds() {
        let contig = Contig::new("chr1".to_string(), b"ATCG".to_vec());
        assert!(contig.slice(0, 2).is_err());
        assert!(contig.slice(1, 5).is_err());
        assert!(contig.slice(3, 2).is_err());
    }

    #[test]
    fn test_reference_genome() {
        let genome = ReferenceGenomeBuilder::new("GRCh38.p14".to_string())
            .add_contig("chr1".to_string(), b"ATCGATCG".to_vec())
            .add_contig("chr2".to_string(), b"GGTTAACC".to_vec())
            .build();

        assert_eq!(genome.assembly, "GRCh38.p14");
        assert_eq!(genome.contig_names().len(), 2);
        assert_eq!(genome.total_length(), 16);
        assert!(genome.verify());
        assert_eq!(genome.slice("chr1", 1, 4).unwrap(), b"ATCG");
    }

    #[test]
    fn test_content_addressing() {
        let genome1 = ReferenceGenomeBuilder::new("test".to_string())
            .add_contig("chr1".to_string(), b"ATCG".to_vec())
            .build();

        let genome2 = ReferenceGenomeBuilder::new("test".to_string())
            .add_contig("chr1".to_string(), b"ATCG".to_vec())
            .build();

        let genome3 = ReferenceGenomeBuilder::new("test".to_string())
            .add_contig("chr1".to_string(), b"ATCC".to_vec())
            .build();

        assert_eq!(genome1.hash(), genome2.hash());
        assert_ne!(genome1.hash(), genome3.hash());
    }
}