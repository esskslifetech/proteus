//! proteus-formats: Unified readers/writers for all genomics formats

pub mod fasta;
pub mod vcf;
pub mod gff;
pub mod bed;
pub mod bam;
pub mod parquet;

pub use fasta::*;
pub use vcf::*;
pub use gff::*;
pub use bed::*;
pub use bam::*;
pub use parquet::*;