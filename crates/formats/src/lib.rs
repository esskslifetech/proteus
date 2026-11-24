//! proteus-formats: Unified readers/writers for all genomics formats

pub mod bam;
pub mod bed;
pub mod fasta;
pub mod gff;
pub mod parquet;
pub mod vcf;

pub use bam::*;
pub use bed::*;
pub use fasta::*;
pub use gff::*;
pub use parquet::*;
pub use vcf::*;
