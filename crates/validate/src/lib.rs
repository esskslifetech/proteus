//! proteus-validate: Validation layers V1-V7

pub mod error;
pub mod v1_schema;
pub mod v2_semantic;
pub mod v3_physical;
pub mod v4_triangulation;
pub mod v5_determinism;
pub mod v6_scientific;
pub mod v7_reproducibility;

pub use error::*;
pub use v1_schema::{SchemaValidator, ValidationResult as SchemaValidationResult};
pub use v2_semantic::{SemanticValidator, ValidationResult as SemanticValidationResult};
pub use v3_physical::{PhysicalValidator, MetabolicReaction, SolverStatus, ValidationResult as PhysicalValidationResult};
pub use v4_triangulation::{Triangulator, ImplementationResult, ArrayResult, TriangulationResult};
pub use v5_determinism::{DeterminismValidator, GoldenFileTest, GoldenResult, DeterminismResult, ProptestConfig, FuzzConfig};
pub use v6_scientific::{BenchmarkDataset, BenchmarkVariant, ClinicalSignificance, ReviewStatus, BenchmarkResult, Prediction, ClassificationMetrics, load_clinvar_benchmark, load_gnomad_constraints, GeneConstraint, generate_report_card};
pub use v7_reproducibility::{RunManifest, DependencyManifest, InputManifest, ContentRef, OutputManifest, EnvironmentManifest, ValidationManifest, ContentStore, generate_nix_flake, generate_dockerfile};