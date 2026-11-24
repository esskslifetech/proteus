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
pub use v3_physical::{
    MetabolicReaction, PhysicalValidator, SolverStatus,
    ValidationResult as PhysicalValidationResult,
};
pub use v4_triangulation::{ArrayResult, ImplementationResult, TriangulationResult, Triangulator};
pub use v5_determinism::{
    DeterminismResult, DeterminismValidator, FuzzConfig, GoldenFileTest, GoldenResult,
    ProptestConfig,
};
pub use v6_scientific::{
    generate_report_card, load_clinvar_benchmark, load_gnomad_constraints, BenchmarkDataset,
    BenchmarkResult, BenchmarkVariant, ClassificationMetrics, ClinicalSignificance, GeneConstraint,
    Prediction, ReviewStatus,
};
pub use v7_reproducibility::{
    generate_dockerfile, generate_nix_flake, ContentRef, ContentStore, DependencyManifest,
    EnvironmentManifest, InputManifest, OutputManifest, RunManifest, ValidationManifest,
};
