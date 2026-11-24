//! proteus-core: Core genome state, variant algebra, and event-sourced revert engine.
//!
//! The reference genome is immutable; edits are overlays. This provides:
//! - O(variants) "instant" mutation application
//! - Exact bit-identical revert via log truncation
//! - Free diff between any two states via log comparison

pub mod algebra;
pub mod error;
pub mod reference;
pub mod state;
pub mod variant;

pub use algebra::*;
pub use error::*;
pub use reference::*;
pub use state::*;
pub use variant::*;
