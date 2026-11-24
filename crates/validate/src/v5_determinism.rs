//! V5: Determinism — seeded RNG, golden-file regression, property tests, fuzzing

use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

#[derive(Debug, Clone)]
pub struct DeterminismValidator {
    seed: u64,
}

impl DeterminismValidator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Create a seeded RNG for reproducible results
    pub fn rng(&self) -> StdRng {
        StdRng::seed_from_u64(self.seed)
    }

    /// Verify that running the same computation twice produces identical results
    pub fn verify_deterministic<F, T>(&self, name: &str, mut f: F) -> DeterminismResult
    where
        F: FnMut() -> T,
        T: PartialEq + std::fmt::Debug,
    {
        let result1 = f();
        let result2 = f();

        if result1 == result2 {
            DeterminismResult::Pass {
                name: name.to_string(),
            }
        } else {
            DeterminismResult::Fail {
                name: name.to_string(),
                first: format!("{:?}", result1),
                second: format!("{:?}", result2),
            }
        }
    }

    /// Verify deterministic serialization
    pub fn verify_serialization<T>(&self, name: &str, value: &T) -> DeterminismResult
    where
        T: serde::Serialize + PartialEq + std::fmt::Debug,
    {
        let json1 = serde_json::to_string(value).unwrap();
        let json2 = serde_json::to_string(value).unwrap();

        if json1 == json2 {
            DeterminismResult::Pass {
                name: name.to_string(),
            }
        } else {
            DeterminismResult::Fail {
                name: name.to_string(),
                first: json1,
                second: json2,
            }
        }
    }
}

/// Golden file regression testing
pub struct GoldenFileTest {
    pub test_name: String,
    pub golden_dir: std::path::PathBuf,
    pub update: bool,
}

impl GoldenFileTest {
    pub fn new(test_name: &str, golden_dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            test_name: test_name.to_string(),
            golden_dir: golden_dir.as_ref().to_path_buf(),
            update: false,
        }
    }

    pub fn with_update(mut self, update: bool) -> Self {
        self.update = update;
        self
    }

    pub fn test<F, T>(&self, mut compute: F) -> GoldenResult
    where
        F: FnMut() -> T,
        T: serde::Serialize + DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let output = compute();
        let golden_path = self.golden_dir.join(format!("{}.golden.json", self.test_name));

        if self.update || !golden_path.exists() {
            std::fs::create_dir_all(&self.golden_dir).ok();
            let json = serde_json::to_string_pretty(&output).unwrap();
            std::fs::write(&golden_path, json).unwrap();
            return GoldenResult::Updated { path: golden_path };
        }

        let golden_json = std::fs::read_to_string(&golden_path).unwrap();
        let golden: T = serde_json::from_str(&golden_json).unwrap();

        if output == golden {
            GoldenResult::Pass { path: golden_path }
        } else {
            GoldenResult::Fail {
                path: golden_path,
                computed: format!("{:?}", output),
                expected: format!("{:?}", golden),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoldenResult {
    Pass { path: std::path::PathBuf },
    Fail { path: std::path::PathBuf, computed: String, expected: String },
    Updated { path: std::path::PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeterminismResult {
    Pass { name: String },
    Fail { name: String, first: String, second: String },
}

impl DeterminismResult {
    pub fn passes(&self) -> bool {
        matches!(self, DeterminismResult::Pass { .. })
    }
}

/// Property-based testing configuration
pub struct ProptestConfig {
    pub cases: usize,
    pub max_shrink_iters: usize,
}

impl Default for ProptestConfig {
    fn default() -> Self {
        Self {
            cases: 1000,
            max_shrink_iters: 100,
        }
    }
}

/// Fuzzing configuration
pub struct FuzzConfig {
    pub runs: usize,
    pub max_len: usize,
    pub timeout_secs: u64,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            runs: 10000,
            max_len: 4096,
            timeout_secs: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism_pass() {
        let validator = DeterminismValidator::new(42);
        let result = validator.verify_deterministic("test", || 42);
        assert!(result.passes());
    }

    #[test]
    fn test_determinism_fail() {
        let mut counter = 0;
        let validator = DeterminismValidator::new(42);
        let result = validator.verify_deterministic("test", || {
            counter += 1;
            counter
        });
        assert!(!result.passes());
    }

    #[test]
    fn test_serialization_deterministic() {
        let validator = DeterminismValidator::new(42);
        let result = validator.verify_serialization("test", &serde_json::json!({"a": 1, "b": 2}));
        assert!(result.passes());
    }
}