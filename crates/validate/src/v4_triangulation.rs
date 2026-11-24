//! V4: Triangulation — cross-implementation agreement

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Triangulator {
    tolerance: f64,
}

impl Triangulator {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Compare results from multiple implementations
    pub fn compare(&self, name: &str, results: &[ImplementationResult]) -> TriangulationResult {
        if results.len() < 2 {
            return TriangulationResult::InsufficientData {
                name: name.to_string(),
                count: results.len(),
            };
        }

        // Extract numeric values
        let values: Vec<f64> = results.iter().filter_map(|r| r.value).collect();

        if values.len() < 2 {
            return TriangulationResult::InsufficientData {
                name: name.to_string(),
                count: values.len(),
            };
        }

        // Compute statistics
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let max_diff: f64 = values
            .iter()
            .map(|v| (v - mean).abs())
            .fold(0.0_f64, f64::max);

        let passes = max_diff <= self.tolerance;

        TriangulationResult::Compared {
            name: name.to_string(),
            implementations: results.iter().map(|r| r.name.clone()).collect(),
            mean,
            std_dev,
            max_deviation: max_diff,
            tolerance: self.tolerance,
            passes,
        }
    }

    /// Compare arrays element-wise
    pub fn compare_arrays(&self, name: &str, results: &[ArrayResult]) -> TriangulationResult {
        if results.len() < 2 {
            return TriangulationResult::InsufficientData {
                name: name.to_string(),
                count: results.len(),
            };
        }

        let len = results[0].values.len();
        if !results.iter().all(|r| r.values.len() == len) {
            return TriangulationResult::MismatchedShapes {
                name: name.to_string(),
            };
        }

        let mut max_diff: f64 = 0.0;
        let mut mean_diff: f64 = 0.0;
        let mut count: usize = 0;

        for i in 0..len {
            let vals: Vec<f64> = results.iter().map(|r| r.values[i]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            for v in &vals {
                let diff = (v - mean).abs();
                max_diff = max_diff.max(diff);
                mean_diff += diff;
                count += 1;
            }
        }

        mean_diff /= count as f64;
        let passes = max_diff <= self.tolerance;

        TriangulationResult::ArrayCompared {
            name: name.to_string(),
            implementations: results.iter().map(|r| r.name.clone()).collect(),
            length: len,
            max_deviation: max_diff,
            mean_deviation: mean_diff,
            tolerance: self.tolerance,
            passes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImplementationResult {
    pub name: String,
    pub value: Option<f64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ArrayResult {
    pub name: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriangulationResult {
    Compared {
        name: String,
        implementations: Vec<String>,
        mean: f64,
        std_dev: f64,
        max_deviation: f64,
        tolerance: f64,
        passes: bool,
    },
    ArrayCompared {
        name: String,
        implementations: Vec<String>,
        length: usize,
        max_deviation: f64,
        mean_deviation: f64,
        tolerance: f64,
        passes: bool,
    },
    InsufficientData {
        name: String,
        count: usize,
    },
    MismatchedShapes {
        name: String,
    },
}

impl TriangulationResult {
    pub fn passes(&self) -> bool {
        match self {
            TriangulationResult::Compared { passes, .. } => *passes,
            TriangulationResult::ArrayCompared { passes, .. } => *passes,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulation_pass() {
        let triangulator = Triangulator::new(1e-6);
        let results = vec![
            ImplementationResult {
                name: "rust".to_string(),
                value: Some(1.0000001),
                metadata: Default::default(),
            },
            ImplementationResult {
                name: "cpp".to_string(),
                value: Some(1.0000002),
                metadata: Default::default(),
            },
            ImplementationResult {
                name: "python".to_string(),
                value: Some(0.9999999),
                metadata: Default::default(),
            },
        ];
        let result = triangulator.compare("test", &results);
        assert!(result.passes());
    }

    #[test]
    fn test_triangulation_fail() {
        let triangulator = Triangulator::new(1e-6);
        let results = vec![
            ImplementationResult {
                name: "rust".to_string(),
                value: Some(1.0),
                metadata: Default::default(),
            },
            ImplementationResult {
                name: "cpp".to_string(),
                value: Some(1.1),
                metadata: Default::default(),
            },
        ];
        let result = triangulator.compare("test", &results);
        assert!(!result.passes());
    }
}
