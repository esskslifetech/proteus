//! V3: Physical/numerical validation — conservation laws, numerical bounds

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct PhysicalValidator;

impl PhysicalValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate mass/charge balance in metabolic models
    pub fn validate_mass_balance(&self, reaction: &MetabolicReaction) -> ValidationResult {
        let mut element_counts = std::collections::HashMap::new();

        for (metabolite, stoich) in &reaction.reactants {
            let formula = parse_chemical_formula(metabolite);
            for (elem, count) in formula {
                *element_counts.entry(elem).or_insert(0) += count * *stoich;
            }
        }

        for (metabolite, stoich) in &reaction.products {
            let formula = parse_chemical_formula(metabolite);
            for (elem, count) in formula {
                *element_counts.entry(elem).or_insert(0) -= count * *stoich;
            }
        }

        for (elem, balance) in element_counts {
            if balance != 0 {
                return ValidationResult::Fail {
                    check: "mass_balance".to_string(),
                    message: format!("Element {} not balanced: net {}", elem, balance),
                };
            }
        }

        ValidationResult::Pass {
            check: "mass_balance".to_string(),
        }
    }

    /// Validate charge balance
    pub fn validate_charge_balance(&self, reaction: &MetabolicReaction) -> ValidationResult {
        let reactant_charge: i32 = reaction
            .reactants
            .iter()
            .map(|(m, s)| get_charge(m) * *s)
            .sum();
        let product_charge: i32 = reaction
            .products
            .iter()
            .map(|(m, s)| get_charge(m) * *s)
            .sum();

        if reactant_charge != product_charge {
            return ValidationResult::Fail {
                check: "charge_balance".to_string(),
                message: format!(
                    "Charge not balanced: reactants={}, products={}",
                    reactant_charge, product_charge
                ),
            };
        }

        ValidationResult::Pass {
            check: "charge_balance".to_string(),
        }
    }

    /// Validate numerical bounds (NaN, Inf, intervals)
    pub fn validate_numerical(
        &self,
        values: &[f64],
        name: &str,
        bounds: Option<(f64, f64)>,
    ) -> ValidationResult {
        for (i, &v) in values.iter().enumerate() {
            if v.is_nan() {
                return ValidationResult::Fail {
                    check: "nan_check".to_string(),
                    message: format!("{}[{}] is NaN", name, i),
                };
            }
            if v.is_infinite() {
                return ValidationResult::Fail {
                    check: "inf_check".to_string(),
                    message: format!("{}[{}] is infinite", name, i),
                };
            }
            if let Some((min, max)) = bounds {
                if v < min || v > max {
                    return ValidationResult::Fail {
                        check: "bounds_check".to_string(),
                        message: format!(
                            "{}[{}] = {} outside bounds [{}, {}]",
                            name, i, v, min, max
                        ),
                    };
                }
            }
        }
        ValidationResult::Pass {
            check: "numerical".to_string(),
        }
    }

    /// Validate solver convergence
    pub fn validate_solver_convergence(
        &self,
        status: SolverStatus,
        tolerance: f64,
    ) -> ValidationResult {
        match status {
            SolverStatus::Optimal => ValidationResult::Pass {
                check: "solver".to_string(),
            },
            SolverStatus::Suboptimal { gap } if gap <= tolerance => ValidationResult::Pass {
                check: "solver".to_string(),
            },
            SolverStatus::Suboptimal { gap } => ValidationResult::Fail {
                check: "solver".to_string(),
                message: format!(
                    "Solver suboptimal with gap {} > tolerance {}",
                    gap, tolerance
                ),
            },
            SolverStatus::Infeasible => ValidationResult::Fail {
                check: "solver".to_string(),
                message: "Solver reported infeasible".to_string(),
            },
            SolverStatus::Unbounded => ValidationResult::Fail {
                check: "solver".to_string(),
                message: "Solver reported unbounded".to_string(),
            },
            SolverStatus::NumericalError => ValidationResult::Fail {
                check: "solver".to_string(),
                message: "Solver reported numerical error".to_string(),
            },
        }
    }
}

/// Chemical formula parsing (simplified)
fn parse_chemical_formula(formula: &str) -> HashMap<String, i32> {
    let mut counts = HashMap::new();
    let mut i = 0;
    let bytes = formula.as_bytes();

    while i < bytes.len() {
        if bytes[i].is_ascii_uppercase() {
            let elem = bytes[i] as char;
            i += 1;
            let mut count_str = String::new();
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                count_str.push(bytes[i] as char);
                i += 1;
            }
            let count = count_str.parse().unwrap_or(1);
            *counts.entry(elem.to_string()).or_insert(0) += count;
        } else {
            i += 1;
        }
    }
    counts
}

fn get_charge(metabolite: &str) -> i32 {
    // Simplified charge lookup
    if metabolite.ends_with("^-") {
        -1
    } else if metabolite.ends_with("^2-") {
        -2
    } else if metabolite.ends_with("^+") {
        1
    } else if metabolite.ends_with("^2+") {
        2
    } else {
        0
    }
}

#[derive(Debug, Clone)]
pub struct MetabolicReaction {
    pub id: String,
    pub reactants: HashMap<String, i32>,
    pub products: HashMap<String, i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolverStatus {
    Optimal,
    Suboptimal { gap: f64 },
    Infeasible,
    Unbounded,
    NumericalError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    Pass { check: String },
    Fail { check: String, message: String },
}

impl ValidationResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, ValidationResult::Pass { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_formula() {
        let counts = parse_chemical_formula("C6H12O6");
        assert_eq!(counts.get("C"), Some(&6));
        assert_eq!(counts.get("H"), Some(&12));
        assert_eq!(counts.get("O"), Some(&6));
    }

    #[test]
    fn test_numerical_validation() {
        let validator = PhysicalValidator::new();
        let result = validator.validate_numerical(&[1.0, 2.0, 3.0], "test", Some((0.0, 10.0)));
        assert!(result.is_pass());

        let result = validator.validate_numerical(&[1.0, f64::NAN, 3.0], "test", None);
        assert!(!result.is_pass());
    }
}
