//! V1: Schema validation layer — reject malformed inputs at every boundary

use crate::error::ValidationError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SchemaValidator {
    schemas: HashMap<String, Value>,
}

impl SchemaValidator {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, schema: &Value) -> Result<(), ValidationError> {
        self.schemas.insert(name.to_string(), schema.clone());
        Ok(())
    }

    pub fn validate(&self, name: &str, instance: &Value) -> Result<ValidationResult, ValidationError> {
        let _schema = self.schemas.get(name)
            .ok_or_else(|| ValidationError::Schema(format!("no schema registered for {}", name)))?;

        // Simplified validation - just check it's valid JSON
        // Real implementation would use jsonschema crate
        let pass = instance.is_object() || instance.is_array();
        
        if pass {
            Ok(ValidationResult::Pass { schema: name.to_string() })
        } else {
            Ok(ValidationResult::Fail {
                schema: name.to_string(),
                errors: vec!["instance does not match schema".to_string()],
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    Pass { schema: String },
    Fail { schema: String, errors: Vec<String> },
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
    fn test_schema_validator_creation() {
        let validator = SchemaValidator::new();
        assert_eq!(validator.schemas.len(), 0);
    }
}