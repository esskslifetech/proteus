//! V6: Scientific benchmarking — ClinVar, gnomAD, curated datasets

use crate::error::ValidationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    pub name: String,
    pub variants: Vec<BenchmarkVariant>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkVariant {
    pub id: String,
    pub chrom: String,
    pub pos: i64,
    pub ref_allele: String,
    pub alt_allele: String,
    pub gene: Option<String>,
    pub consequence: Option<String>,
    pub clinical_significance: ClinicalSignificance,
    pub review_status: ReviewStatus,
    pub source: String,  // "ClinVar", "gnomAD", "HGMD", etc.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClinicalSignificance {
    Pathogenic,
    LikelyPathogenic,
    UncertainSignificance,
    LikelyBenign,
    Benign,
    Conflicting,
    NotProvided,
}

impl ClinicalSignificance {
    pub fn is_pathogenic(&self) -> bool {
        matches!(self, ClinicalSignificance::Pathogenic | ClinicalSignificance::LikelyPathogenic)
    }

    pub fn is_benign(&self) -> bool {
        matches!(self, ClinicalSignificance::Benign | ClinicalSignificance::LikelyBenign)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewStatus {
    NoAssertion,
    NoCriteria,
    SingleSubmitter,
    MultipleSubmittersNoConflict,
    MultipleSubmittersConflict,
    ReviewedByExpertPanel,
    PracticeGuideline,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub dataset: String,
    pub layer: String,
    pub predictions: Vec<Prediction>,
    pub metrics: ClassificationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub variant_id: String,
    pub predicted_pathogenic: bool,
    pub score: f64,  // 0-1 pathogenicity score
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub auc_roc: f64,
    pub auc_pr: f64,
    pub matthews_corr: f64,
}

impl ClassificationMetrics {
    pub fn compute(predictions: &[Prediction], labels: &[bool]) -> Self {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (pred, &label) in predictions.iter().zip(labels) {
            if pred.predicted_pathogenic && label {
                tp += 1;
            } else if pred.predicted_pathogenic && !label {
                fp += 1;
            } else if !pred.predicted_pathogenic && label {
                fn_ += 1;
            } else {
                tn += 1;
            }
        }

        let accuracy = (tp + tn) as f64 / (tp + fp + tn + fn_) as f64;
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        // AUC-ROC (simplified using trapezoidal rule on score-threshold curve)
        let mut sorted: Vec<_> = predictions.iter().zip(labels).collect();
        sorted.sort_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap());

        let mut auc_roc = 0.0;
        let mut prev_fpr = 0.0;
        let mut prev_tpr = 0.0;
        let mut tp_cum = 0;
        let mut fp_cum = 0;
        let total_p = labels.iter().filter(|&&l| l).count();
        let total_n = labels.len() - total_p;

        for (_pred, &label) in &sorted {
            if label {
                tp_cum += 1;
            } else {
                fp_cum += 1;
            }
            let tpr = tp_cum as f64 / total_p as f64;
            let fpr = fp_cum as f64 / total_n as f64;
            auc_roc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_fpr = fpr;
            prev_tpr = tpr;
        }

        // AUC-PR
        let mut auc_pr = 0.0;
        let mut prev_recall = 0.0;
        let mut prev_precision = 1.0;
        tp_cum = 0;
        fp_cum = 0;

        for (_pred, &label) in &sorted {
            if label {
                tp_cum += 1;
            } else {
                fp_cum += 1;
            }
            let recall = tp_cum as f64 / total_p as f64;
            let precision = if tp_cum + fp_cum > 0 { tp_cum as f64 / (tp_cum + fp_cum) as f64 } else { 1.0 };
            auc_pr += (recall - prev_recall) * (precision + prev_precision) / 2.0;
            prev_recall = recall;
            prev_precision = precision;
        }

        // Matthews correlation coefficient
        let matthews_corr = if (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_) > 0 {
            ((tp * tn) as f64 - (fp * fn_) as f64)
                / (((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) as f64).sqrt()
        } else {
            0.0
        };

        Self {
            tp,
            fp,
            tn,
            fn_,
            accuracy,
            precision,
            recall,
            f1,
            auc_roc,
            auc_pr,
            matthews_corr,
        }
    }
}

/// Load ClinVar benchmark dataset (simplified - real implementation would parse VCF/TSV)
pub fn load_clinvar_benchmark(_path: &std::path::Path) -> Result<BenchmarkDataset, ValidationError> {
    // Placeholder - real implementation parses ClinVar VCF
    Ok(BenchmarkDataset {
        name: "ClinVar".to_string(),
        variants: Vec::new(),
        metadata: HashMap::from([
            ("version".to_string(), "2024-01".to_string()),
            ("assembly".to_string(), "GRCh38".to_string()),
        ]),
    })
}

/// Load gnomAD constraint metrics
pub fn load_gnomad_constraints(_path: &std::path::Path) -> Result<HashMap<String, GeneConstraint>, ValidationError> {
    // Placeholder - real implementation parses gnomAD constraint TSV
    Ok(HashMap::new())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneConstraint {
    pub gene: String,
    pub p_li: f64,  // probability of loss-of-function intolerance
    pub oe_lof: f64,  // observed/expected LOF
    pub oe_mis: f64,  // observed/expected missense
    pub z_score: f64,
}

/// Generate benchmark report card
pub fn generate_report_card(results: &[BenchmarkResult]) -> String {
    let mut report = String::new();
    report.push_str("# Proteus Benchmark Report Card\n\n");

    for result in results {
        report.push_str(&format!("## {} ({})\n\n", result.dataset, result.layer));
        report.push_str(&format!("| Metric | Value |\n"));
        report.push_str(&format!("|--------|-------|\n"));
        report.push_str(&format!("| Accuracy | {:.4} |\n", result.metrics.accuracy));
        report.push_str(&format!("| Precision | {:.4} |\n", result.metrics.precision));
        report.push_str(&format!("| Recall | {:.4} |\n", result.metrics.recall));
        report.push_str(&format!("| F1 | {:.4} |\n", result.metrics.f1));
        report.push_str(&format!("| AUC-ROC | {:.4} |\n", result.metrics.auc_roc));
        report.push_str(&format!("| AUC-PR | {:.4} |\n", result.metrics.auc_pr));
        report.push_str(&format!("| MCC | {:.4} |\n", result.metrics.matthews_corr));
        report.push_str("\n");
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_computation() {
        let predictions = vec![
            Prediction { variant_id: "v1".to_string(), predicted_pathogenic: true, score: 0.9, confidence: 0.8 },
            Prediction { variant_id: "v2".to_string(), predicted_pathogenic: true, score: 0.8, confidence: 0.7 },
            Prediction { variant_id: "v3".to_string(), predicted_pathogenic: false, score: 0.3, confidence: 0.6 },
            Prediction { variant_id: "v4".to_string(), predicted_pathogenic: false, score: 0.2, confidence: 0.9 },
        ];
        let labels = vec![true, false, false, true];

        let metrics = ClassificationMetrics::compute(&predictions, &labels);
        assert_eq!(metrics.tp, 1);
        assert_eq!(metrics.fp, 1);
        assert_eq!(metrics.tn, 1);
        assert_eq!(metrics.fn_, 1);
        assert!((metrics.accuracy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_clinical_significance() {
        assert!(ClinicalSignificance::Pathogenic.is_pathogenic());
        assert!(ClinicalSignificance::LikelyPathogenic.is_pathogenic());
        assert!(!ClinicalSignificance::UncertainSignificance.is_pathogenic());
        assert!(ClinicalSignificance::Benign.is_benign());
        assert!(ClinicalSignificance::LikelyBenign.is_benign());
    }
}