//! py-proteus: PyO3 bindings for proteus-core

use pyo3::prelude::*;
// Import core types directly to avoid ambiguity from proteus_validate re-exports
use ::proteus_core::reference::{ReferenceGenome as CoreReferenceGenome, ReferenceGenomeBuilder as CoreReferenceGenomeBuilder};
use ::proteus_core::state::{GenomeState as CoreGenomeState, LayerDiff as CoreLayerDiff};
use ::proteus_core::variant::{VariantEvent as CoreVariantEvent, VariantType as CoreVariantType};
use std::sync::Arc;
use hex;

/// PyO3 wrapper for the core Rust genome state
#[pyclass]
pub struct PyGenomeState {
    inner: CoreGenomeState,
}

#[pymethods]
impl PyGenomeState {
    #[new]
    fn new(reference: &PyReferenceGenome) -> Self {
        Self {
            inner: CoreGenomeState::new(reference.inner.clone()),
        }
    }

    fn apply(&self, event: &PyVariantEvent) -> PyResult<PyGenomeState> {
        let result = self.inner.apply(event.inner.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyGenomeState { inner: result })
    }

    fn apply_batch(&self, events: Vec<PyVariantEvent>) -> PyResult<PyGenomeState> {
        let mut result = self.inner.clone();
        for event in events {
            result = result.apply(event.inner.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        Ok(PyGenomeState { inner: result })
    }

    fn revert(&self, target: usize) -> PyResult<PyGenomeState> {
        let result = self.inner.revert(target)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyGenomeState { inner: result })
    }

    fn revert_all(&self) -> PyGenomeState {
        PyGenomeState {
            inner: self.inner.revert_all(),
        }
    }

    fn revert_last(&self, n: usize) -> PyResult<PyGenomeState> {
        let result = self.inner.revert_last(n)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyGenomeState { inner: result })
    }

    fn applied_variants(&self) -> Vec<PyVariantEvent> {
        self.inner.applied_variants().iter()
            .map(|v| PyVariantEvent { inner: v.clone() })
            .collect()
    }

    fn variant_count(&self) -> usize {
        self.inner.variant_count()
    }

    fn get_sequence(&self, contig: &str) -> PyResult<Vec<u8>> {
        self.inner.get_sequence(contig)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn diff(&self, other: &PyGenomeState) -> PyLayerDiff {
        PyLayerDiff {
            inner: self.inner.diff(&other.inner),
        }
    }

    fn hash(&self) -> String {
        hex::encode(self.inner.hash().as_bytes())
    }

    #[getter]
    fn reference(&self) -> PyReferenceGenome {
        PyReferenceGenome {
            inner: self.inner.reference().clone(),
        }
    }
}

/// PyO3 wrapper for reference genome
#[pyclass]
pub struct PyReferenceGenome {
    inner: Arc<CoreReferenceGenome>,
}

#[pymethods]
impl PyReferenceGenome {
    #[new]
    fn new(assembly: String, contigs: Vec<(String, Vec<u8>)>) -> Self {
        let builder = CoreReferenceGenomeBuilder::new(assembly);
        let builder = contigs.into_iter().fold(builder, |b, (name, seq)| b.add_contig(name, seq));
        Self {
            inner: Arc::new(builder.build()),
        }
    }

    fn contig(&self, name: &str) -> Option<Vec<u8>> {
        self.inner.contig(name).map(|c| c.sequence.clone())
    }

    fn contig_names(&self) -> Vec<String> {
        self.inner.contig_names().to_vec()
    }

    fn total_length(&self) -> usize {
        self.inner.total_length()
    }

    fn hash(&self) -> String {
        hex::encode(self.inner.hash().as_bytes())
    }

    fn verify(&self) -> bool {
        self.inner.verify()
    }

    #[getter]
    fn assembly(&self) -> String {
        self.inner.assembly.clone()
    }
}

/// PyO3 wrapper for variant event
#[pyclass]
#[derive(Clone)]
pub struct PyVariantEvent {
    inner: CoreVariantEvent,
}

#[pymethods]
impl PyVariantEvent {
    #[new]
    fn new(
        id: String,
        variant_type: PyVariantType,
        contig: String,
        start: i64,
        end: i64,
        ref_allele: String,
        alt_allele: String,
        source: String,
    ) -> Self {
        Self {
            inner: CoreVariantEvent {
                id,
                ty: variant_type.into(),
                contig,
                start,
                end,
                ref_allele,
                alt_allele,
                annotations: Default::default(),
                timestamp_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as i64,
                source,
            },
        }
    }

    #[staticmethod]
    fn snv(id: String, contig: String, pos: i64, ref_allele: String, alt_allele: String, source: String) -> Self {
        Self {
            inner: CoreVariantEvent::snv(id, contig, pos, ref_allele, alt_allele, source),
        }
    }

    #[staticmethod]
    fn insertion(id: String, contig: String, pos: i64, inserted_seq: String, source: String) -> Self {
        Self {
            inner: CoreVariantEvent::insertion(id, contig, pos, inserted_seq, source),
        }
    }

    #[staticmethod]
    fn deletion(id: String, contig: String, start: i64, end: i64, deleted_seq: String, source: String) -> Self {
        Self {
            inner: CoreVariantEvent::deletion(id, contig, start, end, deleted_seq, source),
        }
    }

    fn length_delta(&self) -> i64 {
        self.inner.length_delta()
    }

    fn covers(&self, contig: &str, pos: i64) -> bool {
        self.inner.covers(contig, pos)
    }

    #[getter]
    fn variant_type(&self) -> PyVariantType {
        self.inner.ty.into()
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn contig(&self) -> String {
        self.inner.contig.clone()
    }

    #[getter]
    fn start(&self) -> i64 {
        self.inner.start
    }

    #[getter]
    fn end(&self) -> i64 {
        self.inner.end
    }

    #[getter]
    fn ref_allele(&self) -> String {
        self.inner.ref_allele.clone()
    }

    #[getter]
    fn alt_allele(&self) -> String {
        self.inner.alt_allele.clone()
    }

    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    fn timestamp_ns(&self) -> i64 {
        self.inner.timestamp_ns
    }
}

/// Variant type enum - re-export from core with PyO3 support
#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyVariantType {
    Snv,
    Mnv,
    Insertion,
    Deletion,
    Delins,
    Inversion,
    Duplication,
    Translocation,
    Cnv,
    StrExpansion,
    RnaEdit,
    SpliceSite,
}

impl From<CoreVariantType> for PyVariantType {
    fn from(v: CoreVariantType) -> Self {
        match v {
            CoreVariantType::Snv => PyVariantType::Snv,
            CoreVariantType::Mnv => PyVariantType::Mnv,
            CoreVariantType::Insertion => PyVariantType::Insertion,
            CoreVariantType::Deletion => PyVariantType::Deletion,
            CoreVariantType::Delins => PyVariantType::Delins,
            CoreVariantType::Inversion => PyVariantType::Inversion,
            CoreVariantType::Duplication => PyVariantType::Duplication,
            CoreVariantType::Translocation => PyVariantType::Translocation,
            CoreVariantType::Cnv => PyVariantType::Cnv,
            CoreVariantType::StrExpansion => PyVariantType::StrExpansion,
            CoreVariantType::RnaEdit => PyVariantType::RnaEdit,
            CoreVariantType::SpliceSite => PyVariantType::SpliceSite,
        }
    }
}

impl From<PyVariantType> for CoreVariantType {
    fn from(v: PyVariantType) -> Self {
        match v {
            PyVariantType::Snv => CoreVariantType::Snv,
            PyVariantType::Mnv => CoreVariantType::Mnv,
            PyVariantType::Insertion => CoreVariantType::Insertion,
            PyVariantType::Deletion => CoreVariantType::Deletion,
            PyVariantType::Delins => CoreVariantType::Delins,
            PyVariantType::Inversion => CoreVariantType::Inversion,
            PyVariantType::Duplication => CoreVariantType::Duplication,
            PyVariantType::Translocation => CoreVariantType::Translocation,
            PyVariantType::Cnv => CoreVariantType::Cnv,
            PyVariantType::StrExpansion => CoreVariantType::StrExpansion,
            PyVariantType::RnaEdit => CoreVariantType::RnaEdit,
            PyVariantType::SpliceSite => CoreVariantType::SpliceSite,
        }
    }
}

/// PyO3 wrapper for layer diff
#[pyclass]
pub struct PyLayerDiff {
    inner: CoreLayerDiff,
}

#[pymethods]
impl PyLayerDiff {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn total_changes(&self) -> usize {
        self.inner.total_changes()
    }

    #[getter]
    fn reference_changed(&self) -> bool {
        self.inner.reference_changed
    }

    #[getter]
    fn added_variants(&self) -> Vec<PyVariantEvent> {
        self.inner.added_variants.iter()
            .map(|v| PyVariantEvent { inner: v.clone() })
            .collect()
    }

    #[getter]
    fn removed_variants(&self) -> Vec<PyVariantEvent> {
        self.inner.removed_variants.iter()
            .map(|v| PyVariantEvent { inner: v.clone() })
            .collect()
    }

    #[getter]
    fn changed_variants(&self) -> Vec<(PyVariantEvent, PyVariantEvent)> {
        self.inner.changed_variants.iter()
            .map(|(a, b)| (PyVariantEvent { inner: a.clone() }, PyVariantEvent { inner: b.clone() }))
            .collect()
    }
}

/// Python module
#[pymodule]
fn proteus_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGenomeState>()?;
    m.add_class::<PyReferenceGenome>()?;
    m.add_class::<PyVariantEvent>()?;
    m.add_class::<PyVariantType>()?;
    m.add_class::<PyLayerDiff>()?;
    Ok(())
}
