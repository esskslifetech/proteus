"""
Adapter layer: Convert between pure-Python schema types and Rust core (PyO3) types.

This allows the pipeline to use the fast, correct Rust core for genome state management
while keeping the rest of the Python code (schemas, API, reports) unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.schema import (
        GenomeState as PyGenomeState,
        VariantEvent as PyVariantEvent,
        ResourceRef,
        ContentHash,
        LayerDiff as PyLayerDiff,
    )

# Try to import Rust core
try:
    import proteus_core as pc
    RUST_CORE_AVAILABLE = True
except ImportError:
    RUST_CORE_AVAILABLE = False


def variant_type_py_to_rs(py_type):
    """Convert Python VariantEvent.Type to Rust VariantType."""
    from proteus.schema import VariantEvent as PyVariantEvent
    # Use string comparison since enum members aren't directly comparable
    py_str = str(py_type)
    if "SNV" in py_str:
        return pc.PyVariantType.Snv
    elif "MNV" in py_str:
        return pc.PyVariantType.Mnv
    elif "INSERTION" in py_str:
        return pc.PyVariantType.Insertion
    elif "DELETION" in py_str:
        return pc.PyVariantType.Deletion
    elif "DELINS" in py_str:
        return pc.PyVariantType.Delins
    elif "INVERSION" in py_str:
        return pc.PyVariantType.Inversion
    elif "DUPLICATION" in py_str:
        return pc.PyVariantType.Duplication
    elif "TRANSLOCATION" in py_str:
        return pc.PyVariantType.Translocation
    elif "CNV" in py_str:
        return pc.PyVariantType.Cnv
    elif "STR_EXPANSION" in py_str:
        return pc.PyVariantType.StrExpansion
    elif "RNA_EDIT" in py_str:
        return pc.PyVariantType.RnaEdit
    elif "SPLICE_SITE" in py_str:
        return pc.PyVariantType.SpliceSite
    return pc.PyVariantType.Snv


def variant_type_rs_to_py(rs_type):
    """Convert Rust VariantType to Python VariantEvent.Type."""
    from proteus.schema import VariantEvent as PyVariantEvent
    # Use string representation since enum members aren't hashable
    rs_str = str(rs_type)
    if "Snv" in rs_str:
        return PyVariantEvent.Type.SNV
    elif "Mnv" in rs_str:
        return PyVariantEvent.Type.MNV
    elif "Insertion" in rs_str:
        return PyVariantEvent.Type.INSERTION
    elif "Deletion" in rs_str:
        return PyVariantEvent.Type.DELETION
    elif "Delins" in rs_str:
        return PyVariantEvent.Type.DELINS
    elif "Inversion" in rs_str:
        return PyVariantEvent.Type.INVERSION
    elif "Duplication" in rs_str:
        return PyVariantEvent.Type.DUPLICATION
    elif "Translocation" in rs_str:
        return PyVariantEvent.Type.TRANSLOCATION
    elif "Cnv" in rs_str:
        return PyVariantEvent.Type.CNV
    elif "StrExpansion" in rs_str:
        return PyVariantEvent.Type.STR_EXPANSION
    elif "RnaEdit" in rs_str:
        return PyVariantEvent.Type.RNA_EDIT
    elif "SpliceSite" in rs_str:
        return PyVariantEvent.Type.SPLICE_SITE
    return PyVariantEvent.Type.SNV


def variant_py_to_rs(py_var: "PyVariantEvent"):
    """Convert Python VariantEvent to Rust VariantEvent using static methods."""
    if not RUST_CORE_AVAILABLE:
        raise RuntimeError("Rust core not available")
    
    vt = variant_type_py_to_rs(py_var.type)
    
    # Use the appropriate static method based on variant type
    if vt == pc.PyVariantType.Snv:
        return pc.PyVariantEvent.snv(
            py_var.id, py_var.contig, py_var.start, py_var.ref_allele, py_var.alt_allele, py_var.source
        )
    elif vt == pc.PyVariantType.Insertion:
        return pc.PyVariantEvent.insertion(
            py_var.id, py_var.contig, py_var.start, py_var.alt_allele, py_var.source
        )
    elif vt == pc.PyVariantType.Deletion:
        return pc.PyVariantEvent.deletion(
            py_var.id, py_var.contig, py_var.start, py_var.end, py_var.ref_allele, py_var.source
        )
    else:
        # Fallback to SNV
        return pc.PyVariantEvent.snv(
            py_var.id, py_var.contig, py_var.start, py_var.ref_allele, py_var.alt_allele, py_var.source
        )


def variant_rs_to_py(rs_var) -> "PyVariantEvent":
    """Convert Rust VariantEvent to Python VariantEvent."""
    from proteus.schema import VariantEvent as PyVariantEvent
    return PyVariantEvent(
        id=rs_var.id,
        type=variant_type_rs_to_py(rs_var.variant_type),
        contig=rs_var.contig,
        start=rs_var.start,
        end=rs_var.end,
        ref_allele=rs_var.ref_allele,
        alt_allele=rs_var.alt_allele,
        annotations={},
        timestamp_ns=0,
        source=rs_var.source if hasattr(rs_var, 'source') else "unknown",
    )


class RustGenomeStateAdapter:
    """
    Adapter that wraps a Rust GenomeState and provides the same interface
    as the pure-Python GenomeState class.
    """
    
    def __init__(self, rs_state):
        self._rs_state = rs_state
    
    @property
    def reference(self) -> "ResourceRef":
        """Get reference as ResourceRef."""
        from proteus.schema import ResourceRef, ContentHash
        # The Rust reference genome has a hash() method that returns a string
        ref_hash = self._rs_state.reference.hash()
        return ResourceRef(
            hash=ContentHash(algorithm="blake3", digest=ref_hash.encode() if isinstance(ref_hash, str) else ref_hash),
            uri=self._rs_state.reference.assembly if hasattr(self._rs_state.reference, 'assembly') else "",
        )
    
    @property
    def overlays(self) -> list["PyVariantEvent"]:
        """Get applied variants as Python VariantEvents."""
        return [variant_rs_to_py(v) for v in self._rs_state.applied_variants()]
    
    @property
    def head_index(self) -> int:
        return self._rs_state.variant_count()
    
    def hash(self) -> str:
        """Content-addressable hash (hex string)."""
        h = self._rs_state.hash()
        return h.hexdigest() if hasattr(h, 'hexdigest') else h
    
    def hash_bytes(self) -> bytes:
        """Raw 32-byte blake3 digest."""
        h = self._rs_state.hash()
        if hasattr(h, 'digest'):
            return h.digest()
        elif isinstance(h, str):
            return bytes.fromhex(h)
        else:
            # Fallback: convert to bytes
            return bytes(h) if isinstance(h, (bytes, bytearray)) else bytes.fromhex(str(h))
    
    def apply(self, variant: "PyVariantEvent") -> "RustGenomeStateAdapter":
        """Apply a variant and return new adapter."""
        rs_var = variant_py_to_rs(variant)
        new_rs_state = self._rs_state.apply(rs_var)
        return RustGenomeStateAdapter(new_rs_state)
    
    def apply_batch(self, variants: list["PyVariantEvent"]) -> "RustGenomeStateAdapter":
        """Apply multiple variants."""
        rs_vars = [variant_py_to_rs(v) for v in variants]
        new_rs_state = self._rs_state.apply_batch(rs_vars)
        return RustGenomeStateAdapter(new_rs_state)
    
    def revert(self, target: int) -> "RustGenomeStateAdapter":
        """Revert to a previous state."""
        new_rs_state = self._rs_state.revert(target)
        return RustGenomeStateAdapter(new_rs_state)
    
    def revert_all(self) -> "RustGenomeStateAdapter":
        """Revert all variants."""
        return RustGenomeStateAdapter(self._rs_state.revert_all())
    
    def revert_last(self, n: int) -> "RustGenomeStateAdapter":
        """Revert last N variants."""
        new_rs_state = self._rs_state.revert_last(n)
        return RustGenomeStateAdapter(new_rs_state)
    
    def get_sequence(self, contig: str) -> bytes:
        """Get sequence for a contig with all applied variants."""
        return self._rs_state.get_sequence(contig)
    
    def diff(self, other: "RustGenomeStateAdapter"):
        """Diff two genome states (returns Python LayerDiff)."""
        rs_diff = self._rs_state.diff(other._rs_state)
        return layer_diff_rs_to_py(rs_diff)
    
    def diff_raw(self, other: "RustGenomeStateAdapter"):
        """Diff two genome states (returns raw Rust PyLayerDiff)."""
        return self._rs_state.diff(other._rs_state)


def layer_diff_rs_to_py(rs_diff) -> "PyLayerDiff":
    """Convert Rust LayerDiff to Python LayerDiff."""
    from proteus.schema import LayerDiff as PyLayerDiff, DiffEntry
    py_diff = PyLayerDiff(
        layer="L0",
        added={},
        removed={},
        changed={},
    )
    py_diff.added = {v.id: str(v.id) for v in rs_diff.added_variants}
    py_diff.removed = {v.id: str(v.id) for v in rs_diff.removed_variants}
    py_diff.changed = {
        f"{before.id}->{after.id}": DiffEntry(
            before=str(before.id),
            after=str(after.id),
            magnitude=0.0,
            significance="variant_changed",
        )
        for before, after in rs_diff.changed_variants
    }
    return py_diff


def layer_diff_py_to_rs(py_diff: "PyLayerDiff"):
    """Convert Python LayerDiff to Rust LayerDiff."""
    if not RUST_CORE_AVAILABLE:
        raise RuntimeError("Rust core not available")
    # Will need to implement when Rust LayerDiff supports more fields
    return pc.PyLayerDiff()


def load_fasta_contigs(path: str) -> tuple[str, list[tuple[str, bytes]]]:
    """Parse a plain FASTA file into (assembly_name, [(contig, sequence_bytes)])."""
    from pathlib import Path

    contigs: list[tuple[str, bytes]] = []
    name: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if name is not None:
            contigs.append((name, "".join(chunks).upper().encode("ascii")))

    for raw in Path(path).read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].split()[0]
            name, chunks = header, []
        elif name is not None:
            chunks.append(line)
    flush()

    if not contigs:
        raise ValueError(f"no sequences found in FASTA file: {path}")
    return Path(path).stem, contigs


def create_rust_genome_state(reference_uri: str, variants: list["PyVariantEvent"] | None = None) -> RustGenomeStateAdapter:
    """
    Create a Rust-backed GenomeState from a reference genome URI.

    If ``reference_uri`` points to an existing FASTA file it is parsed; otherwise
    a small built-in test genome (``chr1`` = ``ATCGATCGATCG``) is used so the
    pipeline is runnable without external data.
    """
    if not RUST_CORE_AVAILABLE:
        raise RuntimeError("Rust core not available")

    from pathlib import Path

    if Path(reference_uri).is_file() and not reference_uri.lower().endswith((".gz", ".bgz")):
        assembly, contigs = load_fasta_contigs(reference_uri)
    else:
        assembly, contigs = "test", [("chr1", b"ATCGATCGATCG")]

    # PyReferenceGenome takes (assembly_name, list of (contig_name, sequence_bytes))
    ref_genome = pc.PyReferenceGenome(assembly, contigs)
    
    # Create genome state
    rs_state = pc.PyGenomeState(ref_genome)
    
    # Apply variants if provided
    if variants:
        rs_vars = [variant_py_to_rs(v) for v in variants]
        rs_state = rs_state.apply_batch(rs_vars)
    
    return RustGenomeStateAdapter(rs_state)