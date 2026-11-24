# Proteus: Human Genome Mutation Simulation & Viability Analysis

> **Core loop:** load reference → inject variant(s) → simulate reference & mutated across L1–L4 → time-dynamic viability (survival window, point of no return, hysteresis) → revert → per-layer diff → report + evidence bundle

Polyglot (Rust / C++ / Python) system that simulates human DNA mutations across biological layers with event-sourced exact revert, multi-layer validation (V1–V7), and auditable evidence-bundle reporting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Python — orchestration, layers, validation, reports             │
│  Typer CLI · FastAPI · Jinja2 templates · RO-Crate bundle        │
├──────────────────┬──────────────────────────────────────────────┤
│  Rust (core)     │  C++ (kernels, optional)                    │
│  · Genome state  │  · ODE integration (SUNDIALS CVODE / RK4)   │
│  · Variant algebra│  · ΔΔG energy functions                    │
│  · Event log     │  · SIMD statistics (mean, stddev)           │
│  · Exact revert  │  · FBA solver (HiGHS, optional)             │
│  · Content hash  │                                            │
├──────────────────┴──────────────────────────────────────────────┤
│  Glue: PyO3/maturin (Rust→Python) · pybind11 (C++→Python)       │
└─────────────────────────────────────────────────────────────────┘
```

## Simulation Layers

| Layer | Scope | Engine | Fidelity |
|-------|-------|--------|----------|
| **L0** Sequence | DNA/RNA edits | Rust exact string/interval algebra | Exact |
| **L1** Variant semantics | Coding, splicing, regulatory | Deterministic surrogate (rule-based) | Statistical |
| **L2** Protein | Stability (ΔΔG), domains, structure | C++ FoldX-style energy / Python fallback | Approximate |
| **L3** Pathway/Cell | Metabolism, GRN, ODE dynamics | C++ CVODE / RK4 fallback | Model-based |
| **L4** Tissue/Organ | Physiology, neuro proxies | Python compartment model | Coarse |
| **L5** Viability | Composite survival gate | Weighted evidence fusion | Decision layer |

> **Neuro proxy limitation:** L4 tracks named neurophysiological proxies (EEG spectral power, PCI, GNW activation, Φ approximations) as measurable proxies, **not** consciousness or subjective experience. Every report includes an explicit limitations statement.

### Time-dynamic viability (L3b)

L3b integrates mRNA/protein turnover with literature-scale half-lives (mRNA t½ ≈ 9 h, protein t½ ≈ 46 h) and tracks irreversible damage in five categories: energy depletion, proteostasis collapse, oxidative stress, DNA damage, and mitochondrial dysfunction. The pipeline reports:

- **Survival window** — time of the first irreversible event.
- **Point of no return** — the earliest damage type whose threshold duration is crossed.
- **Hysteresis** — after an exact genome revert, irreversible damage persists. The post-revert phenotype is re-simulated and compared with baseline, and a residual viability ceiling is reported (up to 50% loss at maximal residual damage).

Turnover is calibrated to the supplied initial concentrations so an unperturbed baseline is at steady state; variant effects enter as multiplicative synthesis folds (0 = knockout, 0.5 = 50% knockdown).

## Validation Layers (V1–V7)

| # | Layer | Check |
|---|-------|-------|
| **V1** | Schema | Variant event fields valid, type/coordinate bounds |
| **V2** | Semantic | Ref-allele match against reference genome |
| **V3** | Physical | All numeric outputs finite and in range |
| **V4** | Triangulation | Rust core vs Python agree on variant count |
| **V5** | Determinism | Seeded surrogates, event replay identical |
| **V6** | Scientific | Plausibility ranges for pathogenicity, ΔΔG, flux |
| **V7** | Reproducibility | Content-addressed inputs, exact revert verified |

---

## Prerequisites

- **Rust** ≥ 1.80 (`rustup install stable`)
- **Python** ≥ 3.11
- **uv** (recommended for environment management) — `pip install uv`
- **CMake** ≥ 3.20
- **MSYS2** (MinGW64) — for C++ build (Eigen3, SUNDIALS, pybind11, Ninja)
- **Visual Studio Build Tools** (optional, for MSVC C++ build targeting the project's CPython venv)

## Build

### 1. Rust Workspace

```powershell
cd proteus
cargo build --workspace --release
cargo test --workspace           # 39 tests: 25 core + 14 validate
```

### 2. Rust → Python Bindings (PyO3 / maturin)

```powershell
# Install maturin (one-time)
pip install maturin

# Build into the uv venv
cd crates\py-proteus
$env:VIRTUAL_ENV = "C:\path\to\proteus\python\.venv"
maturin develop --release
```

### 3. Python Package

```powershell
cd proteus\python
uv sync
uv run pytest -q                 # 37 passed, 3 skipped (optional C++/SUNDIALS tests skipped when not built)
```

### 4. C++ — MSYS2 Build (MinGW, Python 3.14)

Builds against the MSYS2-provided Python (3.14). Use when you want SUNDIALS CVODE.

```powershell
C:\msys64\usr\bin\bash.exe -lc @"
export PATH=/mingw64/bin:/usr/bin:/bin:$PATH
cd /c/path/to/proteus/cpp
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DEigen3_DIR=/mingw64/share/eigen3/cmake \
  -Dpybind11_DIR=/mingw64/lib/cmake/pybind11 \
  -DSundials_DIR=/mingw64/lib/cmake/sundials \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build . -j4
"@
```

Produces: `build/bin/libproteus_kernels.dll`, `build/bin/libproteus_solvers.dll`, `build/lib/proteus_cpp.cp314-*.pyd`

### 5. C++ — MSVC Build (targeting the project's CPython 3.11 venv)

Builds against MSVC so the `.pyd` is compatible with the project's uv venv. Uses the **RK4 fallback** (SUNDIALS not available).

```powershell
# Requires: Visual Studio 2022+ Build Tools, MSYS2 (for Eigen3 headers)
cd proteus\cpp
cmake -B build-msvc -G "Visual Studio 17 2022" -A x64 ^
  -DEigen3_DIR=C:/msys64/mingw64/share/eigen3/cmake ^
  -Dpybind11_DIR=<venv>/Lib/site-packages/pybind11/share/cmake/pybind11 ^
  -DPYBIND11_FINDPYTHON=ON ^
  -DPython_EXECUTABLE=<venv>/Scripts/python.exe ^
  -DCMAKE_DISABLE_FIND_PACKAGE_Sundials=ON ^
  -DCMAKE_DISABLE_FIND_PACKAGE_HiGHS=ON
cmake --build build-msvc --config Release -j4
```

Produces: `build-msvc/lib/Release/proteus_cpp.cp311-win_amd64.pyd`

**The Python adapter auto-discovers both build directories.** No env vars needed.

## Usage

### Demo Script

```powershell
cd proteus\python
uv run python _demo.py
```

Output:
```
success: True
verdict: Verdict.VIABLE composite: 0.8914
validation: {'V1': 'pass', 'V2': 'pass', 'V3': 'pass', 'V4': 'pass', 'V5': 'pass', 'V6': 'pass', 'V7': 'pass'}
baseline == reverted: True

--- layer diffs ---
  L0: 1 added variants
  L1: 3 changed fields
    max_pathogenicity: 0 -> 0.562615 (delta=0.562615, high)
    mean_pathogenicity: 0 -> 0.562615 (delta=0.562615, high)
    max_severity: 0 -> 5 (delta=5, high)
  L2: 3 changed fields
    max_delta_delta_g: 0 -> 1.93321 (delta=1.93321, high)
    ...
```

### Python API

```python
import asyncio
from proteus.pipeline import run_simulation
from proteus.schema import PipelineConfig, VariantEvent

cfg = PipelineConfig(
    reference_genome_uri="test_genome.fasta",
    variants=[VariantEvent(
        id="v1",
        type=VariantEvent.Type.SNV,
        contig="chr1", start=10, end=10,
        ref_allele="T", alt_allele="A",
        source="demo",
    )],
    produce_report=True,
    output_dir="output",
)
result = asyncio.run(run_simulation(cfg))
print(f"Viability: {result.report.viability.verdict}")
print(f"Baseline == Reverted: {result.report.baseline == result.report.reverted}")
print(f"Survival window: {result.report.survival_window_h} h")
print(f"Point of no return: {result.report.point_of_no_return_event}")
hysteresis = result.report.revert_hysteresis
print(f"Genome revert exact: {hysteresis.get('genome_revert_exact')}")
print(f"Phenotype matches baseline: {hysteresis.get('phenotype_matches_baseline')}")
print(f"Hysteresis index: {hysteresis.get('hysteresis_index')}")
```

## Coordinate Convention

All genomic coordinates are **1-based, inclusive** (VCF convention). Rust `variant.rs` converts internally to 0-based for sequence slicing (`start - 1`).

## Report Formats

| Format | Description |
|--------|-------------|
| **HTML** | Interactive report with fold-out evidence chains, layer results, diffs, and embedded PNG figures |
| **PDF** | Publication-ready document (ReportLab, genome/molecule figures embedded) |
| **LaTeX** | Source for custom compilation |
| **JSON-LD** | Schema.org + PROV-O linked data |
| **RO-Crate** | Research Object crate with `ro-crate-metadata.json`, `run_manifest.json`, `evidence.json`, and all PNG/audio/GIF artifacts by reference |

## Visualization Output

Each run produces genome/molecule-centric figures plus the full dashboard suite:

| File | Content |
|------|---------|
| `visualizations/genome_dna_browser.png` | Reference sequence with injected lesions; contig ideogram; L1 gene consequences |
| `visualizations/genome_dna_helix.png` | Double helix with altered base pairs highlighted in gold/red |
| `visualizations/genome_central_dogma.png` | Codon translation derived directly from L0 reference windows |
| `visualizations/genome_molecular_cascade.png` | Layer-by-layer engine provenance and carried-forward metrics |
| `visualizations/genome_inputs_provenance.png` | What references, model hooks, and compute engines were actually used |
| `visualizations/genome_glycolysis_map.png` | Pathway state with expression-driven color coding |
| `visualizations/dashboard_*.png` | Standard dashboards (viability radar, heatmap, variants, energy, conservation, validation, composite) |
| `visualizations/animation_*.gif` | Mutation-effect and layer-propagation animations |
| `visualizations/sonification_*.wav` | Variant and pathway audio sonifications |
| `report.html` / `report.pdf` | All PNGs embedded; GIFs and WAVs linked by reference |

The pipeline config controls generation via `generate_visualizations: bool = True`; set to `False` for fast CI runs.

## C++ Extension Status

```python
from proteus import cpp_adapter
status = cpp_adapter.cpp_status()
print(f"Available: {status['available']}")
print(f"Module: {status['module_file']}")
```

The extension is optional. Pure-Python fallbacks are used when unavailable. The C++ extension provides:
- **ΔΔG energy function** (FoldX-style, L2 protein stability)
- **ODE integration** (SUNDIALS CVODE with MSYS2 build, RK4 fallback with MSVC build)
- **SIMD statistics** (mean, stddev — useful for benchmarking)

## Limitations

- Variant predictors are **deterministic surrogates** structured for drop-in replacement by trained models (AlphaMissense, ESMFold, FoldX, Human-GEM, PBPK).
- Neuro proxies (L4) model measurable physiologic signals, **not** consciousness.
- Without SUNDIALS, ODE integration uses a fixed-step classical RK4 (suitable for non-stiff pathway models).
- HiGHS LP solver is absent from MSYS2 repos; FBA returns `NumericalError("HiGHS not available")` until available.
- Python `GenomeState` equality (`==`) compares `reference`, `overlays`, and `head_index` structurally.
- Time-dynamic turnover is calibrated to the supplied initial concentrations; absolute concentrations are arbitrary units and only relative dynamics are interpretable.
- Post-revert hysteresis freezes irreversible damage at its point-of-no-return severity and applies a model-assumption viability ceiling (`1 - 0.5 × hysteresis_index`); it is not a measured recovery limit.

## License

Apache-2.0
