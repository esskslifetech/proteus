# Proteus Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records for Proteus.

## ADR-001: Polyglot Architecture (Rust/C++/Python)

**Status:** Accepted
**Date:** 2024-11-18

### Context

Proteus requires:
- Memory safety for genome state manipulation (critical correctness)
- High-performance numerical computation (ODE solvers, FBA, SIMD stats)
- ML inference and orchestration (Python ecosystem)
- User-facing CLI and API (Python ecosystem)

### Decision

We adopt a polyglot architecture:
- **Rust** owns correctness-critical state (genome, variant algebra, validation)
- **C++** owns number crunching (ODE/DAE, FBA, SIMD, BLAS/LAPACK)
- **Python** owns brains and glue (orchestration, API, ML, reports)

### Consequences

**Pros:**
- Memory safety where it matters most (genome state)
- Mature HPC libraries available in C++ (SUNDIALS, HiGHS, Eigen)
- Full ML ecosystem access (PyTorch, transformers)
- User-facing surface in Python (Typer, FastAPI)

**Cons:**
- Three languages to maintain
- Cross-language boundaries require careful ABI design
- Build complexity (Cargo, CMake, Python packaging)

### Alternatives Considered

1. **Pure Rust:** Would miss mature C++ HPC libraries (SUNDIALS, HiGHS)
2. **Pure Python:** Would be too slow for ODE/FBA solvers
3. **C++ with Python bindings only:** Would miss Rust's memory safety for genome state