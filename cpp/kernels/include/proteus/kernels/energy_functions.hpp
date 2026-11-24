#pragma once

#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <optional>
#include <memory>

namespace proteus::kernels {

// Energy functions for protein stability (ΔΔG)

// Amino acid properties
struct AminoAcidProperties {
    char code;
    double hydrophobicity;   // Kyte-Doolittle
    double volume;           // Å³
    double charge;           // at pH 7
    double polarity;
    bool is_aromatic;
    bool is_polar;
    bool is_charged;
};

// Get properties for amino acid code
std::optional<AminoAcidProperties> get_amino_acid_properties(char code);

// Empirical ΔΔG prediction (simplified FoldX-like)
// Returns predicted ΔΔG in kcal/mol
double predict_ddg_empirical(
    const std::string& wild_type_seq,
    std::size_t position,  // 0-based
    char mutant_aa,
    const std::vector<double>& solvent_accessibility,  // per-residue RSA [0,1]
    const std::vector<int>& secondary_structure  // 0=coil, 1=helix, 2=sheet
);

// Statistical potential ΔΔG (knowledge-based)
double predict_ddg_statistical(
    const std::string& wild_type_seq,
    std::size_t position,
    char mutant_aa,
    const std::vector<std::array<double, 3>>& ca_coords  // Cα coordinates
);

// Rosetta-like energy terms (simplified)
struct RosettaEnergyTerms {
    double fa_atr;      // attractive van der Waals
    double fa_rep;      // repulsive van der Waals
    double fa_sol;      // solvation
    double fa_elec;     // electrostatic
    double hbond_sr;    // short-range hbonds
    double hbond_lr;    // long-range hbonds
    double rama;        // Ramachandran
    double p_aa_pp;     // amino acid probability
    double ref;         // reference energy
};

RosettaEnergyTerms compute_rosetta_terms(
    const std::vector<std::array<double, 3>>& coords,
    const std::string& sequence
);

// DeepDDG-style neural network inference (placeholder)
// In practice, this would call into a PyTorch/ONNX model
double predict_ddg_deep(
    const std::string& wild_type_seq,
    std::size_t position,
    char mutant_aa,
    const std::vector<double>& features  // precomputed features
);

// Interface for external energy calculators (FoldX, Rosetta, etc.)
class EnergyCalculator {
public:
    virtual ~EnergyCalculator() = default;
    
    // Compute ΔΔG for a single mutation
    virtual double compute_ddg(
        const std::string& pdb_path,
        const std::string& chain,
        std::size_t position,
        char mutant_aa
    ) = 0;
    
    // Compute energy terms for a structure
    virtual RosettaEnergyTerms compute_terms(const std::string& pdb_path) = 0;
};

// Factory for creating energy calculators
std::unique_ptr<EnergyCalculator> create_foldx_calculator();
std::unique_ptr<EnergyCalculator> create_rosetta_calculator();
std::unique_ptr<EnergyCalculator> create_esm_fold_calculator();

// Mutation scanning - compute ΔΔG for all possible mutations at a position
struct MutationScanResult {
    std::array<double, 20> ddg_values;  // in order: A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
    std::array<char, 20> amino_acids = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'};
};

MutationScanResult scan_position(
    const std::string& wild_type_seq,
    std::size_t position,
    const std::vector<double>& solvent_accessibility,
    const std::vector<int>& secondary_structure
);

} // namespace proteus::kernels