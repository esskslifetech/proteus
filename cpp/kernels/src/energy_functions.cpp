#include <proteus/kernels/energy_functions.hpp>
#include <cmath>
#include <unordered_map>
#include <array>
#include <memory>

namespace proteus::kernels {

// Amino acid properties table (Kyte-Doolittle hydrophobicity, volume, etc.)
static const std::unordered_map<char, AminoAcidProperties> AA_PROPERTIES = {
    {'A', {'A',  1.8,  88.6,  0, 0.0, false, false, false}},
    {'R', {'R', -4.5, 173.4, +1, 0.0, false, true,  true }},
    {'N', {'N', -3.5, 114.1,  0, 0.0, false, true,  false}},
    {'D', {'D', -3.5, 111.1, -1, 0.0, false, true,  true }},
    {'C', {'C',  2.5, 108.5,  0, 0.0, false, false, false}},
    {'Q', {'Q', -3.5, 143.8,  0, 0.0, false, true,  false}},
    {'E', {'E', -3.5, 138.4, -1, 0.0, false, true,  true }},
    {'G', {'G', -0.4,  60.1,  0, 0.0, false, false, false}},
    {'H', {'H', -3.2, 153.2, +1, 0.0, true,  true,  true }},
    {'I', {'I',  4.5, 166.7,  0, 0.0, false, false, false}},
    {'L', {'L',  3.8, 166.7,  0, 0.0, false, false, false}},
    {'K', {'K', -3.9, 168.6, +1, 0.0, false, true,  true }},
    {'M', {'M',  1.9, 162.9,  0, 0.0, false, false, false}},
    {'F', {'F',  2.8, 189.9,  0, 0.0, true,  false, false}},
    {'P', {'P', -1.6, 112.7,  0, 0.0, false, false, false}},
    {'S', {'S', -0.8,  89.0,  0, 0.0, false, true,  false}},
    {'T', {'T', -0.7, 116.1,  0, 0.0, false, true,  false}},
    {'W', {'W', -0.9, 227.8,  0, 0.0, true,  false, false}},
    {'Y', {'Y', -1.3, 193.6,  0, 0.0, true,  true,  false}},
    {'V', {'V',  4.2, 140.0,  0, 0.0, false, false, false}},
};

std::optional<AminoAcidProperties> get_amino_acid_properties(char code) {
    auto it = AA_PROPERTIES.find(static_cast<char>(std::toupper(static_cast<unsigned char>(code))));
    if (it != AA_PROPERTIES.end()) return it->second;
    return std::nullopt;
}

// Empirical ΔΔG prediction (simplified FoldX-like approach)
double predict_ddg_empirical(
    const std::string& wild_type_seq,
    std::size_t position,
    char mutant_aa,
    const std::vector<double>& solvent_accessibility,
    const std::vector<int>& secondary_structure
) {
    if (position >= wild_type_seq.size()) return 0.0;
    
    char wt_aa = wild_type_seq[position];
    if (wt_aa == mutant_aa) return 0.0;
    
    auto wt_props = get_amino_acid_properties(wt_aa);
    auto mut_props = get_amino_acid_properties(mutant_aa);
    if (!wt_props || !mut_props) return 0.0;
    
    double rsa = (position < solvent_accessibility.size()) ? solvent_accessibility[position] : 0.5;
    int ss = (position < secondary_structure.size()) ? secondary_structure[position] : 0;
    
    // Simplified empirical terms (inspired by FoldX)
    double ddg = 0.0;
    
    // Hydrophobicity change (buried residues more sensitive)
    double dh = mut_props->hydrophobicity - wt_props->hydrophobicity;
    ddg += dh * (1.0 - rsa) * 0.5;  // buried: larger effect
    
    // Volume change (packing)
    double dv = mut_props->volume - wt_props->volume;
    ddg += std::abs(dv) * 0.01 * (1.0 - rsa * 0.5);
    
    // Charge change (especially in buried)
    if (wt_props->charge != mut_props->charge) {
        ddg += 2.0 * (1.0 - rsa);  // charge burial penalty
    }
    
    // Secondary structure propensity
    if (ss == 1) {  // helix
        // Helix breakers: P, G
        if (mutant_aa == 'P' || mutant_aa == 'G') ddg += 1.5;
    } else if (ss == 2) {  // sheet
        // Sheet preferences
        if (mutant_aa == 'P') ddg += 2.0;
    }
    
    // Aromatic stacking
    if (wt_props->is_aromatic != mut_props->is_aromatic) {
        ddg += 0.5 * (1.0 - rsa);
    }
    
    return ddg;
}

// Statistical potential (simplified)
double predict_ddg_statistical(
    const std::string& wild_type_seq,
    std::size_t position,
    char mutant_aa,
    const std::vector<std::array<double, 3>>& /*ca_coords*/
) {
    // Placeholder for knowledge-based potential
    // Real implementation would use a precomputed potential (e.g., from PDB)
    return predict_ddg_empirical(wild_type_seq, position, mutant_aa, {}, {});
}

// Rosetta energy terms (simplified computation)
RosettaEnergyTerms compute_rosetta_terms(
    const std::vector<std::array<double, 3>>& coords,
    const std::string& sequence
) {
    RosettaEnergyTerms terms{};
    // kBT = 0.592 kcal/mol at 298K (used for Boltzmann weighting in production)
    
    if (coords.size() != sequence.size() || coords.empty()) return terms;
    
    // Very simplified pairwise energy calculation
    double fa_atr = 0.0, fa_rep = 0.0, fa_sol = 0.0;
    double fa_elec = 0.0, hbond = 0.0;
    
    for (std::size_t i = 0; i < coords.size(); ++i) {
        for (std::size_t j = i + 1; j < coords.size(); ++j) {
            // Distance
            double dx = coords[i][0] - coords[j][0];
            double dy = coords[i][1] - coords[j][1];
            double dz = coords[i][2] - coords[j][2];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist > 12.0) continue;  // cutoff
            
            // Van der Waals (Lennard-Jones 12-6)
            double sigma = 3.5;  // typical
            double eps = 0.2;
            double sr = sigma / dist;
            double sr6 = sr * sr * sr * sr * sr * sr;
            double sr12 = sr6 * sr6;
            fa_atr -= eps * sr6;
            fa_rep += eps * sr12;
            
            // Solvation (simplified)
            fa_sol += 0.01 * std::exp(-dist / 2.0);
            
            // Electrostatics (Coulomb with distance-dependent dielectric)
            auto pi = get_amino_acid_properties(sequence[i]);
            auto pj = get_amino_acid_properties(sequence[j]);
            if (pi && pj && pi->charge != 0 && pj->charge != 0) {
                fa_elec += 332.0 * pi->charge * pj->charge / (dist * 4.0);  // eps=4
            }
            
            // H-bonds (very simplified)
            if (dist < 3.5 && pi && pj && (pi->is_polar || pj->is_polar)) {
                hbond -= 1.0;
            }
        }
    }
    
    terms.fa_atr = fa_atr;
    terms.fa_rep = fa_rep;
    terms.fa_sol = fa_sol;
    terms.fa_elec = fa_elec;
    terms.hbond_sr = hbond * 0.5;
    terms.hbond_lr = hbond * 0.5;
    terms.rama = 0.0;  // would need phi/psi angles
    terms.p_aa_pp = 0.0;
    terms.ref = 0.0;
    
    return terms;
}

// DeepDDG placeholder
double predict_ddg_deep(
    const std::string& wild_type_seq,
    std::size_t position,
    char mutant_aa,
    const std::vector<double>& /*features*/
) {
    // In practice: load ONNX model, run inference
    // For now, return empirical as fallback
    return predict_ddg_empirical(wild_type_seq, position, mutant_aa, {}, {});
}

// Energy calculator interface implementations (stubs)

class FoldXCalculator : public EnergyCalculator {
public:
    double compute_ddg(const std::string&, const std::string&, std::size_t, char) override {
        // Would call FoldX binary
        return 0.0;
    }
    RosettaEnergyTerms compute_terms(const std::string&) override {
        return RosettaEnergyTerms{};
    }
};

class RosettaCalculator : public EnergyCalculator {
public:
    double compute_ddg(const std::string&, const std::string&, std::size_t, char) override {
        // Would call Rosetta
        return 0.0;
    }
    RosettaEnergyTerms compute_terms(const std::string&) override {
        return RosettaEnergyTerms{};
    }
};

class ESMFoldCalculator : public EnergyCalculator {
public:
    double compute_ddg(const std::string&, const std::string&, std::size_t, char) override {
        // Would call ESMFold + energy function
        return 0.0;
    }
    RosettaEnergyTerms compute_terms(const std::string&) override {
        return RosettaEnergyTerms{};
    }
};

std::unique_ptr<EnergyCalculator> create_foldx_calculator() {
    return std::make_unique<FoldXCalculator>();
}

std::unique_ptr<EnergyCalculator> create_rosetta_calculator() {
    return std::make_unique<RosettaCalculator>();
}

std::unique_ptr<EnergyCalculator> create_esm_fold_calculator() {
    return std::make_unique<ESMFoldCalculator>();
}

// Mutation scanning
MutationScanResult scan_position(
    const std::string& wild_type_seq,
    std::size_t position,
    const std::vector<double>& solvent_accessibility,
    const std::vector<int>& secondary_structure
) {
    MutationScanResult result;
    static const std::array<char, 20> AAS = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'};
    
    for (int i = 0; i < 20; ++i) {
        result.amino_acids[i] = AAS[i];
        result.ddg_values[i] = predict_ddg_empirical(
            wild_type_seq, position, AAS[i], 
            solvent_accessibility, secondary_structure
        );
    }
    return result;
}

} // namespace proteus::kernels