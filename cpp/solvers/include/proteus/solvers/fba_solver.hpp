#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <memory>
#include <Eigen/Sparse>

// HiGHS is optional - if not available, use Eigen for LP
#if proteus_USE_HIGHS
#include <Highs.h>
#endif

namespace proteus::solvers {

// Flux Balance Analysis (FBA) using HiGHS

struct MetabolicModel {
    std::string id;
    std::vector<std::string> metabolites;  // metabolite IDs
    std::vector<std::string> reactions;    // reaction IDs
    std::vector<std::string> genes;        // gene IDs
    
    // Stoichiometric matrix S (metabolites × reactions)
    Eigen::SparseMatrix<double> S;
    
    // Bounds
    std::vector<double> lb;  // lower bounds (reactions)
    std::vector<double> ub;  // upper bounds
    
    // Objective
    std::vector<double> c;   // objective coefficients
    std::string objective_id;
    bool maximize = true;
    
    // Gene-reaction rules (simplified)
    std::vector<std::string> gr_rules;
};

struct FBASolution {
    enum class Status {
        Optimal,
        Suboptimal,
        Infeasible,
        Unbounded,
        NumericalError,
        NotSolved
    };
    
    Status status = Status::NotSolved;
    double objective_value = 0.0;
    std::vector<double> flux;           // reaction fluxes
    std::vector<double> reduced_costs;  // reduced costs
    std::vector<double> shadow_prices;  // dual values (metabolites)
    std::string message;
    double solve_time = 0.0;
    int iterations = 0;
};

struct FBAOptions {
    bool verbose = false;
    double feasibility_tol = 1e-9;
    double optimality_tol = 1e-9;
    double dual_feasibility_tol = 1e-9;
    int max_iterations = 10000;
    // HiGHS options are string-based in Highs 1.16+
    std::string simplex_strategy = "dual";  // "dual", "primal", "auto"
    std::string presolve = "on";            // "on", "off", "auto"
    bool compute_duals = true;
};

class FBASolver {
public:
    explicit FBASolver(const FBAOptions& opts = FBAOptions());
    ~FBASolver();

    // Non-copyable, movable
    FBASolver(const FBASolver&) = delete;
    FBASolver& operator=(const FBASolver&) = delete;
    FBASolver(FBASolver&&) noexcept;
    FBASolver& operator=(FBASolver&&) noexcept;

    // Load model from SBML (placeholder - real impl would parse SBML)
    bool load_model(const MetabolicModel& model);
    bool load_model_from_sbml(const std::string& sbml_path);

    // Solve FBA
    FBASolution solve();

    // Solve with modified bounds (e.g., for knockouts)
    FBASolution solve_with_bounds(
        const std::vector<double>&,
        const std::vector<double>&
    );

    // Single gene knockout
    FBASolution single_knockout(const std::string& gene_id);

    // Double gene knockout
    FBASolution double_knockout(const std::string& gene1, const std::string& gene2);

    // Flux Variability Analysis (FVA)
    struct FVAResult {
        std::vector<double> min_flux;
        std::vector<double> max_flux;
    };
    FVAResult flux_variability_analysis(double fraction_of_optimum = 0.9);

    // Parsimonious FBA (pFBA)
    FBASolution parsimonious_fba();

    // Get model info
    const MetabolicModel& model() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience functions
FBASolution solve_fba(const MetabolicModel& model, const FBAOptions& opts = FBAOptions());

// Model builders
MetabolicModel build_recon3d_model();  // Placeholder
MetabolicModel build_human_gem_model();  // Placeholder

// Gene-reaction rule evaluation
bool evaluate_gr_rule(const std::string& rule, const std::unordered_map<std::string, bool>& gene_states);

} // namespace proteus::solvers