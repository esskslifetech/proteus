#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <proteus/kernels/simd_stats.hpp>
#include <proteus/kernels/energy_functions.hpp>
#include <proteus/solvers/ode_solver.hpp>
#include <proteus/solvers/fba_solver.hpp>

namespace py = pybind11;

PYBIND11_MODULE(proteus_cpp, m) {
    m.doc() = "Proteus C++ high-performance kernels and solvers";

    // ========================================================================
    // Kernels: simd_stats
    // ========================================================================
    py::module_ kernels = m.def_submodule("kernels", "SIMD-accelerated statistical kernels");

    kernels.def("mean", [](const std::vector<double>& data) {
        return proteus::kernels::mean_simd(data.data(), data.size());
    }, "Compute mean");

    kernels.def("variance", [](const std::vector<double>& data) {
        return proteus::kernels::variance_simd(data.data(), data.size());
    }, "Compute variance");

    kernels.def("stddev", [](const std::vector<double>& data) {
        return proteus::kernels::stddev_simd(data.data(), data.size());
    }, "Compute standard deviation");

    kernels.def("quantile", [](std::vector<double> data, double q) {
        return proteus::kernels::quantile_simd(data.data(), data.size(), q);
    }, "Compute quantile (modifies input)");

    kernels.def("correlation", [](const std::vector<double>& x, const std::vector<double>& y) {
        return proteus::kernels::correlation_simd(x.data(), y.data(), x.size());
    }, "Compute Pearson correlation");

    kernels.def("histogram", [](const std::vector<double>& data, double min_val, double max_val, size_t bins) {
        return proteus::kernels::histogram_simd(data.data(), data.size(), min_val, max_val, bins);
    }, "Compute histogram");

    kernels.def("euclidean_distance", [](const std::vector<double>& a, const std::vector<double>& b) {
        return proteus::kernels::euclidean_distance_simd(a.data(), b.data(), a.size());
    }, "Euclidean distance");

    kernels.def("manhattan_distance", [](const std::vector<double>& a, const std::vector<double>& b) {
        return proteus::kernels::manhattan_distance_simd(a.data(), b.data(), a.size());
    }, "Manhattan distance");

    kernels.def("cosine_similarity", [](const std::vector<double>& a, const std::vector<double>& b) {
        return proteus::kernels::cosine_similarity_simd(a.data(), b.data(), a.size());
    }, "Cosine similarity");

    // ========================================================================
    // Kernels: energy_functions
    // ========================================================================
    py::module_ energy = m.def_submodule("energy", "Protein energy functions");

    py::class_<proteus::kernels::AminoAcidProperties>(energy, "AminoAcidProperties")
        .def_readonly("code", &proteus::kernels::AminoAcidProperties::code)
        .def_readonly("hydrophobicity", &proteus::kernels::AminoAcidProperties::hydrophobicity)
        .def_readonly("volume", &proteus::kernels::AminoAcidProperties::volume)
        .def_readonly("charge", &proteus::kernels::AminoAcidProperties::charge)
        .def_readonly("polarity", &proteus::kernels::AminoAcidProperties::polarity)
        .def_readonly("is_aromatic", &proteus::kernels::AminoAcidProperties::is_aromatic)
        .def_readonly("is_polar", &proteus::kernels::AminoAcidProperties::is_polar)
        .def_readonly("is_charged", &proteus::kernels::AminoAcidProperties::is_charged);

    energy.def("get_amino_acid_properties", [](char code) {
        auto props = proteus::kernels::get_amino_acid_properties(code);
        if (props) return *props;
        throw py::value_error("Unknown amino acid code");
    }, "Get amino acid properties");

    energy.def("predict_ddg_empirical", &proteus::kernels::predict_ddg_empirical,
        "Predict ΔΔG using empirical method",
        py::arg("wild_type_seq"), py::arg("position"), py::arg("mutant_aa"),
        py::arg("solvent_accessibility") = std::vector<double>{},
        py::arg("secondary_structure") = std::vector<int>{});

    energy.def("predict_ddg_statistical", &proteus::kernels::predict_ddg_statistical,
        "Predict ΔΔG using statistical potential",
        py::arg("wild_type_seq"), py::arg("position"), py::arg("mutant_aa"),
        py::arg("ca_coords"));

    py::class_<proteus::kernels::RosettaEnergyTerms>(energy, "RosettaEnergyTerms")
        .def_readonly("fa_atr", &proteus::kernels::RosettaEnergyTerms::fa_atr)
        .def_readonly("fa_rep", &proteus::kernels::RosettaEnergyTerms::fa_rep)
        .def_readonly("fa_sol", &proteus::kernels::RosettaEnergyTerms::fa_sol)
        .def_readonly("fa_elec", &proteus::kernels::RosettaEnergyTerms::fa_elec)
        .def_readonly("hbond_sr", &proteus::kernels::RosettaEnergyTerms::hbond_sr)
        .def_readonly("hbond_lr", &proteus::kernels::RosettaEnergyTerms::hbond_lr)
        .def_readonly("rama", &proteus::kernels::RosettaEnergyTerms::rama)
        .def_readonly("p_aa_pp", &proteus::kernels::RosettaEnergyTerms::p_aa_pp)
        .def_readonly("ref", &proteus::kernels::RosettaEnergyTerms::ref);

    energy.def("compute_rosetta_terms", &proteus::kernels::compute_rosetta_terms,
        "Compute Rosetta energy terms",
        py::arg("coords"), py::arg("sequence"));

    energy.def("predict_ddg_deep", &proteus::kernels::predict_ddg_deep,
        "Predict ΔΔG using deep learning (placeholder)",
        py::arg("wild_type_seq"), py::arg("position"), py::arg("mutant_aa"),
        py::arg("features") = std::vector<double>{});

    py::class_<proteus::kernels::MutationScanResult>(energy, "MutationScanResult")
        .def_readonly("ddg_values", &proteus::kernels::MutationScanResult::ddg_values)
        .def_readonly("amino_acids", &proteus::kernels::MutationScanResult::amino_acids);

    energy.def("scan_position", &proteus::kernels::scan_position,
        "Scan all mutations at a position",
        py::arg("wild_type_seq"), py::arg("position"),
        py::arg("solvent_accessibility") = std::vector<double>{},
        py::arg("secondary_structure") = std::vector<int>{});

    // EnergyCalculator interface (abstract - not directly exposed)
    // Concrete implementations would be exposed separately

    // ========================================================================
    // Solvers: ODE
    // ========================================================================
    py::module_ solvers = m.def_submodule("solvers", "ODE/DAE and FBA solvers");

    py::class_<proteus::solvers::ODEOptions>(solvers, "ODEOptions")
        .def(py::init<>())
        .def_readwrite("rtol", &proteus::solvers::ODEOptions::rtol)
        .def_readwrite("atol", &proteus::solvers::ODEOptions::atol)
        .def_readwrite("max_step", &proteus::solvers::ODEOptions::max_step)
        .def_readwrite("min_step", &proteus::solvers::ODEOptions::min_step)
        .def_readwrite("max_steps", &proteus::solvers::ODEOptions::max_steps)
        .def_readwrite("use_jacobian", &proteus::solvers::ODEOptions::use_jacobian)
        .def_readwrite("use_dense_jacobian", &proteus::solvers::ODEOptions::use_dense_jacobian);

    py::class_<proteus::solvers::ODEState>(solvers, "ODEState")
        .def_readonly("t", &proteus::solvers::ODEState::t)
        .def_readonly("y", &proteus::solvers::ODEState::y)
        .def_readonly("flag", &proteus::solvers::ODEState::flag)
        .def_readonly("message", &proteus::solvers::ODEState::message);

    py::class_<proteus::solvers::ODESolver>(solvers, "ODESolver")
        .def(py::init<const proteus::solvers::ODEOptions&>())
        .def("solve", [](proteus::solvers::ODESolver& self,
                         py::function f,
                         const Eigen::VectorXd& y0,
                         const std::array<double, 2>& t_span,
                         const std::vector<double>& t_eval) {
            proteus::solvers::ODESystem system =
                [f](double t, const Eigen::VectorXd& y, Eigen::VectorXd& dydt) {
                    py::object result = f(t, y);
                    std::vector<double> values = result.cast<std::vector<double>>();
                    dydt = Eigen::Map<Eigen::VectorXd>(values.data(),
                                                       static_cast<Eigen::Index>(values.size()));
                };
            return self.solve(system, y0, t_span, t_eval);
        }, "Solve ODE system (system(t, y) must return dy/dt as a sequence)",
        py::arg("system"), py::arg("y0"), py::arg("t_span"), py::arg("t_eval") = std::vector<double>{});

    solvers.def("solve_ode", [](py::function f,
                                const Eigen::VectorXd& y0,
                                double t0, double tf,
                                const std::vector<double>& t_eval,
                                const proteus::solvers::ODEOptions& opts) {
            proteus::solvers::ODESystem system =
                [f](double t, const Eigen::VectorXd& y, Eigen::VectorXd& dydt) {
                    py::object result = f(t, y);
                    std::vector<double> values = result.cast<std::vector<double>>();
                    dydt = Eigen::Map<Eigen::VectorXd>(values.data(),
                                                       static_cast<Eigen::Index>(values.size()));
                };
            return proteus::solvers::solve_ode(system, y0, t0, tf, t_eval, opts);
        }, "Solve ODE system (system(t, y) must return dy/dt as a sequence)",
        py::arg("system"), py::arg("y0"), py::arg("t0"), py::arg("tf"),
        py::arg("t_eval") = std::vector<double>{}, py::arg("opts") = proteus::solvers::ODEOptions{});

    // ========================================================================
    // Solvers: FBA
    // ========================================================================
    py::class_<proteus::solvers::MetabolicModel>(solvers, "MetabolicModel")
        .def(py::init<>())
        .def_readwrite("id", &proteus::solvers::MetabolicModel::id)
        .def_readwrite("metabolites", &proteus::solvers::MetabolicModel::metabolites)
        .def_readwrite("reactions", &proteus::solvers::MetabolicModel::reactions)
        .def_readwrite("genes", &proteus::solvers::MetabolicModel::genes)
        .def_readwrite("lb", &proteus::solvers::MetabolicModel::lb)
        .def_readwrite("ub", &proteus::solvers::MetabolicModel::ub)
        .def_readwrite("c", &proteus::solvers::MetabolicModel::c)
        .def_readwrite("objective_id", &proteus::solvers::MetabolicModel::objective_id)
        .def_readwrite("maximize", &proteus::solvers::MetabolicModel::maximize)
        .def_readwrite("gr_rules", &proteus::solvers::MetabolicModel::gr_rules);

    py::enum_<proteus::solvers::FBASolution::Status>(solvers, "FBAStatus")
        .value("OPTIMAL", proteus::solvers::FBASolution::Status::Optimal)
        .value("SUBOPTIMAL", proteus::solvers::FBASolution::Status::Suboptimal)
        .value("INFEASIBLE", proteus::solvers::FBASolution::Status::Infeasible)
        .value("UNBOUNDED", proteus::solvers::FBASolution::Status::Unbounded)
        .value("NUMERICAL_ERROR", proteus::solvers::FBASolution::Status::NumericalError)
        .value("NOT_SOLVED", proteus::solvers::FBASolution::Status::NotSolved);

    py::class_<proteus::solvers::FBASolution>(solvers, "FBASolution")
        .def_readonly("status", &proteus::solvers::FBASolution::status)
        .def_readonly("objective_value", &proteus::solvers::FBASolution::objective_value)
        .def_readonly("flux", &proteus::solvers::FBASolution::flux)
        .def_readonly("reduced_costs", &proteus::solvers::FBASolution::reduced_costs)
        .def_readonly("shadow_prices", &proteus::solvers::FBASolution::shadow_prices)
        .def_readonly("message", &proteus::solvers::FBASolution::message)
        .def_readonly("solve_time", &proteus::solvers::FBASolution::solve_time)
        .def_readonly("iterations", &proteus::solvers::FBASolution::iterations);

    py::class_<proteus::solvers::FBASolver::FVAResult>(solvers, "FVAResult")
        .def_readonly("min_flux", &proteus::solvers::FBASolver::FVAResult::min_flux)
        .def_readonly("max_flux", &proteus::solvers::FBASolver::FVAResult::max_flux);

    py::class_<proteus::solvers::FBAOptions>(solvers, "FBAOptions")
        .def(py::init<>())
        .def_readwrite("verbose", &proteus::solvers::FBAOptions::verbose)
        .def_readwrite("feasibility_tol", &proteus::solvers::FBAOptions::feasibility_tol)
        .def_readwrite("optimality_tol", &proteus::solvers::FBAOptions::optimality_tol)
        .def_readwrite("dual_feasibility_tol", &proteus::solvers::FBAOptions::dual_feasibility_tol)
        .def_readwrite("max_iterations", &proteus::solvers::FBAOptions::max_iterations)
        .def_readwrite("compute_duals", &proteus::solvers::FBAOptions::compute_duals);

    py::class_<proteus::solvers::FBASolver>(solvers, "FBASolver")
        .def(py::init<const proteus::solvers::FBAOptions&>())
        .def("load_model", &proteus::solvers::FBASolver::load_model)
        .def("solve", &proteus::solvers::FBASolver::solve)
        .def("solve_with_bounds", &proteus::solvers::FBASolver::solve_with_bounds)
        .def("single_knockout", &proteus::solvers::FBASolver::single_knockout)
        .def("double_knockout", &proteus::solvers::FBASolver::double_knockout)
        .def("flux_variability_analysis", &proteus::solvers::FBASolver::flux_variability_analysis)
        .def("parsimonious_fba", &proteus::solvers::FBASolver::parsimonious_fba)
        .def("model", &proteus::solvers::FBASolver::model, py::return_value_policy::reference);

    solvers.def("solve_fba", &proteus::solvers::solve_fba,
        "Convenience function to solve FBA",
        py::arg("model"), py::arg("opts") = proteus::solvers::FBAOptions{});

    solvers.def("build_recon3d_model", &proteus::solvers::build_recon3d_model,
        "Build Recon3D model (placeholder)");

    solvers.def("build_human_gem_model", &proteus::solvers::build_human_gem_model,
        "Build Human-GEM model (placeholder)");
}