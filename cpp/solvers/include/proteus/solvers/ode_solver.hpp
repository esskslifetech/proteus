#pragma once

#include <vector>
#include <array>
#include <functional>
#include <string>
#include <optional>
#include <memory>
#include <Eigen/Dense>

// SUNDIALS is optional. When it is unavailable (e.g. an MSVC build without a
// matching SUNDIALS toolchain) the solver falls back to a fixed-step RK4
// integrator, which keeps the C++ acceleration usable for non-stiff models.
#if proteus_USE_SUNDIALS
#include <sundials/sundials_types.h>
#include <sundials/sundials_nvector.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#endif

namespace proteus::solvers {

// ODE/DAE solver using SUNDIALS CVODE

struct ODEOptions {
    double rtol = 1e-6;
    double atol = 1e-9;
    double max_step = 0.0;  // 0 = unlimited
    double min_step = 0.0;
    long int max_steps = 10000;
    bool use_jacobian = true;
    bool use_dense_jacobian = true;
};

struct ODEState {
    std::vector<double> t;
    std::vector<std::vector<double>> y;  // y[step][variable]
    int flag = 0;  // CVODE return flag
    std::string message;
};

using ODESystem = std::function<void(double t, const Eigen::VectorXd& y, Eigen::VectorXd& dydt)>;
using ODEJacobian = std::function<void(double t, const Eigen::VectorXd& y, Eigen::MatrixXd& J)>;

class ODESolver {
public:
    // Exposed (implementation detail) so C callback trampolines can access it
    struct Impl;

    explicit ODESolver(const ODEOptions& opts = ODEOptions());
    ~ODESolver();

    // Non-copyable, movable
    ODESolver(const ODESolver&) = delete;
    ODESolver& operator=(const ODESolver&) = delete;
    ODESolver(ODESolver&&) noexcept;
    ODESolver& operator=(ODESolver&&) noexcept;

    // Solve initial value problem
    // system: dy/dt = f(t, y)
    // y0: initial conditions
    // t_span: [t0, tf]
    // t_eval: optional output times (if empty, uses internal steps)
    ODEState solve(
        const ODESystem& system,
        const Eigen::VectorXd& y0,
        const std::array<double, 2>& t_span,
        const std::vector<double>& t_eval = {}
    );

    // Solve with Jacobian
    ODEState solve(
        const ODESystem& system,
        const ODEJacobian& jacobian,
        const Eigen::VectorXd& y0,
        const std::array<double, 2>& t_span,
        const std::vector<double>& t_eval = {}
    );

    // Solve DAE: F(t, y, y') = 0 (using IDA)
    // Not fully implemented in this stub
    ODEState solve_dae(
        const std::function<void(double, const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::VectorXd&)>& residual,
        const Eigen::VectorXd& y0,
        const Eigen::VectorXd& yp0,
        const std::array<double, 2>& t_span,
        const std::vector<double>& t_eval = {}
    );

private:
    std::unique_ptr<Impl> pimpl_;
};

// Convenience function for simple systems
ODEState solve_ode(
    const ODESystem& system,
    const Eigen::VectorXd& y0,
    double t0, double tf,
    const std::vector<double>& t_eval = {},
    const ODEOptions& opts = ODEOptions()
);

// Stiff system detector
bool is_stiff(const ODESystem& system, const Eigen::VectorXd& y, double t);

} // namespace proteus::solvers