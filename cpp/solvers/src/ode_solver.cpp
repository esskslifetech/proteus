#include <proteus/solvers/ode_solver.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#if proteus_USE_SUNDIALS
#include <sundials/sundials_context.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#endif

namespace proteus::solvers {

struct ODESolver::Impl {
    ODEOptions opts;
#if proteus_USE_SUNDIALS
    void* cvode_mem = nullptr;
    N_Vector y = nullptr;
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;
    SUNContext ctx = nullptr;
#endif
    int n = 0;

    ODESystem system;
    ODEJacobian jacobian;
    std::vector<double> t_eval;
};

#if proteus_USE_SUNDIALS

static int cvode_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    auto* impl = static_cast<ODESolver::Impl*>(user_data);
    int n = impl->n;

    // Copy SUNDIALS buffers into Eigen temporaries (the callback API uses
    // mutable references, which cannot bind to Eigen::Map directly).
    Eigen::Map<const Eigen::VectorXd> y_eig(N_VGetArrayPointer(y), n);
    Eigen::VectorXd y_vec = y_eig;
    Eigen::VectorXd dydt_vec = Eigen::VectorXd::Zero(n);

    try {
        impl->system(static_cast<double>(t), y_vec, dydt_vec);
    } catch (...) {
        return -1;
    }

    Eigen::Map<Eigen::VectorXd> dydt_eig(N_VGetArrayPointer(ydot), n);
    dydt_eig = dydt_vec;
    return 0;
}

static int cvode_jac(sunrealtype t, N_Vector y, N_Vector /*fy*/, SUNMatrix J,
                     void* user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
                     N_Vector /*tmp3*/) {
    auto* impl = static_cast<ODESolver::Impl*>(user_data);
    if (!impl->jacobian) return -1;

    int n = impl->n;
    Eigen::Map<const Eigen::VectorXd> y_eig(N_VGetArrayPointer(y), n);
    Eigen::VectorXd y_vec = y_eig;
    Eigen::MatrixXd J_mat = Eigen::MatrixXd::Zero(n, n);

    try {
        impl->jacobian(static_cast<double>(t), y_vec, J_mat);
    } catch (...) {
        return -1;
    }

    Eigen::Map<Eigen::MatrixXd> J_eig(SUNDenseMatrix_Data(J), n, n);
    J_eig = J_mat;
    return 0;
}

ODESolver::ODESolver(const ODEOptions& opts) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->opts = opts;
    // Serial context for SUNDIALS 7.x
    SUNContext_Create(SUN_COMM_NULL, &pimpl_->ctx);
}

ODESolver::~ODESolver() {
    if (pimpl_->cvode_mem) CVodeFree(&pimpl_->cvode_mem);
    if (pimpl_->y) N_VDestroy(pimpl_->y);
    if (pimpl_->J) SUNMatDestroy(pimpl_->J);
    if (pimpl_->LS) SUNLinSolFree(pimpl_->LS);
    if (pimpl_->ctx) SUNContext_Free(&pimpl_->ctx);
}
#else
ODESolver::ODESolver(const ODEOptions& opts) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->opts = opts;
}

ODESolver::~ODESolver() = default;
#endif

ODESolver::ODESolver(ODESolver&& other) noexcept = default;
ODESolver& ODESolver::operator=(ODESolver&& other) noexcept = default;

namespace {

#if proteus_USE_SUNDIALS
// Run CVODE over requested output times. Returns populated ODEState.
ODEState integrate(ODESolver::Impl& impl, const Eigen::VectorXd& y0,
                   double t0, double tf,
                   const std::vector<double>& t_eval) {
    ODEState state;

    // Clean up any previous solve session
    if (impl.cvode_mem) { CVodeFree(&impl.cvode_mem); impl.cvode_mem = nullptr; }
    if (impl.y) { N_VDestroy(impl.y); impl.y = nullptr; }
    if (impl.J) { SUNMatDestroy(impl.J); impl.J = nullptr; }
    if (impl.LS) { SUNLinSolFree(impl.LS); impl.LS = nullptr; }

    impl.n = static_cast<int>(y0.size());
    if (impl.n <= 0) {
        state.flag = -1;
        state.message = "Empty state vector";
        return state;
    }

    // Create SUNDIALS objects
    impl.y = N_VNew_Serial(impl.n, impl.ctx);
    impl.J = SUNDenseMatrix(impl.n, impl.n, impl.ctx);
    impl.LS = SUNLinSol_Dense(impl.y, impl.J, impl.ctx);
    if (!impl.y || !impl.J || !impl.LS) {
        state.flag = -1;
        state.message = "Failed to allocate SUNDIALS objects";
        return state;
    }

    // Copy initial conditions into the fresh state vector
    auto* y0_data = N_VGetArrayPointer(impl.y);
    for (int i = 0; i < impl.n; ++i) y0_data[i] = y0[i];

    impl.cvode_mem = CVodeCreate(CV_BDF, impl.ctx);
    if (!impl.cvode_mem) {
        state.flag = -1;
        state.message = "CVodeCreate failed";
        return state;
    }

    if (CVodeInit(impl.cvode_mem, cvode_rhs, t0, impl.y) != CV_SUCCESS) {
        state.flag = -1;
        state.message = "CVodeInit failed";
        return state;
    }
    CVodeSStolerances(impl.cvode_mem, impl.opts.rtol, impl.opts.atol);
    if (impl.opts.max_step > 0.0) CVodeSetMaxStep(impl.cvode_mem, impl.opts.max_step);
    if (impl.opts.min_step > 0.0) CVodeSetMinStep(impl.cvode_mem, impl.opts.min_step);
    CVodeSetMaxNumSteps(impl.cvode_mem, impl.opts.max_steps);
    CVodeSetUserData(impl.cvode_mem, &impl);

    if (CVodeSetLinearSolver(impl.cvode_mem, impl.LS, impl.J) != CV_SUCCESS) {
        state.flag = -1;
        state.message = "CVodeSetLinearSolver failed";
        return state;
    }
    if (impl.opts.use_jacobian && impl.jacobian) {
        CVodeSetJacFn(impl.cvode_mem, cvode_jac);
    }

    // Determine output targets
    std::vector<double> targets;
    if (t_eval.empty()) {
        targets.push_back(tf);
    } else {
        for (double t : t_eval) {
            if (t > t0 && t <= tf) targets.push_back(t);
        }
        if (targets.empty() || targets.back() < tf) targets.push_back(tf);
    }

    state.flag = 0;
    double t = t0;

    for (double target : targets) {
        CVodeSetStopTime(impl.cvode_mem, target);
        while (t < target) {
            sunrealtype tret = t;
            int flag = CVode(impl.cvode_mem, target, impl.y, &tret, CV_NORMAL);
            t = static_cast<double>(tret);

            if (flag < 0) {
                state.flag = flag;
                state.message = "CVODE integration failed";
                CVodeGetLastStep(impl.cvode_mem, &tret);
                return state;
            }

            // Record the point
            auto* data = N_VGetArrayPointer(impl.y);
            state.t.push_back(t);
            state.y.emplace_back(data, data + impl.n);

            if (flag == CV_SUCCESS) break;  // reached target
        }
    }

    state.message = "success";
    return state;
}

#else  // !proteus_USE_SUNDIALS

// Fallback fixed-step classical RK4 integrator. Intended for the non-stiff
// pathway/energy models; stiff systems should build with SUNDIALS enabled.
ODEState integrate(ODESolver::Impl& impl, const Eigen::VectorXd& y0,
                   double t0, double tf,
                   const std::vector<double>& t_eval) {
    ODEState state;
    impl.n = static_cast<int>(y0.size());
    if (impl.n <= 0) {
        state.flag = -1;
        state.message = "Empty state vector";
        return state;
    }
    if (tf < t0) {
        state.flag = -1;
        state.message = "t_span is reversed";
        return state;
    }

    std::vector<double> targets;
    if (t_eval.empty()) {
        targets.push_back(tf);
    } else {
        for (double t : t_eval) {
            if (t > t0 && t <= tf) targets.push_back(t);
        }
        if (targets.empty() || targets.back() < tf) targets.push_back(tf);
    }

    const double span = tf - t0;
    double default_step = span > 0.0 ? span / 1000.0 : 1e-3;
    if (impl.opts.max_step > 0.0) default_step = std::min(default_step, impl.opts.max_step);
    default_step = std::max(default_step, impl.opts.min_step > 0.0 ? impl.opts.min_step : 1e-9);

    auto rhs = [&impl](double t, const Eigen::VectorXd& y) {
        Eigen::VectorXd dydt = Eigen::VectorXd::Zero(impl.n);
        impl.system(t, y, dydt);
        return dydt;
    };

    Eigen::VectorXd y = y0;
    double t = t0;
    long int steps = 0;
    const long int max_steps = std::max<long int>(impl.opts.max_steps, 1);

    for (double target : targets) {
        while (t < target) {
            const double h = std::min(default_step, target - t);
            if (steps++ >= max_steps) {
                state.flag = -1;
                state.message = "RK4 step limit exceeded";
                return state;
            }

            const Eigen::VectorXd k1 = rhs(t, y);
            const Eigen::VectorXd k2 = rhs(t + h / 2.0, y + (h / 2.0) * k1);
            const Eigen::VectorXd k3 = rhs(t + h / 2.0, y + (h / 2.0) * k2);
            const Eigen::VectorXd k4 = rhs(t + h, y + h * k3);
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            t += h;
        }

        state.t.push_back(t);
        state.y.emplace_back(y.data(), y.data() + impl.n);
    }

    state.message = "success (RK4 fallback)";
    return state;
}

#endif  // proteus_USE_SUNDIALS

} // namespace

ODEState ODESolver::solve(
    const ODESystem& system,
    const Eigen::VectorXd& y0,
    const std::array<double, 2>& t_span,
    const std::vector<double>& t_eval
) {
    pimpl_->system = system;
    pimpl_->jacobian = nullptr;
    pimpl_->t_eval = t_eval;

    return integrate(*pimpl_, y0, t_span[0], t_span[1], t_eval);
}

ODEState ODESolver::solve(
    const ODESystem& system,
    const ODEJacobian& jacobian,
    const Eigen::VectorXd& y0,
    const std::array<double, 2>& t_span,
    const std::vector<double>& t_eval
) {
    pimpl_->system = system;
    pimpl_->jacobian = jacobian;
    pimpl_->t_eval = t_eval;

    return integrate(*pimpl_, y0, t_span[0], t_span[1], t_eval);
}

ODEState ODESolver::solve_dae(
    const std::function<void(double, const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::VectorXd&)>&,
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    const std::array<double, 2>&,
    const std::vector<double>&
) {
    ODEState state;
    state.flag = -1;
    state.message = "DAE solver not implemented";
    return state;
}

ODEState solve_ode(
    const ODESystem& system,
    const Eigen::VectorXd& y0,
    double t0, double tf,
    const std::vector<double>& t_eval,
    const ODEOptions& opts
) {
    ODESolver solver(opts);
    return solver.solve(system, y0, {t0, tf}, t_eval);
}

bool is_stiff(const ODESystem&, const Eigen::VectorXd&, double) {
    return true;  // Conservative: assume stiff systems by default
}

} // namespace proteus::solvers
