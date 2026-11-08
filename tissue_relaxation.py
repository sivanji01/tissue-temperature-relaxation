"""Numerical modelling of a simplified tissue-temperature relaxation model.

This is an educational model. It is not a clinical prediction tool.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp


def analytical_solution(time, initial_deviation, rate_a, rate_b):
    """Return the analytical temperature-deviation solution."""
    time = np.asarray(time, dtype=float)

    if np.isclose(rate_a, rate_b):
        return initial_deviation * (1 + rate_a * time) * np.exp(-rate_a * time)

    first_term = rate_b * np.exp(-rate_a * time)
    second_term = rate_a * np.exp(-rate_b * time)
    return initial_deviation * (first_term - second_term) / (rate_b - rate_a)


def temperature_ode(time, state, rate_a, rate_b):
    """Return [x', x''] for the first-order form of the model."""
    deviation = state[0]
    deviation_rate = state[1]

    first_derivative = deviation_rate
    second_derivative = -(rate_a + rate_b) * deviation_rate - rate_a * rate_b * deviation

    return np.array([first_derivative, second_derivative])


def explicit_euler(ode_function, time_span, initial_state, step_size, args=()):
    """Solve an ODE system with the forward/Explicit Euler method."""
    start_time, end_time = time_span
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    times = [start_time]
    states = [np.asarray(initial_state, dtype=float)]

    while times[-1] < end_time - 1e-12:
        current_time = times[-1]
        current_state = states[-1]
        step = min(step_size, end_time - current_time)
        slope = ode_function(current_time, current_state, *args)
        next_state = current_state + step * slope
        times.append(current_time + step)
        states.append(next_state)

    return np.asarray(times), np.asarray(states)


def implicit_euler(ode_function, time_span, initial_state, step_size, args=()):
    """Solve an ODE system with the backward/Implicit Euler method."""
    start_time, end_time = time_span
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    times = [start_time]
    states = [np.asarray(initial_state, dtype=float)]

    while times[-1] < end_time - 1e-12:
        current_time = times[-1]
        current_state = states[-1]
        step = min(step_size, end_time - current_time)
        next_time = current_time + step

        # The next state is unknown; root() finds the value that makes this zero.
        def residual(next_state):
            return next_state - current_state - step * ode_function(next_time, next_state, *args)

        result = root(residual, current_state)
        if not result.success:
            raise RuntimeError(result.message)

        times.append(next_time)
        states.append(result.x)

    return np.asarray(times), np.asarray(states)


def rmse(estimate, reference):
    """Calculate root mean square error between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(reference)) ** 2)))


def frobenius_lipschitz_bound(rate_a, rate_b):
    """Return the Frobenius-norm bound for the first-order system matrix."""
    system_matrix = np.array([
        [0.0, 1.0],
        [-rate_a * rate_b, -(rate_a + rate_b)],
    ])
    return float(np.linalg.norm(system_matrix, ord="fro"))


def explicit_euler_stability_limit(rate_a, rate_b):
    """Return the exact Explicit Euler step-size limit for this model."""
    return 2.0 / max(rate_a, rate_b)


def solve_with_bdf(time_span, initial_state, rate_a, rate_b, output_times):
    """Solve the model using SciPy's implicit BDF solver."""
    result = solve_ivp(
        temperature_ode,
        time_span,
        initial_state,
        method="BDF",
        t_eval=output_times,
        args=(rate_a, rate_b),
        rtol=1e-8,
        atol=1e-10,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def main():
    """Run the comparisons and save the figures used in the README."""
    output_times = np.linspace(0.0, 10.0, 1001)
    initial_deviation = 1.0
    initial_state = np.array([initial_deviation, 0.0])
    step_sizes = [0.01, 0.1, 1.0]
    parameter_sets = {
        "Set 1": (0.1, 0.2),
        "Set 2": (0.2, 5.0),
    }

    for name, (rate_a, rate_b) in parameter_sets.items():
        reference = analytical_solution(output_times, initial_deviation, rate_a, rate_b)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(output_times, reference, color="black", linewidth=2, label="Analytical reference")

        for step in step_sizes:
            times, states = explicit_euler(
                temperature_ode, (0.0, 10.0), initial_state, step, (rate_a, rate_b)
            )
            exact_at_times = analytical_solution(times, initial_deviation, rate_a, rate_b)
            error = rmse(states[:, 0], exact_at_times)
            ax.plot(times, states[:, 0], "--", label=f"Explicit Euler, h={step:g} s (RMSE={error:.2e})")

        if name == "Set 2":
            ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized temperature deviation")
        ax.set_title(f"Explicit Euler comparison — {name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"figures/explicit_euler_{name.lower().replace(' ', '_')}.png", dpi=180)
        plt.close(fig)

    # Implicit Euler comparison for the more challenging parameter set.
    rate_a, rate_b = parameter_sets["Set 2"]
    reference = analytical_solution(output_times, initial_deviation, rate_a, rate_b)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(output_times, reference, color="black", linewidth=2, label="Analytical reference")
    for step in step_sizes:
        times, states = implicit_euler(
            temperature_ode, (0.0, 10.0), initial_state, step, (rate_a, rate_b)
        )
        exact_at_times = analytical_solution(times, initial_deviation, rate_a, rate_b)
        error = rmse(states[:, 0], exact_at_times)
        ax.plot(times, states[:, 0], "--", label=f"Implicit Euler, h={step:g} s (RMSE={error:.2e})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized temperature deviation")
    ax.set_title("Implicit Euler comparison — Set 2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/implicit_euler_set_2.png", dpi=180)
    plt.close(fig)

    # BDF comparison for the more challenging parameter set.
    bdf_result = solve_with_bdf((0.0, 10.0), initial_state, rate_a, rate_b, output_times)
    error = rmse(bdf_result.y[0], reference)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(output_times, reference, color="black", linewidth=2, label="Analytical reference")
    ax.plot(bdf_result.t, bdf_result.y[0], "--", label=f"SciPy BDF (RMSE={error:.2e})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized temperature deviation")
    ax.set_title("SciPy BDF comparison — Set 2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/bdf_set_2.png", dpi=180)
    plt.close(fig)

    print("Project figures saved in the figures/ directory.")
    print("Explicit Euler stability limits:")
    for name, (rate_a, rate_b) in parameter_sets.items():
        print(f"  {name}: h < {explicit_euler_stability_limit(rate_a, rate_b):.3f} s")


if __name__ == "__main__":
    main()
