# Tissue Temperature Relaxation: Numerical Modelling

A small scientific-Python project exploring how a simplified tissue-temperature deviation returns to its baseline after a short heating pulse. The project compares an analytical reference solution with Explicit Euler, Implicit Euler and SciPy's BDF solver.

> **Scope:** This is an educational numerical model, not a clinical thermal model or a tool for treatment planning or medical decisions. The parameters are illustrative.

## What this project demonstrates

- Formulating a second-order ordinary differential equation (ODE).
- Converting a second-order ODE into a first-order system.
- Implementing Explicit Euler and Implicit Euler.
- Comparing numerical results with an analytical reference using RMSE.
- Studying stability and step-size restrictions for two parameter sets.
- Using a stiff ODE solver (`BDF`) from SciPy.

## Model

Let `x(t)` denote the normalized temperature deviation from baseline. The model is

\[
 x''(t) + (a+b)x'(t) + abx(t) = 0,
 \qquad x(0)=A,\qquad x'(0)=0.
\]

For different positive rates `a` and `b`, the analytical solution is

\[
 x(t)=\frac{A}{b-a}\left(be^{-at}-ae^{-bt}\right).
\]

For numerical solvers, define `v = x'`. The equivalent first-order system is

\[
 x'=v,\qquad v'=-(a+b)v-abx.
\]

## Experiments

The project compares two illustrative parameter sets:

| Parameter set | a (s⁻¹) | b (s⁻¹) | Explicit Euler stability limit |
|---|---:|---:|---:|
| Set 1 | 0.1 | 0.2 | h < 10.0 s |
| Set 2 | 0.2 | 5.0 | h < 0.4 s |

For this model, the eigenvalues of the first-order system matrix are `-a` and `-b`. Explicit Euler is stable for both modes when

\[
 0 < h < \frac{2}{\max(a,b)}.
\]

The Frobenius norm of the system matrix provides a bound for the Lipschitz constant:

\[
 L_F=\sqrt{1+(ab)^2+(a+b)^2}.
\]

This norm bound and the eigenvalue-based stability limit are related checks, but they are not the same quantity.

## Results

### Explicit Euler — Set 1

All three tested step sizes (0.01 s, 0.1 s and 1.0 s) are below the 10 s stability limit.

![Explicit Euler comparison for parameter set 1](figures/explicit_euler_set_1.png)

### Explicit Euler — Set 2

The exact stability limit is 0.4 s. The 0.01 s and 0.1 s runs are stable; the 1.0 s run is unstable and grows instead of following the decaying reference curve.

![Explicit Euler comparison for parameter set 2](figures/explicit_euler_set_2.png)

### Implicit Euler — Set 2

Implicit Euler remains stable for the tested step sizes. A large step can still reduce accuracy, so stability should not be confused with accuracy.

![Implicit Euler comparison for parameter set 2](figures/implicit_euler_set_2.png)

### SciPy BDF — Set 2

BDF is an implicit, variable-step solver intended for stiff ODEs. It provides a useful comparison with the hand-coded Euler methods.

![BDF comparison for parameter set 2](figures/bdf_set_2.png)

## Run locally

1. Create and activate a Python environment.
2. Install the listed dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the script from the repository root:

   ```bash
   python src/tissue_relaxation.py
   ```

   The figures are written to the `figures/` directory.

4. Or open `notebooks/tissue_temperature_relaxation.ipynb` in Jupyter or VS Code and run the cells in order.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── tissue_relaxation.py
├── notebooks/
│   └── tissue_temperature_relaxation.ipynb
└── figures/
    ├── explicit_euler_set_1.png
    ├── explicit_euler_set_2.png
    ├── implicit_euler_set_2.png
    └── bdf_set_2.png
```

## Notes

This repository presents a university numerical-modelling exercise as a standalone learning project. It does not include the original course notebook template, instructor-provided hidden tests, lecture slides or grading metadata. Before making a complete worked solution public, check that public sharing is permitted by the course rules.
