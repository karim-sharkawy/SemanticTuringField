# Lessons Learned

Several important numerical simulation principles emerged during development (specifically because of all the errors during MVP3).

### Normalize accumulated forces

Summing interactions causes force magnitude to scale with particle count.

Averaging interactions produces more stable behavior.

### Normalize direction vectors

Using displacement directly causes force magnitude to increase with distance.

Separating direction from magnitude yields more physically stable dynamics.

### Numerical stability matters

Particle simulations can become unstable even when the equations are mathematically correct.

Common safeguards include

* velocity caps
* damping
* safe normalization (`+1e-8`)
* finite-value checks
* boundary restoring forces

### Numba prefers explicit loops

Vectorized NumPy is not always the fastest solution.

When using Numba,

simple nested loops often outperform vectorized expressions because they compile directly to optimized machine code.
