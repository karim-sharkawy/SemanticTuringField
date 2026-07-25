# Error Log

This document records significant implementation bugs encountered during development of the Semantic Turing Field (STF), along with their root causes and fixes.

# Error 1: Pygame Circle Center TypeError

```
TypeError: center argument must be a pair of numbers
```

Occurred in MVP 3 during the migration from Matplotlib to Pygame with respect to this code:

```python
pygame.draw.circle(
    self.screen,
    cluster_color(clusters[i]),
    (int(x), int(y)),
    self.particle_radius,
)
```

According to the Pygame documentation, `draw.circle()` expects

```
(tuple(int or float, int or float))
```

for the center argument.

Initially this suggested that `x` or `y` were not numeric.
However, once I checked their types, they both returned `<class 'numpy.float64'>`
which implies the renderer itself was not the source of the problem and only displayed a symptom.

The actual cause came inspection showed particle positions eventually become `nan`, hence the TypeError from pygame.
The renderer was functioning correctly—it merely detected invalid simulation data.

## Resolution

Added runtime validation before rendering.

```python
if not np.isfinite(x) or not np.isfinite(y):
    raise RuntimeError(
        "Particle position became invalid."
    )
```

This immediately exposed the true source of the instability.

# Error 2: Particle Explosion (NaN Positions)

## Symptoms

Simulation ran for about one second before crashing.
Runtime warnings included

```
overflow encountered in scalar multiply

overflow encountered in reduce

invalid value encountered in multiply
```

followed by

```
Invalid particle 330: (nan, nan)
```

## Investigation

The renderer revealed particles leaving the simulation entirely before crashing.

This is because a particle's position eventually became `inf` the `nan` because
the original force equation summed forces from every other particle.

```python
F[i] = np.sum(
    strength[:, None] * diff,
    axis=0,
)
```

Since every particle interacts with every other particle,

```
N particles

↓

N² interactions
```

the magnitude of the accumulated force increased approximately with the vocabulary size.

For 500 particles this meant hundreds of force vectors were added together every frame.

The resulting velocity became extremely large.

Large velocities produced enormous particle positions.

Eventually floating-point overflow occurred

```
1e308

↓

inf

↓

nan
```

## Resolution

The force computation was normalized by averaging instead of summing.

```python
F[i] = np.mean(
    strength[:, None] * direction,
    axis=0,
)
```

Mathematically,

instead of
### need to fix these math symbols
---

\[ F_i ===

\_j f\_ij \]

the implementation became

\[ F_i ===

\_j f\_ij \]

---

[
F_i
===

\sum_j f_{ij}
]

the implementation became

[
F_i
===

\frac{1}{N}
\sum_j
f_{ij}
]

This makes the average force approximately independent of vocabulary size.

# Error 3: Force Grew With Distance

## Original Implementation

```python
diff = pos[i] - pos

F += strength * diff
```

Since

```
||diff||
```

grows as particles separate, the force also grew proportionally.
Particles that drifted away experienced larger forces which made them accelerate even faster.

This created unstable feedback.

## Resolution

Normalize the direction vector.

```python
distance = ...

direction = diff / distance
```

Now `direction` only contains orientation and magnitude is determined entirely by semantic similarity.

Mathematically,
### need to fix these math symbols
instead of

[
F
=

k(x_i-x_j)
]

the implementation became

[
F
=

k
\frac{x_i-x_j}
{|x_i-x_j|}
]

This prevents force magnitude from increasing indefinitely with distance.

# Error 4: Boundary Force Produced NaNs

Runtime warning

```
invalid value encountered in multiply
```

Occurred inside `boundaries.py`
because the boundary force computed

```python
direction =
-position
/
distance
```

but once positions overflowed,
`position = inf` and `distance = inf`

Python then evaluates `inf / inf`, ending in `nan`,
which permanently stops the simulation.

## Resolution

Added safeguards against invalid distances.

Future versions should compute

```python
safe_distance =
max(distance, 1e-8)
```

and ignore non-finite particles before normalization.

# Error 5: Velocity Explosion

Originally the simulation integrated as

```python
velocity += force

position += velocity
```

with no velocity limit.

A single unusually large force could permanently destabilize the simulation.

## Resolution

Velocity capping was introduced.

```python
speed = np.linalg.norm(
    self.vel,
    axis=1,
)

mask = speed > max_speed

self.vel[mask] *= (
    max_speed
    /
    speed[mask]
)[:, None]
```

This limits maximum particle speed while preserving direction.
The cap is applied before updating positions.

# Error 6: Numba Incompatibility

After introducing Numba for Issue #13,
the simulation failed to compile due to:

```
TypingError:

No implementation of function
np.linalg.norm(..., axis=1)
```

Because Numba's `nopython` mode does not support every NumPy API.

Specifically,

```python
np.linalg.norm(
    diff,
    axis=1,
    keepdims=True,
)
```

is unsupported.

## Resolution

Replaced vectorized NumPy operations with explicit loops.

```python
dx = pos[i,0] - pos[j,0]
dy = pos[i,1] - pos[j,1]

distance = np.sqrt(
    dx*dx + dy*dy
) + 1e-8
```

It's more verbose, but results in compiled machine code without Python overhead.