# Design Decisions
This document explains the major engineering decisions behind the Semantic Turing Field (STF), along with the tradeoffs considered during development.

The goal of STF is to create an interactive simulation where semantic relationships can be explored as a dynamic physical system.

# Why GloVe?

The project currently uses GloVe embeddings. This choice was primarily practical rather than theoretical.

GloVe provides:

* Pretrained embeddings that are easy to load
* Good semantic quality
* A lightweight model suitable for interactive visualization
* No dependency on large transformer models

For a project focused on visualization rather than NLP benchmarks, GloVe provides more than enough semantic information.

Could Word2Vec, FastText, or even sentence-transformers have been used instead? Of course.

The architecture of STF is intentionally embedding-agnostic. Any embedding model that produces dense vectors could replace GloVe with minimal code changes.

# Why PCA?

Because it's something I'm already familiar with, so it's a comfort choice.

Although PCA does have some well-known advantages:

* Fast
* Deterministic
* Widely understood
* Available in scikit-learn (another comfort choice)

Many embedding models produce vectors with hundreds of dimensions. The simulation repeatedly computes cosine similarities and semantic forces.

Reducing dimensionality decreases computational cost while preserving much of the semantic structure.

The small information loss is acceptable in my opinion, even if we're accounting for benchmark accuracy.

# Why Use Forces?

Rather than directly clustering words, STF models semantic relationships as physical interactions.

Each word behaves like a particle and semantic similarity determines whether particles attract or repel one another.

This allows clusters to emerge naturally through simulation instead of being explicitly assigned.

The project is inspired by force-directed graph layouts, particle systems, and dynamical systems rather than traditional NLP pipelines.

# Why Attraction and Repulsion?

And of course, if every particle only attracted every other particle, then the system would eventually collapse into a single point. We need both attraction and repulsion.

Introducing a threshold $\beta$ which creates two regimes:

Similarity above the threshold leads to attraction, similarity below the threshold leads to repulsion.

This produces a self-organizing system capable of forming stable semantic structures.

# Why Normalize Force Direction?

The original implementation used

$$
F
=

k(x_i-x_j)
$$

which caused force magnitude to increase with distance.

Large distances therefore produced even larger accelerations.

Instead, STF normalizes direction vectors

$$
\hat d
=
\frac{x_i-x_j}
{|x_i-x_j|}
$$

allowing semantic similarity to determine force magnitude while position determines only direction.

This greatly improves numerical stability.



# Why Average Forces?

Initially every interaction was summed:

$$
F_i
=

\sum_j
f_{ij}
$$

As vocabulary size increased, total force increased proportionally.

Using

$$
F_i
=

\frac1N
\sum_j
f_{ij}
$$

makes the simulation less dependent on the number of particles. Now the same parameters behave similarly for different vocabulary sizes.

# Velocity Damping

Real particle systems often include friction. Here we simply damped the velocity.

Without damping, small numerical errors accumulate and cause particles to gain energy indefinitely.

Velocity damping removes energy every timestep

```python
velocity *= damping
```

to prevent unstable oscillations while still allowing movement.

# Why Soft Boundaries?

Originally particles were clipped to a fixed square.

```python
position = clip(position)
```

This caused particles to bounce unnaturally against invisible walls.

Instead, particles outside a radius experience a gentle restoring force toward the origin.

This creates smoother motion while preventing particles from drifting infinitely far away.

# Why Gravity Waves?

A sentence is represented by the average of its word embeddings.

Cosine similarity is computed between the sentence embedding and every word embedding.

Particles with higher similarity receive an additional force toward the center of the simulation.

Conceptually,

the sentence behaves like a temporary disturbance of the semantic field,

allowing the visualization to react to user input without permanently altering the underlying semantic relationships.

This is one of the defining features of STF.

# Why K-Means?

The simulation itself does not use K-Means. K-Means is only used for visualization.

I simply use it to assign colors to groups of words so that the evolving structures are easier to interpret visually.

# Why Pygame Instead of Matplotlib?

Matplotlib was useful during early prototyping.

However, it is not designed for interactive simulations.

Pygame provides:

* A real-time game loop
* Smooth rendering
* Interactive camera controls
* Keyboard and mouse input
* Better performance

These capabilities make it a better fit for an exploratory visualization.

# Why Numba?

The force computation scales approximately as $O(N^2)$ because every particle interacts with every other particle.

Rather than redesigning the simulation I used `numba` was introduced to compile the nested loops into optimized machine code.

I'm hoping this improves performance, but I haven't actually measured it yet.

# What This Project Is Not

STF is **not** intended to be:

* a state-of-the-art NLP model,
* a cognitively accurate model of language,
* a replacement for transformer-based embeddings,
* or a production semantic search engine.

Instead, it is an experiment in viewing semantic spaces through the lens of dynamical systems and physics.

The emphasis is on intuition, visualization, and exploration rather than predictive performance.

# Future Directions

Several ideas were intentionally left out of the initial implementation to keep the project focused.

Potential extensions include:

* Moving attractors instead of a fixed origin
* Energy-based dynamics
* Barnes-Hut approximation for larger vocabularies
* GPU acceleration
* Dynamic vocabularies (word birth/death)
* Language evolution over time
* Multiple simultaneous sentence disturbances
* 3D semantic fields
* Transformer-based embeddings

Please check Experimental Features (MVP 3.5) milestone for issues on experimental features. Would love to have some contributors work on this if interested!