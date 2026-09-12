# ARCHITECTURE.md

# Semantic Turing Field Architecture

## 1. Overview

The Semantic Turing Field (STF) is a semantic particle simulation in which words are represented as particles in a two-dimensional space and their interactions are determined by relationships between their semantic embeddings.

The system combines three primary components:

1. **Natural language processing** to obtain semantic representations of words.
2. **A force-based simulation engine** to evolve the positions of word particles.
3. **A Pygame visualization layer** to render and interact with the resulting field.

The architecture separates these responsibilities so that the semantic model, simulation engine, and visualization can be developed independently.

At a high level:

```text
                GloVe Embeddings
                       │
                       ▼
              ┌─────────────────┐
              │   NLP Layer     │
              │                 │
              │ Embeddings      │
              │ Semantics       │
              │ Preprocessing   │
              │ Clustering      │
              └────────┬────────┘
                       │
             Semantic information
                       │
                       ▼
              ┌─────────────────┐
              │  Core Layer     │
              │                 │
              │ STFSimulation   │
              │ Forces          │
              │ Gravity Wave    │
              │ Boundaries      │
              └────────┬────────┘
                       │
                  Particle state
                       │
                       ▼
              ┌─────────────────┐
              │ Visualization   │
              │                 │
              │ Renderer        │
              │ Camera          │
              │ Input Handler   │
              └─────────────────┘
```

The application entry point connects these layers but does not contain the underlying semantic or physical logic itself.

---

## 2. Repository Structure

The project is organized around three conceptual layers:

```text
src/
├── core/
│   ├── simulate_engine.py
│   ├── forces.py
│   ├── gravity_wave.py
│   └── boundaries.py
│
├── nlp/
│   ├── embeddings.py
│   ├── semantics.py
│   ├── clustering.py
│   └── text_preprocessing.py
│
├── visualization/
│   ├── renderer.py
│   ├── camera.py
│   ├── input_handler.py
│   └── colors.py
│
└── utils/
    ├── config.py
    └── save_state.py

app.py
```

The exact contents of individual directories may grow over time, but the responsibility of each layer should remain stable.

---

## 3. Application Layer

### `app.py`

`app.py` is the application entry point and orchestrator.

It is responsible for:

* parsing command-line arguments;
* loading or generating embeddings;
* preparing semantic vectors;
* constructing the cosine-similarity matrix;
* performing K-Means clustering;
* creating the `STFSimulation`;
* creating the visualization components;
* running the main Pygame event loop;
* passing user input into the simulation.

It should coordinate components rather than implement their internal algorithms.

The main initialization pipeline is:

```text
Load embeddings
      │
      ▼
Convert embeddings to vectors
      │
      ▼
Reduce vectors to 2D
      │
      ├───────────────┐
      ▼               ▼
Similarity matrix   K-Means
      │               │
      ▼               ▼
STFSimulation      Cluster labels
      │               │
      └───────┬───────┘
              ▼
        Visualization
```

---

# 4. NLP Layer

The NLP layer converts language into numerical representations that the simulation can use.

## 4.1 Embeddings

### `embeddings.py`

This module is responsible for loading the pretrained GloVe vocabulary.

The current system uses a GloVe embedding file and converts the vocabulary into a mapping:

```text
word → embedding vector
```

For example:

```text
"climate" → [v₁, v₂, ..., v₅₀]
"energy"  → [v₁, v₂, ..., v₅₀]
```

These vectors represent semantic information learned from large-scale text.

The simulation does not directly operate on the original high-dimensional embedding vectors as spatial coordinates.

---

## 4.2 Semantic Processing

### `semantics.py`

This module handles operations involving semantic relationships.

Its responsibilities include:

* dimensionality reduction;
* cosine similarity calculations;
* construction of the pairwise similarity matrix;
* computing similarities between an input sentence and vocabulary words.

The pairwise similarity matrix is represented as:

$$
S_{ij} =
\operatorname{cosine\_similarity}
(\mathbf{v}_i,\mathbf{v}_j)
$$

where \(\mathbf{v}_i\) and \(\mathbf{v}_j\) are embedding vectors.

The resulting matrix provides the semantic information required by the force model.

---

## 4.3 Text Preprocessing

### `text_preprocessing.py`

This module handles text entered by the user.

The preprocessing pipeline currently includes:

```text
Input sentence
      │
      ▼
Lowercase
      │
      ▼
Remove punctuation
      │
      ▼
Tokenize
      │
      ▼
Remove stopwords
      │
      ▼
Keep tokens represented in the embedding vocabulary
      │
      ▼
Average valid embeddings
      │
      ▼
Sentence embedding
```

The sentence embedding is computed as the mean of the available word embeddings.

If no valid vocabulary words remain, no gravity-wave force is applied.

---

## 4.4 K-Means Clustering

### `clustering.py`

K-Means provides a separate structural description of the semantic embedding space.

The current pipeline applies K-Means to the word embeddings and assigns each word a cluster ID.

These clusters are used primarily for visualization:

```text
Embedding vectors
       │
       ▼
     K-Means
       │
       ▼
Cluster assignments
       │
       ▼
Particle colors
```

Importantly, K-Means does **not** determine the pairwise force between particles.

The force model continues to use cosine similarity.

Therefore:

> **Cosine similarity determines interaction; K-Means determines visual grouping.**

This distinction is important to the conceptual architecture of STF.

---

# 5. Core Simulation Layer

The core layer contains the actual dynamical system.

## 5.1 `simulate_engine.py`

`STFSimulation` owns the state of the particle field.

For each particle, the simulation maintains:

* position;
* velocity.

The simulation also stores the current model parameters:

* \(\alpha\): interaction strength;
* \(\beta\): similarity threshold;
* \(dt\): timestep;
* damping.

A simulation step follows this general sequence:

```text
Current positions
       │
       ▼
Pairwise semantic forces
       │
       ├──── optional sentence gravity wave
       │
       ├──── boundary force
       │
       ▼
Total force
       │
       ▼
Update velocity
       │
       ▼
Update position
       │
       ▼
Apply damping
       │
       ▼
Limit maximum velocity
       │
       ▼
New particle state
```

The simulation engine does not render particles and does not process keyboard or mouse input.

---

## 5.2 Pairwise Forces

### `forces.py`

This module implements the primary semantic interaction between particles.

For particles \(i\) and \(j\), semantic similarity determines whether their interaction is attractive or repulsive.

The threshold \(\beta\) divides the interaction into two regimes:

```text
similarity > β  → attraction
similarity < β  → repulsion
```

The simulation therefore does not simply pull all semantically related words together.

Instead, similarity creates a signed interaction field that allows both cohesion and separation.

---

## 5.3 Gravity Wave

### `gravity_wave.py`

The gravity wave is an external, user-triggered semantic influence.

When a sentence is entered:

```text
Sentence
   │
   ▼
Preprocessing
   │
   ▼
Sentence embedding
   │
   ▼
Similarity to vocabulary
   │
   ▼
Semantic relevance
   │
   ▼
Gravity-wave force
```

Words that are sufficiently similar to the input sentence receive a stronger attraction toward the center of the field.

The gravity wave therefore provides a mechanism for querying the semantic field dynamically.

It does not modify the underlying word embeddings or the K-Means assignments.

---

## 5.4 Boundary Forces

### `boundaries.py`

Boundary forces keep particles within a useful simulation region.

They are separate from semantic forces because the boundaries are a visualization/simulation constraint rather than a semantic relationship.

---

# 6. Visualization Layer

The visualization layer translates simulation state into an interactive graphical representation.

## 6.1 Renderer

### `renderer.py`

The renderer is responsible for:

* drawing particles;
* assigning particle colors using cluster assignments;
* displaying word labels when particles are hovered;
* displaying simulation parameters;
* displaying control instructions;
* displaying the sentence input area;
* maintaining the visualization frame rate.

The renderer reads simulation state but should not modify the semantic model.

---

## 6.2 Camera

### `camera.py`

The camera converts between world coordinates and screen coordinates.

It handles:

* zooming;
* panning;
* coordinate transformations.

The camera therefore allows the simulation coordinate system to remain independent of the physical pixel dimensions of the Pygame window.

---

## 6.3 Input Handler

### `input_handler.py`

The input handler translates keyboard and mouse events into application actions.

Current interactions include:

* pause/resume;
* single-step simulation;
* reset;
* changing \(\alpha\);
* changing \(\beta\);
* changing damping;
* changing \(dt\);
* saving/loading state;
* camera interaction;
* entering sentences for the gravity wave.

The input handler communicates with the simulation and camera but does not implement their underlying algorithms.

---

# 7. Simulation State

The simulation state is intentionally separate from the visualization.

At any timestep, the important dynamic state can be represented as:

$$
\mathcal{S}(t)
=
\{
\mathbf{x}_1,\ldots,\mathbf{x}_N,
\mathbf{v}_1,\ldots,\mathbf{v}_N
\}
$$

where:

* \(\mathbf{x}_i\) is the position of particle \(i\);
* \(\mathbf{v}_i\) is its velocity.

Static semantic information includes:

$$
\mathcal{M}
=
\{
\mathbf{v}_i,
S_{ij},
c_i
\}
$$

where:

* \(\mathbf{v}_i\) is the embedding of word \(i\);
* \(S_{ij}\) is semantic similarity;
* \(c_i\) is the K-Means cluster assignment.

This distinction is useful because the embeddings and clusters remain fixed during normal simulation, while particle positions and velocities evolve over time.

---

# 8. Coordinate Systems

STF uses two different geometric representations.

## Semantic Embedding Space

The original GloVe vectors exist in a high-dimensional semantic space:

$$
\mathbf{v}_i \in \mathbb{R}^{d}
$$

where the current GloVe embeddings have \(d=50\).

Cosine similarity is computed in this semantic representation.

## Simulation Space

The particles move in a two-dimensional space:

$$
\mathbf{x}_i \in \mathbb{R}^{2}
$$

This space is the dynamic field displayed by Pygame.

Dimensionality reduction is therefore used to create a 2D representation of the embeddings, but the simulation's evolving particle positions are distinct from the original embedding coordinates.

This distinction is fundamental:

> The embedding space describes semantic relationships; the simulation space describes the evolving field.

---

# 9. Parameter Flow

The primary parameters are centralized in `config.py`.

Current defaults include:

| Parameter      | Meaning                                |
| -------------- | -------------------------------------- |
| `MAX_WORDS`    | Number of vocabulary embeddings loaded |
| `ALPHA`        | Pairwise interaction strength          |
| `BETA`         | Attraction/repulsion threshold         |
| `DT`           | Simulation timestep                    |
| `DAMPING`      | Velocity damping                       |
| `NUM_CLUSTERS` | Number of K-Means clusters             |
| `NUM_STEPS`    | Default headless simulation steps      |

The interactive controls expose several of these parameters during runtime.

This makes the simulation useful not only as a visualization but also as an experimental environment for studying how the field responds to different parameter regimes.

---

# 10. State Persistence

### `save_state.py`

Simulation state can be saved and restored.

The persistence layer allows the current simulation and camera state to be stored and later reloaded.

This is intentionally separated from the simulation engine so that serialization does not become part of the physics implementation.

---

# 11. Dependency Direction

The intended dependency direction is:

```text
Application
    │
    ├── NLP
    │
    ├── Core
    │
    └── Visualization

Core
    │
    ├── semantic information
    └── force components

Visualization
    │
    ├── reads simulation state
    └── handles user interaction
```

The core simulation should not depend on Pygame rendering.

The NLP layer should not depend on Pygame.

The renderer should not calculate semantic similarity or physical forces.

This separation allows the simulation to eventually be run without a graphical interface, which is already supported through the headless execution path.

---

# 12. Headless Execution

The application can run without Pygame visualization.

In headless mode:

```text
Load embeddings
      │
      ▼
Construct semantic model
      │
      ▼
Create simulation
      │
      ▼
Run N simulation steps
      │
      ▼
Report simulation statistics
```

This provides a foundation for:

* parameter experiments;
* benchmarking;
* automated tests;
* reproducibility;
* future GIF generation;
* future quantitative analysis.

---

# 13. Visualization and Export

The simulation state is independent of how it is displayed.

This allows the same simulation engine to support multiple outputs:

```text
                  STFSimulation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Pygame        GIF export   Analysis
      rendering       pipeline    scripts
```

For example, a GIF can be generated by repeatedly advancing the simulation, rendering selected states, and combining those frames into an animation.

This should remain an output concern rather than becoming part of the core simulation engine.

---

# 14. Design Principles

The architecture follows several principles:

### Separation of concerns

Semantic processing, simulation, visualization, and persistence have separate responsibilities.

### Reproducibility

Core simulation logic can run without the graphical interface.

### Replaceability

The embedding model, clustering method, force components, and renderer can theoretically be replaced independently.

### Extensibility

New forces should be implemented as independent force components and combined by the simulation engine.

### Interpretability

The relationship between semantic similarity, particle interaction, and visual clustering should remain explicit.

### Experimental control

Important simulation parameters are exposed through configuration and interactive controls.

---

# 15. Conceptual Summary

The Semantic Turing Field can therefore be understood as a pipeline:

$$
\text{Language}
\rightarrow
\text{Embeddings}
\rightarrow
\text{Semantic Relationships}
\rightarrow
\text{Forces}
\rightarrow
\text{Particle Dynamics}
\rightarrow
\text{Visualization}
$$

K-Means provides an additional descriptive layer:

$$
\text{Embeddings}
\rightarrow
\text{K-Means}
\rightarrow
\text{Cluster Labels}
\rightarrow
\text{Visual Appearance}
$$

while the gravity wave provides an interactive query mechanism:

$$
\text{Input Sentence}
\rightarrow
\text{Sentence Embedding}
\rightarrow
\text{Semantic Similarity}
\rightarrow
\text{External Force}
\rightarrow
\text{Field Response}
$$

Together, these components form the current architecture of the Semantic Turing Field.
