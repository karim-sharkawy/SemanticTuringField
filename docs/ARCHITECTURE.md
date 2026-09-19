# ARCHITECTURE.md

# Semantic Turing Field Architecture

# 1. Overview

The Semantic Turing Field (STF) is a semantic particle simulation in which words are represented as particles in a two-dimensional space and their interactions are determined by relationships between their semantic embeddings.

The system combines three primary components:

1. Natural language processing to obtain semantic representations of words.
2. A force-based simulation engine to evolve the positions of word particles.
3. A Pygame visualization layer to render and interact with the resulting field.

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



# 2. Repository Structure

The project is organized around three conceptual layers, with additional
modules supporting offline precomputation, data distribution, and the web
application:

```
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
├── data/
│   └── hf_dataset_registry.py
│
└── utils/
    ├── config.py
    └── save_state.py

scripts/
└── precompute_fields.py

webui/
└── app.py

data/
├── raw GloVe data
└── precomputed/
    ├── 500/
    ├── 1000/
    ├── 2500/
    ├── 5000/
    └── full/

app.py
```

The `src/data/` layer provides access to precomputed STF configurations
stored in the Hugging Face Dataset repository.

The `scripts/` directory contains offline workflows used to generate those
configurations.

The `webui/` directory contains the Streamlit application, which loads
precomputed configurations rather than constructing the semantic field
from the original GloVe data at runtime.

The local `data/precomputed/` directory is therefore an offline build
artifact rather than a runtime dependency of the deployed web application.

The responsibility of each layer remains separated even though the project
supports multiple execution paths.



# 3. Application Layer

The project has two primary application paths:

1. Pygame application — used for local experimentation and interactive
   simulation.
2. Streamlit web application — used to present the precomputed semantic
   fields through a browser-based interface.

## 3.1 `app.py`

`app.py` is the main local Pygame application entry point and orchestrator.

It is responsible for:

* loading or generating the semantic field for local execution;
* preparing the NLP components required by the simulation;
* constructing the `STFSimulation`;
* creating the visualization components;
* running the main Pygame event loop;
* passing user input into the simulation.

It coordinates components rather than implementing their underlying
algorithms.

For local execution, the initialization pipeline is:

```
Load embeddings
      │
      ▼
Prepare semantic vectors
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

## 3.2 `webui/app.py`

`webui/app.py` is the browser-based Streamlit application.

Unlike the local Pygame application, the web application does not construct
the semantic field from the original GloVe embeddings.

Instead, it loads one of the precomputed STF configurations from the
Hugging Face Dataset repository.

The runtime pipeline is:

```
User selects field size
        │
        ▼
Hugging Face Dataset
        │
        ▼
Precomputed STF artifacts
        │
        ▼
Reconstruct simulation state
        │
        ▼
Streamlit + Pygame Renderer
        │
        ▼
Interactive semantic field
```

The supported configurations are:
```
500 words
1,000 words
2,500 words
5,000 words
Full vocabulary
```

The web application still performs simulation steps at runtime.

In particular, sentence input is processed dynamically so that a sentence
can produce a semantic gravity-wave disturbance in the currently loaded
field.

The expensive field-construction operations are therefore performed
offline, while the interactive simulation remains dynamic.



# 4. Offline Precomputation and Data Distribution

STF separates expensive semantic-field construction from runtime
visualization.

The semantic field configurations used by the web application are
generated offline by:

### `scripts/precompute_fields.py`

The precomputation pipeline performs:

```
GloVe embeddings
       │
       ▼
Vocabulary filtering
       │
       ▼
Embedding matrix
       │
       ▼
Semantic vectors
       │
       ├──────────────────┐
       ▼                  ▼
Similarity matrix       K-Means
       │                  │
       └────────┬─────────┘
                ▼
       Initial particle state
                │
                ▼
        Precomputed artifacts
```

The same five configurations are used. 

Each configuration contains the information required to reconstruct the
corresponding STF field without repeating the expensive preprocessing
steps.

Typical artifacts include:
```
embeddings.npy
vecs.npy
similarity.npy
clusters.npy
positions.npy
velocities.npy
words.json
labels.json
metadata.json
```

These artifacts contain both static semantic information and the initial
dynamic particle state.

## 4.1 Hugging Face Dataset

The precomputed configurations are distributed through a Hugging Face
Dataset repository.

The repository is organized by field size:

```
fields/
├── 500/
├── 1000/
├── 2500/
├── 5000/
└── full/
```

Each directory contains the artifacts for that field configuration.

The Hugging Face repository acts as the runtime data source for the
Streamlit application.

This avoids requiring the deployed application to:

* download and parse the original GloVe archive;
* construct the embedding matrix;
* compute the pairwise similarity matrix;
* perform K-Means clustering;
* generate the initial particle state.

These operations are performed once during the offline precomputation
stage.

## 4.2 `hf_dataset_registry.py`

`src/data/hf_dataset_registry.py` provides the interface between STF and
the Hugging Face Dataset repository.

Its responsibilities include:

* defining the supported field configurations;
* downloading precomputed artifacts;
* loading NumPy and JSON data;
* reconstructing the embedding dictionary;
* reconstructing the `STFSimulation`;
* restoring the precomputed initial particle state;
* uploading locally generated configurations to Hugging Face.

The registry therefore separates data distribution from the Streamlit
application itself.

## 4.3 Runtime Caching

The Streamlit application caches loaded field configurations.

When a field is requested for the first time, its artifacts are retrieved
from Hugging Face and loaded into memory.

Subsequent Streamlit reruns can reuse the cached configuration rather than
reconstructing or repeatedly downloading the field.

The architecture is therefore:

```
                 OFFLINE
                    │
                    ▼
              GloVe dataset
                    │
                    ▼
          Precompute STF fields
                    │
                    ▼
             Hugging Face
                    │
                    │
                    ▼
                 RUNTIME
                    │
                    ▼
          Streamlit application
                    │
                    ▼
        Cached precomputed field
                    │
                    ▼
        Dynamic STF simulation
                    │
                    ▼
             Visualization
```

This separation allows the web application to remain lightweight while
preserving the dynamic behavior of the simulation.

# 5. NLP Layer

The NLP layer is responsible for converting the original GloVe vocabulary
into the semantic representations required by STF.

During offline precomputation, these components are used to construct the
semantic information stored in the Hugging Face Dataset repository.

The Streamlit application generally consumes the resulting precomputed
artifacts rather than repeating these operations at runtime.

## 5.1 Embeddings

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



## 5.2 Semantic Processing

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



## 5.3 Text Preprocessing

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



## 5.4 K-Means Clustering

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

Importantly, K-Means does not determine the pairwise force between particles.

The force model continues to use cosine similarity.

Therefore:

> Cosine similarity determines interaction; K-Means determines visual grouping.

This distinction is important to the conceptual architecture of STF.



# 6. Core Simulation Layer

The core layer contains the actual dynamical system.

## 6.1 `simulate_engine.py`

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



## 6.2 Pairwise Forces

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



## 6.3 Gravity Wave

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



## 6.4 Boundary Forces

### `boundaries.py`

Boundary forces keep particles within a useful simulation region.

They are separate from semantic forces because the boundaries are a visualization/simulation constraint rather than a semantic relationship.



# 7. Visualization Layer

The visualization layer translates simulation state into an interactive graphical representation.

## 7.1 Renderer

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



## 7.2 Camera

### `camera.py`

The camera converts between world coordinates and screen coordinates.

It handles:

* zooming;
* panning;
* coordinate transformations.

The camera therefore allows the simulation coordinate system to remain independent of the physical pixel dimensions of the Pygame window.



## 7.3 Input Handler

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



# 8. Simulation State

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



# 9. Coordinate Systems

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



# 10. Parameter Flow

The primary simulation parameters are centralized in `config.py`.

Current defaults include:

| Parameter       | Meaning                               |
|  | - |
| `MAX_WORDS`     | Number of vocabulary embeddings loaded |
| `ALPHA`         | Pairwise interaction strength          |
| `BETA`          | Attraction/repulsion threshold         |
| `DT`            | Simulation timestep                    |
| `DAMPING`       | Velocity damping                       |
| `NUM_CLUSTERS`  | Number of K-Means clusters             |
| `NUM_STEPS`     | Default headless simulation steps      |

For the local Pygame application, several simulation parameters can be
modified interactively through the input handler.

The Streamlit web application intentionally exposes a smaller interface.
Users select one of five precomputed vocabulary configurations rather than
directly modifying the underlying physics parameters.

The web application's field choices are:

THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS
THE SAME AS ALWAYS

The corresponding simulation parameters are fixed during precomputation and
stored as part of the field configuration.

This creates two distinct use cases:

```
Local experimentation
        │
        ├── adjustable simulation parameters
        └── experimental control

Web application
        │
        ├── fixed precomputed configurations
        └── simplified user interaction
```

This separation keeps the web interface focused on exploring the semantic
field rather than exposing implementation-level simulation controls.





# 11. State Persistence

### `save_state.py`

Simulation state can be saved and restored for local experimentation.

The persistence layer allows the current simulation and camera state to be
stored and later reloaded.

This is intentionally separated from the simulation engine so that
serialization does not become part of the physics implementation.

The precomputed field artifacts used by the web application are a separate
form of persistence.

They represent a canonical initial configuration of an STF field, including
its semantic information and initial particle state.

Therefore:

```
save_state.py
     │
     └── local experiment/session state

Hugging Face Dataset
     │
     └── distributable precomputed field configurations
```

The Hugging Face configurations are not intended to capture a user's
individual simulation session. They provide deterministic starting states
from which the web application can run the simulation.



# 12. Dependency Direction

The intended dependency direction is:

```
                    Application
                   /     |      \
                  /      |       \
                 ▼       ▼        ▼
               NLP     Core   Visualization
                 │       ▲
                 │       │
                 └───────┘

          Offline Precomputation
                   │
                   ▼
                 NLP
                   │
                   ▼
                 Core
                   │
                   ▼
          Precomputed artifacts
                   │
                   ▼
            Hugging Face Dataset
                   │
                   ▼
             Streamlit Web UI
```

More specifically,

```
NLP
 │
 ├── embeddings
 ├── semantic processing
 ├── preprocessing
 └── clustering
          │
          ▼
   Precomputation pipeline
          │
          ▼
   Precomputed field data
          │
          ▼
   Hugging Face Dataset
          │
          ▼
   Web application
          │
          ▼
   Core + Visualization
```

The core simulation should not depend on Pygame rendering.

The NLP layer should not depend on Pygame.

The renderer should not calculate semantic similarity or physical forces.

The Hugging Face data layer should provide data to the application without
embedding web-specific logic into the core simulation.

The Streamlit application is therefore an orchestration layer that connects
distributed precomputed data with the existing simulation and visualization
components.

This separation allows the simulation to continue running independently of
the graphical interface and allows the same precomputed semantic fields to
be consumed by different applications in the future.



# 13. Headless Execution

The simulation can run without Pygame visualization.

For local or offline execution, the pipeline is:

```
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

Headless execution provides a foundation for:

* parameter experiments;
* benchmarking;
* automated tests;
* reproducibility;
* GIF generation;
* quantitative analysis;
* offline precomputation.

The web application represents a separate runtime path in which the
semantic field has already been precomputed:

```
Load precomputed field
        │
        ▼
Create simulation
        │
        ▼
Run dynamic simulation steps
        │
        ▼
Optional sentence gravity wave
        │
        ▼
Report or visualize state
```

This distinction prevents expensive semantic preprocessing from being
repeated during normal web application startup.



# 14. Visualization and Export

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



# 15. Design Principles

The architecture follows several principles:

### Separation of concerns

Semantic processing, simulation, visualization, persistence, and data
distribution have separate responsibilities.

### Offline computation

Expensive semantic-field construction is performed offline rather than
during web application startup.

### Reproducibility

Precomputed field configurations use fixed configurations and initial states
so that the web application can consistently reconstruct the same starting
field.

### Replaceability

The embedding model, clustering method, force components, renderer, and data
distribution mechanism can theoretically be replaced independently.

### Extensibility

New forces should be implemented as independent force components and
combined by the simulation engine.

### Interpretability

The relationship between semantic similarity, particle interaction, and
visual clustering should remain explicit.

### Runtime simplicity

The web application should consume precomputed semantic information rather
than reproduce the entire NLP preprocessing pipeline.

### Experimental control

The local application retains adjustable simulation parameters for
experimentation, while the web application presents a controlled set of
precomputed field configurations.



# 16. Conceptual Summary

The offline construction of the Semantic Turing Field can be understood as:

$$
\text{Language}
\rightarrow
\text{Embeddings}
\rightarrow
\text{Semantic Relationships}
\rightarrow
\text{Precomputed Field}
$$

The runtime simulation then evolves that field:

$$
\text{Precomputed Field}
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

The complete architecture can therefore be summarized as:

```
                    OFFLINE
                       │
                       ▼
                GloVe Embeddings
                       │
                       ▼
                 NLP Processing
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Similarity Matrix      K-Means
              │                 │
              └────────┬────────┘
                       ▼
              Precomputed Fields
                       │
                       ▼
              Hugging Face Dataset
                       │
                       │
                    RUNTIME
                       │
                       ▼
              Streamlit Application
                       │
                       ▼
              STF Simulation Engine
                       │
              ┌────────┴─────────┐
              ▼                  ▼
       Semantic Forces     Sentence Gravity
              │                  │
              └────────┬─────────┘
                       ▼
                Particle Dynamics
                       │
                       ▼
                  Visualization
```

This architecture separates expensive semantic preprocessing from runtime
simulation while preserving the interactive behavior that defines the
Semantic Turing Field.

The result is a system that can be explored locally as an experimental
simulation and served through a lightweight web interface using the same
underlying precomputed semantic fields.