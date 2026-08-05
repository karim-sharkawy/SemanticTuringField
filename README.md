# Semantic Turing Field (STF)

The Semantic Turing Field (STF) is an experimental visualization that treats words as interacting particles in a dynamic physical system. STF investigates whether semantic spaces can be modeled as evolving dynamical systems rather than static geometric embeddings.

Sentences perturb the field like external forces, causing related concepts to attract, repel, and self-organize in real time.

## Features

- Physics-inspired 2D semantic simulation
- Interactive sentence perturbations and gravity-wave dynamics
- Fast, modular embedding loading and similarity computation
- Pygame visualization with camera controls
- Streamlit browser demo for real-time sentence exploration

## Architecture

```
Text input  -->  Embedding loader  -->  Similarity matrix
                                      |             |
                                      v             v
                     Semantic simulation engine  -->  Visualization
                                      ^
                                      |
                                 Sentence forces
```

### Components

- `src/nlp/embeddings.py` — load and cache GloVe vectors
- `src/nlp/semantics.py` — compute similarity and sentence influence
- `src/core/simulate_engine.py` — physics integration and field dynamics
- `src/visualization/*` — user controls, renderer, camera, and display
- `webui/app.py` — interactive Streamlit browser demo

## Installation

```bash
git clone https://github.com/karim-sharkawy/SemanticTuringField.git
cd SemanticTuringField
python -m pip install -r requirements.txt
```

## Usage

### Run the desktop visualization

```bash
python run.py
```

### Run with custom word budget

```bash
python -m src.app --words 5000
```

### Headless batch mode

```bash
python run.py --no-viz --steps 5000
```

## Web UI

Open the browser demo using Streamlit:

```bash
cd webui
python -m streamlit run app.py
```

The web demo lets users type a sentence, change the loaded vocabulary size, and observe the semantic field evolution instantly.

## Controls

- `Enter` — begin typing a sentence
- `Return` — submit sentence and trigger gravity-wave influence
- `Space` — pause / resume
- `Right Arrow` — single step while paused
- `R` — reset the simulation
- `Q` / `A` — increase / decrease alpha
- `W` / `S` — increase / decrease beta
- `E` / `D` — increase / decrease damping
- `T` / `G` — increase / decrease timestep
- `F5` — save state
- `F9` — load state
- `Esc` — quit

## Theory

STF treats each word as a particle in a semantic field. Pairwise semantic similarity defines interaction forces, while sentence inputs create gravity-wave perturbations that attract semantically related particles toward the query vector.

This approach is not a conventional embedding projection; it is an exploratory dynamical system that emphasizes how meaning may evolve under external influence.

For more detail, see `docs/semantic_theory.pdf`.

## Future Work

- Add multi-sentence context and phrase-level dynamics
- Support alternative embedding sets and transformer-based cues
- Add audio / microphone-driven semantic perturbations
- Add a web-native visualization with D3 or p5.js
- Extend the physics model with temperature, friction, and stochastic noise

## Dependencies

- `numpy==2.4.4`
- `pygame==2.6.1`
- `scikit-learn==1.8.0`
- `numba==0.66.0`
- `matplotlib==3.10.9`
- `streamlit==1.57.0`

## Citation

The word vectors used in this project are based on GloVe.

- Jeffrey Pennington, Richard Socher, Christopher D. Manning. "GloVe: Global Vectors for Word Representation." EMNLP 2014.

## Notes

- `src/utils/config.py` contains default runtime parameters.
- `run.py` is the project entry point for both visualization and headless execution.
- `webui/app.py` contains the browser demo separate from the Pygame application.
