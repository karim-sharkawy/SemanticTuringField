# Semantic Turing Field (STF)

The Semantic Turing Field (STF) is an experimental visualization that treats words as interacting particles in a dynamic physical system. STF investigates whether semantic spaces can be modeled as evolving dynamical systems rather than static geometric embeddings. While primarily a visualization project, it provides an intuitive framework for exploring semantic organization, contextual perturbations, and emergent structure through concepts borrowed from statistical physics and dynamical systems.

Sentences perturb the field like external forces, causing clusters of related words to reorganize in real time.

<div align="center">
    (GIF of simulation)
</div>

## Overview

Traditional embedding visualizations are static.

STF explores a different question:

> **What if semantic space behaved like a physical system?**

Each word becomes a particle whose motion is governed by interactions derived from semantic similarity. As the simulation evolves, coherent semantic structures naturally emerge.

This project is intended as an exploration of physics-inspired NLP rather than a production language model.

## Features

- Physics-inspired semantic simulation
- Interactive sentence perturbations
- Dynamic clustering of word meanings
- Real-time visualization
- Modular simulation framework

## Example

Input:

"The ocean is calm today."

The sentence produces a semantic disturbance that pulls ocean-related vocabulary together while slightly reshaping the surrounding semantic landscape.

(Insert GIF)

## Theory

The simulation is based on a simple interaction model where semantic similarity determines attractive and repulsive forces between words.

Rather than including the complete mathematical derivation here, the theoretical framework is documented separately.

📄 **See:** `docs/semantic_theory.pdf`

The document covers

- semantic particles
- force equations
- energy interpretation
- gravity-wave sentence perturbations
- simulation dynamics
- future extensions

## Installation

```bash
git clone https://github.com/karim-sharkawy/SemanticTuringField.git
cd SemanticTuringField

pip install -r requirements.txt

python -m srcipts/main.py