# makemore

This repository contains implementations of character-level language models. Two character level language models are implemented to generate baby names using the *names.txt* training dataset. Another character level language model is implemented to generate creative Shakespeare-like text using *tinyshakespeare/input.txt* as the training dataset.

The project covers the implementation of a multilayer perceptron (MLP), a WaveNet-style model, and a Transformer model for generating text at the character level. The implementations for these are written from scratch in Python. I create my own mini version of PyTorch in the process writing the base classes for these architectures from scratch.

## Motivation
Large language models have become a fundamental part of modern-day AI research and Natural Language Processing. This project was inspired by Andrej Karpathy’s makemore series and aims to build a deep understanding of how these models function from the ground up.

Rather than relying on existing deep learning frameworks, I implemented the core building blocks of MLPs, WaveNet, and Transformer models from scratch, creating a mini version of PyTorch along the way. This approach provided hands-on experience with backpropagation, optimization techniques, model training and evaluation and architectural innovations like batch normalization, dropout, causal convolutions, and self-attention.

Through this project, I aimed to develop a strong intuition for designing and debugging deep learning models while deepening my understanding of low-level neural network implementations.


## Features
- Multilayer Perceptron (MLP) Model: A deep neural network trained on character sequences with feedforward linear layers. Implementation of the seminal paper by Bengio et al.: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

- WaveNet-Inspired Model: A convolutional model with hierarchical feature extraction. Implementation of the seminal paper by Google DeepMind: https://arxiv.org/abs/1609.03499

- Transformer Model: An implementation of the GPT-2 architecture with multiple self attention heads, layer norm, and dropout. Implementation of decoder-only version by Vaswani et al.: https://arxiv.org/abs/1706.03762. Also implements ideas from OpenAI GPT-3 release: https://arxiv.org/abs/2005.14165

## Files Overview

- minitorch.py - Core implementation details for the MLP, WaveNet, and Transformer models written from scratch to create a custom mini version of PyTorch.

- mlp.ipynb - Jupyter notebook for training and evaluating the MLP.

- wavenet.ipynb - Jupyter notebook for training and evaluating the WaveNet model.

- gpt2.ipynb - Jupyter notebook for training and evaluating the GPT-2 type transformer model.

- names.txt - Sample dataset of names used for training.


