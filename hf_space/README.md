---
title: Shakespeare Language Model Generator
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🎭 Shakespeare Language Model Generator

Generate Shakespearean text using classical n-grams, neural networks, or GPT models trained on Shakespeare's complete works!

## Features

- **Classical N-grams (Task 2)**: Statistical models using Byte-Pair Encoding with add-one smoothing and backoff
- **Neural N-grams (Task 3)**: Embedding-based neural networks trained on Shakespeare with early stopping  
- **GPT Models (Task 4)**: Transformer-based autoregressive models with causal self-attention

## Model Performance

- **Classical N-grams**: 10.40 PPL (Flatten + 1000 merges + Backoff)
- **Neural N-grams**: 12.51 PPL (Flatten + 1000 merges + 4-gram)
- **GPT Models**: 13.08 PPL (Flatten + 1000 merges)

All models are trained on Shakespeare's complete works and use consistent BPE tokenization.

## Usage

1. Select a model type (Classical N-gram, Neural N-gram, or GPT)
2. Choose a specific model variant
3. Enter your context/prompt text
4. Adjust generation parameters (max length, temperature)
5. Click "Generate Text" to create Shakespearean text!

## Example Prompts

- "to be or not to be"
- "fair is foul and foul is fair"
- "wherefore art thou romeo"
- "shall I compare thee"
- "now is the winter"

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
