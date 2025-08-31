---
title: Shakespeare Language Model Generator
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Generate Shakespearean text using classical n-grams, neural networks, and GPT models
---

# 🎭 Shakespeare Language Model Generator

A comprehensive implementation of language models from classical n-grams to GPT transformers, all trained on Shakespeare's complete works.

## 🚀 Features

- **Classical N-grams**: Statistical models with Byte-Pair Encoding (10.40 PPL best)
- **Neural N-grams**: Embedding-based neural networks (12.51 PPL best)  
- **GPT Models**: Transformer-based autoregressive models (13.08 PPL best)
- **Interactive Generation**: Real-time text generation with customizable parameters

## 📊 Model Performance

| Model Type | Best Performance | Method |
|------------|------------------|---------|
| Classical N-gram | 10.40 PPL | Flatten + 1000 merges + Backoff |
| Neural N-gram | 12.51 PPL | Flatten + 1000 merges + 4-gram |
| GPT | 13.08 PPL | Flatten + 1000 merges |

## 🛠️ Usage

1. Select a model type (Classical, Neural, or GPT)
2. Choose a specific model variant
3. Enter your prompt/context
4. Adjust generation parameters (length, temperature)
5. Click "Generate Text" to create Shakespearean text

## 📚 Research

This implementation demonstrates the evolution from classical statistical methods to modern transformer architectures, showing how architectural innovations enable better understanding and generation of human language.