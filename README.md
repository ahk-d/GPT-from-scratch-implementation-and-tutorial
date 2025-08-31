# GPT from Scratch Implementation

A complete implementation of a GPT (Generative Pre-trained Transformer) model from scratch, including tokenization, language modeling, and transformer architecture.

## Demo

Test the models at: [https://huggingface.co/spaces/ahk-d/shakespeare-gpt](https://huggingface.co/spaces/ahk-d/shakespeare-gpt)

## Project Overview

This project implements a complete GPT model through four progressive tasks:

1. **Task 1: BPE Tokenization** - Byte Pair Encoding for efficient text tokenization
2. **Task 2: N-gram Language Modeling** - Statistical language modeling with n-gram approaches  
3. **Task 3: Neural Bigram Model** - Neural network implementation for bigram language modeling
4. **Task 4: GPT Architecture** - Complete transformer implementation with attention mechanisms and text generation

## Files

- `gpt_from_scratch_final.py` - Main implementation file with all tasks
- `gpt_from_scratch_final.ipynb` - Jupyter notebook version with detailed explanations
- `app.py` - Streamlit web application for model testing
- `requirements.txt` - Python dependencies
- `gpt-report/` - Interactive Next.js report showcasing the implementation

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Or explore the interactive report:
```bash
cd gpt-report
npm install
npm run dev
```

## Model Files

The repository includes several trained model checkpoints:
- `gpt_medium_merges2500_ep25_valloss5.8653_valppl352.59.pt` - Medium GPT model
- `gpt_merges2500_gpt_best_ep1_valloss5.6638_valppl288.24.pt` - Best performing GPT model
- `gpt_small_merges2500_ep24_valloss6.0070_valppl406.25.pt` - Small GPT model
- Various neural network models for different configurations

## Technical Details

- **Language**: Python with PyTorch
- **Tokenization**: Byte Pair Encoding (BPE)
- **Architecture**: Transformer with multi-head attention
- **Training**: Shakespeare dataset with various model sizes
- **Evaluation**: Perplexity and loss metrics

## Interactive Report

The `gpt-report/` directory contains an interactive Next.js application that provides:
- Visual flow diagrams for each task
- Code snippets and implementation details
- Performance metrics and results
- Architecture breakdowns

## Learn More

This implementation demonstrates:
- Transformer architecture from first principles
- Attention mechanisms and causal masking
- Position embeddings and multi-head attention
- Advanced text generation with sampling techniques
- Training optimization and early stopping strategies

The project provides a solid foundation for understanding modern language models and experimenting with transformer architectures.
