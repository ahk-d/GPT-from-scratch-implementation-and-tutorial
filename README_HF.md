# 🎭 Shakespeare Language Model Generator

A comprehensive implementation of language models from classical n-grams to GPT transformers, all trained on Shakespeare's complete works.

## 🚀 Features

- **Classical N-grams**: Statistical models with Byte-Pair Encoding
- **Neural N-grams**: Embedding-based neural networks  
- **GPT Models**: Transformer-based autoregressive models
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

This implementation is part of a comprehensive study comparing classical statistical methods to modern transformer architectures. The models demonstrate the evolution from discrete counting to learned embeddings to flexible attention mechanisms.

## 🔗 Links

- **Hugging Face Space**: [Shakespeare GPT](https://huggingface.co/spaces/ahk-d/shakespeare-gpt)
- **Research Paper**: Full implementation details and results
- **GitHub Repository**: Source code and training scripts

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{shakespeare_gpt_2024,
  title={From Classical N-grams to GPT Transformers: A Comprehensive Study},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
