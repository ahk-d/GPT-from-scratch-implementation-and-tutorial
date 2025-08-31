# GPT from Scratch Implementation and Tutorial

This repository contains a complete implementation of GPT (Generative Pre-trained Transformer) from scratch, following the progression from traditional language models to modern transformer architectures.

## Project Structure

```
├── task1.py              # BPE Tokenization Implementation
├── task2.py              # N-gram Language Models
├── task3.py              # Neural Bigram Models
├── task4.py              # Complete GPT Implementation
├── requirements.txt       # Python dependencies
├── task4_README.md        # Detailed technical report for Task 4
├── Shakespeare_clean_train.txt  # Training data
├── Shakespeare_clean_test.txt   # Test data
└── Shakespeare_clean_full.txt   # Full dataset
```

## Tasks Overview

### Task 1: BPE Tokenization
- Implements Byte Pair Encoding (BPE) from scratch
- Tests different merge counts and normalization techniques
- Evaluates tokens per word and reconstruction quality

**Run with:** `python task1.py`

### Task 2: N-gram Language Models
- Implements n-gram models (n=1..4) with Laplace smoothing
- Uses best BPE configuration from Task 1
- Includes interpolation and perplexity evaluation

**Run with:** `python task2.py`

### Task 3: Neural Bigram Models
- Implements neural bigram model with PyTorch
- Uses embeddings and neural networks
- Includes early stopping and text generation

**Run with:** `python task3.py`

### Task 4: Complete GPT Implementation
- Full GPT model with transformer architecture
- Causal self-attention implementation from scratch
- Comprehensive comparison of all model types
- Hyperparameter tuning and evaluation

**Run with:** `python task4.py`

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Files:**
   Ensure the Shakespeare dataset files are in the root directory:
   - `Shakespeare_clean_train.txt`
   - `Shakespeare_clean_test.txt`
   - `Shakespeare_clean_full.txt`

3. **Run Tasks:**
   ```bash
   # Run individual tasks
   python task1.py
   python task2.py
   python task3.py
   python task4.py
   
   # Or run all tasks sequentially
   python task1.py && python task2.py && python task3.py && python task4.py
   ```

## Key Features

- **Complete Implementation:** From basic tokenization to full GPT model
- **Educational:** Well-documented code with detailed explanations
- **Modular Design:** Each task builds upon the previous one
- **Comprehensive Evaluation:** Perplexity metrics and text generation
- **Hyperparameter Tuning:** Systematic exploration of model configurations

## Results

Each task generates:
- Console output with real-time progress
- Pickle files with detailed results (robust serialization)
- Training plots (for neural models)
- Generated text samples

## Technical Details

- **BPE Merge Counts:** [1000, 2000, 2500] as specified
- **Model Comparison:** N-gram → Neural Bigram → GPT
- **Evaluation:** Perplexity on validation/test sets
- **Text Generation:** Temperature and top-k sampling

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Seaborn

## Notes

- All tasks read data from the current directory (`./`)
- Tasks are designed to run independently
- Results are saved in pickle format for robust serialization
- Training uses early stopping to prevent overfitting

---

For detailed technical documentation, see `task4_README.md` for the complete GPT implementation report.
