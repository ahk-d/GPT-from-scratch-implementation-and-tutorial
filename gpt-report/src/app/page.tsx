'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function Home() {
  const [markdownContent] = useState(`# Building GPT from Scratch: A Comprehensive Implementation Guide

## Abstract

This paper presents a systematic approach to implementing a Generative Pre-trained Transformer (GPT) model from the ground up. We break down the implementation into four fundamental building blocks: Byte Pair Encoding (BPE) tokenization, n-gram language modeling, neural bigram models, and the complete transformer architecture. Each component is implemented and evaluated independently, providing insights into the evolution from statistical to neural language modeling approaches. Our implementation achieves competitive perplexity scores on Shakespeare text data, demonstrating the effectiveness of the transformer architecture for language modeling tasks.

## 1. Introduction

Language models have become fundamental to modern natural language processing, enabling applications ranging from text generation to machine translation. The transformer architecture, introduced by Vaswani et al. (2017), revolutionized the field by replacing recurrent neural networks with self-attention mechanisms. This paper presents a comprehensive implementation of GPT from scratch, systematically building each component to understand the underlying mechanics.

### 1.1 Motivation

Building language models from scratch provides several benefits:
- **Deep Understanding**: Implementing each component reveals the underlying mechanics
- **Educational Value**: Hands-on experience with neural network fundamentals
- **Customization**: Enables modifications for specific use cases
- **Debugging**: Better understanding leads to more effective problem-solving
- **Research Foundation**: Basis for experimenting with new architectures

### 1.2 Key Concepts

This implementation explores four fundamental concepts:
1. **Perplexity**: The primary evaluation metric for language models
2. **BPE Tokenization**: Subword tokenization for efficient text processing
3. **Attention Mechanisms**: Neural mechanisms for focusing on relevant input parts
4. **Transformer Architecture**: Modern neural architecture using self-attention

## 2. Understanding Perplexity: The Language Model's Report Card

### 2.1 Definition and Intuition

Perplexity is the primary metric used to evaluate language models. It measures how "surprised" or "confused" the model is when trying to predict the next word in a sequence. Intuitively, if perplexity = 10, the model is as uncertain as if it had to choose randomly from 10 equally likely options.

### 2.2 Mathematical Formulation

Perplexity is calculated using the following formula:

\`\`\`
Perplexity = exp(-average_log_probability)
\`\`\`

Where the average log probability is computed over all tokens in the test set:

\`\`\`
average_log_probability = (1/N) * Σ log(P(token_i | context_i))
\`\`\`

### 2.3 Properties and Benefits

Perplexity offers several advantages:
- **Interpretable**: Lower values always indicate better performance
- **Language Independent**: Works across different languages and domains
- **Mathematically Sound**: Based on information theory principles
- **Comparable**: Allows direct comparison between different models
- **Practical**: Correlates well with human judgment of text quality

### 2.4 Typical Values

- **Poor Performance**: Perplexity > 100
- **Moderate Performance**: Perplexity 20-100
- **Good Performance**: Perplexity < 20

## 3. Task 1: Byte Pair Encoding (BPE) Tokenization

### 3.1 Background and Motivation

Byte Pair Encoding (BPE) is a subword tokenization algorithm that breaks down text into smaller, meaningful units called tokens. Unlike word-level tokenization that treats each word as a separate unit, BPE can handle unknown words by breaking them into known subword pieces.

Subword tokenization strikes a balance between vocabulary size and token efficiency. It can represent rare words using common subwords while keeping the vocabulary manageable.

### 3.2 Algorithm Description

BPE operates through the following steps:

1. **Character-Level Initialization**: Start with individual characters as the initial vocabulary
2. **Frequency Analysis**: Count how often each pair of adjacent tokens appears in the training corpus
3. **Iterative Merging**: Repeatedly merge the most frequent pair into a new token until reaching the desired vocabulary size
4. **Tokenization**: Apply the learned merges to encode new text into tokens

### 3.3 Example: BPE in Action

Consider the word "hello":
- **Initial**: "hello" → ["h", "e", "l", "l", "o"]
- **After merge 1**: "ll" becomes common → ["h", "e", "ll", "o"]
- **After merge 2**: "he" becomes common → ["he", "ll", "o"]
- **Final**: "hello" → ["he", "ll", "o"]

### 3.4 Implementation Results

Our BPE implementation achieved the following results:

**Best Configuration:**
- Normalization: \`lower_nopunct\`
- Merge count: 2000
- Validation avg tokens/word: 1.3129

**Performance Metrics:**
- Final vocab size: 1927
- Compression ratio: 0.00
- Reconstruct accuracy: 100%

## 4. Task 2: N-gram Language Modeling

### 4.1 Statistical Language Modeling

N-gram models are statistical language models that predict the next word based on the previous N-1 words. They represent one of the simplest yet most effective approaches to language modeling, serving as a baseline for more sophisticated neural methods.

The n-gram assumption states that the probability of a word depends only on the previous N-1 words, not the entire history. This is called the Markov assumption.

### 4.2 Types of N-gram Models

1. **Unigram (n=1)**: P(word) - probability of word regardless of context
2. **Bigram (n=2)**: P(word|previous_word) - probability given previous word
3. **Trigram (n=3)**: P(word|previous_2_words) - probability given previous 2 words
4. **4-gram (n=4)**: P(word|previous_3_words) - probability given previous 3 words

### 4.3 The Sparsity Problem

Higher-order n-grams capture more context but suffer from data sparsity - many n-gram combinations never appear in the training data, leading to zero probabilities.

**Example**: If "the cat sat on the" never appears in training data, P("mat"|"the cat sat on the") = 0, even though it's a perfectly valid continuation.

### 4.4 Smoothing Techniques

#### 4.4.1 Laplace Smoothing (Add-1)
Add 1 to all counts to avoid zero probabilities:
\`\`\`
P(word|context) = (count + 1) / (total + vocab_size)
\`\`\`

#### 4.4.2 Interpolation
Combine probabilities from different n-gram orders:
\`\`\`
P(word) = λ₁P₁(word) + λ₂P₂(word) + λ₃P₃(word)
\`\`\`

### 4.5 Results

**BPE 1000 Merges:**
- n=1: Val=219.48, Test=219.48
- n=2: Val=219.48, Test=219.48
- n=3: Val=321.86, Test=322.78
- n=4: Val=448.41, Test=449.96

**BPE 2000 Merges:**
- n=1: Val=79.50, Test=79.50
- n=2: Val=79.50, Test=79.50
- n=3: Val=121.28, Test=120.70
- n=4: Val=204.40, Test=204.07

## 5. Task 3: Neural Bigram Language Modeling

### 5.1 Bridging Statistical and Neural Approaches

Neural bigram models represent the transition from statistical to neural language modeling. Instead of counting word pairs, we learn distributed representations (embeddings) that capture semantic relationships between words.

The key innovation is that words are represented as dense vectors (embeddings) rather than discrete symbols, allowing the model to learn semantic similarities and generalizations.

### 5.2 Model Architecture

The neural bigram model consists of:
1. **Embedding Layer**: Converts word indices to dense vector representations
2. **Neural Processing**: Learns complex patterns and relationships
3. **Output Projection**: Projects processed embeddings back to vocabulary space

### 5.3 Advantages Over Statistical Models

- **Semantic Understanding**: Can generalize to unseen word combinations
- **Continuous Representations**: Words with similar meanings have similar embeddings
- **Feature Learning**: Automatically discovers useful features from data
- **Scalability**: Can handle large vocabularies efficiently
- **Flexibility**: Easy to extend with additional layers or features

### 5.4 Training Process

**Key Components:**
- **Loss Function**: Cross-entropy loss between predicted and actual next words
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Weight decay and dropout to prevent overfitting
- **Early Stopping**: Stop training when validation perplexity stops improving

### 5.5 Results

**BPE 1000 Merges:**
- Best config: emb_dim=128, lr=0.001, wd=1e-05
- Val perplexity: 29.92
- Test perplexity: 28.53

**BPE 2000 Merges:**
- Best config: emb_dim=128, lr=0.001, wd=0.0001
- Val perplexity: 38.47
- Test perplexity: 33.66

## 6. Task 4: GPT Architecture - The Full Transformer

### 6.1 The Transformer Revolution

The Transformer architecture, introduced in "Attention Is All You Need" (2017), revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms. GPT (Generative Pre-trained Transformer) applies this architecture to language modeling.

The key innovation is that self-attention allows the model to directly model relationships between any positions in the sequence, regardless of distance, enabling parallel processing and better capture of long-range dependencies.

### 6.2 Core Components

1. **Multi-Head Attention**: Multiple attention mechanisms running in parallel, each focusing on different aspects of the input
2. **Positional Encoding**: Adds position information to embeddings since attention is position-agnostic
3. **Feed-Forward Networks**: Two-layer neural networks applied to each position independently
4. **Layer Normalization**: Stabilizes training by normalizing activations within each layer

### 6.3 Self-Attention Mechanism

The self-attention mechanism operates in three steps:

1. **Query, Key, Value Computation**: For each position, compute Query (Q), Key (K), and Value (V) vectors using learned linear transformations
2. **Attention Scores**: Compute attention scores: \`Attention(Q,K,V) = softmax(QK^T/√d_k)V\`, where d_k is the dimension of keys
3. **Causal Masking**: For language modeling, mask future positions to prevent the model from "cheating" by looking ahead

### 6.4 Advantages of Self-Attention

- **Parallelization**: All attention computations can be done simultaneously
- **Long-Range Dependencies**: Direct connections between any positions
- **Interpretability**: Attention weights show which positions the model focuses on
- **Scalability**: Can handle sequences of varying lengths
- **Flexibility**: Easy to modify attention patterns for different tasks

### 6.5 Implementation Status

The complete GPT implementation is currently in progress. This will include:
- Full transformer architecture with multiple layers
- Multi-head attention implementation
- Positional encoding
- Training and evaluation on the complete model

## 7. Discussion and Analysis

### 7.1 Performance Comparison

Our results show a clear progression in model performance:

1. **BPE Tokenization**: Achieved 100% reconstruction accuracy with efficient subword representation
2. **N-gram Models**: Demonstrated the impact of vocabulary size on perplexity (79.50 vs 219.48 for unigrams)
3. **Neural Bigram**: Significantly improved performance over statistical models (28.53 vs 79.50 perplexity)
4. **GPT Architecture**: Expected to achieve state-of-the-art performance (implementation in progress)

### 7.2 Key Insights

1. **Vocabulary Size Matters**: Larger BPE vocabularies (2000 vs 1000 merges) significantly improve n-gram performance
2. **Neural Approaches Superior**: Neural bigram models achieve much lower perplexity than statistical n-grams
3. **Subword Tokenization Effective**: BPE provides a good balance between vocabulary size and token efficiency
4. **Transformer Architecture Promising**: The attention mechanism addresses key limitations of previous approaches

### 7.3 Limitations and Future Work

**Current Limitations:**
- Limited to Shakespeare text data
- Small model sizes compared to production systems
- No comparison with other tokenization methods
- Transformer implementation not yet complete

**Future Work:**
- Implement complete GPT architecture
- Evaluate on larger datasets
- Compare with other tokenization methods (SentencePiece, WordPiece)
- Experiment with different model sizes and architectures
- Analyze attention patterns and model interpretability

## 8. Conclusion

This paper presented a comprehensive implementation of GPT from scratch, systematically building each component from basic tokenization to neural language modeling. Our results demonstrate the effectiveness of modern language modeling approaches and provide insights into the evolution from statistical to neural methods.

Key contributions include:
- Detailed implementation and evaluation of BPE tokenization
- Comprehensive comparison of n-gram and neural language models
- Analysis of the impact of vocabulary size on model performance
- Foundation for implementing the complete transformer architecture

The systematic approach taken in this implementation provides valuable educational insights and serves as a foundation for further research in language modeling and transformer architectures.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

3. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909.

4. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.

## Appendix A: Implementation Details

### A.1 BPE Implementation

\`\`\`python
def train_bpe(text, vocab_size, min_freq=2):
    """Train BPE on text data"""
    # Initialize with character-level vocabulary
    vocab = Counter()
    for word in text.split():
        vocab.update(word)
    
    # Iteratively merge most frequent pairs
    merges = []
    for _ in range(vocab_size - len(vocab)):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=lambda p: pairs[p])
        vocab = merge_vocab(vocab, best_pair)
        merges.append(best_pair)
    
    return merges, vocab
\`\`\`

### A.2 N-gram Model Implementation

\`\`\`python
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(int)
        self.context_counts = defaultdict(int)
    
    def train(self, text):
        """Train n-gram model on text"""
        tokens = text.split()
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = tuple(tokens[i:i+self.n-1])
            self.counts[ngram] += 1
            self.context_counts[context] += 1
    
    def probability(self, context, token):
        """Calculate P(token|context)"""
        ngram = context + (token,)
        return self.counts[ngram] / self.context_counts[context]
\`\`\`

### A.3 Neural Bigram Model Implementation

\`\`\`python
class NeuralBigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        # x: (batch_size, 1) - single token indices
        emb = self.embedding(x)  # (batch_size, 1, embedding_dim)
        emb = emb.squeeze(1)     # (batch_size, embedding_dim)
        logits = self.linear(emb)  # (batch_size, vocab_size)
        return logits
\`\`\`

### A.4 Perplexity Calculation

\`\`\`python
def calculate_perplexity(model, token_stream, bos_token='<s>'):
    """Calculate perplexity on token stream"""
    stream = [bos_token] * (model.n_order - 1) + token_stream
    log_prob_sum = 0.0
    count = 0
    
    for i in range(model.n_order - 1, len(stream)):
        token = stream[i]
        history = tuple(stream[i - model.n_order + 1:i])
        prob = model._calculate_order_probability(token, history)
        if prob > 0:
            log_prob_sum += np.log(prob)
        count += 1
    
    if count == 0:
        return float('inf')
    
    avg_log_prob = log_prob_sum / count
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity
\`\`\`

## 9. Comprehensive Results Report

> **Note:** This document is intended to be viewed as plain text (kept inside a code fence). All numbers below are taken directly from your run logs.

---

### 0) Executive Summary

- **Tokenization (Task 1):** Increasing BPE merges from **1,000 → 2,000** reduces average tokens/word (higher compression) with perfect text reconstruction. Best compression on validation set at **2,000 merges (1.3129 tokens/word)**.
- **Classical n-grams (Task 2):** With limited data and sparse higher-order counts, **unigram** outperformed higher n for both 1,000 and 2,000 merges. Best perplexity at **BPE=2,000: Val/Test≈79.50**.
- **Neural bigram (Task 3 — FIXED):** Strong gains over n-grams. Best overall at **BPE=1,000**, **emb_dim=128**, **wd=1e-5** with **Val≈29.92, Test≈28.53**. BPE=2,000 lags (Val≈38.47, Test≈33.66).
- **GPT (Task 4 — FIXED):** Clear winner. At **BPE=1,000**, **Val≈22.08** (vs. neural bigram 38.91, n-gram 72.01). At **BPE=2,000**, **Val≈28.80**. GPT improves **~43%** over neural bigram and **~69%** over best n-gram on BPE=1,000.

**Big picture:** Token compression (more merges) helps sequence length, but with current data/model, **BPE=1,000** consistently yields **lower perplexity** for neural & GPT models. Capacity and contextual modeling (GPT) dominate simpler models.

---

### 1) Task 1 — BPE Tokenization

**Setup**

- Data slice: **50%** of each split  
  \`train=432,212 chars | valid=25,961 | test=26,024\`
- BPE training vocabulary size driver: **merge_count ∈ {1,000, 2,000}**  
- Normalization: **lower_nopunct**, **aggressive**
- Reconstruction: **True** across splits

**Results**

| Config                                  | Final Vocab | Avg Tokens/Word (Train / Valid / Test) | Reconstruct OK |
|-----------------------------------------|-------------|-----------------------------------------|----------------|
| 1,000 merges • lower_nopunct            | 992         | 1.3995 / **1.4755** / 1.4188            | ✓              |
| 2,000 merges • lower_nopunct            | 1,927       | 1.2260 / **1.3129** / 1.2644            | ✓              |
| 1,000 merges • aggressive               | 992         | 1.3995 / **1.4755** / 1.4188            | ✓              |
| 2,000 merges • aggressive               | 1,927       | 1.2260 / **1.3129** / 1.2644            | ✓              |

**Finding:** Best compression on validation set is **2,000 merges (1.3129 tokens/word)**. However, downstream language modeling (Tasks 2–4) shows **BPE=1,000** achieves **lower perplexity** (better predictive performance), suggesting a trade-off: higher compression vs. model/data regime suitability.

---

### 2) Task 2 — N-gram Language Modeling

**Setup**

- Same 50% data split as Task 1  
- Tokenization: **BPE=1,000** (vocab=992) and **BPE=2,000** (vocab=1,927)  
- Models: interpolated **n ∈ {1,2,3,4}** (grid-searched weights shown by "Best interpolation weights" entries)

**Results (Validation/Test Perplexity)**

**BPE=1,000**
- **n=1:** **219.48 / 219.48**
- n=2: 219.48 / 219.48
- n=3: 321.86 / 322.78
- n=4: 448.41 / 449.96

**BPE=2,000**
- **n=1:** **79.50 / 79.50**
- n=2: 79.50 / 79.50
- n=3: 121.28 / 120.70
- n=4: 204.40 / 204.07

**Observations & Interpretation**

- **Unigram wins** in both tokenization regimes; higher n perform worse due to **sparsity** of higher-order counts with limited data and large vocabularies.  
- **BPE=2,000** produces **much better unigram perplexity** (≈79.5 vs. 219.5) because tokens are longer (closer to words), making unigram distributions more informative.
- Despite this, **n-grams are far outperformed** by neural models (Task 3) and GPT (Task 4) once context modeling is learned.

---

### 3) Task 3 — Neural Bigram Language Modeling (FIXED)

**Setup**

- Device: CPU (per log)  
- Tokenization: **BPE=1,000** (vocab=993) and **BPE=2,000** (vocab=1,929)  
- Data tokens:  
  - **BPE=1,000:** train 193,904 | valid 12,062 | test 11,827  
  - **BPE=2,000:** train 179,881 | valid 11,270 | test 11,072  
- Batch size: 32 | LR: 1e-3 | Weight decay ∈ {1e-5, 1e-4} | Embedding dim ∈ {64, 128}  
- Early stopping applied in some runs

**Best Configurations & Scores**

| BPE | Emb Dim | Weight Decay | **Val PPL** | **Test PPL** |
|-----|---------|--------------|-------------|--------------|
| 1,000 | 128 | **1e-5** | **29.9164** | **28.5338** |
| 2,000 | 128 | **1e-4** | **38.4748** | **33.6593** |

**Notes**

- At **BPE=1,000**, the model converges steadily from ≈991 Val PPL to ~30.  
- At **BPE=2,000**, best Val ≈38.47 with heavier regularization (1e-4), still notably worse than BPE=1,000.
- Relative to Task 2 n-grams, neural bigram reduces perplexity **drastically** (e.g., at BPE=1,000: from 219.5 → 29.9 Val PPL).

**Conclusion:** Learned nonlinear representation of bigram dynamics greatly outperforms count-based models, and **BPE=1,000** is the better operating point in this regime.

---

### 4) Task 4 — GPT Implementation with PyTorch (FIXED)

**Setup**

- **GPU:** CUDA (Device: \`cuda\`)  
- Data slice: **50%** of each split (chars per split as above)  
- Tokenization regimes compared: **BPE=1,000** (vocab=993) vs. **BPE=2,000** (vocab=1,929)  
- Baselines re-run for comparison: **n-grams** and **neural bigram**  
- GPT training: ~**3,000 iterations**; batch counts reported as:  
  - **BPE=1,000:** **189 batches**  
  - **BPE=2,000:** **176 batches**

> **Important:** Logs only report **validation perplexity** for GPT in Task 4; test perplexity is not printed here (unlike Task 3). Comparisons below use **validation** PPL consistently.

**Results — BPE=1,000**

- **N-gram (best Val):** **72.0088** (Test 69.9510)  
- **Neural bigram (best Val):** **38.9074**  
- **GPT (Val):** **22.0831**

**Results — BPE=2,000**

- **N-gram (best Val):** **73.9242** (Test 70.3295)  
- **Neural bigram (best Val):** **39.6324**  
- **GPT (Val):** **28.7979**

**Relative Improvements (Validation PPL)**

- **GPT vs Neural Bigram (BPE=1,000):** 38.91 → 22.08 (**~43%** lower)  
- **GPT vs Best N-gram (BPE=1,000):** 72.01 → 22.08 (**~69%** lower)  
- **BPE sensitivity (GPT):** 28.80 → **22.08** (**~23%** lower PPL for 1,000 vs 2,000 merges)

**Training Dynamics (GPT)**

- Rapid loss/PPL decrease in the first few hundred iterations, then slower improvements—typical of transformer pretraining.  
- Sample generation (BPE=1,000) is coherent at the phrase level:  
  > *"to be or not to say thee yet if you to me speak of …"*  
- **Warnings:** Two harmless Matplotlib legend warnings (no labeled artists).

**Interpretation**

- **GPT decisively outperforms** simpler models by modeling **longer context** and leveraging **self-attention**.  
- **BPE=1,000** again outperforms **BPE=2,000** for GPT. Intuition: with this dataset size and model budget, a **smaller vocabulary** (shorter tokens, longer sequences) improves learning of subword regularities and reduces softmax sparsity.

**Final Comparison Table (Validation PPL)**

| BPE Merges | Best N-gram | Best Neural Bigram | **GPT** |
|------------|-------------|--------------------|---------|
| **1,000**  | 72.0088     | 38.9074            | **22.0831** |
| **2,000**  | 73.9242     | 39.6324            | **28.7979** |

---

### 5) Cross-Task Takeaways

1. **Compression vs. Predictive Performance:**  
   - **Task 1** shows better compression at **2,000 merges** (fewer tokens/word).  
   - **Tasks 3–4** show **lower perplexity at 1,000 merges**—the better choice for this modeling setup.
2. **Model Class Matters (a lot):**  
   - **n-grams** struggle due to sparsity; **neural bigram** makes large gains; **GPT** wins by modeling deeper context and subword structure.
3. **Regularization & Capacity:**  
   - The **best neural bigram** at BPE=2,000 needed stronger weight decay—signaling sensitivity to vocab size and sparsity.  
   - GPT's gains suggest capacity is being effectively used; still room for schedule/regularization tweaks.
4. **Validation vs. Test:**  
   - **Task 3** reports both Val and Test PPL (great!).  
   - **Task 4** currently reports **Val PPL only for GPT**; adding **Test PPL** will complete the picture.

---

### 6) Practical Recommendations

- **For reporting completeness (Task 4):** compute **GPT Test Perplexity** alongside Val PPL for both BPE merges.
- **Training polish (likely low-effort wins):**
  - **Cosine LR decay** + **warmup**;  
  - **Gradient clipping (e.g., 1.0)** to stabilize later iterations;  
  - **Val-best checkpointing** to avoid late-iteration regressions.
- **Data efficiency:** try **sliding/overlapping context windows** (stride < context) to increase distinct training examples per epoch.
- **Ablations to try next:**
  - Vary **context length**, **layers/heads**, and **embed size**;  
  - **Label smoothing** or **weight tying** (if not already);  
  - Explore **dropout** schedules (attention/ffn/emb).
- **BPE choice:** keep **1,000 merges** for now (best perplexity), but revisit if scaling model/data up.

---

### 7) Reproducibility & Artifacts

- **Seeds:** Fixed (good practice). Consider logging seeds into result dicts and saved artifacts.
- **Artifacts produced:**  
  - \`task1_results.pkl\` — BPE stats  
  - \`task2_results.pkl\` — n-gram perplexities & plots  
  - \`task3_fixed_results.pkl\` — neural bigram metrics & plots  
  - \`task4_results.pkl\` — consolidated comparison & plots
- **Plot warnings:**  
  - \`utils.py:541\` & \`utils.py:592\`: "No artists with labels found to put in legend."  
    → Either assign \`label=\` to lines or remove \`ax.legend()\` when unnecessary.

---

### 8) Answers to Task 4 (Direct)

- **Environment:** CUDA GPU (\`cuda\`), 50% data slice (train 432,212 | valid 25,961 | test 26,024 chars).
- **Tokenization:** BPE=1,000 (vocab=993) and BPE=2,000 (vocab=1,929).
- **Per-regime results (Validation PPL):**  
  - **BPE=1,000:**  
    - N-gram best: **72.0088**  
    - Neural bigram best: **38.9074**  
    - **GPT:** **22.0831** (sample text coherent)  
  - **BPE=2,000:**  
    - N-gram best: **73.9242**  
    - Neural bigram best: **39.6324**  
    - **GPT:** **28.7979**
- **Conclusion:** **GPT@BPE=1,000** is the best configuration observed (lowest Val PPL). It outperforms neural bigram by ~**43%** and n-gram by ~**69%** on validation. **Recommendation:** report **GPT Test PPL** and adopt training polish (scheduler, clipping, best-checkpointing) for further gains.

---

### 9) Appendix — Selected Raw Log Highlights (for traceability)

- **GPT@BPE=1,000:**  
  - Training batches: 189; Iter 1500 Val PPL≈16.85 (loss snapshot)  
  - Final reported **Val PPL=22.0831**; sample: "to be or not to say thee yet if you to me speak of…"
- **GPT@BPE=2,000:**  
  - Training batches: 176; **Val PPL=28.7979**; sample Hamlet-style snippet
- **Neural Bigram (Task 3 — FIXED):**  
  - **BPE=1,000:** Best **Val≈29.9164**, **Test≈28.5338** (emb=128, wd=1e-5)  
  - **BPE=2,000:** Best **Val≈38.4748**, **Test≈33.6593** (emb=128, wd=1e-4)

---`);

  return (
    <div className="min-h-screen bg-white">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="prose prose-lg max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={tomorrow}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {markdownContent}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
