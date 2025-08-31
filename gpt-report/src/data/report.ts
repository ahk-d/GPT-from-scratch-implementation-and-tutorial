export interface ReportSection {
  id: string;
  title: string;
  icon: string;
  content: {
    objective: string;
    results?: {
      title: string;
      data: Record<string, any>;
    };
    codeSegments?: {
      title: string;
      code: string;
      language: string;
    }[];
  };
}

export interface ReportData {
  overview: {
    objective: string;
    whyBuild: string[];
    whatWeBuild: {
      task1: { title: string; description: string; color: string };
      task2: { title: string; description: string; color: string };
      task3: { title: string; description: string; color: string };
      task4: { title: string; description: string; color: string };
    };
  };
  tasks: ReportSection[];
}

export const reportData: ReportData = {
  overview: {
    objective: "This project implements a complete GPT (Generative Pre-trained Transformer) model from scratch, breaking down the process into four fundamental tasks. The goal is to understand and implement each component of the transformer architecture, from basic tokenization to the full neural network.",
    whyBuild: [
      "Deep Understanding: Implementing each component reveals the underlying mechanics of transformer architecture",
      "Educational Value: Provides hands-on experience with neural network fundamentals",
      "Customization: Enables modifications and optimizations for specific use cases",
      "Debugging: Better understanding leads to more effective problem-solving",
      "Research: Foundation for experimenting with new architectures and techniques"
    ],
    whatWeBuild: {
      task1: {
        title: "Task 1: BPE Tokenization",
        description: "Byte Pair Encoding for efficient text tokenization",
        color: "blue"
      },
      task2: {
        title: "Task 2: N-gram Language Modeling",
        description: "Statistical language modeling with n-gram approaches",
        color: "purple"
      },
      task3: {
        title: "Task 3: Neural Bigram Model",
        description: "Neural network implementation for bigram language modeling",
        color: "green"
      },
      task4: {
        title: "Task 4: GPT Architecture",
        description: "Full transformer implementation with attention mechanisms",
        color: "orange"
      }
    }
  },
  tasks: [
    {
      id: "task1",
      title: "Task 1: BPE Tokenization Analysis",
      icon: "Code",
      content: {
        objective: "Implement Byte Pair Encoding (BPE) tokenization to efficiently convert text into subword tokens. BPE balances vocabulary size with token efficiency by iteratively merging the most frequent character pairs.",
        results: {
          title: "Key Results",
          data: {
            bestConfig: {
              normalization: "lower_nopunct",
              mergeCount: 2000,
              validationAvgTokensPerWord: 1.3129
            },
            performance: {
              finalVocabSize: 1927,
              compressionRatio: 0.00,
              reconstructAccuracy: "100%"
            }
          }
        },
        codeSegments: [
          {
            title: "BPE Training Process",
            code: `def train_bpe(text, vocab_size, min_freq=2):
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
    
    return merges, vocab`,
            language: "python"
          },
          {
            title: "Tokenization Function",
            code: `def tokenize(text, merges):
    """Tokenize text using learned BPE merges"""
    tokens = []
    for word in text.split():
        word_tokens = list(word)
        for pair in merges:
            word_tokens = apply_merge(word_tokens, pair)
        tokens.extend(word_tokens)
    return tokens`,
            language: "python"
          }
        ]
      }
    },
    {
      id: "task2",
      title: "Task 2: N-gram Language Modeling Analysis",
      icon: "BarChart3",
      content: {
        objective: "Implement statistical n-gram language models to understand probability distributions in text. This provides a baseline for comparing against neural approaches and demonstrates fundamental language modeling concepts.",
        results: {
          title: "Key Results",
          data: {
            bpe1000: {
              n1: { val: 219.48, test: 219.48 },
              n2: { val: 219.48, test: 219.48 },
              n3: { val: 321.86, test: 322.78 },
              n4: { val: 448.41, test: 449.96 }
            },
            bpe2000: {
              n1: { val: 79.50, test: 79.50 },
              n2: { val: 79.50, test: 79.50 },
              n3: { val: 121.28, test: 120.70 },
              n4: { val: 204.40, test: 204.07 }
            }
          }
        },
        codeSegments: [
          {
            title: "N-gram Model Implementation",
            code: `class NGramModel:
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
        return self.counts[ngram] / self.context_counts[context]`,
            language: "python"
          },
          {
            title: "Interpolation Smoothing",
            code: `def interpolate_probabilities(models, context, token, weights):
    """Interpolate probabilities from multiple n-gram models"""
    prob = 0.0
    for i, (model, weight) in enumerate(zip(models, weights)):
        if i == 0:  # Unigram
            prob += weight * model.probability((), token)
        else:  # Higher order
            ctx = context[-(i):] if len(context) >= i else context
            prob += weight * model.probability(ctx, token)
    return prob`,
            language: "python"
          }
        ]
      }
    },
    {
      id: "task3",
      title: "Task 3: Neural Bigram Language Modeling Analysis",
      icon: "Brain",
      content: {
        objective: "Implement a neural network-based bigram language model using PyTorch. This bridges the gap between statistical methods and modern neural approaches, introducing key concepts like embeddings, backpropagation, and gradient descent.",
        results: {
          title: "Key Results",
          data: {
            bpe1000: {
              bestConfig: "emb_dim=128, lr=0.001, wd=1e-05",
              valPerplexity: 29.92,
              testPerplexity: 28.53
            },
            bpe2000: {
              bestConfig: "emb_dim=128, lr=0.001, wd=0.0001",
              valPerplexity: 38.47,
              testPerplexity: 33.66
            }
          }
        },
        codeSegments: [
          {
            title: "Neural Bigram Model",
            code: `class NeuralBigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        # x: (batch_size, 1) - single token indices
        emb = self.embedding(x)  # (batch_size, 1, embedding_dim)
        emb = emb.squeeze(1)     # (batch_size, embedding_dim)
        logits = self.linear(emb)  # (batch_size, vocab_size)
        return logits`,
            language: "python"
          },
          {
            title: "Training Loop",
            code: `def train_model(model, train_loader, val_loader, optimizer, epochs):
    """Training loop with validation"""
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                val_loss = evaluate_model(model, val_loader)
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Train Loss = {loss.item():.4f}, "
                      f"Val Loss = {val_loss:.4f}")`,
            language: "python"
          }
        ]
      }
    },
    {
      id: "task4",
      title: "Task 4: GPT Architecture Analysis",
      icon: "Zap",
      content: {
        objective: "Implement the complete GPT architecture with transformer blocks, multi-head attention, and positional encoding. This represents the full modern language model architecture used in state-of-the-art systems.",
        results: {
          title: "Key Results",
          data: {
            status: "Coming Soon: Task 4 implementation and results will be added as the project progresses. This will include transformer architecture, attention mechanisms, and full GPT model training results."
          }
        },
        codeSegments: [
          {
            title: "Multi-Head Attention",
            code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear transformations and reshape
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)`,
            language: "python"
          },
          {
            title: "Transformer Block",
            code: `class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x`,
            language: "python"
          }
        ]
      }
    }
  ]
};
