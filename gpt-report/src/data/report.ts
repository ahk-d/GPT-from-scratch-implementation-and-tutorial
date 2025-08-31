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
        description: "Complete transformer implementation with attention mechanisms and text generation",
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
            architectures: {
              gptSmall: {
                layers: 6,
                embeddingDim: 256,
                heads: 8,
                chunkSize: 128,
                parameters: "~1.2M"
              },
              gptMedium: {
                layers: 8,
                embeddingDim: 384,
                heads: 12,
                chunkSize: 256,
                parameters: "~3.8M"
              }
            },
            training: {
              batchSize: 32,
              learningRate: "3e-4",
              maxIterations: 1000,
              earlyStopping: "6000 iterations patience"
            },
            features: {
              causalAttention: "Prevents looking at future tokens",
              multiHeadAttention: "8-12 attention heads",
              positionEmbeddings: "Learned positional encoding",
              advancedSampling: "Top-k and top-p (nucleus) sampling"
            }
          }
        },
        codeSegments: [
          {
            title: "Causal Self-Attention Implementation",
            code: `class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # QKV projections
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask (prevent looking at future tokens)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention weights and output
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.output(out)`,
            language: "python"
          },
          {
            title: "Transformer Block with Residual Connections",
            code: `class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Pre-layer normalization (more stable than post-layer norm)
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x`,
            language: "python"
          },
          {
            title: "Complete GPT Model Architecture",
            code: `class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, chunk_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.chunk_size = chunk_size
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(chunk_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_projection = nn.Linear(n_embd, vocab_size, bias=False)
        
    def forward(self, input_tokens):
        B, T = input_tokens.shape
        assert T <= self.chunk_size, f"Sequence length {T} exceeds chunk_size {self.chunk_size}"
        
        # Get embeddings
        token_emb = self.token_embeddings(input_tokens)
        pos_emb = self.position_embeddings(torch.arange(T, device=input_tokens.device))
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.output_projection(x)
        
        return logits`,
            language: "python"
          },
          {
            title: "Advanced Text Generation with Sampling",
            code: `def generate(self, context_tokens, max_tokens, temperature=1.0, top_k=50, top_p=0.9):
    self.eval()
    generated = context_tokens.copy()
    device = next(self.parameters()).device
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Use last chunk_size tokens as context
            input_seq = torch.tensor(generated[-self.chunk_size:], 
                                   dtype=torch.long, device=device).unsqueeze(0)
            
            # Get logits for next token
            logits = self.forward(input_seq)[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits[top_k_indices] = top_k_logits
            
            # Apply top-p (nucleus) filtering  
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
    
    return generated`,
            language: "python"
          }
        ]
      }
    }
  ]
};
