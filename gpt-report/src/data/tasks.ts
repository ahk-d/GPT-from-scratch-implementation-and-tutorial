export interface FunctionNode {
  id: string;
  name: string;
  description: string;
  inputs: string[];
  outputs: string[];
  implementation: string;
  category: 'data' | 'model' | 'training' | 'evaluation' | 'utility';
  complexity: 'low' | 'medium' | 'high';
}

export interface Task {
  id: string;
  title: string;
  description: string;
  functions: FunctionNode[];
  edges: { source: string; target: string; label?: string }[];
}

export const tasks: Task[] = [
  {
    id: 'task1',
    title: 'Task 1: BPE Tokenization',
    description: 'Byte Pair Encoding tokenization with different merge counts and normalization techniques. Evaluates average tokens per word and reconstruction quality.',
    functions: [
      {
        id: 'load_data',
        name: 'load_and_slice_data',
        description: 'Loads Shakespeare data and splits into train/validation/test sets',
        inputs: ['percentage'],
        outputs: ['train_text', 'valid_text', 'test_text'],
        implementation: `def load_and_slice_data(percentage=0.50):
    train_text = take_percentage(load("./Shakespeare_clean_train.txt"), percentage)
    test_text = take_percentage(load("./Shakespeare_clean_test.txt"), percentage)
    
    # Split test data for validation
    test_tokens_list = test_text.split()
    split_point = len(test_tokens_list) // 2
    valid_text = " ".join(test_tokens_list[:split_point])
    test_text = " ".join(test_tokens_list[split_point:])
    
    return train_text, valid_text, test_text`,
        category: 'data',
        complexity: 'low'
      },
      {
        id: 'normalize_text',
        name: 'normalize_text',
        description: 'Applies text normalization (lowercase, punctuation removal)',
        inputs: ['text', 'normalization_type'],
        outputs: ['normalized_text'],
        implementation: `def normalize_text(text, normalization_type):
    if normalization_type == "lower_nopunct":
        text = re.sub(r"[^\\w\\s]", " ", text.lower())
        text = re.sub(r"\\s+", " ", text).strip()
    elif normalization_type == "aggressive":
        text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text.lower())
        text = re.sub(r"\\s+", " ", text).strip()
    return text`,
        category: 'data',
        complexity: 'low'
      },
      {
        id: 'bpe_fit',
        name: 'BPE.fit',
        description: 'Trains BPE model on text corpus with specified number of merges',
        inputs: ['text', 'k_merges', 'norm'],
        outputs: ['trained_bpe_model'],
        implementation: `def fit(self, text, k_merges=1000, norm='lower_nopunct'):
    text = self._norm(text, norm)
    words = text.split()
    print(f"Fitting BPE | words={len(words)} | merges={k_merges} | norm={norm}")
    self._learn(words, k_merges)`,
        category: 'model',
        complexity: 'high'
      },
      {
        id: 'bpe_encode',
        name: 'BPE.encode',
        description: 'Encodes text to tokens using learned BPE merges',
        inputs: ['text', 'norm'],
        outputs: ['tokens'],
        implementation: `def encode(self, text, norm='lower_nopunct'):
    text = self._norm(text, norm)
    out = []
    words = text.split()
    for i, w in enumerate(words):
        pieces = [*w]  # Start with individual characters
        for a, b in self.merges:
            j = 0
            merged = []
            ab = a + b
            while j < len(pieces):
                if j < len(pieces) - 1 and pieces[j] == a and pieces[j+1] == b:
                    merged.append(ab)
                    j += 2
                else:
                    merged.append(pieces[j])
                    j += 1
            pieces = merged
        out.extend(pieces)
        if i < len(words) - 1:
            out.append(' ')
    return out`,
        category: 'model',
        complexity: 'medium'
      },
      {
        id: 'evaluate_bpe',
        name: 'evaluate_bpe_configuration',
        description: 'Evaluates BPE configuration on train/valid/test splits',
        inputs: ['bpe_model', 'train_text', 'valid_text', 'test_text', 'normalization'],
        outputs: ['evaluation_results'],
        implementation: `def evaluate_bpe_configuration(bpe_model, train_text, valid_text, test_text, normalization_technique):
    results = {}
    for split_name, split_text in [("train", train_text), ("valid", valid_text), ("test", test_text)]:
        avg_tpw, reconstruct_ok, num_words = bpe_model.evaluate_tpw(split_text, normalization_technique)
        results[split_name] = {
            "avg_tokens_per_word": avg_tpw,
            "reconstruct_ok": reconstruct_ok,
            "num_words": num_words
        }
    return results`,
        category: 'evaluation',
        complexity: 'medium'
      },
      {
        id: 'save_results',
        name: 'save_results',
        description: 'Saves evaluation results to file',
        inputs: ['results', 'filename'],
        outputs: ['saved_file'],
        implementation: `def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")`,
        category: 'utility',
        complexity: 'low'
      }
    ],
    edges: [
      { source: 'load_data', target: 'normalize_text', label: 'text data' },
      { source: 'normalize_text', target: 'bpe_fit', label: 'normalized text' },
      { source: 'bpe_fit', target: 'bpe_encode', label: 'trained model' },
      { source: 'bpe_encode', target: 'evaluate_bpe', label: 'encoded tokens' },
      { source: 'evaluate_bpe', target: 'save_results', label: 'evaluation results' }
    ]
  },
  {
    id: 'task2',
    title: 'Task 2: N-gram Language Modeling',
    description: 'Implements n-gram models (n=1..4) with Laplace smoothing and interpolation. Tunes weights on validation set and evaluates perplexity.',
    functions: [
      {
        id: 'load_bpe',
        name: 'load_cached_bpe',
        description: 'Loads cached BPE model from Task 1',
        inputs: ['merge_count', 'normalization'],
        outputs: ['bpe_model'],
        implementation: `def load_cached_bpe(merge_count, normalization):
    cache_filename = f"bpe_cache_{merge_count}_{normalization}.pkl"
    try:
        with open(cache_filename, 'rb') as f:
            bpe = pickle.load(f)
        print(f"Loaded cached BPE: {merge_count} merges, {normalization} normalization")
        return bpe
    except FileNotFoundError:
        return None`,
        category: 'data',
        complexity: 'low'
      },
      {
        id: 'tokenize_data',
        name: 'bpe.encode',
        description: 'Tokenizes text data using BPE model',
        inputs: ['text', 'bpe_model'],
        outputs: ['tokens'],
        implementation: `# Uses BPE.encode from Task 1
tokens = bpe.encode(text)`,
        category: 'data',
        complexity: 'low'
      },
      {
        id: 'ngram_fit',
        name: 'NGramLanguageModel.fit',
        description: 'Trains n-gram model on token stream',
        inputs: ['token_stream', 'vocab_size', 'bos_token'],
        outputs: ['trained_ngram_model'],
        implementation: `def fit(self, token_stream, vocab_size, bos_token='<s>'):
    self.vocab_size = vocab_size + 1
    stream = [bos_token] * (self.n_order - 1) + token_stream
    
    for i in range(len(stream)):
        for order in range(1, self.n_order + 1):
            if i - order + 1 < 0:
                continue
            ngram = tuple(stream[i - order + 1:i + 1])
            context = ngram[:-1]
            self.ngram_counts[order - 1][ngram] += 1
            self.context_counts[order - 1][context] += 1`,
        category: 'model',
        complexity: 'medium'
      },
      {
        id: 'tune_weights',
        name: 'tune_interpolation_weights',
        description: 'Tunes interpolation weights on validation set',
        inputs: ['ngram_model', 'valid_tokens', 'bos_token'],
        outputs: ['tuned_weights'],
        implementation: `def tune_interpolation_weights(self, token_stream, bos_token='<s>'):
    stream = [bos_token] * (self.n_order - 1) + token_stream
    best_perplexity = float('inf')
    best_weights = self.interpolation_weights.copy()
    
    for step in INTERPOLATION_STEPS:
        weights = np.zeros(self.n_order)
        for i in range(self.n_order):
            order = i + 1
            if order == self.n_order:
                weights[i] = step
            else:
                weights[i] = (1 - step) * (order / (self.n_order * (self.n_order - 1) / 2))
        weights = weights / np.sum(weights)
        
        self.interpolation_weights = weights
        perplexity = self.calculate_perplexity(token_stream, bos_token)
        
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_weights = weights.copy()
    
    self.interpolation_weights = best_weights`,
        category: 'training',
        complexity: 'medium'
      },
      {
        id: 'calculate_perplexity',
        name: 'calculate_perplexity',
        description: 'Calculates perplexity on token stream',
        inputs: ['ngram_model', 'token_stream', 'bos_token'],
        outputs: ['perplexity'],
        implementation: `def calculate_perplexity(self, token_stream, bos_token='<s>'):
    stream = [bos_token] * (self.n_order - 1) + token_stream
    log_prob_sum = 0.0
    count = 0
    
    for i in range(self.n_order - 1, len(stream)):
        token = stream[i]
        history = tuple(stream[i - self.n_order + 1:i])
        prob = self._calculate_order_probability(token, history)
        if prob > 0:
            log_prob_sum += np.log(prob)
        count += 1
    
    if count == 0:
        return float('inf')
    
    avg_log_prob = log_prob_sum / count
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity`,
        category: 'evaluation',
        complexity: 'medium'
      }
    ],
    edges: [
      { source: 'load_bpe', target: 'tokenize_data', label: 'BPE model' },
      { source: 'tokenize_data', target: 'ngram_fit', label: 'tokens' },
      { source: 'ngram_fit', target: 'tune_weights', label: 'trained model' },
      { source: 'tune_weights', target: 'calculate_perplexity', label: 'tuned model' }
    ]
  },
  {
    id: 'task3',
    title: 'Task 3: Neural Bigram Embeddings',
    description: 'Implements neural bigram model with PyTorch. Includes early stopping, checkpoint saving, and hyperparameter tuning.',
    functions: [
      {
        id: 'prepare_bigram_data',
        name: 'prepare_data_for_training',
        description: 'Prepares bigram data for neural model training',
        inputs: ['token_stream', 'batch_size'],
        outputs: ['prev_token_batches', 'next_token_batches'],
        implementation: `def prepare_data_for_training(token_stream, batch_size):
    bigram_pairs = []
    for i in range(len(token_stream) - 1):
        prev_token = token_stream[i]
        next_token = token_stream[i + 1]
        bigram_pairs.append((prev_token, next_token))
    
    prev_token_batches = []
    next_token_batches = []
    
    for i in range(0, len(bigram_pairs), batch_size):
        batch_pairs = bigram_pairs[i:i + batch_size]
        if len(batch_pairs) == batch_size:
            prev_tokens = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long)
            next_tokens = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long)
            prev_token_batches.append(prev_tokens)
            next_token_batches.append(next_tokens)
    
    return prev_token_batches, next_token_batches`,
        category: 'data',
        complexity: 'medium'
      },
      {
        id: 'neural_bigram_model',
        name: 'NeuralBigramModel',
        description: 'Neural bigram model with embedding and hidden layers',
        inputs: ['vocab_size', 'embedding_dim'],
        outputs: ['model'],
        implementation: `class NeuralBigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.prev_token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, prev_tokens):
        embeddings = self.prev_token_embedding(prev_tokens)
        hidden = self.activation(self.hidden_layer(embeddings))
        logits = self.output_projection(hidden)
        return logits`,
        category: 'model',
        complexity: 'high'
      },
      {
        id: 'train_neural',
        name: 'train_neural_model',
        description: 'Trains neural model with early stopping',
        inputs: ['model', 'prev_batches', 'next_batches', 'optimizer', 'max_iterations', 'patience', 'device'],
        outputs: ['training_history'],
        implementation: `def train_neural_model(model, prev_token_batches, next_token_batches, optimizer, max_iterations, early_stopping_patience, device):
    model.train()
    model.to(device)
    
    history = {'losses': [], 'perplexities': []}
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(max_iterations):
        batch_idx = np.random.randint(0, len(prev_token_batches))
        prev_batch = prev_token_batches[batch_idx].to(device)
        next_batch = next_token_batches[batch_idx].to(device)
        
        optimizer.zero_grad()
        loss = model.calculate_loss(prev_batch, next_batch)
        loss.backward()
        optimizer.step()
        
        history['losses'].append(loss.item())
        perplexity = torch.exp(loss).item()
        history['perplexities'].append(perplexity)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            break
    
    return history`,
        category: 'training',
        complexity: 'high'
      },
      {
        id: 'evaluate_neural',
        name: 'evaluate_model_perplexity',
        description: 'Evaluates neural model perplexity on validation data',
        inputs: ['model', 'prev_batches', 'next_batches', 'device'],
        outputs: ['perplexity'],
        implementation: `def evaluate_model_perplexity(model, prev_token_batches, next_token_batches, device):
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for prev_batch, next_batch in zip(prev_token_batches, next_token_batches):
            prev_batch = prev_batch.to(device)
            next_batch = next_batch.to(device)
            
            loss = model.calculate_loss(prev_batch, next_batch)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = np.exp(avg_loss)
    
    return perplexity`,
        category: 'evaluation',
        complexity: 'medium'
      }
    ],
    edges: [
      { source: 'prepare_bigram_data', target: 'neural_bigram_model', label: 'training data' },
      { source: 'neural_bigram_model', target: 'train_neural', label: 'model' },
      { source: 'train_neural', target: 'evaluate_neural', label: 'trained model' }
    ]
  },
  {
    id: 'task4',
    title: 'Task 4: GPT Implementation',
    description: 'Complete GPT implementation with causal self-attention, transformer blocks, and comprehensive model comparison.',
    functions: [
      {
        id: 'causal_attention',
        name: 'CausalSelfAttention',
        description: 'Multi-head causal self-attention mechanism that prevents looking at future tokens',
        inputs: ['n_embd', 'n_head', 'dropout'],
        outputs: ['attention_layer'],
        implementation: `class CausalSelfAttention(nn.Module):
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
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)`,
        category: 'model',
        complexity: 'high'
      },
      {
        id: 'mlp_layer',
        name: 'MLP',
        description: 'Position-wise feed-forward network with GELU activation',
        inputs: ['n_embd', 'dropout'],
        outputs: ['mlp_layer'],
        implementation: `class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)`,
        category: 'model',
        complexity: 'medium'
      },
      {
        id: 'transformer_block',
        name: 'TransformerBlock',
        description: 'Transformer block with attention and MLP using pre-layer normalization',
        inputs: ['n_embd', 'n_head', 'dropout'],
        outputs: ['transformer_block'],
        implementation: `class TransformerBlock(nn.Module):
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
        category: 'model',
        complexity: 'high'
      },
      {
        id: 'gpt_model',
        name: 'GPTModel',
        description: 'Complete GPT model with transformer architecture and position embeddings',
        inputs: ['vocab_size', 'n_embd', 'n_head', 'n_layer', 'chunk_size', 'dropout'],
        outputs: ['gpt_model'],
        implementation: `class GPTModel(nn.Module):
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
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model created: {n_params:,} parameters")`,
        category: 'model',
        complexity: 'high'
      },
      {
        id: 'prepare_gpt_data',
        name: 'prepare_gpt_data',
        description: 'Prepares overlapping sequences for GPT training with 50% overlap for better data utilization',
        inputs: ['token_stream', 'chunk_size', 'batch_size'],
        outputs: ['input_batches', 'target_batches'],
        implementation: `def prepare_gpt_data(token_stream, chunk_size, batch_size):
    if len(token_stream) < chunk_size + 1:
        print(f"Warning: Token stream too short ({len(token_stream)}) for chunk size {chunk_size}")
        return [], []
    
    # Create overlapping sequences for better data utilization
    sequences = []
    stride = chunk_size // 2  # 50% overlap (reduced from 75%)
    
    for i in range(0, len(token_stream) - chunk_size, stride):
        sequence = token_stream[i:i + chunk_size + 1]
        if len(sequence) == chunk_size + 1:
            sequences.append(sequence)
    
    print(f"Created {len(sequences)} training sequences")
    np.random.shuffle(sequences)
    
    # Create batches
    input_batches = []
    target_batches = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        if len(batch_sequences) >= batch_size // 2:  # Allow smaller final batch
            batch_input = torch.tensor([seq[:-1] for seq in batch_sequences], dtype=torch.long)
            batch_target = torch.tensor([seq[1:] for seq in batch_sequences], dtype=torch.long)
            input_batches.append(batch_input)
            target_batches.append(batch_target)
    
    print(f"Created {len(input_batches)} training batches")
    return input_batches, target_batches`,
        category: 'data',
        complexity: 'medium'
      },
      {
        id: 'train_gpt',
        name: 'train_gpt_model',
        description: 'Trains GPT model with validation monitoring and early stopping',
        inputs: ['model', 'train_batches', 'target_batches', 'valid_batches', 'valid_target_batches', 'optimizer', 'max_iterations', 'device'],
        outputs: ['training_history'],
        implementation: `def train_gpt_model(model, train_batches, target_batches, valid_batches, valid_target_batches, optimizer, max_iterations, device):
    model.train()
    model.to(device)
    
    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loss tracking for early stopping
    best_train_loss = float('inf')
    train_patience_counter = 0
    
    for iteration in range(max_iterations):
        batch_idx = np.random.randint(0, len(train_batches))
        input_batch = train_batches[batch_idx].to(device)
        target_batch = target_batches[batch_idx].to(device)
        
        # Training step
        optimizer.zero_grad()
        loss = model.calculate_loss(input_batch, target_batch)
        
        if torch.isnan(loss):
            print(f"NaN loss at iteration {iteration}! Stopping.")
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record metrics and early stopping logic
        loss_val = loss.item()
        history['losses'].append(loss_val)
        history['perplexities'].append(math.exp(loss_val))`,
        category: 'training',
        complexity: 'high'
      },
      {
        id: 'generate_text',
        name: 'generate',
        description: 'Generates text using trained GPT model with advanced sampling techniques',
        inputs: ['context_tokens', 'max_tokens', 'temperature', 'top_k', 'top_p'],
        outputs: ['generated_tokens'],
        implementation: `def generate(self, context_tokens, max_tokens, temperature=1.0, top_k=50, top_p=0.9):
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
        category: 'evaluation',
        complexity: 'medium'
      }
    ],
    edges: [
      { source: 'causal_attention', target: 'transformer_block', label: 'attention layer' },
      { source: 'mlp_layer', target: 'transformer_block', label: 'feed-forward' },
      { source: 'transformer_block', target: 'gpt_model', label: 'transformer blocks' },
      { source: 'prepare_gpt_data', target: 'gpt_model', label: 'training data' },
      { source: 'gpt_model', target: 'train_gpt', label: 'model' },
      { source: 'train_gpt', target: 'generate_text', label: 'trained model' }
    ]
  }
];
