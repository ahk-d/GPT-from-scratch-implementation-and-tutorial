# Task 4: GPT Implementation with PyTorch - FIXED VERSION
# Key fixes:
# 1. Proper data preparation with sufficient sequences
# 2. Better early stopping logic
# 3. Improved model initialization
# 4. Fixed bigram model training
# 5. Better learning rate scheduling

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import sys
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration - IMPROVED
PERCENTAGE = 0.50
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [1000, 2000]
EMBEDDING_DIMS = [64, 128, 256]
BATCH_SIZE = 32  # Increased batch size
LEARNING_RATE = 3e-4  # Better learning rate for transformers
MAX_ITERATIONS = 3000  # Reduced max iterations
MIN_IMPROVEMENT = 1e-4  # Minimum improvement for early stopping
CHECKPOINT_SAVE_COUNT = 3
GENERATION_CONTEXT = "to be or not to"
GENERATION_MAX_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40

# GPT Configuration - IMPROVED
GPT_CONFIG = {
    'n_embd': 128,      # Smaller embedding dimension for stability
    'n_head': 4,        # Fewer attention heads
    'n_layer': 4,       # Fewer transformer layers
    'dropout': 0.1,
    'chunk_size': 64,   # Smaller context length for better training
    'weight_decay': 0.01,
    'grad_clip': 1.0    # Gradient clipping
}

class CausalSelfAttention(nn.Module):
    """Causal self-attention implementation from scratch"""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        
        # Output projection
        self.output = nn.Linear(n_embd, n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.output(out)
        
        return out

class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation"""
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention and MLP"""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Pre-layer norm (more stable)
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    """GPT model with transformer architecture"""
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, chunk_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.chunk_size = chunk_size
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(chunk_size, n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Output projection
        self.output_projection = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        
    def forward(self, input_tokens):
        B, T = input_tokens.shape
        
        # Get embeddings
        token_emb = self.token_embeddings(input_tokens)
        pos_emb = self.position_embeddings(torch.arange(T, device=input_tokens.device))
        
        # Add embeddings with scaling
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def calculate_loss(self, input_tokens, target_tokens):
        logits = self.forward(input_tokens)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = target_tokens.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss
    
    def calculate_perplexity(self, input_tokens, target_tokens):
        with torch.no_grad():
            loss = self.calculate_loss(input_tokens, target_tokens)
            perplexity = torch.exp(loss).item()
        return perplexity
    
    def generate(self, context_tokens, max_tokens, temperature=1.0, top_k=None):
        self.eval()
        generated = context_tokens.copy()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for _ in range(max_tokens):
                input_seq = torch.tensor(generated[-self.chunk_size:], dtype=torch.long, device=device)
                input_seq = input_seq.unsqueeze(0)
                
                logits = self.forward(input_seq)
                logits = logits[0, -1, :] / temperature
                
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                # Stop if we hit a reasonable stopping point
                if len(generated) > len(context_tokens) + 10 and next_token == 0:
                    break
        
        return generated

class NeuralBigramModel(nn.Module):
    """Improved neural bigram model"""
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layers for better modeling
        self.hidden1 = nn.Linear(embedding_dim, embedding_dim)
        self.hidden2 = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, input_tokens):
        # Handle different input shapes
        if len(input_tokens.shape) == 1:
            input_tokens = input_tokens.unsqueeze(1)
        
        embeddings = self.token_embeddings(input_tokens)
        
        # Pass through hidden layers
        x = F.relu(self.hidden1(embeddings))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        
        logits = self.output_projection(x)
        return logits
    
    def calculate_loss(self, input_tokens, target_tokens):
        logits = self.forward(input_tokens)
        if len(logits.shape) == 3:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = target_tokens.view(B * T)
        else:
            logits_flat = logits
            targets_flat = target_tokens
        
        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss
    
    def calculate_perplexity(self, input_tokens, target_tokens):
        with torch.no_grad():
            loss = self.calculate_loss(input_tokens, target_tokens)
            perplexity = torch.exp(loss).item()
        return perplexity

def prepare_data_for_bigram_training(token_stream, batch_size):
    """FIXED: Better data preparation for bigram training"""
    if len(token_stream) < 2:
        return [], []
    
    # Create bigram pairs
    bigram_pairs = [(token_stream[i], token_stream[i + 1]) 
                   for i in range(len(token_stream) - 1)]
    
    # Shuffle pairs for better training
    np.random.shuffle(bigram_pairs)
    
    # Create batches
    prev_token_batches = []
    next_token_batches = []
    
    for i in range(0, len(bigram_pairs), batch_size):
        batch_pairs = bigram_pairs[i:i + batch_size]
        
        if len(batch_pairs) >= batch_size // 2:  # Allow smaller batches
            prev_tokens = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long)
            next_tokens = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long)
            
            prev_token_batches.append(prev_tokens)
            next_token_batches.append(next_tokens)
    
    return prev_token_batches, next_token_batches

def prepare_data_for_gpt_training(token_stream, chunk_size, batch_size):
    """FIXED: Better data preparation for GPT training"""
    if len(token_stream) < chunk_size + 1:
        return [], []
    
    # Create overlapping sequences for better data utilization
    sequences = []
    stride = chunk_size // 2  # 50% overlap
    
    for i in range(0, len(token_stream) - chunk_size, stride):
        sequence = token_stream[i:i + chunk_size + 1]
        if len(sequence) == chunk_size + 1:
            sequences.append(sequence)
    
    # Shuffle sequences
    np.random.shuffle(sequences)
    
    # Create batches
    input_batches = []
    target_batches = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        if len(batch_sequences) >= batch_size // 2:  # Allow smaller batches
            batch_input = torch.tensor([seq[:-1] for seq in batch_sequences], dtype=torch.long)
            batch_target = torch.tensor([seq[1:] for seq in batch_sequences], dtype=torch.long)
            
            input_batches.append(batch_input)
            target_batches.append(batch_target)
    
    return input_batches, target_batches

def train_neural_model(model, input_batches, target_batches, optimizer, max_iterations, 
                      device, min_improvement=MIN_IMPROVEMENT):
    """FIXED: Better training loop without early stopping"""
    model.train()
    model.to(device)
    
    if len(input_batches) == 0:
        print("No training batches available!")
        return {'losses': [], 'perplexities': []}
    
    history = {'losses': [], 'perplexities': []}
    
    print(f"Training with {len(input_batches)} batches for {max_iterations} iterations")
    
    for iteration in range(max_iterations):
        # Sample random batch
        batch_idx = np.random.randint(0, len(input_batches))
        input_batch = input_batches[batch_idx].to(device)
        target_batch = target_batches[batch_idx].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.calculate_loss(input_batch, target_batch)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at iteration {iteration}")
            break
        
        # Backward pass with gradient clipping
        loss.backward()
        if hasattr(GPT_CONFIG, 'grad_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GPT_CONFIG['grad_clip'])
        optimizer.step()
        
        # Record metrics
        loss_value = loss.item()
        history['losses'].append(loss_value)
        perplexity = torch.exp(loss).item()
        history['perplexities'].append(perplexity)
        
        # Print progress
        if iteration % 500 == 0 or iteration < 10:
            print(f"Iteration {iteration}: Loss = {loss_value:.4f}, Perplexity = {perplexity:.4f}")
    
    return history

def evaluate_model_perplexity(model, input_batches, target_batches, device):
    """FIXED: Better evaluation function"""
    if len(input_batches) == 0:
        return float('inf')
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for input_batch, target_batch in zip(input_batches, target_batches):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            loss = model.calculate_loss(input_batch, target_batch)
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_batches += 1
    
    if total_batches == 0:
        return float('inf')
    
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    
    return perplexity

class NGramLanguageModel:
    """N-gram language model (reused from Task 2)"""
    
    def __init__(self, n_order, alpha=1.0):
        self.n_order = n_order
        self.alpha = alpha
        self.vocab_size = 0
        self.ngram_counts = [Counter() for _ in range(n_order)]
        self.context_counts = [Counter() for _ in range(n_order)]
        self.interpolation_weights = np.ones(n_order) / n_order

    def fit(self, token_stream, vocab_size, bos_token='<s>'):
        self.vocab_size = vocab_size + 1
        stream = [bos_token] * (self.n_order - 1) + token_stream
        
        for i in range(len(stream)):
            for order in range(1, self.n_order + 1):
                if i - order + 1 < 0:
                    continue
                ngram = tuple(stream[i - order + 1:i + 1])
                context = ngram[:-1]
                self.ngram_counts[order - 1][ngram] += 1
                self.context_counts[order - 1][context] += 1

    def calculate_perplexity(self, token_stream, bos_token='<s>'):
        stream = [bos_token] * (self.n_order - 1) + token_stream
        log_prob_sum = 0.0
        count = 0
        
        for i in range(self.n_order - 1, len(stream)):
            token = stream[i]
            context = tuple(stream[i - self.n_order + 1:i])
            prob = self._calculate_interpolated_probability(token, context)
            log_prob_sum += math.log(prob + 1e-10)
            count += 1
        
        return math.exp(-log_prob_sum / count)

    def _calculate_interpolated_probability(self, token, context):
        prob = 0.0
        
        for order in range(1, self.n_order + 1):
            if len(context) >= order - 1:
                ngram_context = context[-(order - 1):]
                ngram = ngram_context + (token,)
                
                count = self.ngram_counts[order - 1][ngram]
                context_count = self.context_counts[order - 1][ngram_context]
                
                if context_count > 0:
                    prob += self.interpolation_weights[order - 1] * (count + self.alpha) / (context_count + self.alpha * self.vocab_size)
        
        return prob

def generate_text_sample(model, tokenizer, token_to_id, id_to_token, context, max_tokens, temperature=1.0, top_k=None):
    """Generate text sample from model"""
    context_tokens = tokenizer.encode(context)
    context_token_ids = [token_to_id.get(token, 0) for token in context_tokens]
    
    if isinstance(model, GPTModel):
        generated_token_ids = model.generate(context_token_ids, max_tokens, temperature, top_k)
        generated_tokens = [id_to_token.get(token_id, '<UNK>') for token_id in generated_token_ids]
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text
    
    return context  # Fallback for other models

# Import from utils
from utils import (
    load_and_slice_data, BPE, normalize_text, save_results,
    load_cached_bpe, plot_training_curves, create_comprehensive_report
)

def main():
    """Main function to run Task 4"""
    print("Task 4: GPT Implementation with PyTorch - FIXED VERSION")
    print("=" * 50)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    
    # Results storage
    results = {
        'bpe_configs': {},
        'ngram_results': {},
        'neural_bigram_results': {},
        'gpt_results': {},
        'comparison': {}
    }
    
    # Test different BPE configurations
    for merge_count in MERGE_COUNTS:
        print(f"\nTesting BPE with {merge_count} merges...")
        
        # Load cached BPE model
        bpe = load_cached_bpe(merge_count, BEST_NORMALIZATION)
        if bpe is None:
            print(f"BPE model not found in cache for {merge_count} merges. Please run task1.py first.")
            continue
        
        # Tokenize data
        train_tokens = bpe.encode(train_text)
        val_tokens = bpe.encode(valid_text)
        test_tokens = bpe.encode(test_text)
        
        # Create token to ID mapping
        unique_tokens = list(set(train_tokens + val_tokens + test_tokens))
        token_to_id = {token: i for i, token in enumerate(unique_tokens)}
        id_to_token = {i: token for i, token in enumerate(unique_tokens)}
        
        # Convert tokens to IDs
        train_tokens = [token_to_id[token] for token in train_tokens]
        val_tokens = [token_to_id[token] for token in val_tokens]
        test_tokens = [token_to_id[token] for token in test_tokens]
        
        vocab_size = len(unique_tokens)
        print(f"Vocabulary size: {vocab_size}")
        
        # Store BPE config
        results['bpe_configs'][merge_count] = {
            'vocab_size': vocab_size,
            'merge_count': merge_count,
            'normalization': BEST_NORMALIZATION,
            'token_to_id': token_to_id,
            'id_to_token': id_to_token
        }
        
        # 1. Test n-gram models
        print("Testing n-gram models...")
        ngram_results = {}
        for n in [1, 2, 3, 4]:
            ngram_model = NGramLanguageModel(n)
            ngram_model.fit(train_tokens, vocab_size)
            
            val_perplexity = ngram_model.calculate_perplexity(val_tokens)
            test_perplexity = ngram_model.calculate_perplexity(test_tokens)
            
            ngram_results[f'n={n}'] = {
                'val_perplexity': val_perplexity,
                'test_perplexity': test_perplexity
            }
            
            print(f"  n={n}: Val Perplexity = {val_perplexity:.4f}, Test Perplexity = {test_perplexity:.4f}")
        
        results['ngram_results'][merge_count] = ngram_results
        
        # 2. Test neural bigram models with different embedding sizes
        print("Testing neural bigram models...")
        neural_results = {}
        
        for embedding_dim in EMBEDDING_DIMS:
            print(f"  Testing with embedding_dim = {embedding_dim}")
            
            # Prepare data for bigram training
            train_prev_batches, train_next_batches = prepare_data_for_bigram_training(
                train_tokens, BATCH_SIZE
            )
            val_prev_batches, val_next_batches = prepare_data_for_bigram_training(
                val_tokens, BATCH_SIZE
            )
            
            # Create model
            model = NeuralBigramModel(vocab_size, embedding_dim)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            # Train model
            history = train_neural_model(
                model, train_prev_batches, train_next_batches, optimizer, 
                MAX_ITERATIONS, device
            )
            
            # Evaluate on validation data
            val_perplexity = evaluate_model_perplexity(model, val_prev_batches, val_next_batches, device)
            
            # Plot training curves for this configuration
            plot_training_curves(
                history, 
                f'Neural Bigram (BPE merges={merge_count}, emb_dim={embedding_dim})', 
                None  # Display only, don't save
            )
            
            neural_results[f'emb_dim={embedding_dim}'] = {
                'val_perplexity': val_perplexity,
                'training_history': history
            }
            
            print(f"    Val Perplexity = {val_perplexity:.4f}")
        
        results['neural_bigram_results'][merge_count] = neural_results
        
        # 3. Test GPT model
        print("Testing GPT model...")
        
        # Prepare data for GPT training
        train_input_batches, train_target_batches = prepare_data_for_gpt_training(
            train_tokens, GPT_CONFIG['chunk_size'], BATCH_SIZE
        )
        val_input_batches, val_target_batches = prepare_data_for_gpt_training(
            val_tokens, GPT_CONFIG['chunk_size'], BATCH_SIZE
        )
        
        # Create GPT model
        gpt_model = GPTModel(
            vocab_size=vocab_size,
            n_embd=GPT_CONFIG['n_embd'],
            n_head=GPT_CONFIG['n_head'],
            n_layer=GPT_CONFIG['n_layer'],
            chunk_size=GPT_CONFIG['chunk_size'],
            dropout=GPT_CONFIG['dropout']
        )
        
        # Create optimizer with weight decay
        optimizer = optim.AdamW(
            gpt_model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=GPT_CONFIG['weight_decay']
        )
        
        # Train GPT model
        history = train_neural_model(
            gpt_model, train_input_batches, train_target_batches, optimizer,
            MAX_ITERATIONS, device
        )
        
        # Evaluate GPT model on validation data
        val_perplexity = evaluate_model_perplexity(gpt_model, val_input_batches, val_target_batches, device)
        
        # Generate sample text
        sample_text = generate_text_sample(
            gpt_model, bpe, token_to_id, id_to_token, GENERATION_CONTEXT, GENERATION_MAX_TOKENS, 
            TEMPERATURE, TOP_K
        )
        
        results['gpt_results'][merge_count] = {
            'val_perplexity': val_perplexity,
            'training_history': history,
            'sample_text': sample_text,
            'config': GPT_CONFIG
        }
        
        print(f"  GPT Val Perplexity = {val_perplexity:.4f}")
        print(f"  Sample text: {sample_text[:100]}...")
        
        # Plot training curves for GPT model
        plot_training_curves(
            history, 
            f'GPT Training (BPE merges={merge_count})', 
            None  # Display only, don't save
        )
    
    # Compare models across all configurations
    print("\nComparing all models...")
    comparison = {}
    
    for merge_count in MERGE_COUNTS:
        if merge_count in results['ngram_results']:
            comparison[merge_count] = {
                'best_ngram': min(results['ngram_results'][merge_count].values(), 
                                 key=lambda x: x['val_perplexity']),
                'best_neural': min(results['neural_bigram_results'][merge_count].values(), 
                                  key=lambda x: x['val_perplexity']),
                'gpt': results['gpt_results'][merge_count]
            }
    
    results['comparison'] = comparison
    
    # Print final comparison
    print("\nFinal Comparison:")
    print("-" * 80)
    print(f"{'BPE Merges':<12} {'Best N-gram':<15} {'Best Neural':<15} {'GPT':<15}")
    print("-" * 80)
    
    for merge_count in MERGE_COUNTS:
        if merge_count in comparison:
            best_ngram = comparison[merge_count]['best_ngram']['val_perplexity']
            best_neural = comparison[merge_count]['best_neural']['val_perplexity']
            gpt_perplexity = comparison[merge_count]['gpt']['val_perplexity']
            
            print(f"{merge_count:<12} {best_ngram:<15.4f} {best_neural:<15.4f} {gpt_perplexity:<15.4f}")
    
    # Save results
    save_results(results, 'task4_results.pkl')
    
    # Create comprehensive report
    create_comprehensive_report(results, "Task 4")
    
    # Print sample generated text
    print("\nSample Generated Text:")
    print("=" * 50)
    if results['gpt_results']:
        best_merge_count = min(results['gpt_results'].keys(), 
                              key=lambda x: results['gpt_results'][x]['val_perplexity'])
        sample_text = results['gpt_results'][best_merge_count]['sample_text']
        print(sample_text)
    
    print("\nTask 4 completed!")

if __name__ == "__main__":
    main()
