# Task 4: GPT Implementation with PyTorch
# - Complete implementation from n-gram to GPT
# - BPE tokenization with merge counts [1000, 2000, 2500]
# - Causal self-attention implementation from scratch
# - Hyperparameter tuning and evaluation
# - Comprehensive comparison of all models
# - Text generation with temperature and top-k sampling

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

# Colab compatibility
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Running on Google Colab")
except:
    print("Running locally")

# Import from utils
from utils import (
    load_and_slice_data, BPE, normalize_text, save_results,
    load_cached_bpe, plot_training_curves, create_comprehensive_report
)

# Configuration
PERCENTAGE = 0.50                    # 0.01=1%, 0.05=5%, 1.0=full - Using 50% for larger dataset
BEST_NORMALIZATION = "lower_nopunct"  # Best from Task 1
MERGE_COUNTS = [1000, 2000]     # Only 1000 and 2000 merge counts
EMBEDDING_DIMS = [64, 128, 256]       # Different embedding sizes to test
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 5000
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_SAVE_COUNT = 3
GENERATION_CONTEXT = "to be or not to"
GENERATION_MAX_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40

# GPT Configuration (increased for better performance)
GPT_CONFIG = {
    'n_embd': 192,      # Embedding dimension (increased)
    'n_head': 6,        # Number of attention heads (increased)
    'n_layer': 6,       # Number of transformer layers (increased)
    'dropout': 0.1,     # Dropout rate
    'chunk_size': 128,  # Context length (increased)
    'weight_decay': 0.01 # Weight decay for regularization
}

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention implementation from scratch
    """
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        """
        What it does: Initializes causal self-attention layer
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            dropout (float): Dropout rate
        Returns:
            None
        """
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
        
        # Register causal mask
        self.register_buffer('causal_mask', None)
        
    def forward(self, x):
        """
        What it does: Forward pass through causal self-attention
        Args:
            x (torch.Tensor): Input tensor (B, T, n_embd)
        Returns:
            torch.Tensor: Output tensor (B, T, n_embd)
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)    # (B, n_head, T, head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        
        # Apply causal mask
        if self.causal_mask is None or self.causal_mask.size(0) != T:
            self.causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            self.causal_mask = self.causal_mask.to(x.device)
        
        scores = scores.masked_fill(self.causal_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        
        # Output projection
        out = self.output(out)
        
        return out

class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation
    """
    
    def __init__(self, n_embd, dropout=0.1):
        """
        What it does: Initializes MLP layer
        Args:
            n_embd (int): Embedding dimension
            dropout (float): Dropout rate
        Returns:
            None
        """
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        What it does: Forward pass through MLP
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.fc1(x)
        x = F.gelu(x)  # Using GELU as specified
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with causal self-attention and MLP
    """
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        """
        What it does: Initializes transformer block
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            dropout (float): Dropout rate
        Returns:
            None
        """
        super().__init__()
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        """
        What it does: Forward pass through transformer block
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention with residual connection
        x = x + self.attention(self.ln1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x

class GPTModel(nn.Module):
    """
    GPT model with transformer architecture
    """
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, chunk_size, dropout=0.1):
        """
        What it does: Initializes GPT model
        Args:
            vocab_size (int): Size of vocabulary
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_layer (int): Number of transformer layers
            chunk_size (int): Context length
            dropout (float): Dropout rate
        Returns:
            None
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.chunk_size = chunk_size
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings
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
        
        # Tie weights (use proper weight tying)
        self.output_projection.weight = self.token_embeddings.weight
        
    def forward(self, input_tokens):
        """
        What it does: Forward pass through GPT model
        Args:
            input_tokens (torch.Tensor): Input token indices (B, T)
        Returns:
            torch.Tensor: Logits for next token prediction (B, T, vocab_size)
        """
        B, T = input_tokens.shape
        
        # Get token and position embeddings
        token_emb = self.token_embeddings(input_tokens)  # (B, T, n_embd)
        pos_emb = self.position_embeddings(torch.arange(T, device=input_tokens.device))  # (T, n_embd)
        
        # Add embeddings
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.output_projection(x)  # (B, T, vocab_size)
        
        return logits
    
    def calculate_loss(self, input_tokens, target_tokens):
        """
        What it does: Calculates loss for training
        Args:
            input_tokens (torch.Tensor): Input token indices
            target_tokens (torch.Tensor): Target token indices
        Returns:
            torch.Tensor: Loss value
        """
        logits = self.forward(input_tokens)  # (B, T, vocab_size)
        
        # Reshape for cross entropy loss
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)  # (B*T, vocab_size)
        targets_flat = target_tokens.view(B * T)  # (B*T)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        return loss
    
    def calculate_perplexity(self, input_tokens, target_tokens):
        """
        What it does: Calculates perplexity
        Args:
            input_tokens (torch.Tensor): Input token indices
            target_tokens (torch.Tensor): Target token indices
        Returns:
            float: Perplexity value
        """
        with torch.no_grad():
            loss = self.calculate_loss(input_tokens, target_tokens)
            perplexity = torch.exp(loss).item()
        return perplexity
    
    def generate(self, context_tokens, max_tokens, temperature=1.0, top_k=None):
        """
        What it does: Generates text using the model
        Args:
            context_tokens (list): Starting context tokens
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
        Returns:
            list: Generated token sequence
        """
        self.eval()
        generated = context_tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Prepare input - ensure it's a tensor
                if isinstance(generated, list):
                    input_seq = torch.tensor(generated[-self.chunk_size:], dtype=torch.long, device=next(self.parameters()).device)
                else:
                    input_seq = generated[-self.chunk_size:]
                
                input_seq = input_seq.unsqueeze(0)  # Add batch dimension
                
                # Get logits
                logits = self.forward(input_seq)  # (1, T, vocab_size)
                logits = logits[0, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # Stop if we hit the end token
                if next_token == 0:  # Assuming 0 is the end token
                    break
        
        return generated

class NeuralBigramModel(nn.Module):
    """
    Neural bigram model - predicts next token given previous token
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        What it does: Initializes neural bigram model
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
        Returns:
            None
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Previous token embedding
        self.prev_token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layer for better modeling
        self.hidden_layer = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
        
        # Output projection to vocabulary size
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, prev_tokens):
        """
        What it does: Forward pass through the model
        Args:
            prev_tokens (torch.Tensor): Previous token indices (B,)
        Returns:
            torch.Tensor: Logits for next token prediction (B, vocab_size)
        """
        # Get embeddings for previous tokens
        embeddings = self.prev_token_embedding(prev_tokens)  # (B, embedding_dim)
        
        # Pass through hidden layer
        hidden = self.activation(self.hidden_layer(embeddings))  # (B, embedding_dim)
        
        # Project to vocabulary size
        logits = self.output_projection(hidden)  # (B, vocab_size)
        
        return logits
    
    def calculate_loss(self, prev_tokens, next_tokens):
        """
        What it does: Calculates loss for training
        Args:
            prev_tokens (torch.Tensor): Previous token indices (B,)
            next_tokens (torch.Tensor): Next token indices (B,)
        Returns:
            torch.Tensor: Loss value
        """
        logits = self.forward(prev_tokens)  # (B, vocab_size)
        
        # Calculate cross entropy loss
        loss = nn.functional.cross_entropy(logits, next_tokens)
        
        return loss
    
    def calculate_perplexity(self, prev_tokens, next_tokens):
        """
        What it does: Calculates perplexity
        Args:
            prev_tokens (torch.Tensor): Previous token indices (B,)
            next_tokens (torch.Tensor): Next token indices (B,)
        Returns:
            float: Perplexity value
        """
        with torch.no_grad():
            loss = self.calculate_loss(prev_tokens, next_tokens)
            perplexity = torch.exp(loss).item()
        return perplexity

def prepare_data_for_bigram_training(token_stream, batch_size):
    """
    What it does: Prepares bigram data for training neural models
    Args:
        token_stream (list): List of token IDs
        batch_size (int): Batch size for training
    Returns:
        tuple: (prev_token_batches, next_token_batches)
    """
    # Create bigram pairs: (prev_token, next_token)
    bigram_pairs = []
    for i in range(len(token_stream) - 1):
        prev_token = token_stream[i]
        next_token = token_stream[i + 1]
        bigram_pairs.append((prev_token, next_token))
    
    # Create batches
    prev_token_batches = []
    next_token_batches = []
    
    for i in range(0, len(bigram_pairs), batch_size):
        batch_pairs = bigram_pairs[i:i + batch_size]
        
        if len(batch_pairs) == batch_size:  # Only full batches
            prev_tokens = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long)
            next_tokens = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long)
            
            prev_token_batches.append(prev_tokens)
            next_token_batches.append(next_tokens)
    
    return prev_token_batches, next_token_batches

def prepare_data_for_gpt_training(token_stream, chunk_size, batch_size):
    """
    What it does: Prepares data for training GPT models
    Args:
        token_stream (list): List of token IDs
        chunk_size (int): Context length
        batch_size (int): Batch size for training
    Returns:
        tuple: (input_batches, target_batches)
    """
    # Create chunks of tokens
    chunks = []
    for i in range(0, len(token_stream) - chunk_size, chunk_size):
        chunk = token_stream[i:i + chunk_size + 1]  # +1 for target
        if len(chunk) == chunk_size + 1:  # Ensure full chunk
            chunks.append(chunk)
    
    # Create batches
    input_batches = []
    target_batches = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        if len(batch_chunks) == batch_size:  # Only full batches
            # Input: all tokens except last
            batch_input = torch.tensor([chunk[:-1] for chunk in batch_chunks], dtype=torch.long)
            # Target: all tokens except first
            batch_target = torch.tensor([chunk[1:] for chunk in batch_chunks], dtype=torch.long)
            
            input_batches.append(batch_input)
            target_batches.append(batch_target)
    
    return input_batches, target_batches
    """
    N-gram language model (reused from Task 2)
    """
    
    def __init__(self, n_order, alpha=1.0):
        """
        What it does: Initializes n-gram language model
        Args:
            n_order (int): Order of n-gram (1=unigram, 2=bigram, etc.)
            alpha (float): Laplace smoothing parameter
        Returns:
            None
        """
        self.n_order = n_order
        self.alpha = alpha
        self.vocab_size = 0
        self.ngram_counts = [Counter() for _ in range(n_order)]
        self.context_counts = [Counter() for _ in range(n_order)]
        self.interpolation_weights = np.ones(n_order) / n_order

    def fit(self, token_stream, vocab_size, bos_token='<s>'):
        """
        What it does: Trains the n-gram model on token stream
        Args:
            token_stream (list): List of tokens
            vocab_size (int): Size of vocabulary
            bos_token (str): Beginning of sequence token
        Returns:
            None
        """
        self.vocab_size = vocab_size + 1  # Include BOS in smoothing
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
        """
        What it does: Calculates perplexity on token stream
        Args:
            token_stream (list): List of tokens
            bos_token (str): Beginning of sequence token
        Returns:
            float: Perplexity value
        """
        stream = [bos_token] * (self.n_order - 1) + token_stream
        log_prob_sum = 0.0
        count = 0
        
        for i in range(self.n_order - 1, len(stream)):
            token = stream[i]
            history = tuple(stream[i - self.n_order + 1:i])
            prob = self._calculate_order_probability(token, history)
            if prob > 0:
                log_prob_sum += math.log(prob)
            count += 1
        
        if count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / count
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity

    def _calculate_order_probability(self, token, history):
        """
        What it does: Calculates probability using interpolation across all orders
        Args:
            token (str): Token to predict
            history (tuple): Previous tokens
        Returns:
            float: Interpolated probability
        """
        vocab_size = max(1, self.vocab_size)
        alpha = self.alpha
        probability = 0.0
        
        for order in range(1, self.n_order + 1):
            context = history[-(order - 1):] if order > 1 else tuple()
            ngram = tuple(list(context) + [token])
            ngram_count = self.ngram_counts[order - 1].get(ngram, 0)
            context_count = self.context_counts[order - 1].get(context, 0)
            
            # Laplace smoothing
            smoothed_prob = (ngram_count + alpha) / (context_count + alpha * vocab_size)
            probability += self.interpolation_weights[order - 1] * smoothed_prob
            
        return probability

class NeuralBigramModel(nn.Module):
    """
    Neural bigram model (reused from Task 3)
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        What it does: Initializes neural bigram model
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
        Returns:
            None
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Token embedding table
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output projection to vocabulary size
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_tokens):
        """
        What it does: Forward pass through the model
        Args:
            input_tokens (torch.Tensor): Input token indices (B,) or (B, T)
        Returns:
            torch.Tensor: Logits for next token prediction (B, vocab_size) or (B, T, vocab_size)
        """
        # Handle both 1D and 2D input
        if len(input_tokens.shape) == 1:
            # Single token per batch: (B,) -> (B, 1)
            input_tokens = input_tokens.unsqueeze(1)
        
        # Get embeddings for input tokens
        embeddings = self.token_embeddings(input_tokens)  # (B, T, embedding_dim)
        
        # Project to vocabulary size
        logits = self.output_projection(embeddings)  # (B, T, vocab_size)
        
        return logits
    
    def calculate_loss(self, input_tokens, target_tokens):
        """
        What it does: Calculates loss for training
        Args:
            input_tokens (torch.Tensor): Input token indices
            target_tokens (torch.Tensor): Target token indices
        Returns:
            torch.Tensor: Loss value
        """
        logits = self.forward(input_tokens)  # (B, T, vocab_size)
        
        # Reshape for cross entropy loss
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)  # (B*T, vocab_size)
        targets_flat = target_tokens.view(B * T)  # (B*T)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        return loss
    
    def calculate_perplexity(self, input_tokens, target_tokens):
        """
        What it does: Calculates perplexity
        Args:
            input_tokens (torch.Tensor): Input token indices
            target_tokens (torch.Tensor): Target token indices
        Returns:
            float: Perplexity value
        """
        with torch.no_grad():
            loss = self.calculate_loss(input_tokens, target_tokens)
            perplexity = torch.exp(loss).item()
        return perplexity

def prepare_data_for_training(token_stream, chunk_size, batch_size):
    """
    What it does: Prepares data for training neural models
    Args:
        token_stream (list): List of tokens
        chunk_size (int): Context length
        batch_size (int): Batch size
    Returns:
        tuple: (input_batches, target_batches)
    """
    # Create sequences
    sequences = []
    for i in range(0, len(token_stream) - chunk_size, chunk_size):
        sequence = token_stream[i:i + chunk_size + 1]
        if len(sequence) == chunk_size + 1:
            sequences.append(sequence)
    
    # Create batches
    input_batches = []
    target_batches = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        if len(batch_sequences) == batch_size:
            # Prepare input and target
            batch_input = torch.tensor([seq[:-1] for seq in batch_sequences], dtype=torch.long)
            batch_target = torch.tensor([seq[1:] for seq in batch_sequences], dtype=torch.long)
            
            input_batches.append(batch_input)
            target_batches.append(batch_target)
    
    return input_batches, target_batches

class NGramLanguageModel:
    """
    N-gram language model (reused from Task 2)
    """
    
    def __init__(self, n_order, alpha=1.0):
        """
        What it does: Initializes n-gram language model
        Args:
            n_order (int): Order of n-gram (1=unigram, 2=bigram, etc.)
            alpha (float): Laplace smoothing parameter
        Returns:
            None
        """
        self.n_order = n_order
        self.alpha = alpha
        self.vocab_size = 0
        self.ngram_counts = [Counter() for _ in range(n_order)]
        self.context_counts = [Counter() for _ in range(n_order)]
        self.interpolation_weights = np.ones(n_order) / n_order

    def fit(self, token_stream, vocab_size, bos_token='<s>'):
        """
        What it does: Trains the n-gram model on token stream
        Args:
            token_stream (list): List of tokens
            vocab_size (int): Size of vocabulary
            bos_token (str): Beginning of sequence token
        Returns:
            None
        """
        self.vocab_size = vocab_size + 1  # Include BOS in smoothing
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
        """
        What it does: Calculates perplexity on token stream
        Args:
            token_stream (list): List of tokens
            bos_token (str): Beginning of sequence token
        Returns:
            float: Perplexity value
        """
        stream = [bos_token] * (self.n_order - 1) + token_stream
        
        log_prob_sum = 0.0
        count = 0
        
        for i in range(self.n_order - 1, len(stream)):
            token = stream[i]
            context = tuple(stream[i - self.n_order + 1:i])
            
            # Calculate interpolated probability
            prob = self._calculate_interpolated_probability(token, context)
            log_prob_sum += math.log(prob + 1e-10)
            count += 1
        
        return math.exp(-log_prob_sum / count)

    def _calculate_interpolated_probability(self, token, context):
        """
        What it does: Calculates interpolated probability
        Args:
            token: Token to predict
            context: Context tokens
        Returns:
            float: Interpolated probability
        """
        prob = 0.0
        
        for order in range(1, self.n_order + 1):
            if len(context) >= order - 1:
                ngram_context = context[-(order - 1):]
                ngram = ngram_context + (token,)
                
                # Laplace smoothing
                count = self.ngram_counts[order - 1][ngram]
                context_count = self.context_counts[order - 1][ngram_context]
                
                if context_count > 0:
                    prob += self.interpolation_weights[order - 1] * (count + self.alpha) / (context_count + self.alpha * self.vocab_size)
        
        return prob

def train_neural_model(model, input_batches, target_batches, optimizer, max_iterations, 
                      early_stopping_patience, device):
    """
    What it does: Trains a neural model with early stopping
    Args:
        model: Neural model to train
        input_batches: Input batches
        target_batches: Target batches
        optimizer: Optimizer
        max_iterations: Maximum training iterations
        early_stopping_patience: Early stopping patience
        device: Device to train on
    Returns:
        dict: Training history
    """
    model.train()
    model.to(device)
    
    history = {'losses': [], 'perplexities': []}
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(max_iterations):
        # Sample random batch
        batch_idx = np.random.randint(0, len(input_batches))
        input_batch = input_batches[batch_idx].to(device)
        target_batch = target_batches[batch_idx].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.calculate_loss(input_batch, target_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record metrics
        history['losses'].append(loss.item())
        perplexity = torch.exp(loss).item()
        history['perplexities'].append(perplexity)
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at iteration {iteration}")
            break
        
        # Print progress
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}, Perplexity = {perplexity:.4f}")
    
    return history

def evaluate_model_perplexity(model, input_batches, target_batches, device):
    """
    What it does: Evaluates model perplexity on validation data
    Args:
        model: Model to evaluate
        input_batches: Input batches
        target_batches: Target batches
        device: Device to evaluate on
    Returns:
        float: Average perplexity
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for input_batch, target_batch in zip(input_batches, target_batches):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            loss = model.calculate_loss(input_batch, target_batch)
            total_loss += loss.item() * input_batch.numel()
            total_tokens += input_batch.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def generate_text_sample(model, tokenizer, token_to_id, id_to_token, context, max_tokens, temperature=1.0, top_k=None):
    """
    What it does: Generates text sample from model
    Args:
        model: Model to generate from
        tokenizer: Tokenizer for encoding/decoding
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        context (str): Starting context
        max_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature
        top_k (int): Top-k sampling parameter
    Returns:
        str: Generated text
    """
    # Tokenize context
    context_tokens = tokenizer.encode(context)
    context_token_ids = [token_to_id.get(token, 0) for token in context_tokens]
    
    # Generate
    if isinstance(model, GPTModel):
        generated_token_ids = model.generate(context_token_ids, max_tokens, temperature, top_k)
    else:
        # For other models, use simple sampling
        generated_token_ids = context_token_ids.copy()
        for _ in range(max_tokens):
            # This is a simplified generation for non-GPT models
            # In practice, you'd implement proper generation for each model type
            break
    
    # Convert back to tokens and decode
    generated_tokens = [id_to_token.get(token_id, '<UNK>') for token_id in generated_token_ids]
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def plot_training_history(history, title, save_path):
    """
    What it does: Plots training history
    Args:
        history (dict): Training history
        title (str): Plot title
        save_path (str): Path to save plot
    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['losses'])
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot perplexity
    ax2.plot(history['perplexities'])
    ax2.set_title(f'{title} - Perplexity')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    # Optionally save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
    
    plt.close()

def main():
    """
    What it does: Main function to run Task 4
    Args:
        None
    Returns:
        None
    """
    print("Task 4: GPT Implementation with PyTorch")
    print("=" * 50)
    
    # Set device - use CPU for stability, GPU if explicitly requested
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        # Use CPU for stability (MPS can be unstable on macOS)
        device = torch.device("cpu")
        print("Using CPU (MPS disabled for stability)")
    
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
        
        # Store BPE config (without the BPE object for JSON serialization)
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
            
            # Initialize weights properly
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            model.apply(init_weights)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            # Train model
            history = train_neural_model(
                model, train_prev_batches, train_next_batches, optimizer, 
                MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device
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
        
        # Initialize weights properly to prevent numerical instability
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        gpt_model.apply(init_weights)
        
        # Create optimizer with weight decay
        optimizer = optim.AdamW(
            gpt_model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=GPT_CONFIG['weight_decay']
        )
        
        # Train GPT model
        history = train_neural_model(
            gpt_model, train_input_batches, train_target_batches, optimizer,
            MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device
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
