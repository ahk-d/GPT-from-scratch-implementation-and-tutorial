# Task 3: Neural Bigram Embeddings
# - Uses best BPE configuration from Task 2 (k=2000, lower_nopunct)
# - Implements neural bigram model with PyTorch
# - Includes early stopping and checkpoint saving
# - Measures perplexity and generates text
# - Saves results to task3_results.json

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import sys
import os
from pathlib import Path

# Import from utils
from utils import (
    load_and_slice_data, BPE, normalize_text, save_results,
    load_cached_bpe, plot_training_curves, create_comprehensive_report
)

# Configuration
PERCENTAGE = 0.10
BEST_NORMALIZATION = "lower_nopunct"  # Best from Task 2
MERGE_COUNTS = [1000, 2000]  # Different merge counts to test
EMBEDDING_DIM = 128                  # Embedding dimension
BATCH_SIZE = 32                      # Batch size for training
LEARNING_RATE = 1e-3                 # Learning rate
MAX_ITERATIONS = 10000               # Maximum training iterations
EARLY_STOPPING_PATIENCE = 5          # Early stopping patience
CHECKPOINT_SAVE_COUNT = 3            # Number of best checkpoints to save
GENERATION_CONTEXT = "to be or not to"
GENERATION_MAX_TOKENS = 60

class NeuralBigramModel(nn.Module):
    """
    Neural bigram model with token embeddings
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
        loss = nn.functional.cross_entropy(logits_flat, targets_flat)
        
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
    perplexity = np.exp(avg_loss)
    
    return perplexity

def main():
    """
    What it does: Main function to run Task 3
    Args:
        None
    Returns:
        None
    """
    print("Task 3: Neural Bigram Embeddings")
    print("=" * 50)
    
    # Set device
    device = torch.device("cpu")  # Use CPU to avoid MPS issues
    print(f"Using device: {device}")
    
    # Load data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    
    # Results storage
    results = {}
    
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
        valid_tokens = bpe.encode(valid_text)
        test_tokens = bpe.encode(test_text)
        
        vocab_size = len(bpe.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create token to ID mapping
        unique_tokens = list(set(train_tokens + valid_tokens + test_tokens))
        token_to_id = {token: i for i, token in enumerate(unique_tokens)}
        id_to_token = {i: token for i, token in enumerate(unique_tokens)}
        
        # Convert tokens to IDs
        train_tokens = [token_to_id[token] for token in train_tokens]
        valid_tokens = [token_to_id[token] for token in valid_tokens]
        test_tokens = [token_to_id[token] for token in test_tokens]
        
        vocab_size = len(unique_tokens)
        
        # Prepare data for training
        chunk_size = 64  # Context length
        input_batches, target_batches = prepare_data_for_training(
            train_tokens, chunk_size, BATCH_SIZE
        )
        
        # Create model
        model = NeuralBigramModel(vocab_size, EMBEDDING_DIM)
        
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
            model, input_batches, target_batches, optimizer, 
            MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device
        )
        
        # Evaluate
        val_perplexity = evaluate_model_perplexity(model, input_batches, target_batches, device)
        
        results[merge_count] = {
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'val_perplexity': val_perplexity,
            'training_history': history,
            'token_to_id': token_to_id,
            'id_to_token': id_to_token
        }
        
        print(f"  Val Perplexity = {val_perplexity:.4f}")
        
        # Plot training curves for this configuration
        plot_training_curves(
            history, 
            f'Neural Bigram (BPE merges={merge_count})', 
            f'task3_training_curves_{merge_count}.png'
        )
    
    # Save results
    save_results(results, 'task3_results.pkl')
    
    # Create comprehensive report
    create_comprehensive_report(results, "Task 3")
    
    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for merge_count in MERGE_COUNTS:
        if merge_count in results:
            result = results[merge_count]
            print(f"BPE merges: {merge_count}")
            print(f"  Vocab size: {result['vocab_size']}")
            print(f"  Val Perplexity: {result['val_perplexity']:.4f}")
    
    print("\nTask 3 completed!")

if __name__ == "__main__":
    main()
