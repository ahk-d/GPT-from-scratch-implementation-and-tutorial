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
import matplotlib.pyplot as plt

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
PERCENTAGE = 0.50                       # 0.01=1%, 0.05=5%, 1.0=full - Using 50% for larger dataset
BEST_NORMALIZATION = "lower_nopunct"  # Best from Task 2
MERGE_COUNTS = [1000, 2000]           # Different merge counts to test (reduced for speed)

# Hyperparameter search space (reduced for efficiency)
EMBEDDING_DIMS = [128]                # Embedding dimensions to test
BATCH_SIZES = [32]                   # Batch sizes to test
LEARNING_RATES = [1e-3]              # Learning rates to test
MAX_ITERATIONS = 5000                # Maximum training iterations (reduced for speed)
EARLY_STOPPING_PATIENCE = 3          # Early stopping patience (reduced for speed)
WEIGHT_DECAY_VALUES = [1e-4]         # Weight decay values to test

# Generation settings
GENERATION_CONTEXT = "to be or not to"
GENERATION_MAX_TOKENS = 60

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

def prepare_data_for_training(token_stream, batch_size):
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

def train_neural_model(model, prev_token_batches, next_token_batches, optimizer, max_iterations, 
                      early_stopping_patience, device):
    """
    What it does: Trains a neural bigram model with early stopping
    Args:
        model: Neural model to train
        prev_token_batches: Previous token batches
        next_token_batches: Next token batches
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
        batch_idx = np.random.randint(0, len(prev_token_batches))
        prev_batch = prev_token_batches[batch_idx].to(device)
        next_batch = next_token_batches[batch_idx].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.calculate_loss(prev_batch, next_batch)
        
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

def evaluate_model_perplexity(model, prev_token_batches, next_token_batches, device):
    """
    What it does: Evaluates model perplexity on validation data
    Args:
        model: Model to evaluate
        prev_token_batches: Previous token batches
        next_token_batches: Next token batches
        device: Device to evaluate on
    Returns:
        float: Average perplexity
    """
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
    
    return perplexity

def main():
    """
    What it does: Main function to run Task 3 with comprehensive hyperparameter search
    Args:
        None
    Returns:
        None
    """
    print("Task 3: Neural Bigram Language Modeling with Hyperparameter Search")
    print("=" * 70)
    
    # Set device - use CPU for stability, GPU if explicitly requested
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        # Use CPU for stability (MPS can be unstable on macOS)
        device = torch.device("cpu")
        print("Using CPU (MPS disabled for stability)")
    
    print(f"Device: {device}")
    
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
        
        # Hyperparameter search
        best_config = None
        best_perplexity = float('inf')
        hyperparameter_results = {}
        
        total_configs = len(EMBEDDING_DIMS) * len(BATCH_SIZES) * len(LEARNING_RATES) * len(WEIGHT_DECAY_VALUES)
        config_count = 0
        
        print(f"  Testing {total_configs} hyperparameter configurations...")
        
        for emb_dim in EMBEDDING_DIMS:
            for batch_size in BATCH_SIZES:
                for lr in LEARNING_RATES:
                    for weight_decay in WEIGHT_DECAY_VALUES:
                        config_count += 1
                        config_key = f"emb_dim={emb_dim}_batch={batch_size}_lr={lr}_wd={weight_decay}"
                        
                        print(f"    Config {config_count}/{total_configs}: {config_key}")
                        
                        # Prepare data for training and validation - FIXED
                        train_prev_batches, train_next_batches = prepare_data_for_training(
                            train_tokens, batch_size
                        )
                        valid_prev_batches, valid_next_batches = prepare_data_for_training(
                            valid_tokens, batch_size
                        )
                        
                        # Create model
                        model = NeuralBigramModel(vocab_size, emb_dim)
                        
                        # Initialize weights properly
                        def init_weights(m):
                            if isinstance(m, nn.Linear):
                                torch.nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                    torch.nn.init.zeros_(m.bias)
                            elif isinstance(m, nn.Embedding):
                                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                        
                        model.apply(init_weights)
                        
                        # Create optimizer with weight decay
                        optimizer = optim.AdamW(
                            model.parameters(), 
                            lr=lr, 
                            weight_decay=weight_decay
                        )
                        
                        # Train model
                        history = train_neural_model(
                            model, train_prev_batches, train_next_batches, optimizer, 
                            MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device
                        )
                        
                        # Evaluate on validation data - FIXED
                        val_perplexity = evaluate_model_perplexity(
                            model, valid_prev_batches, valid_next_batches, device
                        )
                        
                        # Store results
                        hyperparameter_results[config_key] = {
                            'embedding_dim': emb_dim,
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'weight_decay': weight_decay,
                            'val_perplexity': val_perplexity,
                            'training_history': history,
                            'final_loss': history['losses'][-1] if history['losses'] else float('inf')
                        }
                        
                        print(f"      Val Perplexity = {val_perplexity:.4f}")
                        
                        # Update best configuration
                        if val_perplexity < best_perplexity:
                            best_perplexity = val_perplexity
                            best_config = config_key
                        
                        # Clean up to save memory
                        del model, optimizer, history
        
        # Train best model on full data and evaluate on test set
        if best_config:
            print(f"\n  Re-training best configuration on full data: {best_config}")
            best_result = hyperparameter_results[best_config]
            
            # Extract best hyperparameters
            best_emb_dim = best_result['embedding_dim']
            best_batch_size = best_result['batch_size']
            best_lr = best_result['learning_rate']
            best_weight_decay = best_result['weight_decay']
            
            # Prepare test data
            test_prev_batches, test_next_batches = prepare_data_for_training(
                test_tokens, best_batch_size
            )
            
            # Re-prepare training data with best batch size
            train_prev_batches, train_next_batches = prepare_data_for_training(
                train_tokens, best_batch_size
            )
            
            # Create and train final model
            final_model = NeuralBigramModel(vocab_size, best_emb_dim)
            final_model.apply(lambda m: init_weights(m) if hasattr(m, 'weight') else None)
            
            final_optimizer = optim.AdamW(
                final_model.parameters(), 
                lr=best_lr, 
                weight_decay=best_weight_decay
            )
            
            # Train final model
            final_history = train_neural_model(
                final_model, train_prev_batches, train_next_batches, final_optimizer,
                MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device
            )
            
            # Evaluate on test set
            test_perplexity = evaluate_model_perplexity(
                final_model, test_prev_batches, test_next_batches, device
            )
            
            print(f"  Final test perplexity: {test_perplexity:.4f}")
            
            # Add test results to best config
            hyperparameter_results[best_config]['test_perplexity'] = test_perplexity
            hyperparameter_results[best_config]['final_training_history'] = final_history
        
        # Store results for this merge count
        results[merge_count] = {
            'vocab_size': vocab_size,
            'token_to_id': token_to_id,
            'id_to_token': id_to_token,
            'best_config': best_config,
            'best_perplexity': best_perplexity,
            'hyperparameter_results': hyperparameter_results
        }
        
        print(f"  Best configuration: {best_config}")
        print(f"  Best validation perplexity: {best_perplexity:.4f}")
        if best_config and 'test_perplexity' in hyperparameter_results[best_config]:
            print(f"  Test perplexity: {hyperparameter_results[best_config]['test_perplexity']:.4f}")
        
        # Plot training curves for best configuration
        if best_config and best_config in hyperparameter_results:
            best_history = hyperparameter_results[best_config]['training_history']
            plot_training_curves(
                best_history, 
                f'Neural Bigram (BPE merges={merge_count}, {best_config})'
            )
    
    # Save results
    save_results(results, 'task3_results.pkl')
    
    # Create comprehensive report
    create_comprehensive_report(results, "Task 3")
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    for merge_count in MERGE_COUNTS:
        if merge_count in results:
            result = results[merge_count]
            print(f"BPE merges: {merge_count}")
            print(f"  Vocab size: {result['vocab_size']}")
            print(f"  Best config: {result['best_config']}")
            print(f"  Best validation perplexity: {result['best_perplexity']:.4f}")
            
            # Show test perplexity if available
            if result['best_config'] and 'test_perplexity' in result['hyperparameter_results'][result['best_config']]:
                test_perp = result['hyperparameter_results'][result['best_config']]['test_perplexity']
                print(f"  Test perplexity: {test_perp:.4f}")
            
            # Show top 3 configurations
            configs = result['hyperparameter_results']
            sorted_configs = sorted(configs.items(), key=lambda x: x[1]['val_perplexity'])
            print(f"  Top 3 configurations:")
            for i, (config, config_result) in enumerate(sorted_configs[:3]):
                print(f"    {i+1}. {config}: {config_result['val_perplexity']:.4f}")
    
    print("\nTask 3 completed!")

if __name__ == "__main__":
    main()