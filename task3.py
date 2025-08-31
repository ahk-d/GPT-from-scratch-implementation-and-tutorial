# Task 3: Neural Bigram Embeddings - FIXED VERSION
# Key fixes:
# 1. Proper early stopping patience
# 2. Better learning rate scheduling
# 3. Validation-based early stopping
# 4. More reasonable hyperparameters

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
    load_cached_bpe
)

# FIXED Configuration
PERCENTAGE = 0.01
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [1000, 2000]

# FIXED Hyperparameters - more reasonable ranges
EMBEDDING_DIMS = [64, 128]
BATCH_SIZES = [32]
LEARNING_RATES = [1e-3]
MAX_ITERATIONS = 10000  # Increased from 5000
EARLY_STOPPING_PATIENCE = 1000  # Much more reasonable - was 3!
WEIGHT_DECAY_VALUES = [1e-5, 1e-4]
VALIDATION_INTERVAL = 100  # Check validation every N iterations

class NeuralBigramModel(nn.Module):
    """
    FIXED: Neural bigram model with better initialization and architecture
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Previous token embedding with better initialization
        self.prev_token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Simpler architecture - remove unnecessary complexity
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Better initialization
        self._init_weights()
        
    def _init_weights(self):
        """Proper weight initialization"""
        # Embedding initialization
        nn.init.normal_(self.prev_token_embedding.weight, mean=0.0, std=0.02)
        
        # Linear layer initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, prev_tokens):
        """
        Forward pass - simplified architecture
        """
        embeddings = self.prev_token_embedding(prev_tokens)  # (B, embedding_dim)
        logits = self.output_projection(embeddings)  # (B, vocab_size)
        return logits
    
    def calculate_loss(self, prev_tokens, next_tokens):
        """Calculate cross entropy loss"""
        logits = self.forward(prev_tokens)
        loss = nn.functional.cross_entropy(logits, next_tokens)
        return loss

def prepare_data_fixed(token_stream, batch_size, device):
    """
    FIXED: Better data preparation with proper batching
    """
    # Create bigram pairs
    bigram_pairs = [(token_stream[i], token_stream[i + 1]) 
                   for i in range(len(token_stream) - 1)]
    
    # Shuffle for better training
    np.random.shuffle(bigram_pairs)
    
    # Create batches
    batches = []
    for i in range(0, len(bigram_pairs) - batch_size + 1, batch_size):
        batch_pairs = bigram_pairs[i:i + batch_size]
        
        prev_tokens = torch.tensor([pair[0] for pair in batch_pairs], 
                                 dtype=torch.long, device=device)
        next_tokens = torch.tensor([pair[1] for pair in batch_pairs], 
                                 dtype=torch.long, device=device)
        
        batches.append((prev_tokens, next_tokens))
    
    return batches

def train_neural_model_fixed(model, train_batches, valid_batches, optimizer, 
                           max_iterations, patience, device, validation_interval=100):
    """
    IMPROVED: Training with better overfitting detection and early stopping
    """
    model.train()
    
    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Track overfitting indicators
    train_val_gap = []
    min_improvement = 1e-4
    
    for iteration in range(max_iterations):
        # Sample random training batch
        batch_idx = np.random.randint(0, len(train_batches))
        prev_batch, next_batch = train_batches[batch_idx]
        
        # Training step
        optimizer.zero_grad()
        loss = model.calculate_loss(prev_batch, next_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Record training metrics
        history['losses'].append(loss.item())
        history['perplexities'].append(torch.exp(loss).item())
        
        # Validation check
        if iteration % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_prev, val_next in valid_batches[:10]:  # Sample of validation batches
                    val_loss += model.calculate_loss(val_prev, val_next).item()
                val_loss /= min(len(valid_batches), 10)
            
            history['val_losses'].append(val_loss)
            history['val_perplexities'].append(np.exp(val_loss))
            
            # Calculate train-val gap for overfitting detection
            current_train_loss = loss.item()
            gap = current_train_loss - val_loss
            train_val_gap.append(gap)
            
            print(f"Iteration {iteration}: Train Loss = {current_train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Perplexity = {np.exp(val_loss):.2f}, "
                  f"Gap = {gap:.4f}")
            
            # Improved early stopping with multiple conditions
            should_stop = False
            stop_reason = ""
            
            # 1. Standard patience-based stopping
            if val_loss < best_val_loss - min_improvement:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += validation_interval
                
            if patience_counter >= patience:
                should_stop = True
                stop_reason = f"patience exceeded ({patience})"
            
            # 2. Overfitting detection: train loss decreasing but val loss increasing
            if len(history['val_losses']) >= 3:
                recent_train_losses = history['losses'][-3:]
                recent_val_losses = history['val_losses'][-3:]
                
                train_decreasing = recent_train_losses[-1] < recent_train_losses[0]
                val_increasing = recent_val_losses[-1] > recent_val_losses[0]
                
                if train_decreasing and val_increasing:
                    should_stop = True
                    stop_reason = "overfitting detected (train decreasing, val increasing)"
            
            # 3. Train-val gap becoming too large (overfitting)
            if len(train_val_gap) >= 2 and train_val_gap[-1] < -0.5:  # Train loss much lower than val
                should_stop = True
                stop_reason = "overfitting detected (large train-val gap)"
            
            if should_stop:
                print(f"Early stopping at iteration {iteration}: {stop_reason}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
                
            model.train()
        
        # Progress reporting
        elif iteration % 1000 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}, "
                  f"Perplexity = {torch.exp(loss).item():.2f}")
    
    return history

def evaluate_model_fixed(model, batches, device):
    """FIXED: Proper model evaluation"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for prev_batch, next_batch in batches:
            loss = model.calculate_loss(prev_batch, next_batch)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = np.exp(avg_loss)
    return perplexity

def main():
    """
    FIXED: Main function with proper training and evaluation
    """
    print("Task 3: Neural Bigram Language Modeling (FIXED VERSION)")
    print("=" * 70)
    
    # Use CPU for consistent results
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    
    results = {}
    
    # Test different BPE configurations
    for merge_count in MERGE_COUNTS:
        print(f"\n{'='*50}")
        print(f"Testing BPE with {merge_count} merges")
        print(f"{'='*50}")
        
        # Load BPE model
        bpe = load_cached_bpe(merge_count, BEST_NORMALIZATION)
        if bpe is None:
            print(f"BPE model not found. Please run Task 1 first.")
            continue
        
        # Tokenize data
        train_tokens = bpe.encode(train_text)
        valid_tokens = bpe.encode(valid_text)
        test_tokens = bpe.encode(test_text)
        
        # Use BPE model's vocabulary and token mappings
        token_to_id = bpe.token2id
        id_to_token = bpe.id2token
        
        # Convert to IDs with fallback for missing tokens
        def safe_token_to_id(token):
            return token_to_id.get(token, 0)  # Use 0 as fallback for unknown tokens
        
        train_ids = [safe_token_to_id(token) for token in train_tokens]
        valid_ids = [safe_token_to_id(token) for token in valid_tokens]
        test_ids = [safe_token_to_id(token) for token in test_tokens]
        
        vocab_size = len(bpe.vocab)  # Use BPE model's vocabulary size
        print(f"Vocabulary size: {vocab_size}")
        print(f"Training tokens: {len(train_ids)}")
        print(f"Validation tokens: {len(valid_ids)}")
        print(f"Test tokens: {len(test_ids)}")
        
        # Hyperparameter search
        best_config = None
        best_val_perplexity = float('inf')
        hyperparameter_results = {}
        
        total_configs = len(EMBEDDING_DIMS) * len(BATCH_SIZES) * len(LEARNING_RATES) * len(WEIGHT_DECAY_VALUES)
        config_count = 0
        
        print(f"\nTesting {total_configs} hyperparameter configurations...")
        
        for emb_dim in EMBEDDING_DIMS:
            for batch_size in BATCH_SIZES:
                for lr in LEARNING_RATES:
                    for weight_decay in WEIGHT_DECAY_VALUES:
                        config_count += 1
                        config_key = f"emb_dim={emb_dim}_batch={batch_size}_lr={lr}_wd={weight_decay}"
                        
                        print(f"\nConfig {config_count}/{total_configs}: {config_key}")
                        
                        # Prepare data
                        train_batches = prepare_data_fixed(train_ids, batch_size, device)
                        valid_batches = prepare_data_fixed(valid_ids, batch_size, device)
                        
                        print(f"  Training batches: {len(train_batches)}")
                        print(f"  Validation batches: {len(valid_batches)}")
                        
                        # Create and train model
                        model = NeuralBigramModel(vocab_size, emb_dim)
                        model.to(device)
                        
                        optimizer = optim.AdamW(
                            model.parameters(), 
                            lr=lr, 
                            weight_decay=weight_decay
                        )
                        
                        # Train model
                        history = train_neural_model_fixed(
                            model, train_batches, valid_batches, optimizer,
                            MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device, 
                            VALIDATION_INTERVAL
                        )
                        
                        # Final validation evaluation
                        val_perplexity = evaluate_model_fixed(model, valid_batches, device)
                        
                        print(f"  Final validation perplexity: {val_perplexity:.4f}")
                        
                        # Store results
                        hyperparameter_results[config_key] = {
                            'embedding_dim': emb_dim,
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'weight_decay': weight_decay,
                            'val_perplexity': val_perplexity,
                            'training_history': history,
                            'checkpoint_path': f'task3_{merge_count}_{config_key}.pt'  # Better naming convention
                        }
                        
                        # Save the trained model checkpoint
                        torch.save(model.state_dict(), f'task3_{merge_count}_{config_key}.pt')
                        
                        # Update best configuration
                        if val_perplexity < best_val_perplexity:
                            best_val_perplexity = val_perplexity
                            best_config = config_key
                        
                        # Clean up
                        del model, optimizer
        
        # Test evaluation with best model
        if best_config:
            print(f"\n{'='*30}")
            print(f"Testing best configuration: {best_config}")
            print(f"{'='*30}")
            
            best_result = hyperparameter_results[best_config]
            
            # Recreate best model for test evaluation
            model = NeuralBigramModel(vocab_size, best_result['embedding_dim'])
            model.to(device)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=best_result['learning_rate'],
                weight_decay=best_result['weight_decay']
            )
            
            # Retrain on full training data
            train_batches = prepare_data_fixed(train_ids, best_result['batch_size'], device)
            valid_batches = prepare_data_fixed(valid_ids, best_result['batch_size'], device)
            test_batches = prepare_data_fixed(test_ids, best_result['batch_size'], device)
            
            final_history = train_neural_model_fixed(
                model, train_batches, valid_batches, optimizer,
                MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device,
                VALIDATION_INTERVAL
            )
            
            # Test evaluation
            test_perplexity = evaluate_model_fixed(model, test_batches, device)
            print(f"Final test perplexity: {test_perplexity:.4f}")
            
            # Update results with test perplexity
            hyperparameter_results[best_config]['test_perplexity'] = test_perplexity
            hyperparameter_results[best_config]['final_training_history'] = final_history
            hyperparameter_results[best_config]['final_checkpoint_path'] = f'task3_{merge_count}_{best_config}_final.pt'
            
            # Save the final best model
            torch.save(model.state_dict(), f'task3_{merge_count}_{best_config}_final.pt')
        
        # Store results for this merge count
        results[merge_count] = {
            'vocab_size': vocab_size,
            'best_config': best_config,
            'best_perplexity': best_val_perplexity,
            'hyperparameter_results': hyperparameter_results
        }
    
    # Save and report results
    save_results(results, 'task3_fixed_results.pkl')
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    for merge_count in MERGE_COUNTS:
        if merge_count in results:
            result = results[merge_count]
            print(f"\nBPE merges: {merge_count}")
            print(f"  Vocab size: {result['vocab_size']}")
            print(f"  Best config: {result['best_config']}")
            print(f"  Best validation perplexity: {result['best_perplexity']:.4f}")
            
            if result['best_config'] and 'test_perplexity' in result['hyperparameter_results'][result['best_config']]:
                test_perp = result['hyperparameter_results'][result['best_config']]['test_perplexity']
                print(f"  Test perplexity: {test_perp:.4f}")
    
    print("\nTask 3 (FIXED) completed!")

if __name__ == "__main__":
    main()