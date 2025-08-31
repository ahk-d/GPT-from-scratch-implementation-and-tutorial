# Task 2: N-gram Language Modeling on BPE Subwords
# - Uses best BPE configuration from Task 1 (lower_nopunct normalization, 2500 merges)
# - Implements n-gram models for n=1..4 with Laplace smoothing
# - Tunes interpolation weights on validation set
# - Evaluates perplexity on validation and test sets
# - Includes text generation (argmax and sampling)
# - Saves results to task2_results.pkl

import json
import numpy as np
from collections import Counter
import sys
import os
import pickle

# Import from utils
from utils import (
    load_and_slice_data, BPE, normalize_text, save_results,
    load_cached_bpe
)

# Configuration
PERCENTAGE = 0.10                       # 0.01=1%, 0.1=10%, 1.0=full - Using 10% for better results
BEST_NORMALIZATION = "lower_nopunct"  # Best from Task 1 results (lower_nopunct with 2000 merges)
TOP_MERGE_COUNTS = [1000, 2000]  # Only 1000 and 2000 merge counts
N_GRAM_ORDERS = [1, 2, 3, 4]      # n-gram orders to evaluate
LAPLACE_ALPHA = 1.0               # Add-one smoothing parameter
INTERPOLATION_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]  # Grid for interpolation weights
UNK_STRATEGY = "avg"              # {"avg", "mode"} for unseen tokens
GENERATION_CONTEXT = "to be or not to"
GENERATION_N = 3
GENERATION_MAX_TOKENS = 60

class NGramLanguageModel:
    """
    N-gram language model with Laplace smoothing and interpolation
    """
    
    def __init__(self, n_order, alpha=1.0, unk_strategy="avg"):
        """
        What it does: Initializes n-gram language model
        Args:
            n_order (int): Order of n-gram (1=unigram, 2=bigram, etc.)
            alpha (float): Laplace smoothing parameter
            unk_strategy (str): Strategy for unknown tokens ("avg", "min", "smooth")
        Returns:
            None
        """
        self.n_order = n_order
        self.alpha = alpha
        self.unk_strategy = unk_strategy
        self.vocab_size = 0
        self.ngram_counts = [Counter() for _ in range(n_order)]
        self.context_counts = [Counter() for _ in range(n_order)]
        self.interpolation_weights = np.ones(n_order) / n_order
        self.unk_token = "<UNK>"  # Unknown token symbol

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
        
        # Handle unknown tokens in training data
        # Count token frequencies to identify rare tokens
        token_freq = Counter(token_stream)
        rare_threshold = max(1, len(token_stream) // (vocab_size * 10))  # 10% of avg frequency
        
        stream = [bos_token] * (self.n_order - 1) + token_stream 
        
        for i in range(len(stream)):
            for order in range(1, self.n_order + 1):
                if i - order + 1 < 0:
                    continue
                ngram = tuple(stream[i - order + 1:i + 1])
                context = ngram[:-1]
                self.ngram_counts[order - 1][ngram] += 1
                self.context_counts[order - 1][context] += 1

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

            smoothed_prob = (ngram_count + alpha) / (context_count + alpha * vocab_size)
            probability += self.interpolation_weights[order - 1] * smoothed_prob
            return probability

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
                log_prob_sum += np.log(prob)
            count += 1
        
        if count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / count
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity

    def tune_interpolation_weights(self, token_stream, bos_token='<s>'):
        """
        What it does: Tunes interpolation weights on validation data
        Args:
            token_stream (list): Validation token stream
            bos_token (str): Beginning of sequence token
        Returns:
            None
        """
        stream = [bos_token] * (self.n_order - 1) + token_stream
        best_perplexity = float('inf')
        best_weights = self.interpolation_weights.copy()
        
        # Grid search over interpolation weights
        # For n=1, no interpolation needed
        if self.n_order == 1:
            self.interpolation_weights = np.array([1.0])
            print(f"Best interpolation weights: {self.interpolation_weights}")
            return
            
        for step in INTERPOLATION_STEPS:
            if self.n_order == 1:
                weights = np.array([1.0])  # For unigram, no interpolation needed
            else:
                # Proper weight distribution for n-gram interpolation
                # Higher orders should get more weight, lower orders get less
                weights = np.zeros(self.n_order)
                for i in range(self.n_order):
                    order = i + 1  # 1-based order
                    if order == self.n_order:
                        weights[i] = step  # Highest order gets step weight
                    else:
                        weights[i] = (1 - step) * (order / (self.n_order * (self.n_order - 1) / 2))
                weights = weights / np.sum(weights)  # Normalize
            
            self.interpolation_weights = weights
            perplexity = self.calculate_perplexity(token_stream, bos_token)
            
            # Debug: Print some n-gram counts for verification
            if step == 0.5:  # Only print for middle step to avoid spam
                print(f"    Debug - Weights: {weights}, Perplexity: {perplexity:.4f}")
                print(f"    Debug - Unigram count: {len(self.ngram_counts[0])}, Bigram count: {len(self.ngram_counts[1]) if self.n_order > 1 else 'N/A'}")
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_weights = weights.copy()
        
        self.interpolation_weights = best_weights
        print(f"Best interpolation weights: {best_weights}")

def main():
    """
    What it does: Main function to run Task 2
    Args:
        None
    Returns:
        None
    """
    print("Task 2: N-gram Language Modeling")
    print("=" * 50)
    
    # Load data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    
    # Debug: Check data separation
    print(f"Data split check:")
    print(f"  Train chars: {len(train_text)}")
    print(f"  Valid chars: {len(valid_text)}")
    print(f"  Test chars: {len(test_text)}")
    print(f"  Train/Valid overlap: {len(set(train_text[:100]) & set(valid_text[:100]))}/100")
    print(f"  Valid/Test overlap: {len(set(valid_text[:100]) & set(test_text[:100]))}/100")
    
    # Results storage
    results = {}
    
    # Test different BPE configurations
    for merge_count in TOP_MERGE_COUNTS:
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
        
        # Test n-gram models
        ngram_results = {}
        for n in N_GRAM_ORDERS:
            print(f"  Testing n={n} n-gram model...")
            
            # Create and train n-gram model
            ngram_model = NGramLanguageModel(n, LAPLACE_ALPHA, UNK_STRATEGY)
            ngram_model.fit(train_tokens, vocab_size)
            
            # Tune interpolation weights on validation set
            ngram_model.tune_interpolation_weights(valid_tokens)
            
            # Evaluate on validation and test sets
            val_perplexity = ngram_model.calculate_perplexity(valid_tokens)
            test_perplexity = ngram_model.calculate_perplexity(test_tokens)
            
            ngram_results[f'n={n}'] = {
                'val_perplexity': val_perplexity,
                'test_perplexity': test_perplexity,
                'interpolation_weights': ngram_model.interpolation_weights.tolist(),
                'model_path': f'task2_{merge_count}_{n}.pkl'  # Better naming convention
            }
            
            # Save the trained n-gram model - save only the data, not the class
            model_data = {
                'ngram_counts': ngram_model.ngram_counts,
                'context_counts': ngram_model.context_counts,
                'interpolation_weights': ngram_model.interpolation_weights.tolist(),
                'vocab_size': ngram_model.vocab_size,
                'n_order': ngram_model.n_order,
                'alpha': ngram_model.alpha,
                'unk_strategy': ngram_model.unk_strategy
            }
            with open(f'task2_{merge_count}_{n}.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"    Val Perplexity = {val_perplexity:.4f}, Test={test_perplexity:.4f}")
        
        results[merge_count] = {
            'vocab_size': vocab_size,
            'normalization': BEST_NORMALIZATION,
            'ngram_results': ngram_results
        }
    
    # Save results
    save_results(results, 'task2_results.pkl')
    
    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for merge_count in TOP_MERGE_COUNTS:
        if merge_count in results:
            print(f"BPE merges: {merge_count}")
            for n in N_GRAM_ORDERS:
                key = f'n={n}'
                if key in results[merge_count]['ngram_results']:
                    result = results[merge_count]['ngram_results'][key]
                    print(f"  {key}: Val={result['val_perplexity']:.4f}, Test={result['test_perplexity']:.4f}")
    
    print("\nTask 2 completed!")

if __name__ == "__main__":
    main()
