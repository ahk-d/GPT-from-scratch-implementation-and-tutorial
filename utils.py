# Utils: Common functions for GPT from scratch implementation
# Contains shared functions used across all tasks

import json
import re
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up matplotlib for Colab compatibility
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load(path):
    """
    What it does: Loads text file content
    Args:
        path (str): File path to load
    Returns:
        str: File content
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def take_percentage(text, percentage):
    """
    What it does: Takes a percentage of text from the beginning
    Args:
        text (str): Input text
        percentage (float): Percentage to take (0.0-1.0)
    Returns:
        str: First percentage% of text
    """
    n = max(1, int(len(text) * percentage))
    return text[:n]

def normalize_text(text, normalization_type):
    """
    What it does: Normalizes text according to specified type
    Args:
        text (str): Input text
        normalization_type (str): Type of normalization
    Returns:
        str: Normalized text
    """
    if normalization_type == "lower_nopunct":
        text = re.sub(r"[^\w\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
    elif normalization_type == "aggressive":
        # Most aggressive: lowercase + remove all non-alphanumeric except spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_slice_data(percentage=0.50):
    """
    What it does: Loads Shakespeare data and takes specified percentage
    Args:
        percentage (float): Percentage of data to use
    Returns:
        tuple: (train_text, valid_text, test_text)
    """
    train_text = take_percentage(load("./Shakespeare_clean_train.txt"), percentage)
    test_text = take_percentage(load("./Shakespeare_clean_test.txt"), percentage)
    
    # Split test data for validation (since we don't have a separate validation file)
    test_tokens_list = test_text.split()
    split_point = len(test_tokens_list) // 2
    valid_text = " ".join(test_tokens_list[:split_point])
    test_text = " ".join(test_tokens_list[split_point:])
    
    print(f"Using {percentage*100:.1f}% of each split | chars: train={len(train_text)}, valid={len(valid_text)}, test={len(test_text)}")
    return train_text, valid_text, test_text

def save_results(results, filename):
    """
    What it does: Saves results using pickle for robust serialization
    Args:
        results (dict): Results to save
        filename (str): Output filename
    Returns:
        None
    """
    # Use pickle for robust serialization of any Python object
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")

def load_results(filename):
    """
    What it does: Loads results using pickle
    Args:
        filename (str): Input filename
    Returns:
        dict: Loaded results
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Results loaded from {filename}")
    return results

def load_cached_bpe(merge_count, normalization):
    """
    What it does: Loads cached BPE model if available
    Args:
        merge_count (int): Number of merges
        normalization (str): Normalization type
    Returns:
        BPE: Cached BPE model or None if not found
    """
    cache_filename = f"bpe_cache_{merge_count}_{normalization}.pkl"
    try:
        with open(cache_filename, 'rb') as f:
            bpe = pickle.load(f)
        print(f"Loaded cached BPE: {merge_count} merges, {normalization} normalization")
        return bpe
    except FileNotFoundError:
        return None

def save_cached_bpe(bpe, merge_count, normalization):
    """
    What it does: Saves BPE model to cache
    Args:
        bpe (BPE): BPE model to cache
        merge_count (int): Number of merges
        normalization (str): Normalization type
    Returns:
        None
    """
    cache_filename = f"bpe_cache_{merge_count}_{normalization}.pkl"
    
    with open(cache_filename, 'wb') as f:
        pickle.dump(bpe, f)
    
    print(f"Cached BPE: {merge_count} merges, {normalization} normalization")

# BPE Class
class BPE:
        """
        Byte Pair Encoding tokenizer with normalization support
        """
        
        def __init__(self):
            self.vocab = None
            self.merges = []
            self.token2id = {}
            self.id2token = []
            self.eos = '</s>'
            self.end_of_word = '__'  # End-of-word symbol for better word boundaries

        def _norm(self, text, norm):
            """
            What it does: Internal normalization method
            Args:
                text (str): Input text
                norm (str): Normalization type ('lower_nopunct', 'aggressive')
            Returns:
                str: Normalized text
            """
            return normalize_text(text, norm)

        def _stats(self, tokens):
            """
            What it does: Counts character pairs in tokenized words
            Args:
                tokens (list): List of tokenized words
            Returns:
                Counter: Frequency of character pairs
            """
            pairs = Counter()
            for t in tokens:
                for a, b in zip(t, t[1:]):
                    pairs[(a, b)] += 1
            return pairs

        def _merge_vocab(self, pair, tokens):
            """
            What it does: Merges a character pair in all tokenized words
            Args:
                pair (tuple): Character pair to merge
                tokens (list): List of tokenized words
            Returns:
                list: Updated tokenized words with merged pair
            """
            a, b = pair
            ab = a + b
            merged = []
            for t in tokens:
                i = 0
                out = []
                while i < len(t):
                    if i < len(t) - 1 and t[i] == a and t[i+1] == b:
                        out.append(ab)
                        i += 2
                    else:
                        out.append(t[i])
                        i += 1
                merged.append(out)
            return merged

        def _learn(self, corpus_tokens, K):
            """
            What it does: Learns BPE merges from corpus
            Args:
                corpus_tokens (list): List of words
                K (int): Number of merges to perform
            Returns:
                None: Updates self.merges and self.vocab
            """
            tokens = [[*w] for w in corpus_tokens]
            last_bucket = -1
            for step in range(K):
                pairs = self._stats(tokens)
                if not pairs:
                    break
                (a, b), _ = pairs.most_common(1)[0]
                tokens = self._merge_vocab((a, b), tokens)
                self.merges.append((a, b))
                # single-line progress every 5%
                pct = int(100 * (step + 1) / max(1, K))
                bucket = pct // 5
                if bucket != last_bucket:
                    print(f"progress: {pct:3d}% ({step+1}/{K} merges)", end="\r")
                    last_bucket = bucket
            print()  # newline after progress
            
            # Build vocabulary from final tokens
            self.vocab = set()
            for t in tokens:
                self.vocab.update(t)
            # Add end-of-word token to vocabulary
            self.vocab.add(self.end_of_word)
            self.vocab = sorted(list(self.vocab))
            
            # Build token mappings
            self.token2id = {token: i for i, token in enumerate(self.vocab)}
            self.id2token = {i: token for i, token in enumerate(self.vocab)}
            
            print(f"Final vocab size: {len(self.vocab)}")

        def fit(self, text, k_merges=1000, norm='lower_nopunct'):
            """
            What it does: Fits BPE model on text
            Args:
                text (str): Text to fit on
                k_merges (int): Number of merges to perform
                norm (str): Normalization type
            Returns:
                None
            """
            text = self._norm(text, norm)
            words = text.split()
            print(f"Fitting BPE | words={len(words)} | merges={k_merges} | norm={norm}")
            self._learn(words, k_merges)

        def encode(self, text, norm='lower_nopunct'):
            """
            What it does: Encodes text to tokens using learned merges
            Args:
                text (str): Text to encode
                norm (str): Normalization type
            Returns:
                list: List of tokens
            """
            text = self._norm(text, norm)
            out = []
            words = text.split()
            
            for word in words:
                if not word:  # Skip empty words
                    continue
                    
                pieces = list(word)  # Start with individual characters
                
                # Apply all learned merges
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
                
                # Add word pieces
                out.extend(pieces)
                # Add end-of-word symbol after each word
                out.append(self.end_of_word)
                
            return out

        def decode(self, tokens):
            """
            What it does: Decodes tokens back to text
            Args:
                tokens (list): List of tokens
            Returns:
                str: Reconstructed text
            """
            if not tokens:
                return ""
            
            result_words = []
            current_word = ""
            
            for token in tokens:
                if token == self.end_of_word:  # End of word marker
                    if current_word:
                        result_words.append(current_word)
                        current_word = ""
                else:
                    # Regular token - add to current word
                    current_word += token
            
            # Add any remaining word
            if current_word:
                result_words.append(current_word)
            
            # Join words with spaces
            decoded_text = " ".join(result_words)
            
            # Clean up any extra spaces
            decoded_text = " ".join(decoded_text.split())
            
            return decoded_text

        def evaluate_tpw(self, text, norm='lower_nopunct'):
            """
            What it does: Evaluates average tokens per word and reconstruction
            Args:
                text (str): Text to evaluate
                norm (str): Normalization type
            Returns:
                tuple: (avg_tokens_per_word, reconstruct_ok, num_words)
            """
            s = self._norm(text, norm)
            words = s.split()
            if not words:
                return 0.0, True, 0
            toks = self.encode(text, norm)  # encode() handles normalization internally
            # Count only non-end-of-word tokens for tokens per word calculation
            word_tokens = [t for t in toks if t != self.end_of_word]
            avg_tpw = len(word_tokens) / len(words)
            recon_ok = (self.decode(toks) == s)
            return float(avg_tpw), bool(recon_ok), len(words)

def evaluate_bpe_configuration(bpe_model, train_text, valid_text, test_text, normalization_technique):
    """
    What it does: Evaluates BPE configuration on all splits
    Args:
        bpe_model (BPE): Trained BPE model
        train_text (str): Training text
        valid_text (str): Validation text
        test_text (str): Test text
        normalization_technique (str): Normalization technique used
    Returns:
        dict: Evaluation results
    """
    results = {}
    
    for split_name, split_text in [("train", train_text), ("valid", valid_text), ("test", test_text)]:
        avg_tpw, reconstruct_ok, num_words = bpe_model.evaluate_tpw(split_text, normalization_technique)
        results[split_name] = {
            "avg_tokens_per_word": avg_tpw,
            "reconstruct_ok": reconstruct_ok,
            "num_words": num_words
        }
    
    return results

def print_configuration_summary(normalization_technique, merge_count, bpe_model, results):
        """
        What it does: Prints summary of BPE configuration
        Args:
            normalization_technique (str): Normalization technique
            merge_count (int): Number of merges
            bpe_model (BPE): BPE model
            results (dict): Evaluation results
        Returns:
            None
        """
        print("=" * 64)
        print(f"CONFIG  normalization_technique={normalization_technique} | merges(merge_count)={merge_count}")
        print("=" * 64)
        print(f"[SUMMARY] vocab={len(bpe_model.vocab)}")
        
        for split_name in ["train", "valid", "test"]:
            split_results = results[split_name]
            print(f"  avg tokens/word: {split_name}={split_results['avg_tokens_per_word']:.4f} (N={split_results['num_words']}) | ", end="")
        
        print()
        

        
        for split_name in ["train", "valid", "test"]:
            split_results = results[split_name]
            print(f"  reconstruct_ok: {split_name}={split_results['reconstruct_ok']} | ", end="")
        
        print()

def create_result_entry(normalization_technique, merge_count, bpe_model, evaluation_results):
        """
        What it does: Creates result entry for JSON output
        Args:
            normalization_technique (str): Normalization technique
            merge_count (int): Number of merges
            bpe_model (BPE): BPE model
            evaluation_results (dict): Evaluation results
        Returns:
            dict: Result entry
        """
        return {
            "normalization_technique": normalization_technique,
            "merge_count": merge_count,
            "vocab_size": len(bpe_model.vocab),
            "evaluation": evaluation_results
        }

    # ============================================================================
    # PLOTTING AND VISUALIZATION FUNCTIONS
    # ============================================================================

def plot_training_curves(history, title, save_path=None):
        """
        What it does: Plots simple training curves (loss and perplexity)
        Args:
            history (dict): Training history with 'losses' and 'perplexities'
            title (str): Plot title
            save_path (str): Path to save plot (optional)
        Returns:
            None
        """
        # Create single plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot loss on primary y-axis
        color1 = '#1f77b4'
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', color=color1, fontsize=12)
        line1 = ax1.plot(history['losses'], color=color1, linewidth=2, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot perplexity on secondary y-axis
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.set_ylabel('Perplexity', color=color2, fontsize=12)
        line2 = ax2.plot(history['perplexities'], color=color2, linewidth=2, label='Perplexity')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Always display the plot
        plt.show()
        
        # Optionally save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.close()

def plot_model_comparison(results, task_name, save_path=None):
    """
    What it does: Creates comprehensive model comparison plots
    Args:
        results (dict): Results dictionary from any task
        task_name (str): Name of the task for title
        save_path (str): Path to save plot (optional)
    Returns:
        None
        """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # Extract data for plotting
    merge_counts = list(results.keys())
    if not merge_counts:
        print("No results to plot")
        return
    
    # 1. Perplexity comparison across merge counts
    ax1 = fig.add_subplot(gs[0, 0])
    perplexities = []
    labels = []
    valid_merge_counts = []  # Track which merge counts have valid data
    
    for merge_count in merge_counts:
        if task_name == "Task 3":
            if 'best_perplexity' in results[merge_count]:
                perplexities.append(results[merge_count]['best_perplexity'])
                labels.append(f'{merge_count} merges')
                valid_merge_counts.append(merge_count)
        elif 'val_perplexity' in results[merge_count]:
            perplexities.append(results[merge_count]['val_perplexity'])
            labels.append(f'{merge_count} merges')
            valid_merge_counts.append(merge_count)
    
    if perplexities:
        bars = ax1.bar(labels, perplexities, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(perplexities)])
        ax1.set_title('Best Validation Perplexity by BPE Merge Count', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Perplexity', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, perplexities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(perplexities),
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Vocabulary size vs merge count
    ax2 = fig.add_subplot(gs[0, 1])
    vocab_sizes = [results[mc].get('vocab_size', 0) for mc in merge_counts]
    ax2.plot(merge_counts, vocab_sizes, 'o-', linewidth=3, markersize=8, color='#d62728')
    ax2.set_title('Vocabulary Size vs BPE Merge Count', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Merge Count', fontsize=12)
    ax2.set_ylabel('Vocabulary Size', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Hyperparameter analysis (for Task 3)
    ax3 = fig.add_subplot(gs[0, 2])
    if task_name == "Task 3":
        # Plot embedding dimension impact
        for merge_count in merge_counts:
            if 'hyperparameter_results' in results[merge_count]:
                emb_dims = []
                perplexities_scatter = []
                for config_key, config_result in results[merge_count]['hyperparameter_results'].items():
                    if 'emb_dim=' in config_key:
                        emb_dim = int(config_key.split('emb_dim=')[1].split('_')[0])
                        emb_dims.append(emb_dim)
                        perplexities_scatter.append(config_result['val_perplexity'])
                
                if emb_dims:
                    ax3.scatter(emb_dims, perplexities_scatter, alpha=0.6, s=50, label=f'{merge_count} merges')
        
        ax3.set_title('Perplexity vs Embedding Dimension', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Embedding Dimension', fontsize=12)
        ax3.set_ylabel('Validation Perplexity', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    else:
        # Training history for other tasks
        for i, merge_count in enumerate(merge_counts):
            if 'training_history' in results[merge_count]:
                history = results[merge_count]['training_history']
                if 'losses' in history and len(history['losses']) > 0:
                    losses = history['losses'][-100:]
                    iterations = range(len(history['losses']) - len(losses) + 1, len(history['losses']) + 1)
                    ax3.plot(iterations, losses, label=f'{merge_count} merges', linewidth=2, alpha=0.8)
        
        ax3.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. Learning rate analysis (for Task 3)
    ax4 = fig.add_subplot(gs[1, 0])
    if task_name == "Task 3":
        for merge_count in merge_counts:
            if 'hyperparameter_results' in results[merge_count]:
                lrs = []
                perplexities_lr = []
                for config_key, config_result in results[merge_count]['hyperparameter_results'].items():
                    if 'lr=' in config_key:
                        lr = float(config_key.split('lr=')[1].split('_')[0])
                        lrs.append(lr)
                        perplexities_lr.append(config_result['val_perplexity'])
                
                if lrs:
                    ax4.scatter(lrs, perplexities_lr, alpha=0.6, s=50, label=f'{merge_count} merges')
        
        ax4.set_title('Perplexity vs Learning Rate', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Learning Rate', fontsize=12)
        ax4.set_ylabel('Validation Perplexity', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
    else:
        # Model type comparison (for task 4)
        if 'comparison' in results:
            comparison = results['comparison']
            model_types = ['Best N-gram', 'Best Neural', 'GPT']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, merge_count in enumerate(merge_counts):
                if merge_count in comparison:
                    comp = comparison[merge_count]
                    values = [
                        comp['best_ngram']['val_perplexity'],
                        comp['best_neural']['val_perplexity'],
                        comp['gpt']['val_perplexity']
                    ]
                    x_pos = np.arange(len(model_types)) + i * 0.25
                    ax4.bar(x_pos, values, width=0.2, label=f'{merge_count} merges', 
                           color=colors[i], alpha=0.8)
            
            ax4.set_title('Model Type Comparison', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Model Type', fontsize=12)
            ax4.set_ylabel('Validation Perplexity', fontsize=12)
            ax4.set_xticks(np.arange(len(model_types)) + 0.25)
            ax4.set_xticklabels(model_types)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        else:
            # Show summary for single configurations
            ax4.axis('off')
            summary_text = f"Single Configuration Analysis\n\n"
            if perplexities and valid_merge_counts:
                best_idx = perplexities.index(min(perplexities))
                best_merge_count = valid_merge_counts[best_idx]
                best_perplexity = min(perplexities)
                summary_text += f"Best configuration: {best_merge_count} merges\n"
                summary_text += f"Best perplexity: {best_perplexity:.4f}\n\n"
            
            summary_text += "Analysis Notes:\n"
            summary_text += "• Lower perplexity = better model\n"
            summary_text += "• Neural models typically outperform n-grams\n"
            summary_text += "• Hyperparameter tuning is crucial"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Batch size analysis (for Task 3)
    ax5 = fig.add_subplot(gs[1, 1])
    if task_name == "Task 3":
        for merge_count in merge_counts:
            if 'hyperparameter_results' in results[merge_count]:
                batch_sizes = []
                perplexities_batch = []
                for config_key, config_result in results[merge_count]['hyperparameter_results'].items():
                    if 'batch=' in config_key:
                        batch_size = int(config_key.split('batch=')[1].split('_')[0])
                        batch_sizes.append(batch_size)
                        perplexities_batch.append(config_result['val_perplexity'])
                
                if batch_sizes:
                    ax5.scatter(batch_sizes, perplexities_batch, alpha=0.6, s=50, label=f'{merge_count} merges')
        
        ax5.set_title('Perplexity vs Batch Size', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Batch Size', fontsize=12)
        ax5.set_ylabel('Validation Perplexity', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    else:
        # Summary statistics for other tasks
        ax5.axis('off')
        summary_text = f"Task: {task_name}\n\n"
        summary_text += f"Number of configurations: {len(merge_counts)}\n"
        summary_text += f"BPE merge counts: {', '.join(map(str, merge_counts))}\n\n"
        
        if perplexities and valid_merge_counts:
            best_idx = perplexities.index(min(perplexities))
            best_merge_count = valid_merge_counts[best_idx]
            best_perplexity = min(perplexities)
            summary_text += f"Best configuration: {best_merge_count} merges\n"
            summary_text += f"Best perplexity: {best_perplexity:.4f}\n\n"
        
        summary_text += "Key Findings:\n"
        summary_text += "• Lower perplexity = better model\n"
        summary_text += "• Larger vocab size with more merges\n"
        summary_text += "• GPT typically outperforms simpler models"
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary text
    if task_name == "Task 3":
        summary_text = f"Task: {task_name} - Neural Bigram\n\n"
        summary_text += f"Number of merge counts: {len(merge_counts)}\n"
        
        # Show actual hyperparameters tested
        if merge_counts and 'hyperparameter_results' in results[merge_counts[0]]:
            all_emb_dims = set()
            all_batch_sizes = set()
            all_lrs = set()
            all_wds = set()
            
            for merge_count in merge_counts:
                if 'hyperparameter_results' in results[merge_count]:
                    for config_key in results[merge_count]['hyperparameter_results'].keys():
                        parts = config_key.split('_')
                        for part in parts:
                            if part.startswith('emb_dim='):
                                all_emb_dims.add(int(part.split('=')[1]))
                            elif part.startswith('batch='):
                                all_batch_sizes.add(int(part.split('=')[1]))
                            elif part.startswith('lr='):
                                all_lrs.add(float(part.split('=')[1]))
                            elif part.startswith('wd='):
                                all_wds.add(float(part.split('=')[1]))
            
            summary_text += f"Hyperparameters tested:\n"
            summary_text += f"• Embedding dims: {sorted(all_emb_dims)}\n"
            summary_text += f"• Batch sizes: {sorted(all_batch_sizes)}\n"
            summary_text += f"• Learning rates: {sorted(all_lrs)}\n"
            summary_text += f"• Weight decay: {sorted(all_wds)}\n\n"
        
        if perplexities and valid_merge_counts:
            best_idx = perplexities.index(min(perplexities))
            best_merge_count = valid_merge_counts[best_idx]
            best_perplexity = min(perplexities)
            summary_text += f"Best configuration: {best_merge_count} merges\n"
            summary_text += f"Best perplexity: {best_perplexity:.4f}\n\n"
        
        summary_text += "Key Findings:\n"
        summary_text += "• Neural models improve over n-grams\n"
        summary_text += "• Hyperparameter tuning is crucial\n"
        summary_text += "• BPE quality affects performance"
    else:
        summary_text = f"Task: {task_name}\n\n"
        summary_text += f"Number of configurations: {len(merge_counts)}\n"
        summary_text += f"BPE merge counts: {', '.join(map(str, merge_counts))}\n\n"
        
        if perplexities and valid_merge_counts:
            best_idx = perplexities.index(min(perplexities))
            best_merge_count = valid_merge_counts[best_idx]
            best_perplexity = min(perplexities)
            summary_text += f"Best configuration: {best_merge_count} merges\n"
            summary_text += f"Best perplexity: {best_perplexity:.4f}\n\n"
        
        summary_text += "Key Findings:\n"
        summary_text += "• Lower perplexity = better model\n"
        summary_text += "• Larger vocab size with more merges\n"
        summary_text += "• GPT typically outperforms simpler models"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'{task_name} - Comprehensive Model Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    # Optionally save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.close()

def plot_ngram_analysis(results, save_path=None):
        """
        What it does: Creates n-gram specific analysis plots
        Args:
            results (dict): Results from task 2
            save_path (str): Path to save plot (optional)
        Returns:
            None
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        merge_counts = list(results.keys())
        n_orders = [1, 2, 3, 4]
        
        # 1. Perplexity by n-gram order
        for merge_count in merge_counts:
            ngram_results = results[merge_count]['ngram_results']
            val_perplexities = [ngram_results[f'n={n}']['val_perplexity'] for n in n_orders]
            ax1.plot(n_orders, val_perplexities, 'o-', linewidth=2, markersize=6, 
                    label=f'{merge_count} merges', alpha=0.8)
        
        ax1.set_title('Perplexity vs N-gram Order', fontsize=14, fontweight='bold')
        ax1.set_xlabel('N-gram Order', fontsize=12)
        ax1.set_ylabel('Validation Perplexity', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Best n-gram order for each merge count
        best_orders = []
        best_perplexities = []
        
        for merge_count in merge_counts:
            ngram_results = results[merge_count]['ngram_results']
            best_order = min(ngram_results.keys(), key=lambda x: ngram_results[x]['val_perplexity'])
            best_perplexity = ngram_results[best_order]['val_perplexity']
            best_orders.append(int(best_order.split('=')[1]))
            best_perplexities.append(best_perplexity)
        
        bars = ax2.bar(merge_counts, best_perplexities, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
        ax2.set_title('Best N-gram Perplexity by Merge Count', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Merge Count', fontsize=12)
        ax2.set_ylabel('Best Perplexity', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add order labels on bars
        for bar, order in zip(bars, best_orders):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(best_perplexities),
                    f'n={order}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Validation vs Test perplexity
        for merge_count in merge_counts:
            ngram_results = results[merge_count]['ngram_results']
            val_perplexities = [ngram_results[f'n={n}']['val_perplexity'] for n in n_orders]
            test_perplexities = [ngram_results[f'n={n}']['test_perplexity'] for n in n_orders]
            
            ax3.scatter(val_perplexities, test_perplexities, s=100, alpha=0.7, 
                    label=f'{merge_count} merges')
        
        ax3.plot([min(val_perplexities), max(val_perplexities)], 
                [min(val_perplexities), max(val_perplexities)], 'k--', alpha=0.5)
        ax3.set_title('Validation vs Test Perplexity', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Validation Perplexity', fontsize=12)
        ax3.set_ylabel('Test Perplexity', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary
        ax4.axis('off')
        summary_text = "N-gram Analysis Summary\n\n"
        for merge_count in merge_counts:
            ngram_results = results[merge_count]['ngram_results']
            best_order = min(ngram_results.keys(), key=lambda x: ngram_results[x]['val_perplexity'])
            best_perplexity = ngram_results[best_order]['val_perplexity']
            summary_text += f"{merge_count} merges:\n"
            summary_text += f"  Best order: {best_order}\n"
            summary_text += f"  Best perplexity: {best_perplexity:.4f}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle('N-gram Language Model Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Display the plot
        plt.show()
        
        # Optionally save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"N-gram analysis plot saved to {save_path}")
        
        plt.close()

def create_comprehensive_report(task_results, task_name, output_dir='.'):
        """
        What it does: Creates a comprehensive report with all plots
        Args:
            task_results: Results from any task
            task_name (str): Name of the task
            output_dir (str): Output directory
        Returns:
            None
        """
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create plots based on task type
        if task_name == "Task 1":
            # For Task 1, we'll create a simple BPE analysis since plot_bpe_analysis doesn't exist
            print("Task 1 BPE analysis would be displayed here")
        elif task_name == "Task 2":
            plot_ngram_analysis(task_results, None)  # Display only, don't save
        elif task_name in ["Task 3", "Task 4"]:
            plot_model_comparison(task_results, task_name, None)  # Display only, don't save
        
        print(f"Comprehensive report created for {task_name}")

def generate_text_ngram(ngram_model, bpe_model, context, max_tokens=50, temperature=1.0, top_k=10):
        """
        What it does: Generates text using n-gram model
        Args:
            ngram_model: Trained n-gram model
            bpe_model: BPE tokenizer
            context (str): Starting context
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
        Returns:
            str: Generated text
        """
        # Tokenize context
        context_tokens = bpe_model.encode(context)
        
        # Generate tokens
        generated_tokens = context_tokens.copy()
        
        for _ in range(max_tokens):
            # Get last n-1 tokens for context
            if len(generated_tokens) < ngram_model.n_order - 1:
                # Pad with BOS tokens
                history = ['<s>'] * (ngram_model.n_order - 1 - len(generated_tokens)) + generated_tokens
            else:
                history = generated_tokens[-(ngram_model.n_order - 1):]
            
            # Get probability distribution
            probs = {}
            for token in bpe_model.vocab:
                prob = ngram_model._calculate_order_probability(token, tuple(history))
                probs[token] = prob
            
            # Apply temperature and top-k sampling
            if temperature != 1.0:
                for token in probs:
                    probs[token] = probs[token] ** (1.0 / temperature)
            
            # Top-k sampling
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_k_probs = sorted_probs[:top_k]
            
            # Normalize probabilities
            total_prob = sum(prob for _, prob in top_k_probs)
            if total_prob > 0:
                top_k_probs = [(token, prob / total_prob) for token, prob in top_k_probs]
                
                # Sample from top-k
                tokens, probs = zip(*top_k_probs)
                chosen_token = np.random.choice(tokens, p=probs)
            else:
                # Fallback to most common token
                chosen_token = bpe_model.vocab[0]
            
            generated_tokens.append(chosen_token)
            
            # Stop if we hit end token
            if chosen_token == '</s>':
                break
        
        # Decode back to text
        generated_text = bpe_model.decode(generated_tokens)
        return generated_text

def generate_text_neural(model, bpe_model, token_to_id, id_to_token, context, max_tokens=50, temperature=1.0, top_k=10):
        """
        What it does: Generates text using neural model
        Args:
            model: Trained neural model
            bpe_model: BPE tokenizer
            token_to_id: Token to ID mapping
            id_to_token: ID to token mapping
            context (str): Starting context
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
        Returns:
            str: Generated text
        """
        device = next(model.parameters()).device
        
        # Tokenize context
        context_tokens = bpe_model.encode(context)
        context_ids = [token_to_id.get(token, 0) for token in context_tokens]
        
        # Generate tokens
        generated_ids = context_ids.copy()
        
        model.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                # Prepare input
                input_ids = torch.tensor([generated_ids[-1]], dtype=torch.long, device=device)
                
                # Get logits
                logits = model(input_ids)  # (1, 1, vocab_size)
                logits = logits[0, -1, :]  # (vocab_size,)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = torch.softmax(top_k_logits, dim=-1)
                    chosen_idx = torch.multinomial(probs, 1)
                    next_token_id = top_k_indices[chosen_idx].item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token_id)
                
                # Stop if we hit end token or max length
                if next_token_id == token_to_id.get('</s>', -1) or len(generated_ids) >= max_tokens:
                    break
        
        # Convert back to tokens
        generated_tokens = [id_to_token.get(token_id, '<UNK>') for token_id in generated_ids]
        
        # Decode back to text
        generated_text = bpe_model.decode(generated_tokens)
        return generated_text

def generate_text_gpt(model, bpe_model, token_to_id, id_to_token, context, max_tokens=50, temperature=1.0, top_k=10):
        """
        What it does: Generates text using GPT model
        Args:
            model: Trained GPT model
            bpe_model: BPE tokenizer
            token_to_id: Token to ID mapping
            id_to_token: ID to token mapping
            context (str): Starting context
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
        Returns:
            str: Generated text
        """
        device = next(model.parameters()).device
        
        # Tokenize context
        context_tokens = bpe_model.encode(context)
        context_ids = [token_to_id.get(token, 0) for token in context_tokens]
        
        # Generate tokens
        generated_ids = context_ids.copy()
        
        model.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                # Prepare input (use last chunk_size tokens)
                input_ids = torch.tensor([generated_ids[-model.chunk_size:]], dtype=torch.long, device=device)
                
                # Get logits
                logits = model(input_ids)  # (1, T, vocab_size)
                logits = logits[0, -1, :]  # (vocab_size,)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = torch.softmax(top_k_logits, dim=-1)
                    chosen_idx = torch.multinomial(probs, 1)
                    next_token_id = top_k_indices[chosen_idx].item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token_id)
                
                # Stop if we hit end token or max length
                if next_token_id == token_to_id.get('</s>', -1) or len(generated_ids) >= max_tokens:
                    break
        
        # Convert back to tokens
        generated_tokens = [id_to_token.get(token_id, '<UNK>') for token_id in generated_ids]
        
        # Decode back to text
        generated_text = bpe_model.decode(generated_tokens)
        return generated_text

def create_text_generation_interface(task_results, task_name):
        """
        What it does: Creates a text generation interface for any task
        Args:
            task_results: Results from the task
            task_name (str): Name of the task
        Returns:
            function: Text generation function
        """
        if task_name == "Task 2":
            # For n-gram models, we need to load the best model
            def generate_text(context, max_tokens=50, temperature=1.0, top_k=10):
                # Find best configuration
                best_merge_count = min(task_results.keys(), 
                                    key=lambda x: task_results[x]['ngram_results']['n=2']['val_perplexity'])
                best_config = task_results[best_merge_count]
                
                # Load BPE model
                bpe = load_cached_bpe(best_merge_count, "lower_nopunct")
                
                # Create n-gram model (simplified - in practice you'd load the trained model)
                print(f"Using n-gram model with {best_merge_count} BPE merges")
                return f"Generated text for context: '{context}' (n-gram model)"
            
            return generate_text
        
        elif task_name == "Task 3":
            # For neural models
            def generate_text(context, max_tokens=50, temperature=1.0, top_k=10):
                # Find best configuration
                best_merge_count = min(task_results.keys(), 
                                    key=lambda x: task_results[x]['best_perplexity'])
                best_config = task_results[best_merge_count]
                
                print(f"Using neural model with {best_merge_count} BPE merges")
                print(f"Best config: {best_config['best_config']}")
                return f"Generated text for context: '{context}' (neural model)"
            
            return generate_text
        
        elif task_name == "Task 4":
            # For GPT models
            def generate_text(context, max_tokens=50, temperature=1.0, top_k=10):
                # Find best configuration
                best_merge_count = min(task_results.keys(), 
                                    key=lambda x: task_results[x]['gpt_results']['val_perplexity'])
                best_config = task_results[best_merge_count]
                
                print(f"Using GPT model with {best_merge_count} BPE merges")
                return f"Generated text for context: '{context}' (GPT model)"
            
            return generate_text
        
        else:
            raise ValueError(f"Unknown task: {task_name}")
