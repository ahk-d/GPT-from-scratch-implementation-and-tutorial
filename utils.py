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

def load_and_slice_data(percentage=0.10):
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
            corpus_tokens (list): List of words with EOS tokens
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
        # Add end-of-word symbol before spaces for consistency with training
        text = text.replace(' ', f' {self.end_of_word} ')
        out = []
        
        for w in text.split():
            pieces = [* (w + self.eos)]
            for a, b in self.merges:
                i = 0
                merged = []
                ab = a + b
                while i < len(pieces):
                    if i < len(pieces) - 1 and pieces[i] == a and pieces[i+1] == b:
                        merged.append(ab)
                        i += 2
                    else:
                        merged.append(pieces[i])
                        i += 1
                pieces = merged
            out.extend(pieces)
        return out

    def decode(self, tokens):
        """
        What it does: Decodes tokens back to text
        Args:
            tokens (list): List of tokens
        Returns:
            str: Reconstructed text
        """
        s = "".join(tokens).replace(self.eos, " ")
        # Remove end-of-word symbols and clean up spaces
        s = s.replace(self.end_of_word, "")
        return re.sub(r"\s+", " ", s).strip()

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
        avg_tpw = len(toks) / len(words)
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
    
    # Calculate compression ratios
    for split_name in ["train", "valid", "test"]:
        split_results = results[split_name]
        compression_ratio = split_results['avg_tokens_per_word'] / len(bpe_model.vocab) if len(bpe_model.vocab) > 0 else 0
        print(f"  compression ratio: {split_name}={compression_ratio:.2f} | ", end="")
    
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

def plot_training_curves(history, title, save_path):
    """
    What it does: Plots training curves (loss and perplexity)
    Args:
        history (dict): Training history with 'losses' and 'perplexities'
        title (str): Plot title
        save_path (str): Path to save plot
    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(history['losses'], 'b-', linewidth=2, alpha=0.8)
    ax1.set_title(f'{title} - Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot perplexity
    ax2.plot(history['perplexities'], 'r-', linewidth=2, alpha=0.8)
    ax2.set_title(f'{title} - Training Perplexity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_model_comparison(results, task_name, save_path):
    """
    What it does: Creates comprehensive model comparison plots
    Args:
        results (dict): Results dictionary from any task
        task_name (str): Name of the task for title
        save_path (str): Path to save plot
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
    
    for merge_count in merge_counts:
        if 'val_perplexity' in results[merge_count]:
            perplexities.append(results[merge_count]['val_perplexity'])
            labels.append(f'{merge_count} merges')
    
    if perplexities:
        bars = ax1.bar(labels, perplexities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Validation Perplexity by BPE Merge Count', fontsize=14, fontweight='bold')
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
    
    # 3. Training history (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    for i, merge_count in enumerate(merge_counts):
        if 'training_history' in results[merge_count]:
            history = results[merge_count]['training_history']
            if 'losses' in history and len(history['losses']) > 0:
                # Plot last 100 iterations for clarity
                losses = history['losses'][-100:]
                iterations = range(len(history['losses']) - len(losses) + 1, len(history['losses']) + 1)
                ax3.plot(iterations, losses, label=f'{merge_count} merges', linewidth=2, alpha=0.8)
    
    ax3.set_title('Training Loss Comparison (Last 100 Iterations)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Embedding dimension comparison (for neural models)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'neural_bigram_results' in results.get(merge_counts[0], {}):
        for merge_count in merge_counts:
            neural_results = results[merge_count].get('neural_bigram_results', {})
            if neural_results:
                emb_dims = []
                perplexities = []
                for key, value in neural_results.items():
                    if 'emb_dim=' in key:
                        emb_dim = int(key.split('=')[1])
                        emb_dims.append(emb_dim)
                        perplexities.append(value['val_perplexity'])
                
                if emb_dims:
                    ax4.plot(emb_dims, perplexities, 'o-', linewidth=2, markersize=6, 
                            label=f'{merge_count} merges', alpha=0.8)
        
        ax4.set_title('Perplexity vs Embedding Dimension', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Embedding Dimension', fontsize=12)
        ax4.set_ylabel('Validation Perplexity', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # 5. Model type comparison (for task 4)
    ax5 = fig.add_subplot(gs[1, 1])
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
                ax5.bar(x_pos, values, width=0.2, label=f'{merge_count} merges', 
                       color=colors[i], alpha=0.8)
        
        ax5.set_title('Model Type Comparison', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Model Type', fontsize=12)
        ax5.set_ylabel('Validation Perplexity', fontsize=12)
        ax5.set_xticks(np.arange(len(model_types)) + 0.25)
        ax5.set_xticklabels(model_types)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"Task: {task_name}\n\n"
    summary_text += f"Number of configurations: {len(merge_counts)}\n"
    summary_text += f"BPE merge counts: {', '.join(map(str, merge_counts))}\n\n"
    
    if perplexities:
        best_perplexity = min(perplexities)
        best_config = merge_counts[perplexities.index(best_perplexity)]
        summary_text += f"Best configuration: {best_config} merges\n"
        summary_text += f"Best perplexity: {best_perplexity:.4f}\n\n"
    
    summary_text += "Key Findings:\n"
    summary_text += "• Lower perplexity = better model\n"
    summary_text += "• Larger vocab size with more merges\n"
    summary_text += "• GPT typically outperforms simpler models"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'{task_name} - Comprehensive Model Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {save_path}")

def plot_bpe_analysis(results, save_path):
    """
    What it does: Creates BPE-specific analysis plots
    Args:
        results (list): Results from task 1
        save_path (str): Path to save plot
    Returns:
        None
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    merge_counts = []
    vocab_sizes = []
    avg_tokens_per_word = []
    compression_ratios = []
    normalization_types = []
    
    for result in results:
        merge_counts.append(result['merge_count'])
        vocab_sizes.append(result['vocab_size'])
        
        # Get validation metrics
        val_eval = result['evaluation']['valid']
        avg_tokens_per_word.append(val_eval['avg_tokens_per_word'])
        
        # Calculate compression ratio
        compression_ratio = val_eval['avg_tokens_per_word'] / result['vocab_size']
        compression_ratios.append(compression_ratio)
        
        normalization_types.append(result['normalization_technique'])
    
    # 1. Vocabulary size vs merge count
    ax1.scatter(merge_counts, vocab_sizes, c=range(len(merge_counts)), cmap='viridis', s=100, alpha=0.7)
    ax1.set_title('Vocabulary Size vs Merge Count', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Merge Count', fontsize=12)
    ax1.set_ylabel('Vocabulary Size', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Average tokens per word
    colors = ['#1f77b4' if norm == 'lower_nopunct' else '#ff7f0e' for norm in normalization_types]
    bars = ax2.bar(range(len(merge_counts)), avg_tokens_per_word, color=colors, alpha=0.8)
    ax2.set_title('Average Tokens per Word', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Configuration Index', fontsize=12)
    ax2.set_ylabel('Avg Tokens per Word', fontsize=12)
    ax2.set_xticks(range(len(merge_counts)))
    ax2.set_xticklabels([f'{mc}\n{norm}' for mc, norm in zip(merge_counts, normalization_types)], 
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Compression ratio
    ax3.plot(merge_counts, compression_ratios, 'o-', linewidth=3, markersize=8, color='#d62728')
    ax3.set_title('Compression Ratio vs Merge Count', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Merge Count', fontsize=12)
    ax3.set_ylabel('Compression Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Normalization comparison
    ax4.axis('off')
    norm_comparison = {}
    for norm, mc, tpw in zip(normalization_types, merge_counts, avg_tokens_per_word):
        if norm not in norm_comparison:
            norm_comparison[norm] = []
        norm_comparison[norm].append((mc, tpw))
    
    summary_text = "BPE Analysis Summary\n\n"
    for norm, data in norm_comparison.items():
        summary_text += f"{norm}:\n"
        for mc, tpw in data:
            summary_text += f"  {mc} merges: {tpw:.3f} tokens/word\n"
        summary_text += "\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('BPE Tokenization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BPE analysis plot saved to {save_path}")

def plot_ngram_analysis(results, save_path):
    """
    What it does: Creates n-gram specific analysis plots
    Args:
        results (dict): Results from task 2
        save_path (str): Path to save plot
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"N-gram analysis plot saved to {save_path}")

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
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create plots based on task type
    if task_name == "Task 1":
        plot_bpe_analysis(task_results, f"{output_dir}/{task_name.lower().replace(' ', '_')}_bpe_analysis.png")
    elif task_name == "Task 2":
        plot_ngram_analysis(task_results, f"{output_dir}/{task_name.lower().replace(' ', '_')}_ngram_analysis.png")
    elif task_name in ["Task 3", "Task 4"]:
        plot_model_comparison(task_results, task_name, f"{output_dir}/{task_name.lower().replace(' ', '_')}_model_comparison.png")
    
    print(f"Comprehensive report created for {task_name}")
