# Utils: Common functions for GPT from scratch implementation
# Contains shared functions used across all tasks
# MODIFIED: BPE class simplified for better n-gram modeling

import json
import re
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle

# Unified generation context for all tasks
GEN_CONTEXT = "to be or not to be "


def normalize_generation_context(context):
    """Ensure context ends appropriately for generation"""
    context = context.strip()
    if not context.endswith(' '):
        context += ' '  # Add space for natural continuation
    return context

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

# BPE Class - MODIFIED for better n-gram modeling
class BPE:
    """BPE tokenizer - MODIFIED to treat spaces naturally for n-gram modeling"""
    
    def __init__(self):
        self.vocab = set()
        self.merges = []
        self.token2id = {}
        self.id2token = {}
        self.eos = '</s>'
        # REMOVED: self.end_of_word - no more special tokens!

    def _norm(self, text, norm, keep_edge_spaces=False):
        if norm == "lower_nopunct":
            text = re.sub(r"[^\w\s]", " ", text.lower())
            text = re.sub(r"\s+", " ", text)
        elif norm == "aggressive":
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
            text = re.sub(r"\s+", " ", text)
        # Only strip when we *don't* want to preserve the boundary
        if not keep_edge_spaces:
            text = text.strip()
        return text

    def _stats(self, tokens):
        """Count adjacent token pairs"""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def _merge_vocab(self, pair, tokens):
        """Merge all instances of pair in token sequence"""
        a, b = pair
        ab = a + b
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(ab)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _learn(self, text, K):
        """Learn BPE merges from character sequence"""
        # MODIFIED: Start with all characters including spaces
        tokens = list(text)
        
        last_bucket = -1
        
        for step in range(K):
            pairs = self._stats(tokens)
            if not pairs:
                break
                
            (a, b), _ = pairs.most_common(1)[0]
            tokens = self._merge_vocab((a, b), tokens)
            self.merges.append((a, b))
            
            pct = int(100 * (step + 1) / max(1, K))
            bucket = pct // 5
            if bucket != last_bucket:
                print(f"progress: {pct:3d}% ({step+1}/{K} merges)", end="\r")
                last_bucket = bucket
        print()
        
        # Build vocabulary from final tokens
        self.vocab = sorted(list(set(tokens)))
        
        # Build token mappings
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for i, token in enumerate(self.vocab)}
        
        print(f"Final vocab size: {len(self.vocab)}")

    def fit(self, text, k_merges=1000, norm='lower_nopunct'):
        text = self._norm(text, norm, keep_edge_spaces=False)  # unchanged behavior
        print(f"Fitting BPE | chars={len(text)} | merges={k_merges} | norm={norm}")
        self._learn(text, k_merges)

    def encode(self, text, norm='lower_nopunct', preserve_edge_spaces=True):
        # Preserve trailing space so the next token starts after a space
        text = self._norm(text, norm, keep_edge_spaces=preserve_edge_spaces)
        if preserve_edge_spaces and not text.endswith(' '):
            text += ' '
        tokens = list(text)
        for a, b in self.merges:
            tokens = self._merge_vocab((a, b), tokens)
        return tokens

    def decode(self, tokens):
        """MODIFIED: Simply join tokens - spaces preserved naturally"""
        return "".join(tokens)

    def evaluate_tpw(self, text, norm='lower_nopunct'):
        """Calculate tokens per word"""
        s = self._norm(text, norm)
        words = s.split()
        if not words:
            return 0.0, True, 0
        
        toks = self.encode(text, norm)
        # Count non-space tokens for TPW calculation
        word_tokens = [t for t in toks if t.strip()]
        avg_tpw = len(word_tokens) / len(words) if words else 0.0
        recon_ok = (self.decode(toks).strip() == s.strip())
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