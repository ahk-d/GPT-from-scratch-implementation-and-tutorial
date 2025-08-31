#!/usr/bin/env python3
"""
Test file for BPE (Byte Pair Encoding) encoding and decoding
Tests the BPE implementation with various scenarios
"""

import sys
import os
import json
from collections import Counter

# Add the current directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import BPE, normalize_text

def test_bpe_basic():
    """Test basic BPE functionality"""
    print("=== Testing Basic BPE Functionality ===")
    
    # Sample text for testing
    sample_texts = [
        "hello world",
        "the quick brown fox",
        "artificial intelligence",
        "machine learning is fun",
        "hello world hello world"  # Test repetition
    ]
    
    # Initialize BPE with small vocabulary for testing
    bpe = BPE()
    
    # Test training
    print("Training BPE on sample texts...")
    # Join texts with spaces for BPE.fit which expects a single string
    combined_text = " ".join(sample_texts)
    bpe.fit(combined_text, k_merges=50)
    
    print(f"Vocabulary size: {len(bpe.vocab)}")
    print(f"Number of merges: {len(bpe.merges)}")
    
    # Test encoding and decoding
    for text in sample_texts:
        print(f"\nOriginal: '{text}'")
        
        # Encode
        encoded = bpe.encode(text)
        print(f"Encoded: {encoded}")
        
        # Decode
        decoded = bpe.decode(encoded)
        print(f"Decoded: '{decoded}'")
        
        # Check if encoding/decoding is reversible
        assert decoded == text, f"Encoding/decoding failed for: {text}"
        print("✓ Encoding/decoding successful")

def test_bpe_merges():
    """Test BPE merge operations"""
    print("\n=== Testing BPE Merge Operations ===")
    
    # Create a simple corpus
    corpus = [
        "low lower lowest",
        "newer newest",
        "wider width",
        "low lower lowest",
        "newer newest",
        "wider width"
    ]
    
    bpe = BPE()
    # Join corpus texts for BPE.fit
    combined_corpus = " ".join(corpus)
    bpe.fit(combined_corpus, k_merges=20)
    
    print(f"Vocabulary size: {len(bpe.vocab)}")
    print(f"Number of merges: {len(bpe.merges)}")
    
    # Show some merges
    print("\nFirst 10 merges:")
    for i, pair in enumerate(bpe.merges[:10]):
        print(f"  {i+1}: {pair[0]}{pair[1]} -> {pair[0] + pair[1]}")
    
    # Test encoding with merges
    test_text = "lowest newest width"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"\nTest text: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    assert decoded == test_text, "Merge test failed"

def test_bpe_edge_cases():
    """Test BPE edge cases"""
    print("\n=== Testing BPE Edge Cases ===")
    
    bpe = BPE()
    
    # Test empty text
    print("Testing empty text...")
    empty_encoded = bpe.encode("")
    empty_decoded = bpe.decode(empty_encoded)
    assert empty_decoded == "", "Empty text handling failed"
    print("✓ Empty text handled correctly")
    
    # Test single character
    print("Testing single character...")
    single_encoded = bpe.encode("a")
    single_decoded = bpe.decode(single_encoded)
    assert single_decoded == "a", "Single character handling failed"
    print("✓ Single character handled correctly")
    
    # Test special characters
    print("Testing special characters...")
    special_text = "hello@world.com"
    special_encoded = bpe.encode(special_text)
    special_decoded = bpe.decode(special_encoded)
    print(f"  Original: '{special_text}'")
    print(f"  Encoded: {special_encoded}")
    print(f"  Decoded: '{special_decoded}'")
    # Note: BPE normalizes text by default, so special chars may be removed
    # We'll test that encoding/decoding is reversible for the normalized version
    normalized_special = normalize_text(special_text, "lower_nopunct")
    assert special_decoded == normalized_special, "Special characters handling failed"
    print("✓ Special characters handled correctly")

def test_bpe_normalization():
    """Test BPE with text normalization"""
    print("\n=== Testing BPE with Text Normalization ===")
    
    # Test different normalization strategies (only those supported by utils.normalize_text)
    normalization_strategies = ["lower_nopunct", "aggressive"]
    
    for strategy in normalization_strategies:
        print(f"\nTesting normalization: {strategy}")
        
        # Sample text with mixed case and punctuation
        sample_text = "Hello, World! How are you?"
        
        # Normalize text
        normalized = normalize_text(sample_text, strategy)
        print(f"Original: '{sample_text}'")
        print(f"Normalized ({strategy}): '{normalized}'")
        
        # Train BPE on normalized text
        bpe = BPE()
        bpe.fit(normalized, k_merges=30)
        
        # Encode and decode
        encoded = bpe.encode(normalized)
        decoded = bpe.decode(encoded)
        
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        
        assert decoded == normalized, f"Normalization test failed for {strategy}"
        print(f"✓ {strategy} normalization successful")

def test_bpe_vocabulary():
    """Test BPE vocabulary management"""
    print("\n=== Testing BPE Vocabulary Management ===")
    
    # Create corpus with repeated patterns
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "the fish swam in the pond"
    ]
    
    bpe = BPE()
    
    # Test different merge counts
    merge_counts = [10, 25, 50]
    
    for merges in merge_counts:
        print(f"\nTesting with {merges} merges:")
        
        # Reset BPE for each test
        bpe = BPE()
        # Join corpus texts for BPE.fit
        combined_corpus = " ".join(corpus)
        bpe.fit(combined_corpus, k_merges=merges)
        
        print(f"  Vocabulary size: {len(bpe.vocab)}")
        print(f"  Number of merges: {len(bpe.merges)}")
        
        # Test encoding
        test_text = "the cat sat"
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        
        print(f"  Test: '{test_text}' -> {encoded} -> '{decoded}'")
        assert decoded == test_text, f"Vocabulary test failed for {merges} merges"
        print(f"  ✓ {merges} merges successful")

def test_bpe_consistency():
    """Test BPE consistency across multiple runs"""
    print("\n=== Testing BPE Consistency ===")
    
    corpus = [
        "machine learning is fascinating",
        "deep learning neural networks",
        "artificial intelligence research"
    ]
    
    # Train BPE multiple times
    results = []
    for run in range(3):
        bpe = BPE()
        # Join corpus texts for BPE.fit
        combined_corpus = " ".join(corpus)
        bpe.fit(combined_corpus, k_merges=40)
        
        # Test encoding
        test_text = "machine learning"
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        
        results.append({
            'vocab_size': len(bpe.vocab),
            'merges': len(bpe.merges),
            'encoded': encoded,
            'decoded': decoded
        })
        
        print(f"Run {run+1}: vocab_size={results[-1]['vocab_size']}, merges={results[-1]['merges']}")
    
    # Check consistency
    vocab_sizes = [r['vocab_size'] for r in results]
    merge_counts = [r['merges'] for r in results]
    
    print(f"Vocabulary sizes: {vocab_sizes}")
    print(f"Merge counts: {merge_counts}")
    
    # All runs should produce the same results
    assert len(set(vocab_sizes)) == 1, "Vocabulary sizes not consistent"
    assert len(set(merge_counts)) == 1, "Merge counts not consistent"
    print("✓ BPE consistency verified")

def main():
    """Run all BPE tests"""
    print("Starting BPE Tests...\n")
    
    try:
        test_bpe_basic()
        test_bpe_merges()
        test_bpe_edge_cases()
        test_bpe_normalization()
        test_bpe_vocabulary()
        test_bpe_consistency()
        
        print("\n🎉 All BPE tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
