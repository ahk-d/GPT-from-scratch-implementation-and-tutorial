#!/usr/bin/env python3
"""
Simple BPE Demonstration
Shows BPE encoding and decoding with interesting examples
"""

import sys
import os

# Add the current directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import BPE, normalize_text

def demo_bpe_basic():
    """Demonstrate basic BPE functionality"""
    print("=== BPE Basic Demonstration ===\n")
    
    # Sample text with repeated patterns
    text = "the cat sat on the mat while the dog ran in the park"
    
    print(f"Original text: '{text}'")
    print(f"Text length: {len(text)} characters")
    print(f"Word count: {len(text.split())} words")
    
    # Initialize and train BPE
    bpe = BPE()
    print(f"\nTraining BPE with 30 merges...")
    bpe.fit(text, k_merges=30)
    
    print(f"Vocabulary size: {len(bpe.vocab)}")
    print(f"Number of merges: {len(bpe.merges)}")
    
    # Show some merges
    print(f"\nFirst 10 merges:")
    for i, (a, b) in enumerate(bpe.merges[:10]):
        print(f"  {i+1:2d}: '{a}' + '{b}' -> '{a + b}'")
    
    # Encode the text
    encoded = bpe.encode(text)
    print(f"\nEncoded tokens ({len(encoded)} tokens):")
    print(f"  {encoded}")
    
    # Decode back to text
    decoded = bpe.decode(encoded)
    print(f"\nDecoded text: '{decoded}'")
    
    # Verify reversibility
    print(f"✓ Encoding/decoding reversible: {decoded == normalize_text(text, 'lower_nopunct')}")
    
    return bpe

def demo_bpe_subwords():
    """Demonstrate BPE subword tokenization"""
    print("\n=== BPE Subword Tokenization ===\n")
    
    # Words that should benefit from subword tokenization
    words = [
        "artificial",
        "intelligence", 
        "machine",
        "learning",
        "neural",
        "networks",
        "transformer",
        "attention"
    ]
    
    # Train BPE on these words
    text = " ".join(words)
    bpe = BPE()
    bpe.fit(text, k_merges=50)
    
    print(f"Training on technical vocabulary with {len(bpe.merges)} merges")
    print(f"Vocabulary size: {len(bpe.vocab)}")
    
    # Show how each word is tokenized
    print(f"\nSubword tokenization:")
    for word in words:
        encoded = bpe.encode(word)
        # Remove end-of-word tokens for display
        tokens = [t for t in encoded if t != bpe.end_of_word]
        print(f"  '{word:12}' -> {tokens}")
    
    return bpe

def demo_bpe_efficiency():
    """Demonstrate BPE efficiency improvements"""
    print("\n=== BPE Efficiency Analysis ===\n")
    
    # Text with varying levels of repetition
    texts = [
        "hello world",  # Simple, no repetition
        "the quick brown fox jumps over the lazy dog",  # Some repetition
        "artificial intelligence artificial intelligence artificial intelligence",  # High repetition
    ]
    
    for i, text in enumerate(texts):
        print(f"Text {i+1}: '{text}'")
        
        # Train BPE
        bpe = BPE()
        bpe.fit(text, k_merges=20)
        
        # Encode
        encoded = bpe.encode(text)
        word_tokens = [t for t in encoded if t != bpe.end_of_word]
        
        print(f"  Original: {len(text.split())} words, {len(text)} chars")
        print(f"  Encoded:  {len(word_tokens)} tokens")
        print(f"  Compression: {len(word_tokens) / len(text.split()):.2f} tokens/word")
        print()

def demo_bpe_merges():
    """Demonstrate BPE merge process"""
    print("\n=== BPE Merge Process ===\n")
    
    # Simple example to show merge progression
    text = "low lower lowest newer newest"
    
    print(f"Training text: '{text}'")
    print(f"Initial characters: {list(set(''.join(text.split())))}")
    
    # Train with different numbers of merges
    for merges in [5, 10, 15]:
        bpe = BPE()
        bpe.fit(text, k_merges=merges)
        
        print(f"\nWith {merges} merges:")
        print(f"  Vocabulary size: {len(bpe.vocab)}")
        print(f"  Actual merges: {len(bpe.merges)}")
        
        # Show merges
        print(f"  Merges:")
        for i, (a, b) in enumerate(bpe.merges):
            print(f"    {i+1:2d}: '{a}' + '{b}' -> '{a + b}'")
        
        # Test encoding
        test_word = "lowest"
        encoded = bpe.encode(test_word)
        word_tokens = [t for t in encoded if t != bpe.end_of_word]
        print(f"  '{test_word}' -> {word_tokens}")

def main():
    """Run BPE demonstrations"""
    print("🚀 BPE (Byte Pair Encoding) Demonstration\n")
    
    try:
        # Run demonstrations
        demo_bpe_basic()
        demo_bpe_subwords()
        demo_bpe_efficiency()
        demo_bpe_merges()
        
        print("\n✨ BPE demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
