#!/usr/bin/env python3
"""
Test script to check BPE word boundary handling
Tests if BPE generates "word__" instead of "word, __"
"""

from utils import BPE

def test_bpe_word_boundaries():
    """Test BPE word boundary handling"""
    
    # Create a simple test text with punctuation
    test_text = "hello, world! how are you?"
    
    print("Testing BPE word boundary handling")
    print("=" * 50)
    print(f"Original text: '{test_text}'")
    
    # Test with lower_nopunct normalization
    print(f"\n--- Testing with 'lower_nopunct' normalization ---")
    bpe = BPE()
    bpe.fit(test_text, k_merges=10, norm='lower_nopunct')
    
    # Encode the text
    tokens = bpe.encode(test_text, norm='lower_nopunct')
    print(f"Encoded tokens: {tokens}")
    
    # Decode back
    decoded = bpe.decode(tokens)
    print(f"Decoded text: '{decoded}'")
    
    # Check if we have "word__" pattern
    has_word_underscore = any('__' in token for token in tokens)
    print(f"Has 'word__' pattern: {has_word_underscore}")
    
    # Test with aggressive normalization
    print(f"\n--- Testing with 'aggressive' normalization ---")
    bpe2 = BPE()
    bpe2.fit(test_text, k_merges=10, norm='aggressive')
    
    # Encode the text
    tokens2 = bpe2.encode(test_text, norm='aggressive')
    print(f"Encoded tokens: {tokens2}")
    
    # Decode back
    decoded2 = bpe2.decode(tokens2)
    print(f"Decoded text: '{decoded2}'")
    
    # Check if we have "word__" pattern
    has_word_underscore2 = any('__' in token for token in tokens2)
    print(f"Has 'word__' pattern: {has_word_underscore2}")
    
    # Test with a more complex example
    print(f"\n--- Testing with Shakespeare text ---")
    with open("Shakespeare_clean_test.txt", "r", encoding="utf-8") as f:
        shakespeare_sample = f.read()[:500]  # First 500 chars
    
    print(f"Shakespeare sample: '{shakespeare_sample[:100]}...'")
    
    bpe3 = BPE()
    bpe3.fit(shakespeare_sample, k_merges=50, norm='lower_nopunct')
    
    # Encode a small portion
    sample_text = shakespeare_sample.split('.')[0] + "."
    print(f"Sample sentence: '{sample_text}'")
    
    tokens3 = bpe3.encode(sample_text, norm='lower_nopunct')
    print(f"Encoded tokens: {tokens3}")
    
    # Check for word boundary patterns
    word_boundaries = []
    for token in tokens3:
        if token.endswith('__'):
            word_boundaries.append(token)
    
    print(f"Word boundary tokens: {word_boundaries}")
    
    # Decode back
    decoded3 = bpe3.decode(tokens3)
    print(f"Decoded text: '{decoded3}'")

if __name__ == "__main__":
    test_bpe_word_boundaries()
