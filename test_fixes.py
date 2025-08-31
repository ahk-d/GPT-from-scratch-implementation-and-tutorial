#!/usr/bin/env python3
"""
Test script to verify all fixes work correctly
Run this after applying the fixes to make sure everything works
"""

import sys
import os

def test_bpe_fixes():
    """Test that BPE encoding/decoding works properly"""
    print("Testing BPE fixes...")
    print("=" * 50)
    
    try:
        from utils import load_cached_bpe
        
        # Load BPE model
        bpe = load_cached_bpe(2000, "lower_nopunct")
        if bpe is None:
            print("❌ No BPE model found. Run task1.py first.")
            return False
        
        # Test basic encoding/decoding
        test_text = "to be or not to be that is the question"
        print(f"Original: '{test_text}'")
        
        # Encode
        tokens = bpe.encode(test_text)
        print(f"Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"Tokens: {tokens}")
        
        # Decode
        decoded = bpe.decode(tokens)
        print(f"Decoded: '{decoded}'")
        
        # Check reconstruction
        normalized_original = bpe._norm(test_text, 'lower_nopunct')
        if decoded == normalized_original:
            print("✅ BPE encoding/decoding works correctly!")
            return True
        else:
            print("❌ BPE reconstruction failed!")
            print(f"Expected: '{normalized_original}'")
            print(f"Got:      '{decoded}'")
            return False
            
    except Exception as e:
        print(f"❌ Error testing BPE: {e}")
        return False

def test_generation():
    """Test text generation"""
    print("\nTesting text generation...")
    print("=" * 50)
    
    try:
        # Test with a simple mock
        from utils import load_cached_bpe
        
        bpe = load_cached_bpe(2000, "lower_nopunct")
        if bpe is None:
            print("❌ No BPE model found. Run task1.py first.")
            return False
        
        # Test token-by-token simulation
        context = "to be or not"
        tokens = bpe.encode(context)
        
        # Simulate adding some tokens
        additional_tokens = ['to', '__', 'be', '__']
        all_tokens = tokens + additional_tokens
        
        decoded = bpe.decode(all_tokens)
        print(f"Context: '{context}'")
        print(f"After adding tokens: '{decoded}'")
        
        if " " in decoded:  # Should have spaces between words
            print("✅ Token generation produces proper word spacing!")
            return True
        else:
            print("❌ Generated text lacks proper word spacing")
            return False
            
    except Exception as e:
        print(f"❌ Error testing generation: {e}")
        return False

def test_model_loading():
    """Test that models can be loaded"""
    print("\nTesting model loading...")
    print("=" * 50)
    
    try:
        import os
        
        # Check for result files
        task2_results = os.path.exists("task2_results.pkl")
        task3_results = os.path.exists("task3_fixed_results.pkl")
        
        print(f"Task 2 results available: {'✅' if task2_results else '❌'}")
        print(f"Task 3 results available: {'✅' if task3_results else '❌'}")
        
        if task2_results or task3_results:
            print("✅ At least some model results are available!")
            return True
        else:
            print("❌ No model results found. Run task2.py and/or task3.py first.")
            return False
            
    except Exception as e:
        print(f"❌ Error checking model files: {e}")
        return False

def main():
    print("🔧 Testing GPT-from-scratch fixes...")
    print("=" * 60)
    
    # Run tests
    bpe_ok = test_bpe_fixes()
    gen_ok = test_generation()
    model_ok = test_model_loading()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"BPE encoding/decoding: {'✅ PASS' if bpe_ok else '❌ FAIL'}")
    print(f"Text generation:       {'✅ PASS' if gen_ok else '❌ FAIL'}")
    print(f"Model availability:    {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if bpe_ok and gen_ok:
        print("\n🎉 Core fixes are working! You can now:")
        print("   1. Run: python generate_text.py --model task3_neural_bigram_2000 --context 'to be or not to be'")
        print("   2. The generated text should have proper word spacing")
        print("   3. Words should be separated instead of running together")
    else:
        print("\n⚠️  Some issues remain. Please:")
        print("   1. Make sure you've applied the BPE method fixes to utils.py")
        print("   2. Replace generate_text.py with the fixed version")
        print("   3. Ensure you have run task1.py to create BPE models")

if __name__ == "__main__":
    main()