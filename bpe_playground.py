#!/usr/bin/env python3
"""
Interactive BPE Playground
Experiment with BPE encoding and decoding interactively
"""

import sys
import os

# Add the current directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import BPE, normalize_text

def interactive_bpe():
    """Interactive BPE playground"""
    print("🎮 BPE Interactive Playground")
    print("=" * 50)
    print("Type 'quit' to exit, 'help' for commands")
    print()
    
    global bpe
    bpe = None
    
    while True:
        try:
            command = input("BPE> ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                print("Goodbye! 👋")
                break
                
            elif command == 'help':
                print_help()
                
            elif command == 'train':
                train_bpe_interactive()
                
            elif command == 'encode':
                if bpe is None:
                    print("❌ No BPE model trained yet. Use 'train' first.")
                    continue
                encode_text_interactive(bpe)
                
            elif command == 'decode':
                if bpe is None:
                    print("❌ No BPE model trained yet. Use 'train' first.")
                    continue
                decode_tokens_interactive(bpe)
                
            elif command == 'info':
                if bpe is None:
                    print("❌ No BPE model trained yet. Use 'train' first.")
                    continue
                show_bpe_info(bpe)
                
            elif command == 'demo':
                run_demo()
                
            elif command == '':
                continue
                
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_help():
    """Print help information"""
    print("\n📚 Available Commands:")
    print("  train   - Train a new BPE model on text")
    print("  encode  - Encode text to tokens")
    print("  decode  - Decode tokens back to text")
    print("  info    - Show BPE model information")
    print("  demo    - Run a quick demonstration")
    print("  help    - Show this help message")
    print("  quit    - Exit the playground")
    print()

def train_bpe_interactive():
    """Train BPE model interactively"""
    print("\n🚀 Training BPE Model")
    print("-" * 30)
    
    # Get training text
    print("Enter training text (or press Enter for sample text):")
    text = input("Text: ").strip()
    
    if not text:
        text = "the quick brown fox jumps over the lazy dog while the cat sleeps peacefully on the warm mat"
        print(f"Using sample text: '{text}'")
    
    # Get number of merges
    while True:
        try:
            merges_input = input("Number of merges (default 50): ").strip()
            if not merges_input:
                merges = 50
                break
            merges = int(merges_input)
            if merges > 0:
                break
            print("❌ Number of merges must be positive")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Get normalization strategy
    print("Normalization strategy:")
    print("  1. lower_nopunct (default)")
    print("  2. aggressive")
    norm_choice = input("Choice (1-2, default 1): ").strip()
    
    if norm_choice == "2":
        norm = "aggressive"
    else:
        norm = "lower_nopunct"
    
    print(f"\nTraining BPE with {merges} merges and {norm} normalization...")
    
    # Train BPE
    global bpe
    bpe = BPE()
    bpe.fit(text, k_merges=merges, norm=norm)
    
    print(f"✅ BPE model trained successfully!")
    print(f"   Vocabulary size: {len(bpe.vocab)}")
    print(f"   Number of merges: {len(bpe.merges)}")

def encode_text_interactive(bpe):
    """Encode text interactively"""
    print("\n🔤 Encoding Text to Tokens")
    print("-" * 30)
    
    text = input("Enter text to encode: ").strip()
    if not text:
        print("❌ No text provided")
        return
    
    print(f"\nOriginal text: '{text}'")
    
    # Encode
    encoded = bpe.encode(text)
    word_tokens = [t for t in encoded if t != bpe.end_of_word]
    
    print(f"Encoded tokens ({len(encoded)} total, {len(word_tokens)} word tokens):")
    print(f"  {encoded}")
    
    # Show token breakdown
    print(f"\nToken breakdown:")
    current_word = ""
    word_tokens = []
    
    for token in encoded:
        if token == bpe.end_of_word:
            if current_word:
                print(f"    '{current_word}' -> {word_tokens}")
                current_word = ""
                word_tokens = []
        else:
            current_word += token
            word_tokens.append(token)
    
    if current_word:
        print(f"    '{current_word}' -> {word_tokens}")

def decode_tokens_interactive(bpe):
    """Decode tokens interactively"""
    print("\n🔤 Decoding Tokens to Text")
    print("-" * 30)
    print("Enter tokens separated by spaces (e.g., 'hello __ world __'):")
    
    tokens_input = input("Tokens: ").strip()
    if not tokens_input:
        print("❌ No tokens provided")
        return
    
    # Parse tokens
    tokens = tokens_input.split()
    
    print(f"\nInput tokens: {tokens}")
    
    # Decode
    decoded = bpe.decode(tokens)
    print(f"Decoded text: '{decoded}'")
    
    # Verify by re-encoding
    re_encoded = bpe.encode(decoded)
    print(f"Re-encoded: {re_encoded}")
    
    # Check if reversible
    if re_encoded == tokens:
        print("✅ Perfect reversibility!")
    else:
        print("⚠️  Some information may be lost in encoding/decoding")

def show_bpe_info(bpe):
    """Show BPE model information"""
    print("\n📊 BPE Model Information")
    print("-" * 30)
    print(f"Vocabulary size: {len(bpe.vocab)}")
    print(f"Number of merges: {len(bpe.merges)}")
    print(f"End-of-word symbol: '{bpe.end_of_word}'")
    
    print(f"\nVocabulary (first 20 tokens):")
    for i, token in enumerate(bpe.vocab[:20]):
        print(f"  {i:2d}: '{token}'")
    
    if len(bpe.vocab) > 20:
        print(f"  ... and {len(bpe.vocab) - 20} more")
    
    print(f"\nMerges (first 15):")
    for i, (a, b) in enumerate(bpe.merges[:15]):
        print(f"  {i+1:2d}: '{a}' + '{b}' -> '{a + b}'")
    
    if len(bpe.merges) > 15:
        print(f"  ... and {len(bpe.merges) - 15} more")

def run_demo():
    """Run a quick demonstration"""
    print("\n🎬 Quick BPE Demonstration")
    print("-" * 30)
    
    # Sample text
    text = "artificial intelligence machine learning"
    
    print(f"Training on: '{text}'")
    
    # Train BPE
    demo_bpe = BPE()
    demo_bpe.fit(text, k_merges=20)
    
    print(f"Vocabulary size: {len(demo_bpe.vocab)}")
    print(f"Merges: {len(demo_bpe.merges)}")
    
    # Test encoding/decoding
    test_text = "artificial intelligence"
    encoded = demo_bpe.encode(test_text)
    decoded = demo_bpe.decode(encoded)
    
    print(f"\nTest: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"✓ Reversible: {decoded == normalize_text(test_text, 'lower_nopunct')}")



if __name__ == "__main__":
    print("Welcome to the BPE Interactive Playground!")
    print("This playground lets you experiment with Byte Pair Encoding.")
    print()
    
    # Initialize global BPE variable
    bpe = None
    
    try:
        interactive_bpe()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
