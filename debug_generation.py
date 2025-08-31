#!/usr/bin/env python3
"""
Debug script to test BPE decoding and token generation
"""

import pickle
import torch
from utils import load_cached_bpe
from task3 import NeuralBigramModel

def test_bpe_decoding():
    """Test BPE encoding and decoding"""
    print("Testing BPE encoding and decoding...")
    
    # Load BPE model
    bpe = load_cached_bpe(2000, "lower_nopunct")
    if bpe is None:
        print("BPE model not found!")
        return
    
    # Test text
    test_text = "to be or not to be"
    print(f"Original text: '{test_text}'")
    
    # Encode
    tokens = bpe.encode(test_text)
    print(f"Encoded tokens: {tokens}")
    
    # Decode
    decoded = bpe.decode(tokens)
    print(f"Decoded text: '{decoded}'")
    
    # Test token mappings
    print(f"Token to ID mapping: {bpe.token2id}")
    print(f"ID to token mapping: {bpe.id2token}")
    
    # Test with some specific tokens
    test_tokens = ["to", "be", "__", "or", "not", "__"]
    print(f"Test tokens: {test_tokens}")
    decoded_test = bpe.decode(test_tokens)
    print(f"Decoded test: '{decoded_test}'")

def test_neural_generation():
    """Test neural bigram generation"""
    print("\nTesting neural bigram generation...")
    
    # Load BPE model
    bpe = load_cached_bpe(2000, "lower_nopunct")
    if bpe is None:
        print("BPE model not found!")
        return
    
    # Load trained model
    vocab_size = len(bpe.vocab)
    model = NeuralBigramModel(vocab_size, 64)
    
    # Try to load the best model
    model_path = "neural_bigram_2000_emb_dim=64_batch=32_lr=0.001_wd=0.0001_final.pt"
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test generation
    context_text = "to be or not to be"
    context_tokens = bpe.encode(context_text)
    print(f"Context tokens: {context_tokens}")
    
    # Convert to IDs
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    print(f"Context IDs: {context_ids}")
    
    # Generate
    with torch.no_grad():
        generated_ids = list(context_ids)
        for i in range(20):
            last_id = torch.tensor([generated_ids[-1]], dtype=torch.long)
            logits = model(last_id).squeeze(0)
            next_id = torch.argmax(logits).item()
            generated_ids.append(next_id)
    
    print(f"Generated IDs: {generated_ids}")
    
    # Convert back to tokens
    generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
    print(f"Generated tokens: {generated_tokens}")
    
    # Decode
    generated_text = bpe.decode(generated_tokens)
    print(f"Generated text: '{generated_text}'")

if __name__ == "__main__":
    test_bpe_decoding()
    test_neural_generation()
