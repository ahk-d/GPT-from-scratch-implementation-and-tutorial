#!/usr/bin/env python3
"""
Debug BPE token mappings
"""

import pickle
from utils import load_cached_bpe

def debug_bpe_mappings():
    """Debug BPE token mappings"""
    print("Debugging BPE token mappings...")
    
    # Load BPE model
    bpe = load_cached_bpe(1000, "lower_nopunct")
    if not bpe:
        print("Could not load BPE model")
        return
    
    print(f"BPE vocab size: {len(bpe.vocab)}")
    print(f"End-of-word token: '{bpe.end_of_word}'")
    
    # Check if BPE has token mappings
    if hasattr(bpe, 'token2id') and bpe.token2id:
        print("BPE has token2id mapping")
        token_to_id = bpe.token2id
        id_to_token = bpe.id2token
    else:
        print("BPE does not have token2id mapping, creating from vocab")
        vocab = list(bpe.vocab)
        token_to_id = {token: i for i, token in enumerate(vocab)}
        id_to_token = {i: token for i, token in enumerate(vocab)}
    
    print(f"Token to ID mapping size: {len(token_to_id)}")
    print(f"ID to token mapping size: {len(id_to_token)}")
    
    # Check end-of-word token mapping
    eow_token = bpe.end_of_word
    if eow_token in token_to_id:
        eow_id = token_to_id[eow_token]
        print(f"End-of-word token '{eow_token}' -> ID {eow_id}")
        print(f"ID {eow_id} -> token '{id_to_token.get(eow_id, 'NOT_FOUND')}'")
    else:
        print(f"End-of-word token '{eow_token}' NOT found in token_to_id!")
        print(f"Available tokens: {list(token_to_id.keys())[:10]}...")
    
    # Test encoding and decoding
    test_text = "the quick brown fox"
    print(f"\nTesting with text: '{test_text}'")
    
    tokens = bpe.encode(test_text)
    print(f"Encoded tokens: {tokens}")
    
    # Convert to IDs
    token_ids = []
    for token in tokens:
        if token in token_to_id:
            token_ids.append(token_to_id[token])
        else:
            print(f"WARNING: Token '{token}' not found in token_to_id!")
            token_ids.append(0)  # fallback
    
    print(f"Token IDs: {token_ids}")
    
    # Convert back to tokens
    decoded_tokens = []
    for token_id in token_ids:
        if token_id in id_to_token:
            decoded_tokens.append(id_to_token[token_id])
        else:
            print(f"WARNING: ID {token_id} not found in id_to_token!")
            decoded_tokens.append('<UNK>')
    
    print(f"Decoded tokens: {decoded_tokens}")
    
    # Final decode
    final_text = bpe.decode(decoded_tokens)
    print(f"Final text: '{final_text}'")
    print(f"Correct: {final_text == test_text}")

if __name__ == "__main__":
    debug_bpe_mappings()
