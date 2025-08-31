#!/usr/bin/env python3
"""
Check BPE model details
"""

from utils import load_cached_bpe

def check_bpe_models():
    """Check BPE model details"""
    print("Checking BPE models...")
    
    for merge_count in [1000, 2000]:
        for norm in ["lower_nopunct", "aggressive"]:
            print(f"\nBPE {merge_count} merges, {norm} normalization:")
            bpe = load_cached_bpe(merge_count, norm)
            if bpe:
                print(f"  Vocab size: {len(bpe.vocab)}")
                print(f"  End-of-word token: '{bpe.end_of_word}'")
                print(f"  End-of-word in vocab: {bpe.end_of_word in bpe.vocab}")
                if hasattr(bpe, 'token2id'):
                    print(f"  Token2ID size: {len(bpe.token2id)}")
                    if bpe.end_of_word in bpe.token2id:
                        eow_id = bpe.token2id[bpe.end_of_word]
                        print(f"  End-of-word ID: {eow_id}")
            else:
                print("  Not found")

if __name__ == "__main__":
    check_bpe_models()
