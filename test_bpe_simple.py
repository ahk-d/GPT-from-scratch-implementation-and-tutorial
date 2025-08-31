from utils import load_cached_bpe

def test_bpe():
    bpe = load_cached_bpe(2000, "lower_nopunct")
    if bpe:
        text = "So shall I, love; and so, I pray, be you:"
        encoded = bpe.encode(text)
        decoded = bpe.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Vocab size: {len(bpe.vocab)}")
        
    else:
        print("BPE not found")

test_bpe()
