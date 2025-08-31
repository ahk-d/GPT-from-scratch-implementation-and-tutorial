from utils import load_cached_bpe

def test_bpe():
    bpe = load_cached_bpe(2000, "lower_nopunct")
    if bpe:
        text = "to be or not to be"
        encoded = bpe.encode(text)
        decoded = bpe.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Vocab size: {len(bpe.vocab)}")
    else:
        print("BPE not found")

test_bpe()
