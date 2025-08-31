# Task 2: N-gram Language Modeling on BPE Subwords (FIXED)
# - Uses cached BPE models from Task 1
# - Fits 1..4 gram models with Laplace smoothing
# - Evaluates perplexity on valid/test
# - Generates sample text
# - Saves models + results

import numpy as np
from collections import Counter
import pickle
from utils import (
    normalize_generation_context,
    load_and_slice_data,
    load_cached_bpe,
    save_results,
    GEN_CONTEXT,
)

# Config
PERCENTAGE = 1.0                         # use 100% of dataset splits
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [500, 1000, 2000, 2500]
N_GRAM_ORDERS = [1, 2, 3, 4]
LAPLACE_ALPHA = 1.0
GEN_MAX_TOKENS = 20
GEN_TEMPERATURE = 0.5
BOS = "<s>"


class NGramModel:
    """N-gram model with Laplace smoothing (FIXED VERSION)"""

    def __init__(self, n_order, alpha=1.0):
        self.n = n_order
        self.alpha = alpha
        self.vocab = set()              # tokens actually seen in data (+ BOS)
        self.vocab_size = 0
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def fit(self, tokens, bos=BOS):
        """Build vocab/counts from actual tokens (not full BPE vocab)."""
        # Only tokens we actually see, plus BOS used for padding
        self.vocab = set(tokens) | {bos}
        self.vocab = sorted(list(self.vocab))
        self.vocab_size = len(self.vocab)

        print(f"    Actual vocab size from data: {self.vocab_size}")

        # Pad with BOS tokens and count n-grams
        padded_tokens = [bos] * (self.n - 1) + tokens
        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i + self.n])
            context = ngram[:-1] if self.n > 1 else tuple()
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

        print(f"    Total n-grams: {sum(self.ngram_counts.values())}")
        print(f"    Unique n-grams: {len(self.ngram_counts)}")

    def prob(self, token, history):
        """P(token | history) with Laplace smoothing."""
        if self.n == 1:
            context = tuple()
            ngram = (token,)
        else:
            k = self.n - 1
            context = tuple(history[-k:]) if len(history) >= k else tuple(history)
            ngram = context + (token,)

        numerator = self.ngram_counts.get(ngram, 0) + self.alpha
        denominator = self.context_counts.get(context, 0) + self.alpha * self.vocab_size
        return numerator / denominator

    def perplexity(self, tokens, bos=BOS):
        """Compute perplexity on a token sequence."""
        padded_tokens = [bos] * (self.n - 1) + tokens
        log_prob_sum = 0.0
        token_count = 0

        for i in range(self.n - 1, len(padded_tokens)):
            current_token = padded_tokens[i]
            history = padded_tokens[max(0, i - self.n + 1):i]
            p = self.prob(current_token, history)
            # p should never be zero with Laplace; still guard
            if p > 0.0:
                log_prob_sum += np.log(p)
                token_count += 1

        if token_count == 0:
            return float("inf")
        return np.exp(-log_prob_sum / token_count)

    def generate(self, bpe, context, max_tokens=30, temperature=0.7, bos=BOS):
        """Sample tokens autoregressively and decode with BPE.

        IMPORTANT: preserve trailing space in the context so the first
        sampled token respects the word boundary.
        """
        context = normalize_generation_context(context)
        # Preserve edge spaces during generation to keep the boundary
        context_tokens = bpe.encode(context, preserve_edge_spaces=True)

        tokens = [bos] * (self.n - 1) + context_tokens
        generated_count = 0

        while generated_count < max_tokens:
            history = tokens[-(self.n - 1):] if self.n > 1 else []

            # probs over non-BOS vocab tokens
            vocab_tokens = [t for t in self.vocab if t != bos]
            probs = np.array([self.prob(t, history) for t in vocab_tokens], dtype=np.float64)

            if probs.size == 0 or probs.sum() == 0.0:
                break

            if temperature != 1.0:
                # Temperature on probabilities (simple and stable here)
                probs = probs ** (1.0 / temperature)

            probs = probs + 1e-12
            probs = probs / probs.sum()

            next_token = np.random.choice(vocab_tokens, p=probs)
            tokens.append(next_token)
            generated_count += 1

        # Remove BOS padding and decode
        generated_tokens = tokens[(self.n - 1):]
        return bpe.decode(generated_tokens)


def main():
    print("Task 2: N-gram Language Modeling (FIXED)")
    print("=" * 60)

    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    results = {}

    for merges in MERGE_COUNTS:
        print(f"\nUsing BPE merges={merges}")
        bpe = load_cached_bpe(merges, BEST_NORMALIZATION)
        if bpe is None:
            print(f"ERROR: Run Task 1 first to cache BPE with {merges} merges.")
            continue

        # Encode all splits
        # For dataset encoding we do NOT need to preserve edge spaces.
        train_tokens = bpe.encode(train_text, preserve_edge_spaces=False)
        valid_tokens = bpe.encode(valid_text, preserve_edge_spaces=False)
        test_tokens = bpe.encode(test_text, preserve_edge_spaces=False)

        print(f"  BPE vocab size: {len(bpe.vocab)}")
        print(f"  Train tokens: {len(train_tokens)}")
        print(f"  Unique tokens in train: {len(set(train_tokens))}")

        results[merges] = {"bpe_vocab_size": len(bpe.vocab), "ngram_results": {}}

        for n in N_GRAM_ORDERS:
            print(f"  Training {n}-gram...")
            model = NGramModel(n, LAPLACE_ALPHA)
            model.fit(train_tokens)  # Uses actual vocab from data

            val_ppl = model.perplexity(valid_tokens)
            test_ppl = model.perplexity(test_tokens)

            results[merges]["ngram_results"][f"n={n}"] = {
                "actual_vocab_size": model.vocab_size,
                "val_perplexity": float(val_ppl),
                "test_perplexity": float(test_ppl),
                "model_file": f"task2_fixed_{merges}_{n}.pkl",
            }

            # Save model
            with open(f"task2_fixed_{merges}_{n}.pkl", "wb") as f:
                pickle.dump(model, f)

            print(f"    Actual vocab: {model.vocab_size} | val_ppl={val_ppl:.2f} | test_ppl={test_ppl:.2f}")

            # Generate sample (preserving edge spaces in the context)
            try:
                generated = model.generate(
                    bpe,
                    GEN_CONTEXT,                 # e.g., "fair is foul and "
                    max_tokens=GEN_MAX_TOKENS,
                    temperature=GEN_TEMPERATURE,
                )
                print(f"    [Generated]: {generated}")
            except Exception as e:
                print(f"    [Generation failed]: {e}")

    save_results(results, "task2_fixed_results.pkl")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    for merges, info in results.items():
        print(f"\nBPE merges={merges} (BPE vocab={info['bpe_vocab_size']})")
        for n_key, stats in info["ngram_results"].items():
            actual_vocab = stats["actual_vocab_size"]
            val_ppl = stats["val_perplexity"]
            test_ppl = stats["test_perplexity"]
            print(f"  {n_key}: actual_vocab={actual_vocab} | val={val_ppl:.2f} | test={test_ppl:.2f}")


if __name__ == "__main__":
    main()
