# Task 2: N-gram Language Modeling on BPE Subwords
# - Uses cached BPE models from Task 1
# - Fits 1..4 gram models with Laplace smoothing
# - Evaluates perplexity on valid/test
# - Generates sample text
# - Saves models + results

import numpy as np
from collections import Counter
import pickle
from utils import load_and_slice_data, load_cached_bpe, save_results, GEN_CONTEXT

# Config
PERCENTAGE = 0.1               # 10% of dataset
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [1000, 2000]      # only 2000 and 3000
N_GRAM_ORDERS = [1, 2, 3, 4]
LAPLACE_ALPHA = 0.1
GEN_MAX_TOKENS = 40
GEN_TEMPERATURE = 0.7

class NGramModel:
    """N-gram model with Laplace smoothing"""

    def __init__(self, n_order, alpha=0.1):
        self.n = n_order
        self.alpha = alpha
        self.vocab_size = 0
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def fit(self, tokens, vocab_size, bos="<s>"):
        self.vocab_size = vocab_size
        stream = [bos] * (self.n - 1) + tokens
        for i in range(len(stream) - self.n + 1):
            ngram = tuple(stream[i:i+self.n])
            ctx = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[ctx] += 1

    def prob(self, token, history):
        ctx = tuple(history[-(self.n-1):]) if self.n > 1 else tuple()
        ngram = ctx + (token,)
        num = self.ngram_counts.get(ngram, 0) + self.alpha
        den = self.context_counts.get(ctx, 0) + self.alpha * self.vocab_size
        return num / den

    def perplexity(self, tokens, bos="<s>"):
        stream = [bos] * (self.n - 1) + tokens
        log_sum, count = 0.0, 0
        for i in range(self.n-1, len(stream)):
            token = stream[i]
            hist = stream[i-self.n+1:i]
            p = self.prob(token, hist)
            log_sum += np.log(p)
            count += 1
        return np.exp(-log_sum / max(1, count))

    def generate(self, bpe, context, max_tokens=30, temperature=0.7):
        tokens = bpe.encode(context)
        for _ in range(max_tokens):
            hist = tokens[-(self.n-1):]
            probs = np.array([self.prob(tok, hist) for tok in bpe.vocab])
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
            next_tok = np.random.choice(bpe.vocab, p=probs)
            tokens.append(next_tok)
        return bpe.decode(tokens)

def main():
    print("Task 2: N-gram Language Modeling")
    print("="*60)

    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    results = {}

    for merges in MERGE_COUNTS:
        print(f"\nUsing BPE merges={merges}")
        bpe = load_cached_bpe(merges, BEST_NORMALIZATION)
        if bpe is None:
            print(f"ERROR: Run Task 1 first to cache BPE with {merges} merges.")
            continue

        train_toks = bpe.encode(train_text)
        valid_toks = bpe.encode(valid_text)
        test_toks  = bpe.encode(test_text)
        vocab_size = len(bpe.vocab)

        print(f"  Vocab size={vocab_size}, train tokens={len(train_toks)}")

        results[merges] = {"vocab_size": vocab_size, "ngram_results": {}}

        for n in N_GRAM_ORDERS:
            print(f" Training {n}-gram...")
            model = NGramModel(n, LAPLACE_ALPHA)
            model.fit(train_toks, vocab_size)

            val_ppl = model.perplexity(valid_toks)
            test_ppl = model.perplexity(test_toks)

            results[merges]["ngram_results"][f"n={n}"] = {
                "val_perplexity": val_ppl,
                "test_perplexity": test_ppl,
                "model_file": f"task2_{merges}_{n}.pkl"
            }

            with open(f"task2_{merges}_{n}.pkl", "wb") as f:
                pickle.dump(model, f)

            print(f"   val_ppl={val_ppl:.2f} | test_ppl={test_ppl:.2f}")

            generated = model.generate(bpe, GEN_CONTEXT, GEN_MAX_TOKENS, GEN_TEMPERATURE)
            print(f"   [Generated sample]: {generated}")

    save_results(results, "task2_results.pkl")

    print("\nSummary:")
    for merges, info in results.items():
        print(f" merges={merges}, vocab={info['vocab_size']}")
        for n, stats in info["ngram_results"].items():
            print(f"   {n}: val_ppl={stats['val_perplexity']:.2f}, test_ppl={stats['test_perplexity']:.2f}")

if __name__ == "__main__":
    main()
