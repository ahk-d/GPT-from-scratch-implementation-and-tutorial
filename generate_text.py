#!/usr/bin/env python3
"""
Text Generation from Best Models — Fixed & Robust
- Loads best models from each task using saved artifacts
- Uses stable tokenizer <-> id mappings
- Adds temperature / top-k sampling
- Gracefully handles missing checkpoints
"""

import os
import pickle
import math
import torch
import numpy as np

from utils import load_cached_bpe
from task2 import NGramLanguageModel
from task3 import NeuralBigramModel
from task4 import GPTModel

# ----------------------------
# Utilities
# ----------------------------

def load_task_results(task_num):
    """Load results dict saved during training for a specific task."""
    for name in (f"task{task_num}_results.pkl", f"task{task_num}_fixed_results.pkl"):
        if os.path.exists(name):
            with open(name, "rb") as f:
                return pickle.load(f)
    print(f"[warn] Results file for task {task_num} not found.")
    return None


def get_token_mappings(bpe):
    """
    Returns (token_to_id, id_to_token, vocab_size) using the tokenizer's
    authoritative mapping. Falls back gently if not present.
    """
    if hasattr(bpe, "token_to_id") and isinstance(bpe.token_to_id, dict):
        token_to_id = bpe.token_to_id
    else:
        # Fallback: build deterministic mapping from vocab sequence
        # (Assumes bpe.vocab is an ordered list/tuple)
        vocab = list(getattr(bpe, "vocab"))
        token_to_id = {tok: i for i, tok in enumerate(vocab)}

    id_to_token = {i: t for t, i in token_to_id.items()}
    vocab_size = len(token_to_id)
    return token_to_id, id_to_token, vocab_size


def encode_tokens(bpe, token_to_id, text):
    # bpe.encode should return a list of tokens; map to ids with unknowns to 0
    tokens = bpe.encode(text)
    return [token_to_id.get(t, 0) for t in tokens]


def decode_tokens(bpe, id_to_token, ids):
    tokens = [id_to_token.get(i, "<UNK>") for i in ids]
    return bpe.decode(tokens)


def resolve_special_ids(token_to_id):
    """
    Return (bos_id, eos_id, eow_id) if present; otherwise (None, None, None).
    Recognized keys: <BOS>/<EOS>, <s></s>, BOS/EOS, and '__' for end-of-word.
    """
    bos_keys = ["<BOS>", "<s>", "BOS"]
    eos_keys = ["<EOS>", "</s>", "EOS"]
    eow_keys = ["__"]  # end-of-word symbol
    bos_id = next((token_to_id[k] for k in bos_keys if k in token_to_id), None)
    eos_id = next((token_to_id[k] for k in eos_keys if k in token_to_id), None)
    eow_id = next((token_to_id[k] for k in eow_keys if k in token_to_id), None)
    return bos_id, eos_id, eow_id


def sample_from_logits(logits, temperature=1.0, top_k=None):
    """
    logits: 1D tensor [vocab]
    """
    logits = logits.float()
    if temperature is not None and temperature > 0.0 and temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < logits.numel():
        vals, idx = torch.topk(logits, k=top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[idx] = logits[idx]
        logits = mask

    probs = torch.softmax(logits, dim=-1)
    # Numerical safety: if probs is NaN or sums to 0, fallback to uniform
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        probs = torch.ones_like(probs) / probs.numel()
    return torch.multinomial(probs, 1).item()


# ----------------------------
# N-gram: loading & generation
# ----------------------------

def load_trained_ngram(best_n, artifact_dict):
    """
    Tries to construct an NGramLanguageModel and load its counts/params from artifacts.
    Expected optional keys in artifact_dict:
      - 'counts_path' OR 'model_path' (pickle or custom)
    """
    model = NGramLanguageModel(best_n)
    counts_path = artifact_dict.get("counts_path")
    model_path = artifact_dict.get("model_path")
    loaded = False

    try:
        if counts_path and os.path.exists(counts_path):
            # Expect your class to have a loader; otherwise unpickle fields
            if hasattr(model, "load_counts"):
                model.load_counts(counts_path)
            else:
                with open(counts_path, "rb") as f:
                    data = pickle.load(f)
                # Try common attribute names:
                for attr in ("counts", "counts_n", "ngram_counts"):
                    if attr in data:
                        setattr(model, attr, data[attr])
                if "vocab_size" in data:
                    model.vocab_size = data["vocab_size"]
            loaded = True
        elif model_path and os.path.exists(model_path):
            # A fully pickled model instance:
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            # Shallow copy learned attributes
            for k, v in loaded_model.__dict__.items():
                setattr(model, k, v)
            loaded = True
    except Exception as e:
        print(f"[warn] Failed to load n-gram artifact: {e}")

    return model, loaded


def generate_ngram_ids(ngram_model, id_to_token, token_to_id, context_ids, max_tokens=50, eos_id=None, eow_id=None):
    """
    Generic n-gram sampling using the model's _calculate_order_probability(token, context_tokens)
    interface. We compute a probability over all tokens each step.
    If the method or learned params are missing -> backs off to uniform.
    """
    n = getattr(ngram_model, "n_order", 1)
    vocab_size = len(id_to_token)
    ids_to_tokens = [id_to_token[i] for i in range(vocab_size)]
    generated = list(context_ids) if context_ids else []

    has_calc = hasattr(ngram_model, "_calculate_order_probability")

    for _ in range(max_tokens):
        # Derive token context in *tokens* (strings), length n-1
        if n > 1:
            ctx_ids = generated[-(n - 1):] if len(generated) >= (n - 1) else generated
        else:
            ctx_ids = []
        ctx_tokens = [id_to_token.get(i, "<UNK>") for i in ctx_ids]

        probs = np.zeros(vocab_size, dtype=np.float64)
        if has_calc:
            # Build distribution by querying each possible token
            total = 0.0
            for i, tok in enumerate(ids_to_tokens):
                try:
                    p = float(ngram_model._calculate_order_probability(tok, tuple(ctx_tokens)))
                except Exception:
                    p = 0.0
                probs[i] = max(p, 0.0)
                total += probs[i]
            if not math.isfinite(total) or total <= 0.0:
                probs[:] = 1.0 / vocab_size  # uniform fallback
            else:
                probs /= total
        else:
            probs[:] = 1.0 / vocab_size

        next_id = int(np.random.choice(vocab_size, p=probs))
        generated.append(next_id)
        
        # Don't add end-of-word token here - it should be handled by the model itself
        # The BPE encoder already adds end-of-word tokens after each word
        
        if eos_id is not None and next_id == eos_id:
            break

    return generated


# ----------------------------
# Neural Bigram: loading & generation
# ----------------------------

def load_trained_neural_bigram(vocab_size, emb_dim, artifact_dict, device="cpu"):
    """
    Expected optional keys:
      - 'checkpoint_path' (torch save)
      - 'state_dict_key' (e.g., 'model' if saved as {'model': state_dict, ...})
    """
    model = NeuralBigramModel(vocab_size, emb_dim)
    ckpt = artifact_dict.get("checkpoint_path")
    key = artifact_dict.get("state_dict_key", "model")
    loaded = False
    try:
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                model.load_state_dict(state[key])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            else:
                model.load_state_dict(state)
            loaded = True
    except Exception as e:
        print(f"[warn] Failed to load Neural Bigram checkpoint: {e}")

    model.eval()
    return model, loaded


def generate_neural_bigram_ids(model, context_ids, max_tokens=50, eos_id=None, temperature=0.9, top_k=50):
    if not context_ids:
        raise ValueError("Neural bigram generation requires at least one context token id.")
    generated = list(context_ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            last = torch.tensor([generated[-1]], dtype=torch.long)
            logits = model(last).squeeze(0)  # [vocab]
            next_id = sample_from_logits(logits, temperature=temperature, top_k=top_k)
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
    return generated


# ----------------------------
# GPT: loading & generation
# ----------------------------

def load_trained_gpt(model_config, artifact_dict, device="cpu"):
    """
    Expected:
      - model_config (dict) used to construct GPTModel
      - artifact_dict['checkpoint_path']
      - optional artifact_dict['state_dict_key'] (default 'model')
    """
    model = GPTModel(**model_config)
    ckpt = artifact_dict.get("checkpoint_path")
    key = artifact_dict.get("state_dict_key", "model")
    loaded = False
    try:
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                model.load_state_dict(state[key])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            else:
                model.load_state_dict(state)
            loaded = True
    except Exception as e:
        print(f"[warn] Failed to load GPT checkpoint: {e}")

    model.eval()
    return model, loaded


def generate_gpt_ids(model, context_ids, max_tokens=50, eos_id=None, temperature=0.9, top_k=40, device="cpu"):
    ids = list(context_ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
            logits = model(x)[:, -1, :].squeeze(0)  # [vocab]
            next_id = sample_from_logits(logits.cpu(), temperature=temperature, top_k=top_k)
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
    return ids


# ----------------------------
# Main
# ----------------------------

def main():
    print("Text Generation from Best Models")
    print("=" * 50)

    # User-editable generation settings
    context_text = "to be or not to"
    max_tokens = 50
    device = "cpu"  # set to "cuda" if your models/checkpoints were trained for GPU and CUDA is available

    print(f"Context: '{context_text}'")
    print(f"Max tokens to generate: {max_tokens}\n")

    # ---------------- Task 1: Tokenization summary ----------------
    print("Task 1: BPE Tokenization")
    print("-" * 30)
    task1_results = load_task_results(1)
    if task1_results:
        # Pick best by valid avg_tokens_per_word
        try:
            best_config = min(task1_results, key=lambda x: x['evaluation']['valid']['avg_tokens_per_word'])
            val = best_config['evaluation']['valid']['avg_tokens_per_word']
            print("Best BPE configuration:")
            print(f"  Normalization: {best_config['normalization_technique']}")
            print(f"  Merge count: {best_config['merge_count']}")
            print(f"  Validation avg tokens/word: {val:.4f}")
        except Exception:
            print("[warn] Could not parse Task 1 results structure.")
    print()

    # ---------------- Task 2: N-gram ----------------
    print("Task 2: N-gram Language Modeling")
    print("-" * 30)
    task2_results = load_task_results(2)

    if task2_results:
        for merge_count in [1000, 2000]:
            if merge_count not in task2_results:
                continue

            try:
                ngram_results = task2_results[merge_count]['ngram_results']
                # Find best by val perplexity
                best_key, best_val = min(
                    ngram_results.items(),
                    key=lambda kv: kv[1].get('val_perplexity', float('inf'))
                )
                best_n = int(best_key.split('=')[1]) if '=' in best_key else ngram_results[best_key].get('n', 1)
                best_info = ngram_results[best_key]
                print(f"BPE {merge_count} merges - Best n-gram (n={best_n}):")
                if 'val_perplexity' in best_info:
                    print(f"  Validation perplexity: {best_info['val_perplexity']:.4f}")

                # Load BPE & mappings
                bpe = load_cached_bpe(merge_count, "lower_nopunct")
                if not bpe:
                    print("  [warn] Could not load cached BPE.")
                    print()
                    continue

                token_to_id, id_to_token, vocab_size = get_token_mappings(bpe)
                bos_id, eos_id, eow_id = resolve_special_ids(token_to_id)
                context_ids = encode_tokens(bpe, token_to_id, context_text)

                                 # Attempt to load trained n-gram counts/state
                 model, ok = load_trained_ngram(best_n, best_info)
                 if not ok:
                     print("  [warn] Trained n-gram artifact not found; falling back to uniform/backoff sampling.")
                 else:
                     print("  [info] Loaded trained n-gram model successfully.")

                # Generate ids then decode
                gen_ids = generate_ngram_ids(model, id_to_token, token_to_id, context_ids, max_tokens=max_tokens, eos_id=eos_id, eow_id=eow_id)
                gen_text = decode_tokens(bpe, id_to_token, gen_ids)
                print(f"  Generated text: {gen_text}")
                print()

            except Exception as e:
                print(f"  [warn] Failed to run n-gram for merges={merge_count}: {e}\n")

    # ---------------- Task 3: Neural Bigram ----------------
    print("Task 3: Neural Bigram Language Modeling")
    print("-" * 30)
    task3_results = load_task_results(3)

    if task3_results:
        for merge_count in [1000, 2000]:
            if merge_count not in task3_results:
                continue

            try:
                data = task3_results[merge_count]
                neural_results = data['hyperparameter_results']
                best_config = data.get('best_config')
                if not best_config or best_config not in neural_results:
                    print(f"BPE {merge_count} merges - No best config recorded.")
                    print()
                    continue

                best_info = neural_results[best_config]
                print(f"BPE {merge_count} merges - Best neural bigram:")
                print(f"  Config: {best_config}")
                if 'val_perplexity' in best_info:
                    print(f"  Validation perplexity: {best_info['val_perplexity']:.4f}")

                # BPE & ids
                bpe = load_cached_bpe(merge_count, "lower_nopunct")
                if not bpe:
                    print("  [warn] Could not load cached BPE.")
                    print()
                    continue

                token_to_id, id_to_token, vocab_size = get_token_mappings(bpe)
                bos_id, eos_id, eow_id = resolve_special_ids(token_to_id)
                context_ids = encode_tokens(bpe, token_to_id, context_text)
                if not context_ids:
                    # if context is empty after encoding, seed with BOS if available
                    if bos_id is not None:
                        context_ids = [bos_id]
                    else:
                        print("  [warn] Empty context for neural bigram; skipping.")
                        print()
                        continue

                emb_dim = best_info.get('embedding_dim', 128)
                                 model, ok = load_trained_neural_bigram(vocab_size, emb_dim, best_info, device=device)
                 if not ok:
                     print("  [warn] Missing neural bigram checkpoint; skipping generation.\n")
                     continue
                 else:
                     print("  [info] Loaded trained neural bigram model successfully.")

                gen_ids = generate_neural_bigram_ids(
                    model,
                    context_ids,
                    max_tokens=max_tokens,
                    eos_id=eos_id,
                    temperature=0.9,
                    top_k=50
                )
                gen_text = decode_tokens(bpe, id_to_token, gen_ids)
                print(f"  Generated text: {gen_text}\n")

            except Exception as e:
                print(f"  [warn] Failed to run neural bigram for merges={merge_count}: {e}\n")

    # ---------------- Task 4: GPT ----------------
    print("Task 4: GPT Language Modeling")
    print("-" * 30)
    task4_results = load_task_results(4)

    if task4_results and 'gpt_results' in task4_results:
        for merge_count in [1000, 2000]:
            try:
                if merge_count not in task4_results['gpt_results']:
                    continue
                ginfo = task4_results['gpt_results'][merge_count]
                print(f"BPE {merge_count} merges - GPT model:")
                if 'val_perplexity' in ginfo:
                    print(f"  Validation perplexity: {ginfo['val_perplexity']:.4f}")

                bpe = load_cached_bpe(merge_count, "lower_nopunct")
                if not bpe:
                    print("  [warn] Could not load cached BPE.")
                    print()
                    continue

                token_to_id, id_to_token, vocab_size = get_token_mappings(bpe)
                bos_id, eos_id, eow_id = resolve_special_ids(token_to_id)
                context_ids = encode_tokens(bpe, token_to_id, context_text)
                # Some GPT setups expect BOS at start:
                if bos_id is not None and (len(context_ids) == 0 or context_ids[0] != bos_id):
                    # Prepend BOS only if your training used it; safe default is to leave as-is.
                    pass

                if 'checkpoint_path' not in ginfo or 'model_config' not in ginfo:
                    # If training saved a ready-made sample text, print it
                    if 'sample_text' in ginfo:
                        print(f"  Generated text: {ginfo['sample_text']}\n")
                    else:
                        print("  [warn] Missing GPT checkpoint/model_config; skipping generation.\n")
                    continue

                model, ok = load_trained_gpt(ginfo['model_config'], ginfo, device=device)
                if not ok:
                    print("  [warn] Failed to load GPT checkpoint; skipping generation.\n")
                    continue

                gen_ids = generate_gpt_ids(
                    model,
                    context_ids,
                    max_tokens=max_tokens,
                    eos_id=eos_id,
                    temperature=0.9,
                    top_k=40,
                    device=device
                )
                gen_text = decode_tokens(bpe, id_to_token, gen_ids)
                print(f"  Generated text: {gen_text}\n")

            except Exception as e:
                print(f"  [warn] Failed to run GPT for merges={merge_count}: {e}\n")
    else:
        print("[warn] Task 4 results not found or missing 'gpt_results'.\n")

    print("Text generation completed!")


if __name__ == "__main__":
    main()
