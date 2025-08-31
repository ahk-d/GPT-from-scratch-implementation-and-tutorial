# Task 3: Neural Bigram Language Modeling (FIXED for new BPE)
# - Uses cached BPE models from Task 1
# - Trains neural bigram models with different hyperparameters
# - Evaluates perplexity on valid/test
# - Generates sample text directly
# - Saves models + results

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import normalize_generation_context, load_and_slice_data, load_cached_bpe, save_results, GEN_CONTEXT

# Config - ADJUSTED for better performance
PERCENTAGE = 1                           # use 100% like your run
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [500, 1000, 2000]
EMBEDDING_DIMS = [64]
BATCH_SIZES = [32]
LEARNING_RATES = [1e-3, 5e-4]
MAX_ITERATIONS = 10000
EARLY_STOPPING_PATIENCE = 2000
WEIGHT_DECAY_VALUES = [1e-5]
VALIDATION_INTERVAL = 100

# Generation config
GEN_MAX_TOKENS = 20
GEN_TEMPERATURE = 0.7

# Create plots directory
os.makedirs('task3_plots', exist_ok=True)

class NeuralBigramModel(nn.Module):
    """Neural bigram model with embedding + linear projection"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.prev_token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.prev_token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    def forward(self, prev_tokens):
        embeddings = self.prev_token_embedding(prev_tokens)
        logits = self.output_projection(embeddings)
        return logits
    def calculate_loss(self, prev_tokens, next_tokens):
        logits = self.forward(prev_tokens)
        return nn.functional.cross_entropy(logits, next_tokens)

def prepare_data(token_stream, batch_size, device):
    """Prepare bigram (x_t -> x_{t+1}) batches"""
    pairs = [(token_stream[i], token_stream[i+1]) for i in range(len(token_stream) - 1)]
    np.random.shuffle(pairs)
    batches = []
    for i in range(0, len(pairs) - batch_size + 1, batch_size):
        chunk = pairs[i:i + batch_size]
        prev_tokens = torch.tensor([p[0] for p in chunk], dtype=torch.long, device=device)
        next_tokens = torch.tensor([p[1] for p in chunk], dtype=torch.long, device=device)
        batches.append((prev_tokens, next_tokens))
    return batches

def train_model(model, train_batches, valid_batches, optimizer, 
               max_iterations, patience, device, validation_interval=100):
    model.train()
    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    for iteration in range(max_iterations):
        bi = np.random.randint(0, len(train_batches))
        prev_batch, next_batch = train_batches[bi]
        optimizer.zero_grad()
        loss = model.calculate_loss(prev_batch, next_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history['losses'].append(loss.item())
        history['perplexities'].append(torch.exp(loss).item())
        if iteration % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                take = min(len(valid_batches), 10)
                for vb_prev, vb_next in valid_batches[:take]:
                    val_loss += model.calculate_loss(vb_prev, vb_next).item()
                val_loss /= max(1, take)
            history['val_losses'].append(val_loss)
            history['val_perplexities'].append(np.exp(val_loss))
            if iteration % (validation_interval * 5) == 0:
                print(f"    Iter {iteration}: Train {loss.item():.4f} | Val {val_loss:.4f} | Val PPL {np.exp(val_loss):.2f}")
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += validation_interval
            if patience_counter >= patience:
                print(f"    Early stopping at iter {iteration}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
            model.train()
    return history

def plot_training_history(history, merges, lr, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['losses'], label='Train Loss', alpha=0.7)
    if history['val_losses']:
        val_x = np.arange(0, len(history['losses']), VALIDATION_INTERVAL)
        val_x = val_x[:len(history['val_losses'])]
        ax1.plot(val_x, history['val_losses'], label='Val Loss', marker='o', markersize=3)
    ax1.set_title(f'Loss (Merges={merges}, LR={lr})')
    ax1.set_xlabel('Iterations'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history['perplexities'], label='Train Perplexity', alpha=0.7)
    if history['val_perplexities']:
        ax2.plot(val_x, history['val_perplexities'], label='Val Perplexity', marker='o', markersize=3)
    ax2.set_title(f'Perplexity (Merges={merges}, LR={lr})')
    ax2.set_xlabel('Iterations'); ax2.set_ylabel('Perplexity'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def evaluate_model(model, batches, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for prev_batch, next_batch in batches:
            total_loss += model.calculate_loss(prev_batch, next_batch).item()
            total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    return float(np.exp(avg_loss))

def generate_text(model, bpe, context, max_tokens=30, temperature=0.7):
    """Generation that respects your edge-space handling from utils"""
    model.eval()
    device = next(model.parameters()).device
    context = normalize_generation_context(context)
    # Keep trailing space for boundary; your utils.encode adds one if missing.
    context_tokens = bpe.encode(context, preserve_edge_spaces=True)
    context_ids = [bpe.token2id[t] for t in context_tokens if t in bpe.token2id]  # after rebuild, these should all exist
    if not context_ids:
        context_ids = [0]
    generated_ids = context_ids[:]
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated_ids[-1]], dtype=torch.long, device=device)
            logits = model(inp)
            logits = logits[0] if logits.ndim == 2 else logits
            if temperature > 0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            generated_ids.append(nxt)
    toks = [bpe.id2token[i] for i in generated_ids]
    return bpe.decode(toks).rstrip()

def main():
    print("Task 3: Neural Bigram Language Modeling (FIXED)")
    print("=" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42); np.random.seed(42)

    # Load data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)

    results = {}
    for merges in MERGE_COUNTS:
        print(f"\nBPE merges={merges}")
        bpe = load_cached_bpe(merges, BEST_NORMALIZATION)
        if bpe is None:
            print(f"ERROR: Run Task 1 first to cache BPE with {merges} merges.")
            continue

        # --- Encode splits (utils handles spacing well) ----------------------
        train_tokens = bpe.encode(train_text)   # default preserve_edge_spaces=True in utils
        valid_tokens = bpe.encode(valid_text)
        test_tokens  = bpe.encode(test_text)

        # >>> CRITICAL FIX: rebuild vocab/id maps from actual tokens produced
        #     (also include GEN_CONTEXT tokens so generation never hits unknowns)
        ctx_tokens = bpe.encode(normalize_generation_context(GEN_CONTEXT), preserve_edge_spaces=True)
        all_tokens = sorted(set(train_tokens) | set(valid_tokens) | set(test_tokens) | set(ctx_tokens))
        bpe.vocab = all_tokens
        bpe.token2id = {tok: i for i, tok in enumerate(bpe.vocab)}
        bpe.id2token = {i: tok for tok, i in bpe.token2id.items()}
        vocab_size = len(bpe.vocab)
        print(f"  Vocab size={vocab_size}, train tokens={len(train_tokens)}")

        # >>> map tokens to ids with the rebuilt maps (no unknowns anymore)
        train_ids = [bpe.token2id[t] for t in train_tokens]
        valid_ids = [bpe.token2id[t] for t in valid_tokens]
        test_ids  = [bpe.token2id[t] for t in test_tokens]

        results[merges] = {"vocab_size": vocab_size, "lr_results": {}}

        for lr in LEARNING_RATES:
            print(f"  Training with LR={lr}")
            emb_dim, batch_size, weight_decay = EMBEDDING_DIMS[0], BATCH_SIZES[0], WEIGHT_DECAY_VALUES[0]
            print(f"    Config: emb_dim={emb_dim}, batch={batch_size}")

            # Prepare data
            train_batches = prepare_data(train_ids, batch_size, device)
            valid_batches = prepare_data(valid_ids, batch_size, device)
            test_batches  = prepare_data(test_ids,  batch_size, device)
            print(f"    Prepared {len(train_batches)} training batches, {len(valid_batches)} validation batches")

            # Model + optimizer
            model = NeuralBigramModel(vocab_size, emb_dim).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Train
            history = train_model(model, train_batches, valid_batches, optimizer,
                                  MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device, VALIDATION_INTERVAL)

            # Plot
            plot_path = f'task3_plots/task3_merges{merges}_lr{lr}.png'
            plot_training_history(history, merges, lr, plot_path)
            print(f"    Plot saved: {plot_path}")

            # Evaluate
            val_perplexity = evaluate_model(model, valid_batches, device)
            test_perplexity = evaluate_model(model, test_batches, device)
            print(f"    val_ppl={val_perplexity:.2f} | test_ppl={test_perplexity:.2f}")

            # Save results
            results[merges]["lr_results"][lr] = {
                "val_perplexity": val_perplexity,
                "test_perplexity": test_perplexity,
                "final_train_loss": history['losses'][-1] if history['losses'] else float('inf'),
                "best_val_loss": min(history['val_losses']) if history['val_losses'] else float('inf')
            }

            # Generate
            try:
                generated = generate_text(model, bpe, GEN_CONTEXT, GEN_MAX_TOKENS, GEN_TEMPERATURE)
                print(f"    [Generated]: {generated}")
                results[merges]["lr_results"][lr]["generated_sample"] = generated
            except Exception as e:
                print(f"    [Generation failed]: {e}")
                results[merges]["lr_results"][lr]["generated_sample"] = "Generation failed"

            # Save model
            model_path = f"task3_{merges}_lr{lr}_final.pt"
            torch.save(model.state_dict(), model_path)
            results[merges]["lr_results"][lr]["model_path"] = model_path

    # Save results
    save_results(results, "task3_results.pkl")

    print("\n" + "="*60)
    print("SUMMARY:")
    for merges, info in results.items():
        print(f"\nBPE merges={merges} (vocab={info['vocab_size']})")
        for lr, stats in info["lr_results"].items():
            print(f"  LR={lr}: val_ppl={stats['val_perplexity']:.2f} | test_ppl={stats['test_perplexity']:.2f}")
    print("\nTask 3 completed!")

if __name__ == "__main__":
    main()
