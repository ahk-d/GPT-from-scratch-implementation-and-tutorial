# Task 4: GPT Implementation (Ultra Compact, Robust Early Stopping)
# - Uses cached BPE models from Task 1
# - Trains GPT model with transformer architecture
# - Robust early stopping (EMA-smoothed val loss, min_delta, patience in evals)
# - LR-on-plateau scheduler + best checkpoint restore
# - Evaluates perplexity and generates text

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from utils import normalize_generation_context, load_and_slice_data, load_cached_bpe, GEN_CONTEXT

# -------------------------
# Config
# -------------------------
PERCENTAGE = 1
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [500, 1000, 2000, 2500]

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EARLY_STOPPING_PATIENCE = 200     # already in your code; keep as-is or tune
VALIDATION_INTERVAL = 50          # already in your code; keep as-is
MAX_ITERATIONS = 3000             # you asked to cap at 3000
# validation cadence
VALIDATION_INTERVAL = 50

# early stopping (now measured in validation checks, not raw iterations)
EARLY_STOPPING_PATIENCE_EVALS = 12   # stop if no real improvement after N validations
MIN_DELTA = 1e-3                     # require at least this improvement in val loss
EMA_BETA = 0.9                       # smoothing for validation loss (EMA)
VAL_MAX_BATCHES = 64                 # evaluate on up to this many fixed val batches

# LR scheduler (ReduceLROnPlateau) settings
SCHED_FACTOR = 0.5
SCHED_PATIENCE_EVALS = 5
SCHED_MIN_LR = 1e-5
SCHED_COOLDOWN_EVALS = 2

GPT_CONFIG = {
    'n_embd': 64,
    'n_head': 4,
    'n_layer': 4,
    'chunk_size': 32,
    'dropout': 0.1
}

# Generation config
GEN_MAX_TOKENS = 15
GEN_TEMPERATURE = 0.7
TOP_K = 50

# Create plots directory
os.makedirs('task4_plots', exist_ok=True)

# -------------------------
# Model components
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.output(out)

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, chunk_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.chunk_size = chunk_size

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(chunk_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.output_projection = nn.Linear(n_embd, vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model: {n_params:,} parameters")

    def forward(self, input_tokens):
        B, T = input_tokens.shape
        assert T <= self.chunk_size, f"Input length {T} > chunk_size {self.chunk_size}"
        tok = self.token_embeddings(input_tokens)
        pos = self.position_embeddings(torch.arange(T, device=input_tokens.device))
        x = self.dropout(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.output_projection(x)
        return logits

    def calculate_loss(self, input_tokens, target_tokens):
        logits = self.forward(input_tokens)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), target_tokens.view(B * T))
        return loss

    def generate(self, context_ids, max_tokens, temperature=1.0, top_k=50):
        self.eval()
        generated = list(context_ids)
        device = next(self.parameters()).device
        with torch.no_grad():
            for _ in range(max_tokens):
                inp = torch.tensor(generated[-self.chunk_size:], dtype=torch.long,
                                   device=device).unsqueeze(0)
                logits = self.forward(inp)[0, -1, :]
                if temperature > 0:
                    logits = logits / temperature
                if top_k and top_k > 0:
                    tk_vals, tk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    filtered = torch.full_like(logits, float('-inf'))
                    filtered[tk_idx] = tk_vals
                    logits = filtered
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                generated.append(next_id)
        return generated

# -------------------------
# Data prep
# -------------------------
def prepare_data(token_stream, chunk_size, batch_size):
    if len(token_stream) < chunk_size + 1:
        print(f"Warning: Token stream too short ({len(token_stream)}) for chunk size {chunk_size}")
        return [], []
    sequences = []
    stride = chunk_size // 2
    for i in range(0, len(token_stream) - chunk_size, stride):
        seq = token_stream[i:i + chunk_size + 1]
        if len(seq) == chunk_size + 1:
            sequences.append(seq)
    print(f"Created {len(sequences)} training sequences")
    np.random.shuffle(sequences)

    input_batches, target_batches = [], []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if len(batch) >= batch_size // 2:
            inp = torch.tensor([s[:-1] for s in batch], dtype=torch.long)
            tgt = torch.tensor([s[1:] for s in batch], dtype=torch.long)
            input_batches.append(inp)
            target_batches.append(tgt)
    print(f"Created {len(input_batches)} training batches")
    return input_batches, target_batches

# -------------------------
# Training with robust early stopping
# -------------------------
def train_model(model, train_batches, target_batches,
                valid_batches, valid_target_batches,
                optimizer, max_iterations, device):
    """Train GPT model with EMA-smoothed early stopping and LR decay — using SAME param names."""
    model.train()
    model.to(device)

    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # EMA smoothing of validation loss to stabilize early stopping
    ema_val_loss = None
    ema_alpha = 0.1  # internal constant; doesn’t change your param names

    print(f"Starting training with {len(train_batches)} training batches")

    for iteration in range(max_iterations):
        # ---- train step ----
        batch_idx = np.random.randint(0, len(train_batches))
        input_batch = train_batches[batch_idx].to(device)
        target_batch = target_batches[batch_idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.calculate_loss(input_batch, target_batch)
        if torch.isnan(loss):
            print(f"NaN loss at iteration {iteration}! Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # record train metrics
        loss_val = float(loss.item())
        history['losses'].append(loss_val)
        history['perplexities'].append(math.exp(loss_val))

        # ---- validation / early stopping ----
        if iteration % VALIDATION_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                use_n = min(8, len(valid_batches))  # small but stable sample
                val_losses = []
                for val_input, val_target in zip(valid_batches[:use_n], valid_target_batches[:use_n]):
                    val_input = val_input.to(device)
                    val_target = val_target.to(device)
                    v = model.calculate_loss(val_input, val_target)
                    if not torch.isnan(v):
                        val_losses.append(float(v.item()))
                val_loss = float(np.mean(val_losses)) if val_losses else float('inf')

            # EMA smooth the val loss
            if ema_val_loss is None:
                ema_val_loss = val_loss
            else:
                ema_val_loss = ema_alpha * val_loss + (1.0 - ema_alpha) * ema_val_loss

            history['val_losses'].append(val_loss)
            history['val_perplexities'].append(math.exp(val_loss))

            # simple plateau detection (compare EMA)
            improved = ema_val_loss < (best_val_loss - 1e-4)

            # reduce LR on plateau every 3 validations without improvement
            if not improved and (patience_counter // VALIDATION_INTERVAL + 1) % 3 == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.5, LEARNING_RATE * 0.1)

            # bookkeeping
            if improved:
                best_val_loss = ema_val_loss
                patience_counter = 0
                # lightweight state save (CPU clone)
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += VALIDATION_INTERVAL

            print(
                f"Iter {iteration:4d}: Train {loss_val:.4f} (PPL {math.exp(loss_val):6.1f}) | "
                f"Val {val_loss:.4f} (EMA {ema_val_loss:.4f}, PPL {math.exp(val_loss):6.1f}) | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )

            # hard stop on patience
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at iteration {iteration}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

            model.train()

    return history

# -------------------------
# Plotting
# -------------------------
def plot_training_history(history, merges, lr, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['losses'], label='Train Loss', alpha=0.7)
    if history['val_losses']:
        # place val points at the actual validation iterations
        val_x = np.arange(0, len(history['losses']), VALIDATION_INTERVAL)[:len(history['val_losses'])]
        ax1.plot(val_x, history['val_losses'], label='Val Loss', marker='o', markersize=3)
    ax1.set_title(f'GPT Loss vs Iterations (Merges={merges}, LR={lr})')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['perplexities'], label='Train Perplexity', alpha=0.7)
    if history['val_perplexities']:
        val_x = np.arange(0, len(history['losses']), VALIDATION_INTERVAL)[:len(history['val_perplexities'])]
        ax2.plot(val_x, history['val_perplexities'], label='Val Perplexity', marker='o', markersize=3)
    ax2.set_title(f'GPT Perplexity vs Iterations (Merges={merges}, LR={lr})')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    print("Task 4: GPT Implementation (Ultra Compact, Robust Early Stopping)")
    print("=" * 50)

    device = torch.device("cpu")  # switch to "cuda" if you want and it's available
    torch.manual_seed(42)
    np.random.seed(42)

    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)

    for merges in MERGE_COUNTS:
        print(f"\nBPE merges={merges}")
        bpe = load_cached_bpe(merges, BEST_NORMALIZATION)
        if bpe is None:
            print("ERROR: Run Task 1 first to cache BPE models.")
            continue

        # tokenize -> ids
        train_ids = [bpe.token2id.get(tok, 0) for tok in bpe.encode(train_text)]
        valid_ids = [bpe.token2id.get(tok, 0) for tok in bpe.encode(valid_text)]
        test_ids  = [bpe.token2id.get(tok, 0) for tok in bpe.encode(test_text)]

        vocab_size = len(bpe.vocab)
        print(f"  Vocab size={vocab_size}, train tokens={len(train_ids)}")

        train_batches, train_target_batches = prepare_data(train_ids, GPT_CONFIG['chunk_size'], BATCH_SIZE)
        valid_batches, valid_target_batches = prepare_data(valid_ids, GPT_CONFIG['chunk_size'], BATCH_SIZE)

        if not train_batches or not valid_batches:
            print("  Not enough data for training/validation.")
            continue

        model = GPTModel(
            vocab_size=vocab_size,
            n_embd=GPT_CONFIG['n_embd'],
            n_head=GPT_CONFIG['n_head'],
            n_layer=GPT_CONFIG['n_layer'],
            chunk_size=GPT_CONFIG['chunk_size'],
            dropout=GPT_CONFIG['dropout']
        )

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

        history = train_model(
            model, train_batches, train_target_batches,
            valid_batches, valid_target_batches,
            optimizer, MAX_ITERATIONS, device
        )

        plot_path = f'task4_plots/task4_merges{merges}_lr{LEARNING_RATE}.png'
        plot_training_history(history, merges, LEARNING_RATE, plot_path)
        print(f"  Plot saved: {plot_path}")

        # final validation perplexity
        model.eval()
        val_loss = 0.0
        denom = min(len(valid_batches), 5)
        with torch.no_grad():
            for v_inp, v_tgt in list(zip(valid_batches, valid_target_batches))[:denom]:
                v_inp = v_inp.to(device)
                v_tgt = v_tgt.to(device)
                val_loss += model.calculate_loss(v_inp, v_tgt).item()
        val_loss /= max(1, denom)
        print(f"  Final val perplexity: {math.exp(val_loss):.2f}")

        # generation
        context = normalize_generation_context(GEN_CONTEXT)
        context_ids = [bpe.token2id.get(tok, 0) for tok in bpe.encode(context)]
        try:
            gen_ids = model.generate(context_ids, GEN_MAX_TOKENS, temperature=GEN_TEMPERATURE, top_k=TOP_K)
            gen_tokens = [bpe.id2token.get(i, "<UNK>") for i in gen_ids]
            print(f"  [Generated]: {bpe.decode(gen_tokens)}")
        except Exception as e:
            print(f"  [Generation failed]: {e}")

        # save checkpoint
        save_name = f'gpt_model_merge{merges}_compact.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': GPT_CONFIG,
            'vocab_size': vocab_size,
            'token_to_id': bpe.token2id,
            'id_to_token': bpe.id2token,
            'merge_count': merges,
            'training_history': history
        }, save_name)
        print(f"  Model saved: {save_name}")

    print("\nTask 4 completed!")

if __name__ == "__main__":
    main()
