# Task 4: GPT Implementation (Ultra Compact)
# - Uses cached BPE models from Task 1
# - Trains GPT model with transformer architecture
# - Evaluates perplexity and generates text
# - Self-contained with integrated generation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from utils import load_and_slice_data, load_cached_bpe, save_results, GEN_CONTEXT

# Config
PERCENTAGE = 1  # Tiny percentage for testing
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [1000, 2000, 3000, 5000]  # Only test one merge count
BATCH_SIZE = 16  # Smaller batch for testing
LEARNING_RATE = 3e-4
MAX_ITERATIONS = 2000  # Reduced for testing
EARLY_STOPPING_PATIENCE = 200
VALIDATION_INTERVAL = 50

# Single GPT config
GPT_CONFIG = {
    'n_embd': 32,     # Small embedding
    'n_head': 2,      # Small attention heads
    'n_layer': 2,     # Small number of layers
    'chunk_size': 16, # Short sequences
    'dropout': 0.1
}

# Generation config
GEN_MAX_TOKENS = 15
GEN_TEMPERATURE = 0.7

# Create plots directory
os.makedirs('task4_plots', exist_ok=True)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # QKV projections
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention weights and output
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.output(out)

class MLP(nn.Module):
    """Position-wise feed-forward network"""
    
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
    """Single transformer block"""
    
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
    """GPT model"""
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, chunk_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.chunk_size = chunk_size
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(chunk_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) 
            for _ in range(n_layer)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_projection = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model: {n_params:,} parameters")
        
    def forward(self, input_tokens):
        B, T = input_tokens.shape
        assert T <= self.chunk_size
        
        # Get embeddings
        token_emb = self.token_embeddings(input_tokens)
        pos_emb = self.position_embeddings(torch.arange(T, device=input_tokens.device))
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.output_projection(x)
        
        return logits
    
    def calculate_loss(self, input_tokens, target_tokens):
        """Calculate cross-entropy loss"""
        logits = self.forward(input_tokens)
        B, T, C = logits.shape
        
        logits_flat = logits.view(B * T, C)
        targets_flat = target_tokens.view(B * T)
        
        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss

    def generate(self, context_tokens, max_tokens, temperature=1.0, top_k=50):
        """Generate text"""
        self.eval()
        generated = context_tokens.copy()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for step in range(max_tokens):
                input_seq = torch.tensor(generated[-self.chunk_size:], 
                                       dtype=torch.long, device=device).unsqueeze(0)
                
                logits = self.forward(input_seq)[0, -1, :]
                
                # Temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    filtered_logits = torch.full_like(logits, float('-inf'))
                    filtered_logits[top_k_indices] = top_k_logits
                    logits = filtered_logits
                
                # Sampling
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
        
        return generated

def prepare_data(token_stream, chunk_size, batch_size):
    """Prepare sequences for GPT training"""
    if len(token_stream) < chunk_size + 1:
        print(f"Warning: Token stream too short ({len(token_stream)}) for chunk size {chunk_size}")
        return [], []
    
    # Create overlapping sequences
    sequences = []
    stride = chunk_size // 2  # 50% overlap
    
    for i in range(0, len(token_stream) - chunk_size, stride):
        sequence = token_stream[i:i + chunk_size + 1]
        if len(sequence) == chunk_size + 1:
            sequences.append(sequence)
    
    print(f"Created {len(sequences)} training sequences")
    np.random.shuffle(sequences)
    
    # Create batches
    input_batches = []
    target_batches = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        if len(batch_sequences) >= batch_size // 2:  # Allow smaller final batch
            batch_input = torch.tensor([seq[:-1] for seq in batch_sequences], dtype=torch.long)
            batch_target = torch.tensor([seq[1:] for seq in batch_sequences], dtype=torch.long)
            input_batches.append(batch_input)
            target_batches.append(batch_target)
    
    print(f"Created {len(input_batches)} training batches")
    return input_batches, target_batches

def train_model(model, train_batches, target_batches, valid_batches, valid_target_batches, optimizer, max_iterations, device):
    """Train GPT model"""
    model.train()
    model.to(device)
    
    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting training with {len(train_batches)} training batches")
    
    for iteration in range(max_iterations):
        batch_idx = np.random.randint(0, len(train_batches))
        input_batch = train_batches[batch_idx].to(device)
        target_batch = target_batches[batch_idx].to(device)
        
        # Training step
        optimizer.zero_grad()
        loss = model.calculate_loss(input_batch, target_batch)
        
        if torch.isnan(loss):
            print(f"NaN loss at iteration {iteration}! Stopping.")
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record metrics
        loss_val = loss.item()
        history['losses'].append(loss_val)
        history['perplexities'].append(math.exp(loss_val))
        
        # Validation check
        if iteration % VALIDATION_INTERVAL == 0:
            model.eval()
            val_loss = 0.0
            val_batches_used = 0
            
            with torch.no_grad():
                for val_input, val_target in zip(valid_batches[:3], valid_target_batches[:3]):
                    val_input = val_input.to(device)
                    val_target = val_target.to(device)
                    batch_val_loss = model.calculate_loss(val_input, val_target)
                    
                    if not torch.isnan(batch_val_loss):
                        val_loss += batch_val_loss.item()
                        val_batches_used += 1
            
            if val_batches_used > 0:
                val_loss /= val_batches_used
                val_perplexity = math.exp(val_loss)
                
                history['val_losses'].append(val_loss)
                history['val_perplexities'].append(val_perplexity)
                
                print(f"Iter {iteration:4d}: Train Loss = {loss_val:.4f} "
                      f"(PPL = {math.exp(loss_val):6.1f}), "
                      f"Val Loss = {val_loss:.4f} (PPL = {val_perplexity:6.1f})")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += VALIDATION_INTERVAL
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at iteration {iteration}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            model.train()
    
    return history

def plot_training_history(history, merges, lr, save_path):
    """Plot loss and perplexity over iterations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    ax1.plot(history['losses'], label='Train Loss', alpha=0.7)
    if history['val_losses']:
        val_x = np.arange(0, len(history['losses']), VALIDATION_INTERVAL)
        ax1.plot(val_x, history['val_losses'], label='Val Loss', marker='o', markersize=3)
    ax1.set_title(f'GPT Loss vs Iterations (Merges={merges}, LR={lr})')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot perplexity
    ax2.plot(history['perplexities'], label='Train Perplexity', alpha=0.7)
    if history['val_perplexities']:
        ax2.plot(val_x, history['val_perplexities'], label='Val Perplexity', marker='o', markersize=3)
    ax2.set_title(f'GPT Perplexity vs Iterations (Merges={merges}, LR={lr})')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("Task 4: GPT Implementation (Ultra Compact)")
    print("=" * 50)
    
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    
    for merges in MERGE_COUNTS:
        print(f"\nBPE merges={merges}")
        bpe = load_cached_bpe(merges, BEST_NORMALIZATION)
        if bpe is None:
            print(f"ERROR: Run Task 1 first.")
            continue
        
        # Tokenize and convert to IDs
        train_ids = [bpe.token2id.get(token, 0) for token in bpe.encode(train_text)]
        valid_ids = [bpe.token2id.get(token, 0) for token in bpe.encode(valid_text)]
        test_ids = [bpe.token2id.get(token, 0) for token in bpe.encode(test_text)]
        
        vocab_size = len(bpe.vocab)
        print(f"  Vocab size={vocab_size}, train tokens={len(train_ids)}")
        
        # Prepare data
        train_batches, train_target_batches = prepare_data(train_ids, GPT_CONFIG['chunk_size'], BATCH_SIZE)
        valid_batches, valid_target_batches = prepare_data(valid_ids, GPT_CONFIG['chunk_size'], BATCH_SIZE)
        
        if not train_batches:
            print(f"  No training data available")
            continue
        
        # Create GPT model
        model = GPTModel(
            vocab_size=vocab_size,
            n_embd=GPT_CONFIG['n_embd'],
            n_head=GPT_CONFIG['n_head'],
            n_layer=GPT_CONFIG['n_layer'],
            chunk_size=GPT_CONFIG['chunk_size'],
            dropout=GPT_CONFIG['dropout']
        )
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        
        # Train model
        history = train_model(model, train_batches, train_target_batches,
                            valid_batches, valid_target_batches, optimizer,
                            MAX_ITERATIONS, device)
        
        # Plot training history
        plot_path = f'task4_plots/task4_merges{merges}_lr{LEARNING_RATE}.png'
        plot_training_history(history, merges, LEARNING_RATE, plot_path)
        print(f"  Plot saved: {plot_path}")
        
        # Evaluate final perplexity
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_input, val_target in zip(valid_batches[:5], valid_target_batches[:5]):
                val_input = val_input.to(device)
                val_target = val_target.to(device)
                val_loss += model.calculate_loss(val_input, val_target).item()
            val_loss /= min(len(valid_batches), 5)
        
        final_perplexity = math.exp(val_loss)
        print(f"  Final val perplexity: {final_perplexity:.2f}")
        
        # Generate sample text
        context_tokens = bpe.encode(GEN_CONTEXT)
        context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
        
        try:
            generated_ids = model.generate(context_ids, GEN_MAX_TOKENS, temperature=GEN_TEMPERATURE, top_k=50)
            generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
            generated_text = bpe.decode(generated_tokens)
            print(f"  [Generated]: {generated_text}")
        except Exception as e:
            print(f"  [Generation failed]: {e}")
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': GPT_CONFIG,
            'vocab_size': vocab_size,
            'token_to_id': bpe.token2id,
            'id_to_token': bpe.id2token,
            'merge_count': merges,
            'training_history': history
        }, f'gpt_model_merge{merges}_compact.pt')
        
        print(f"  Model saved: gpt_model_merge{merges}_compact.pt")
    
    print("\nTask 4 completed!")

if __name__ == "__main__":
    main()
