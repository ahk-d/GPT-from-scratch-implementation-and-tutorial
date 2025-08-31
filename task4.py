# Task 4: Pure GPT Implementation - GPT MODELS ONLY
# No model comparisons, no imports from other tasks
# Just focus on implementing and training GPT (transformer) models

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle

# Only import from utils (shared utilities)
from utils import (
    load_and_slice_data, save_results, load_cached_bpe
)

# Configuration
PERCENTAGE = 0.01
BEST_NORMALIZATION = "lower_nopunct"  
MERGE_COUNTS = [1000, 2000]
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_ITERATIONS = 15000
EARLY_STOPPING_PATIENCE = 2000
VALIDATION_INTERVAL = 200

# GPT Configurations to test
GPT_CONFIGS = [
    {
        'name': 'GPT-Small',
        'n_embd': 256,
        'n_head': 8, 
        'n_layer': 6,
        'dropout': 0.1,
        'chunk_size': 128,
    },
    {
        'name': 'GPT-Medium', 
        'n_embd': 384,
        'n_head': 12,
        'n_layer': 8,
        'dropout': 0.1,
        'chunk_size': 256,
    }
]

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism"""
    
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
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask (prevent looking at future tokens)
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
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP"""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Pre-layer normalization (more stable than post-layer norm)
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    """GPT (Generative Pre-trained Transformer) model"""
    
    def __init__(self, vocab_size, n_embd, n_head, n_layer, chunk_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.chunk_size = chunk_size
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(chunk_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_projection = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT Model created: {n_params:,} parameters")
        
    def _init_weights(self):
        """Initialize weights using GPT-2 style initialization"""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        
    def forward(self, input_tokens):
        """
        Forward pass
        Args:
            input_tokens: (batch_size, sequence_length) tensor of token IDs
        Returns:
            logits: (batch_size, sequence_length, vocab_size) tensor
        """
        B, T = input_tokens.shape
        assert T <= self.chunk_size, f"Sequence length {T} exceeds chunk_size {self.chunk_size}"
        
        # Get embeddings
        token_emb = self.token_embeddings(input_tokens)  # (B, T, n_embd)
        pos_emb = self.position_embeddings(torch.arange(T, device=input_tokens.device))  # (T, n_embd)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)  # (B, T, n_embd)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.output_projection(x)  # (B, T, vocab_size)
        
        return logits
    
    def calculate_loss(self, input_tokens, target_tokens):
        """Calculate cross-entropy loss for language modeling"""
        logits = self.forward(input_tokens)  # (B, T, vocab_size)
        B, T, C = logits.shape
        
        # Flatten for cross-entropy
        logits_flat = logits.view(B * T, C)
        targets_flat = target_tokens.view(B * T)
        
        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss
    
    def calculate_perplexity(self, input_tokens, target_tokens):
        """Calculate perplexity"""
        with torch.no_grad():
            loss = self.calculate_loss(input_tokens, target_tokens)
            perplexity = torch.exp(loss).item()
        return perplexity
    
    def generate(self, context_tokens, max_tokens, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text using the GPT model
        Args:
            context_tokens: List of token IDs to start with
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability <= top_p
        Returns:
            List of token IDs (including context)
        """
        self.eval()
        generated = context_tokens.copy()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Use last chunk_size tokens as context
                input_seq = torch.tensor(generated[-self.chunk_size:], 
                                       dtype=torch.long, device=device).unsqueeze(0)
                
                # Get logits for next token
                logits = self.forward(input_seq)[0, -1, :]  # (vocab_size,)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Apply top-p (nucleus) filtering  
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
        
        return generated

def prepare_gpt_data(token_stream, chunk_size, batch_size):
    """Prepare sequences for GPT training"""
    if len(token_stream) < chunk_size + 1:
        print(f"Warning: Token stream too short ({len(token_stream)}) for chunk size {chunk_size}")
        return [], []
    
    # Create overlapping sequences for better data utilization
    sequences = []
    stride = chunk_size // 4  # 75% overlap
    
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

def train_gpt_model(model, train_batches, target_batches, valid_batches, valid_target_batches, optimizer, max_iterations, device):
    """Train GPT model with validation monitoring"""
    model.train()
    model.to(device)
    
    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting training with {len(train_batches)} training batches")
    
    for iteration in range(max_iterations):
        # Sample random training batch
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
                # Use subset of validation batches for speed
                for val_input, val_target in zip(valid_batches[:10], valid_target_batches[:10]):
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
                
                print(f"Iter {iteration:6d}: Train Loss = {loss_val:.4f} "
                      f"(PPL = {math.exp(loss_val):6.1f}), "
                      f"Val Loss = {val_loss:.4f} (PPL = {val_perplexity:6.1f})")
                
                # Early stopping with model saving
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

def main():
    """Main function - Pure GPT implementation and training"""
    print("Task 4: Pure GPT Implementation")
    print("=" * 60)
    print("Focus: Implement and train GPT (transformer) models")
    print("Architecture: Multi-head self-attention + feed-forward layers")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)
    results = {}
    
    for merge_count in MERGE_COUNTS:
        print(f"\n{'='*50}")
        print(f"Training GPT with {merge_count} BPE merges")
        print(f"{'='*50}")
        
        # Load BPE tokenizer
        bpe = load_cached_bpe(merge_count, BEST_NORMALIZATION)
        if bpe is None:
            print(f" BPE model not found for {merge_count} merges. Run task1.py first.")
            continue
        
        # Tokenize and convert to IDs
        train_tokens = bpe.encode(train_text)
        valid_tokens = bpe.encode(valid_text)
        
        token_to_id = bpe.token2id
        id_to_token = bpe.id2token
        
        train_ids = [token_to_id.get(token, 0) for token in train_tokens]
        valid_ids = [token_to_id.get(token, 0) for token in valid_tokens]
        
        vocab_size = len(bpe.vocab)
        print(f"Vocabulary size: {vocab_size:,}")
        print(f"Training tokens: {len(train_ids):,}")
        print(f"Validation tokens: {len(valid_ids):,}")
        
        # Test different GPT configurations
        merge_results = {}
        
        for config in GPT_CONFIGS:
            config_name = config['name']
            print(f"\n--- Training {config_name} ---")
            print(f"Architecture: {config['n_layer']} layers, {config['n_embd']} dim, {config['n_head']} heads")
            
            # Prepare training data
            train_batches, train_target_batches = prepare_gpt_data(
                train_ids, config['chunk_size'], BATCH_SIZE
            )
            valid_batches, valid_target_batches = prepare_gpt_data(
                valid_ids, config['chunk_size'], BATCH_SIZE
            )
            
            if not train_batches:
                print(f" No training data for {config_name}")
                continue
            
            # Create GPT model
            gpt_model = GPTModel(
                vocab_size=vocab_size,
                n_embd=config['n_embd'],
                n_head=config['n_head'],
                n_layer=config['n_layer'],
                chunk_size=config['chunk_size'],
                dropout=config['dropout']
            )
            
            # Create optimizer
            optimizer = optim.AdamW(
                gpt_model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=0.01
            )
            
            # Train model
            history = train_gpt_model(
                gpt_model, train_batches, train_target_batches,
                valid_batches, valid_target_batches, optimizer,
                MAX_ITERATIONS, device
            )
            
            # Save the trained model
            model_path = f'gpt_model_{merge_count}_{config_name.lower().replace("-", "_")}.pt'
            torch.save({
                'model_state_dict': gpt_model.state_dict(),
                'config': config,
                'vocab_size': vocab_size,
                'token_to_id': token_to_id,
                'id_to_token': id_to_token,
                'bpe_merge_count': merge_count
            }, model_path)
            
            # Generate sample text to verify model works
            context = "to be or not to be"
            context_tokens = bpe.encode(context)
            context_ids = [token_to_id.get(token, 0) for token in context_tokens]
            
            # Test different generation strategies
            generation_samples = {}
            
            try:
                # Conservative generation
                generated_ids = gpt_model.generate(
                    context_ids, 30, temperature=0.7, top_k=50, top_p=0.9
                )
                generated_tokens = [id_to_token.get(id, "<UNK>") for id in generated_ids]
                generation_samples['conservative'] = bpe.decode(generated_tokens)
                
                # Creative generation  
                generated_ids = gpt_model.generate(
                    context_ids, 30, temperature=1.0, top_k=100, top_p=0.95
                )
                generated_tokens = [id_to_token.get(id, "<UNK>") for id in generated_ids]
                generation_samples['creative'] = bpe.decode(generated_tokens)
                
            except Exception as e:
                print(f"Generation failed: {e}")
                generation_samples = {'error': str(e)}
            
            # Store results
            merge_results[config_name] = {
                'training_history': history,
                'generation_samples': generation_samples,
                'config': config,
                'model_path': model_path,
                'final_val_loss': history['val_losses'][-1] if history['val_losses'] else float('inf'),
                'final_val_perplexity': history['val_perplexities'][-1] if history['val_perplexities'] else float('inf')
            }
            
            print(f" {config_name} training completed!")
            print(f"Final validation perplexity: {merge_results[config_name]['final_val_perplexity']:.2f}")
            print(f"Model saved to: {model_path}")
            if 'conservative' in generation_samples:
                print(f"Sample: {generation_samples['conservative']}")
        
        results[merge_count] = {
            'vocab_size': vocab_size,
            'gpt_results': merge_results
        }
    
    # Save complete results
    save_results(results, 'task4_gpt_results.pkl')
    
    # Print final summary
    print(f"\n{'='*60}")
    print(" GPT TRAINING COMPLETED!")
    print(f"{'='*60}")
    
    print("Models trained:")
    for merge_count in MERGE_COUNTS:
        if merge_count in results:
            print(f"\n BPE Merges: {merge_count}")
            gpt_results = results[merge_count]['gpt_results']
            
            for config_name, config_result in gpt_results.items():
                val_perplexity = config_result.get('final_val_perplexity', float('inf'))
                model_path = config_result.get('model_path', 'N/A')
                print(f"  {config_name:12}: PPL = {val_perplexity:7.2f} | {model_path}")
    
    print(f"\n🔧 How to use your trained GPT models:")
    print(f"python generate_text.py --model task4_gpt_2000 --context 'to be or not to be'")
    print(f"\n💡 GPT models use transformer architecture with:")
    print(f"- Multi-head self-attention (looks at ALL previous tokens)")
    print(f"- Multiple transformer layers for complex pattern learning")
    print(f"- Position embeddings for sequence understanding")
    print(f"- Advanced generation with top-k and top-p sampling")

if __name__ == "__main__":
    main()