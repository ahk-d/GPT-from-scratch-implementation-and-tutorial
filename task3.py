# Task 3: Neural Bigram Language Modeling (Compact Version)
# - Uses cached BPE models from Task 1
# - Trains neural bigram models with different hyperparameters
# - Evaluates perplexity on valid/test
# - Generates sample text directly
# - Saves models + results

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_and_slice_data, load_cached_bpe, save_results, GEN_CONTEXT

# Config
PERCENTAGE = 0.1
BEST_NORMALIZATION = "lower_nopunct"
MERGE_COUNTS = [1000, 2000, 3000]  # Only test one merge count
EMBEDDING_DIMS = [64]   # Only test one embedding dim
BATCH_SIZES = [32]
LEARNING_RATES = [5e-4, 1e-4, 5e-5] # Only test one learning rate
MAX_ITERATIONS = 2000   # Reduced iterations
EARLY_STOPPING_PATIENCE = 500
WEIGHT_DECAY_VALUES = [1e-5]
VALIDATION_INTERVAL = 100

# Generation config
GEN_MAX_TOKENS = 20
GEN_TEMPERATURE = 0.7

class NeuralBigramModel(nn.Module):
    """Neural bigram model with embedding + linear projection"""
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Previous token embedding
        self.prev_token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Proper weight initialization"""
        nn.init.normal_(self.prev_token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, prev_tokens):
        """Forward pass"""
        embeddings = self.prev_token_embedding(prev_tokens)
        logits = self.output_projection(embeddings)
        return logits
    
    def calculate_loss(self, prev_tokens, next_tokens):
        """Calculate cross entropy loss"""
        logits = self.forward(prev_tokens)
        loss = nn.functional.cross_entropy(logits, next_tokens)
        return loss

def prepare_data(token_stream, batch_size, device):
    """Prepare bigram data batches"""
    # Create bigram pairs
    bigram_pairs = [(token_stream[i], token_stream[i + 1]) 
                   for i in range(len(token_stream) - 1)]
    
    # Shuffle for better training
    np.random.shuffle(bigram_pairs)
    
    # Create batches
    batches = []
    for i in range(0, len(bigram_pairs) - batch_size + 1, batch_size):
        batch_pairs = bigram_pairs[i:i + batch_size]
        
        prev_tokens = torch.tensor([pair[0] for pair in batch_pairs], 
                                 dtype=torch.long, device=device)
        next_tokens = torch.tensor([pair[1] for pair in batch_pairs], 
                                 dtype=torch.long, device=device)
        
        batches.append((prev_tokens, next_tokens))
    
    return batches

def train_model(model, train_batches, valid_batches, optimizer, 
               max_iterations, patience, device, validation_interval=100):
    """Train model with early stopping"""
    model.train()
    
    history = {'losses': [], 'val_losses': [], 'perplexities': [], 'val_perplexities': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for iteration in range(max_iterations):
        # Sample random training batch
        batch_idx = np.random.randint(0, len(train_batches))
        prev_batch, next_batch = train_batches[batch_idx]
        
        # Training step
        optimizer.zero_grad()
        loss = model.calculate_loss(prev_batch, next_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Record training metrics
        history['losses'].append(loss.item())
        history['perplexities'].append(torch.exp(loss).item())
        
        # Validation check
        if iteration % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_prev, val_next in valid_batches[:10]:
                    val_loss += model.calculate_loss(val_prev, val_next).item()
                val_loss /= min(len(valid_batches), 10)
            
            history['val_losses'].append(val_loss)
            history['val_perplexities'].append(np.exp(val_loss))
            
            print(f"Iteration {iteration}: Train Loss = {loss.item():.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Perplexity = {np.exp(val_loss):.2f}")
            
            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += validation_interval
                
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iteration}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
                
            model.train()
        
        # Progress reporting
        elif iteration % 1000 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}, "
                  f"Perplexity = {torch.exp(loss).item():.2f}")
    
    return history

def evaluate_model(model, batches, device):
    """Evaluate model perplexity"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for prev_batch, next_batch in batches:
            loss = model.calculate_loss(prev_batch, next_batch)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = np.exp(avg_loss)
    return perplexity

def generate_text(model, bpe, context, max_tokens=30, temperature=0.7):
    """Generate text using the neural bigram model"""
    model.eval()
    
    # Encode context
    context_tokens = bpe.encode(context)
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    
    if not context_ids:
        context_ids = [0]
    
    generated_ids = context_ids.copy()
    
    with torch.no_grad():
        for step in range(max_tokens):
            current_token_id = generated_ids[-1]
            input_tensor = torch.tensor([current_token_id], dtype=torch.long)
            
            # Get logits
            logits = model(input_tensor)
            if len(logits.shape) == 2:
                logits = logits[0, :]
            
            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            if next_token_id >= len(bpe.vocab):
                break
            
            generated_ids.append(next_token_id)
    
    # Decode generated tokens
    generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
    return bpe.decode(generated_tokens)

def main():
    print("Task 3: Neural Bigram Language Modeling (Ultra Compact)")
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
        
        # Single configuration training
        emb_dim, batch_size, lr, weight_decay = EMBEDDING_DIMS[0], BATCH_SIZES[0], LEARNING_RATES[0], WEIGHT_DECAY_VALUES[0]
        print(f"  Training: emb_dim={emb_dim}, batch={batch_size}, lr={lr}")
        
        # Prepare data
        train_batches = prepare_data(train_ids, batch_size, device)
        valid_batches = prepare_data(valid_ids, batch_size, device)
        test_batches = prepare_data(test_ids, batch_size, device)
        
        # Create and train model
        model = NeuralBigramModel(vocab_size, emb_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train
        history = train_model(model, train_batches, valid_batches, optimizer,
                            MAX_ITERATIONS, EARLY_STOPPING_PATIENCE, device, VALIDATION_INTERVAL)
        
        # Evaluate
        val_perplexity = evaluate_model(model, valid_batches, device)
        test_perplexity = evaluate_model(model, test_batches, device)
        print(f"  val_ppl={val_perplexity:.2f} | test_ppl={test_perplexity:.2f}")
        
        # Generate sample
        generated = generate_text(model, bpe, GEN_CONTEXT, GEN_MAX_TOKENS, GEN_TEMPERATURE)
        print(f"  [Generated]: {generated}")
        
        # Save model
        torch.save(model.state_dict(), f"task3_{merges}_final.pt")
    
    print("\nTask 3 completed!")

if __name__ == "__main__":
    main()
