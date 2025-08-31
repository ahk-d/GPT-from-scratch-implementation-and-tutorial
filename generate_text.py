#!/usr/bin/env python3
"""
Simple Text Generation Script
Usage: python generate_text.py --model MODEL --context "text" [options]
"""

import argparse
import pickle
import torch
import numpy as np
import os
from utils import load_cached_bpe
from task2 import NGramLanguageModel
from task3 import NeuralBigramModel

def load_model(task_num, merge_count):
    """Load the best model for a given task and merge count."""
    # Load results
    results_file = f"task{task_num}_fixed_results.pkl" if task_num == 3 else f"task{task_num}_results.pkl"
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    if merge_count not in results:
        raise ValueError(f"No results for merge count {merge_count}")
    
    # Load BPE
    bpe = load_cached_bpe(merge_count, "lower_nopunct")
    
    if task_num == 2:
        # N-gram model
        ngram_results = results[merge_count]['ngram_results']
        best_key = min(ngram_results.items(), key=lambda x: x[1].get('val_perplexity', float('inf')))[0]
        best_n = int(best_key.split('=')[1]) if '=' in best_key else 1
        
        # Load the actual trained model
        model_path = ngram_results[best_key].get('model_path')
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Fallback: create new model
            model = NGramLanguageModel(best_n)
        
        return model, bpe, best_n
    
    elif task_num == 3:
        # Neural bigram model
        data = results[merge_count]
        best_config = data['best_config']
        best_info = data['hyperparameter_results'][best_config]
        
        vocab_size = len(bpe.vocab)
        emb_dim = best_info['embedding_dim']
        model = NeuralBigramModel(vocab_size, emb_dim)
        
        # Load checkpoint
        checkpoint_path = best_info.get('final_checkpoint_path') or best_info.get('checkpoint_path')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, bpe, best_config
    
    else:
        raise ValueError(f"Unsupported task: {task_num}")

def generate_text(model, bpe, context, max_tokens=50, temperature=0.8):
    """Generate text using the specified model."""
    # Encode context
    context_tokens = bpe.encode(context)
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    

    
    # Generate text
    with torch.no_grad():
        generated_ids = context_ids.copy()
        for _ in range(max_tokens):
            if hasattr(model, 'forward'):
                # Neural model
                last_id = torch.tensor([generated_ids[-1]], dtype=torch.long)
                logits = model(last_id).squeeze(0)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                # N-gram model - use actual n-gram probabilities
                n = getattr(model, 'n', 1)
                if len(generated_ids) >= n:
                    # Get context for n-gram
                    context_ids_ngram = generated_ids[-(n-1):] if n > 1 else []
                    context_tokens_ngram = [bpe.id2token.get(id, "<UNK>") for id in context_ids_ngram]
                    
                    # Calculate probabilities for all possible next tokens
                    vocab_size = len(bpe.vocab)
                    probs = np.zeros(vocab_size)
                    
                    for i in range(vocab_size):
                        token = bpe.id2token.get(i, "<UNK>")
                        try:
                            if hasattr(model, '_calculate_order_probability'):
                                prob = model._calculate_order_probability(token, tuple(context_tokens_ngram))
                            else:
                                # Fallback to unigram if method not available
                                prob = 1.0 / vocab_size
                        except:
                            prob = 1.0 / vocab_size
                        probs[i] = max(prob, 0.0)
                    
                    # Normalize and sample
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs = np.ones(vocab_size) / vocab_size
                    
                    next_id = np.random.choice(vocab_size, p=probs)
                else:
                    # Not enough context, use unigram
                    next_id = np.random.randint(0, len(bpe.vocab))
            
            generated_ids.append(next_id)
        
        # Decode
        generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
        
        # Better decoding: handle subword tokens properly
        decoded_text = ""
        current_word = ""
        
        for token in generated_tokens:
            if token == '__':  # End of word
                if current_word:
                    decoded_text += current_word + " "
                    current_word = ""
            else:
                current_word += token
        
        # Add the last word if any
        if current_word:
            decoded_text += current_word
        
        # Clean up
        decoded_text = decoded_text.strip()
        
        return decoded_text

def main():
    parser = argparse.ArgumentParser(description='Simple text generation')
    parser.add_argument('--model', required=True, help='Model: task2_ngram_2000, task3_neural_bigram_2000, etc.')
    parser.add_argument('--context', default="to be or not to be", help='Context text')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Parse model spec
    parts = args.model.split('_')
    task_num = int(parts[0].replace('task', ''))
    merge_count = int(parts[-1])  # Last part is merge count
    
    print(f"Generating with {args.model}")
    print(f"Context: '{args.context}'")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print("-" * 50)
    
    try:
        model, bpe, config = load_model(task_num, merge_count)
        generated = generate_text(model, bpe, args.context, args.max_tokens, args.temperature)
        print(f"Generated: {generated}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
