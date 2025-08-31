#!/usr/bin/env python3
"""
Direct Text Generation from Model Files
Usage: python generate_direct.py --model MODEL_FILE --context "text" [options]
"""

import argparse
import pickle
import torch
import numpy as np
import os
from utils import load_cached_bpe
from task2 import NGramLanguageModel
from task3 import NeuralBigramModel

def load_model_direct(model_path):
    """Load model directly from file path."""
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    # Determine model type and parameters from filename
    filename = os.path.basename(model_path)
    parts = filename.replace('.pt', '').replace('.pkl', '').split('_')
    
    if filename.startswith('neural_bigram_'):
        # Neural bigram model
        merge_count = int(parts[2])
        emb_dim = int(parts[4].split('=')[1])
        
        # Load BPE
        bpe = load_cached_bpe(merge_count, "lower_nopunct")
        if bpe is None:
            raise ValueError(f"BPE model not found for merge count {merge_count}")
        
        # Create and load model
        vocab_size = len(bpe.vocab)
        model = NeuralBigramModel(vocab_size, emb_dim)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, bpe, "neural_bigram"
        
    elif filename.startswith('ngram_model_'):
        # N-gram model
        merge_count = int(parts[2])
        n = int(parts[3])
        
        # Load BPE
        bpe = load_cached_bpe(merge_count, "lower_nopunct")
        if bpe is None:
            raise ValueError(f"BPE model not found for merge count {merge_count}")
        
        # Load n-gram model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, bpe, "ngram"
        
    else:
        raise ValueError(f"Unknown model type: {filename}")

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

def list_available_models():
    """List all available model files."""
    models = []
    for file in os.listdir('.'):
        if file.endswith('.pt') or file.endswith('.pkl'):
            if file.startswith('neural_bigram_') or file.startswith('ngram_model_'):
                models.append(file)
    return sorted(models)

def main():
    parser = argparse.ArgumentParser(description='Direct text generation from model files')
    parser.add_argument('--model', help='Model file path')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--context', default="to be or not to be", help='Context text')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available model files:")
        print("-" * 50)
        models = list_available_models()
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        return
    
    if not args.model:
        print("Error: Please specify a model with --model or use --list to see available models")
        return
    
    print(f"Generating with model: {args.model}")
    print(f"Context: '{args.context}'")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print("-" * 50)
    
    try:
        model, bpe, model_type = load_model_direct(args.model)
        print(f"Loaded {model_type} model successfully")
        generated = generate_text(model, bpe, args.context, args.max_tokens, args.temperature)
        print(f"Generated: {generated}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
