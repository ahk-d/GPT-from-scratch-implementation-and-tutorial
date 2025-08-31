#!/usr/bin/env python3
"""
Simple Text Generation from Existing Models
Usage: python generate_from_models.py --model MODEL --context "text" [options]
"""

import argparse
import pickle
import torch
import numpy as np
import os
from utils import load_cached_bpe
from task3 import NeuralBigramModel

def load_neural_bigram_model(model_path, merge_count):
    """Load a neural bigram model from checkpoint."""
    # Load BPE
    bpe = load_cached_bpe(merge_count, "lower_nopunct")
    if bpe is None:
        raise ValueError(f"BPE model not found for merge count {merge_count}")
    
    # Determine embedding dimension from model path
    if "emb_dim=64" in model_path:
        emb_dim = 64
    elif "emb_dim=128" in model_path:
        emb_dim = 128
    else:
        emb_dim = 64  # default
    
    # Create model
    vocab_size = len(bpe.vocab)
    model = NeuralBigramModel(vocab_size, emb_dim)
    
    # Load checkpoint
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, bpe

def generate_text(model, bpe, context, max_tokens=50, temperature=0.8):
    """Generate text using the neural bigram model."""
    # Encode context
    context_tokens = bpe.encode(context)
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    
    # Generate text
    with torch.no_grad():
        generated_ids = context_ids.copy()
        for _ in range(max_tokens):
            # Neural model
            last_id = torch.tensor([generated_ids[-1]], dtype=torch.long)
            logits = model(last_id).squeeze(0)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
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
    """List all available neural bigram models."""
    models = []
    for file in os.listdir('.'):
        if file.startswith('neural_bigram_') and file.endswith('.pt'):
            # Parse model info
            parts = file.replace('.pt', '').split('_')
            merge_count = parts[2]
            emb_dim = parts[4].split('=')[1]
            batch_size = parts[5].split('=')[1]
            lr = parts[6].split('=')[1]
            wd = parts[7].split('=')[1]
            is_final = 'final' in file
            
            models.append({
                'file': file,
                'merge_count': merge_count,
                'emb_dim': emb_dim,
                'batch_size': batch_size,
                'lr': lr,
                'wd': wd,
                'is_final': is_final
            })
    
    return models

def main():
    parser = argparse.ArgumentParser(description='Generate text from existing neural bigram models')
    parser.add_argument('--model', help='Model file path (e.g., neural_bigram_1000_emb_dim=64_batch=32_lr=0.001_wd=0.0001_final.pt)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--context', default="to be or not to be", help='Context text')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available neural bigram models:")
        print("-" * 50)
        models = list_available_models()
        for i, model in enumerate(models, 1):
            final_str = " (FINAL)" if model['is_final'] else ""
            print(f"{i}. {model['file']}{final_str}")
            print(f"   Merge count: {model['merge_count']}, Embedding dim: {model['emb_dim']}")
            print(f"   Batch size: {model['batch_size']}, LR: {model['lr']}, WD: {model['wd']}")
            print()
        return
    
    if not args.model:
        print("Error: Please specify a model with --model or use --list to see available models")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Parse merge count from model filename
    parts = args.model.split('_')
    merge_count = int(parts[2])
    
    print(f"Generating with model: {args.model}")
    print(f"Context: '{args.context}'")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print("-" * 50)
    
    try:
        model, bpe = load_neural_bigram_model(args.model, merge_count)
        generated = generate_text(model, bpe, args.context, args.max_tokens, args.temperature)
        print(f"Generated: {generated}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
