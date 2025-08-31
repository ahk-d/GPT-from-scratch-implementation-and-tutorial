#!/usr/bin/env python3
"""
Simplified Text Generation Script
Supports: N-gram models, Neural Bigram models, and GPT models
Usage: 
  python generate_text.py --model task2_ngram_2000 --context "text"
  python generate_text.py --model task3_neural_bigram_2000 --context "text"  
  python generate_text.py --model task4_gpt_2000 --context "text"
  python generate_text.py --compare --context "text"
  python generate_text.py --list
"""

import argparse
import pickle
import torch
import numpy as np
import os
import glob
from collections import Counter

from utils import load_cached_bpe

def show_available_models():
    """Show what model files are available"""
    print("AVAILABLE MODEL FILES:")
    print("=" * 50)
    
    # N-gram models
    ngram_files = glob.glob("task2_*.pkl")
    if ngram_files:
        print(f"N-gram models: {len(ngram_files)} files")
        for f in sorted(ngram_files):
            print(f"   {f}")
    else:
        print("N-gram models: None found")
    
    # Neural Bigram models
    neural_files = glob.glob("task3_*_final.pt")
    if neural_files:
        print(f"Neural Bigram models: {len(neural_files)} files")
        for f in sorted(neural_files):
            print(f"   {f}")
    else:
        print("Neural Bigram models: None found")
    
    # GPT models
    gpt_files = glob.glob("gpt_model_merge*.pt")
    if gpt_files:
        print(f"GPT models: {len(gpt_files)} files")
        for f in sorted(gpt_files):
            print(f"   {f}")
    else:
        print("GPT models: None found")
    
    print("=" * 50)

def import_models():
    """Import model classes"""
    try:
        from task2 import NGramLanguageModel
        print("Successfully imported NGramLanguageModel from task2")
    except ImportError as e:
        print(f"Import error for task2: {e}")
        NGramLanguageModel = None
    
    try:
        from task3 import NeuralBigramModel  
        print("Successfully imported NeuralBigramModel from task3")
    except ImportError as e:
        print(f"Import error for task3: {e}")
        NeuralBigramModel = None
    
    try:
        from task4 import GPTModel
        print("Successfully imported GPTModel from task4")
    except ImportError as e:
        print(f"Import error for task4: {e}")
        GPTModel = None
    
    return NGramLanguageModel, NeuralBigramModel, GPTModel

def load_ngram_model(merge_count):
    """Load N-gram model from saved data"""
    try:
        NGramLanguageModel, _, _ = import_models()
        if NGramLanguageModel is None:
            return None, None, None
        
        # Find model files
        pattern = f"task2_{merge_count}_*.pkl"
        model_files = glob.glob(pattern)
        
        if not model_files:
            print(f"No n-gram model files found for {merge_count} merges")
            return None, None, None
        
        print(f"Found {len(model_files)} n-gram model files")
        
        # Load the first available model
        model_file = model_files[0]
        n_value = int(model_file.split('_')[-1].replace('.pkl', ''))
        
        print(f"Loading: {model_file} (n={n_value})")
        
        # Create new model and load data
        vocab_size = 1000 if merge_count == 1000 else 2000
        model = NGramLanguageModel(vocab_size, n_value)
        
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load model attributes
        model.ngram_counts = data['ngram_counts']
        model.context_counts = data['context_counts']
        model.interpolation_weights = np.array(data['interpolation_weights'])
        model.vocab_size = data['vocab_size']
        model.alpha = data['alpha']
        model.unk_strategy = data['unk_strategy']
        
        print(f"Successfully loaded n-gram model with n={n_value}")
        return model, n_value, None
        
    except Exception as e:
        print(f"Could not load N-gram model: {e}")
        return None, None, None

def load_neural_bigram_model(merge_count):
    """Load Neural Bigram model from saved checkpoint"""
    try:
        _, NeuralBigramModel, _ = import_models()
        if NeuralBigramModel is None:
            return None, None, None
        
        # Find final model files
        pattern = f"task3_{merge_count}_emb_dim=*_final.pt"
        model_files = glob.glob(pattern)
        
        if not model_files:
            print(f"No neural bigram model files found for {merge_count} merges")
            return None, None, None
        
        model_file = model_files[0]
        print(f"Loading: {model_file}")
        
        # Extract configuration from filename
        parts = model_file.replace('.pt', '').split('_')
        emb_dim = int([p for p in parts if p.startswith('emb_dim=')][0].split('=')[1])
        
        # Load BPE for vocab size
        bpe = load_cached_bpe(merge_count, "lower_nopunct")
        if bpe is None:
            return None, None, None
        
        vocab_size = len(bpe.vocab)
        print(f"Creating neural bigram model: vocab_size={vocab_size}, emb_dim={emb_dim}")
        
        # Create and load model
        model = NeuralBigramModel(vocab_size, emb_dim)
        checkpoint = torch.load(model_file, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        print(f"Successfully loaded neural bigram model")
        return model, f"emb_dim={emb_dim}", None
        
    except Exception as e:
        print(f"Could not load Neural Bigram model: {e}")
        return None, None, None

def load_gpt_model(merge_count):
    """Load GPT model from saved checkpoint"""
    try:
        # Find GPT model files
        pattern = f"gpt_model_merge{merge_count}_gpt_*.pt"
        model_files = glob.glob(pattern)
        
        if not model_files:
            print(f"No GPT model files found for {merge_count} merges")
            return None, None, None
        
        model_file = model_files[0]
        print(f"Loading: {model_file}")
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location='cpu')
        config = checkpoint.get('model_config', checkpoint.get('config'))
        vocab_size = checkpoint.get('vocab_size')
        
        if not config or not vocab_size:
            print("ERROR: Incomplete checkpoint data")
            return None, None, None
        
        # Import and create model
        from task4 import GPTModel
        model = GPTModel(
            vocab_size=vocab_size,
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            chunk_size=config['chunk_size'],
            dropout=config.get('dropout', 0.1)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Successfully loaded GPT model")
        return model, config, None
        
    except Exception as e:
        print(f"Could not load GPT model: {e}")
        return None, None, None

def generate_text_ngram(model, bpe, context, max_tokens=50, temperature=0.8):
    """Generate text using N-gram model"""
    context_tokens = bpe.encode(context, norm='lower_nopunct')
    generated_tokens = context_tokens.copy()
    
    for step in range(max_tokens):
        n = getattr(model, 'n_order', 3)
        if len(generated_tokens) >= n - 1:
            context_ngram = generated_tokens[-(n-1):] if n > 1 else []
            
            vocab_size = len(bpe.vocab)
            probs = np.zeros(vocab_size)
            
            for i, token in enumerate(bpe.vocab):
                try:
                    if hasattr(model, '_calculate_order_probability'):
                        prob = model._calculate_order_probability(token, tuple(context_ngram))
                    else:
                        prob = 1.0 / vocab_size
                except:
                    prob = 1.0 / vocab_size
                probs[i] = max(prob, 1e-10)
            
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(vocab_size) / vocab_size
            
            next_token_idx = np.random.choice(vocab_size, p=probs)
            next_token = bpe.vocab[next_token_idx]
            generated_tokens.append(next_token)
        else:
            next_token = np.random.choice(bpe.vocab)
            generated_tokens.append(next_token)
    
    return bpe.decode(generated_tokens)

def generate_text_neural_bigram(model, bpe, context, max_tokens=50, temperature=0.8):
    """Generate text using Neural Bigram model"""
    model.eval()
    
    context_tokens = bpe.encode(context, norm='lower_nopunct')
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    
    if not context_ids:
        return context
    
    generated_ids = context_ids.copy()
    
    with torch.no_grad():
        for step in range(max_tokens):
            current_token_id = generated_ids[-1]
            input_tensor = torch.tensor([current_token_id], dtype=torch.long)
            
            try:
                logits = model(input_tensor)
                
                if len(logits.shape) == 3:
                    logits = logits[0, -1, :]
                elif len(logits.shape) == 2:
                    logits = logits[0, :]
                
                if temperature > 0:
                    logits = logits / temperature
                
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                
                if next_token_id >= len(bpe.vocab):
                    break
                
                generated_ids.append(next_token_id)
                
            except Exception as e:
                print(f"Error in neural generation: {e}")
                break
    
    generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
    return bpe.decode(generated_tokens)

def generate_text_gpt(model, bpe, context, max_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
    """Generate text using GPT model"""
    model.eval()
    
    context_tokens = bpe.encode(context, norm='lower_nopunct')
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    
    if not context_ids:
        return context
    
    try:
        generated_ids = model.generate(
            context_ids, max_tokens, temperature=temperature, 
            top_k=top_k, top_p=top_p
        )
        generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
        return bpe.decode(generated_tokens)
    except Exception as e:
        print(f"Error in GPT generation: {e}")
        return context

def main():
    parser = argparse.ArgumentParser(description='Text generation with trained models')
    parser.add_argument('--model', help='Model: task2_ngram_2000, task3_neural_bigram_2000, task4_gpt_2000')
    parser.add_argument('--context', default="to be or not to be", help='Context text')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (GPT only)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling (GPT only)')
    parser.add_argument('--list', action='store_true', help='List all available model files')
    
    args = parser.parse_args()
    
    if args.list:
        show_available_models()
        return
    
    if not args.model:
        print("ERROR: Must specify --model")
        print("Examples:")
        print("  python generate_text.py --model task2_ngram_2000 --context 'hello world'")
        print("  python generate_text.py --model task3_neural_bigram_2000 --context 'hello world'")
        print("  python generate_text.py --model task4_gpt_2000 --context 'hello world'")
        print("  python generate_text.py --list")
        return
    
    # Parse model specification
    parts = args.model.split('_')
    if len(parts) < 3:
        print("ERROR: Invalid model format. Expected: taskN_modeltype_mergecount")
        return
    
    task_num = int(parts[0].replace('task', ''))
    merge_count = int(parts[-1])
    
    print(f"Generating with {args.model}")
    print(f"Context: '{args.context}'")
    print(f"Settings: max_tokens={args.max_tokens}, temperature={args.temperature}")
    if task_num == 4:
        print(f"GPT settings: top_k={args.top_k}, top_p={args.top_p}")
    print("-" * 50)
    
    # Load BPE
    bpe = load_cached_bpe(merge_count, "lower_nopunct")
    if bpe is None:
        print(f"ERROR: No BPE model found for {merge_count} merges. Run task1.py first.")
        return
    
    try:
        if task_num == 2:
            model, config, perplexity = load_ngram_model(merge_count)
            if model:
                generated = generate_text_ngram(model, bpe, args.context, args.max_tokens, args.temperature)
                print(f"Generated: {generated}")
            else:
                print("ERROR: Could not load N-gram model")
        
        elif task_num == 3:
            model, config, perplexity = load_neural_bigram_model(merge_count)
            if model:
                generated = generate_text_neural_bigram(model, bpe, args.context, args.max_tokens, args.temperature)
                print(f"Generated: {generated}")
                print(f"Model config: {config}")
            else:
                print("ERROR: Could not load Neural Bigram model")
        
        elif task_num == 4:
            model, config, perplexity = load_gpt_model(merge_count)
            if model:
                generated = generate_text_gpt(
                    model, bpe, args.context, args.max_tokens, 
                    args.temperature, args.top_k, args.top_p
                )
                print(f"Generated: {generated}")
                if config:
                    print(f"Model config: {config['n_layer']} layers, {config['n_embd']} dim, {config['n_head']} heads")
            else:
                print("ERROR: Could not load GPT model")
        
        else:
            print(f"ERROR: Unsupported task number: {task_num}")
    
    except Exception as e:
        print(f"ERROR: Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()