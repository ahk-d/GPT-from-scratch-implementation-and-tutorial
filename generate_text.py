#!/usr/bin/env python3
"""
Simple Text Generation Script
Usage: 
  python generate_text.py --task 2 --context "text"
  python generate_text.py --task 3 --context "text"  
  python generate_text.py --task 4 --context "text"
  python generate_text.py --all --context "text"
  python generate_text.py --list
"""

import argparse
import pickle
import torch
import numpy as np
import glob
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
    gpt_files = glob.glob("gpt_model_*.pt")
    if gpt_files:
        print(f"GPT models: {len(gpt_files)} files")
        for f in sorted(gpt_files):
            print(f"   {f}")
    else:
        print("GPT models: None found")
    
    print("=" * 50)

def generate_text_ngram(model, bpe, context, max_tokens=30, temperature=0.8):
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

def generate_text_neural_bigram(model, bpe, context, max_tokens=30, temperature=0.8):
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

def generate_text_gpt(model, bpe, context, max_tokens=30, temperature=0.8):
    """Generate text using GPT model"""
    model.eval()
    
    context_tokens = bpe.encode(context, norm='lower_nopunct')
    context_ids = [bpe.token2id.get(token, 0) for token in context_tokens]
    
    if not context_ids:
        return context
    
    try:
        generated_ids = model.generate(
            context_ids, max_tokens, temperature=temperature
        )
        generated_tokens = [bpe.id2token.get(id, "<UNK>") for id in generated_ids]
        return bpe.decode(generated_tokens)
    except Exception as e:
        print(f"Error in GPT generation: {e}")
        return context

def main():
    parser = argparse.ArgumentParser(description='Simple text generation')
    parser.add_argument('--task', type=int, help='Task: 2 (N-gram), 3 (Neural), 4 (GPT)')
    parser.add_argument('--all', action='store_true', help='Generate from all available trained models')
    parser.add_argument('--context', default="to be or not to be", help='Context text')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--list', action='store_true', help='List all available model files')
    
    args = parser.parse_args()
    
    if args.list:
        show_available_models()
        return
    
    if not args.task and not args.all:
        print("ERROR: Must specify --task (2, 3, or 4) or --all")
        print("Examples:")
        print("  python generate_text.py --task 2 --context 'hello world'")
        print("  python generate_text.py --task 3 --context 'hello world'")
        print("  python generate_text.py --task 4 --context 'hello world'")
        print("  python generate_text.py --all --context 'hello world'")
        print("  python generate_text.py --list")
        return
    
    if args.all:
        print(f"Generating from ALL available trained models...")
        print(f"Context: '{args.context}'")
        print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
        print("=" * 60)
    else:
        print(f"Task {args.task}: Generating text from '{args.context}'")
        print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
        print("-" * 50)
    
    try:
        if args.all:
            # Generate from all available trained models
            # Task 2: N-gram models
            all_ngram_files = glob.glob("task2_*.pkl")
            ngram_files = [f for f in all_ngram_files if not f.endswith('_results.pkl') and not f.endswith('_final.pkl')]
            if ngram_files:
                print(f"\n--- TASK 2: N-gram Models ({len(ngram_files)} found) ---")
                for model_file in sorted(ngram_files):
                    print(f"\nLoading: {model_file}")
                    try:
                        # Extract merge count from filename
                        merge_count = 2000  # default
                        if '_' in model_file:
                            parts = model_file.split('_')
                            if len(parts) >= 2 and parts[1].isdigit():
                                merge_count = int(parts[1])
                        
                        print(f"Extracted: merge_count={merge_count}")
                        
                        # Load BPE with the correct merge count
                        bpe = load_cached_bpe(merge_count, "lower_nopunct")
                        if not bpe:
                            print(f"ERROR: No BPE model found for {merge_count} merges. Skipping...")
                            continue
                        
                        # Load model
                        from task2 import NGramLanguageModel
                        with open(model_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        n_order = int(model_file.split('_')[-1].replace('.pkl', ''))
                        model = NGramLanguageModel(n_order, 1.0)
                        model.ngram_counts = data['ngram_counts']
                        model.context_counts = data['context_counts']
                        model.interpolation_weights = data['interpolation_weights']
                        model.vocab_size = data['vocab_size']
                        
                        generated = generate_text_ngram(model, bpe, args.context, args.max_tokens, args.temperature)
                        print(f"Generated: {generated}")
                        
                    except Exception as e:
                        print(f"ERROR loading {model_file}: {e}")
                        continue
            else:
                print("\n--- TASK 2: No N-gram models found ---")
            
            # Task 3: Neural Bigram models
            neural_files = glob.glob("task3_*_final.pt")
            if neural_files:
                print(f"\n--- TASK 3: Neural Bigram Models ({len(neural_files)} found) ---")
                for model_file in sorted(neural_files):
                    print(f"\nLoading: {model_file}")
                    try:
                        # Parse merge_count and emb_dim from filename
                        import re
                        m = re.search(r"task3_(\d+)_emb_dim=(\d+)_", model_file)
                        if not m:
                            print(f"ERROR: Could not parse merge_count/emb_dim from {model_file}. Skipping...")
                            continue
                        
                        merge_count = int(m.group(1))
                        emb_dim = int(m.group(2))
                        
                        # Load the matching BPE cache
                        bpe = load_cached_bpe(merge_count, "lower_nopunct")
                        if not bpe:
                            print(f"ERROR: No BPE cache for {merge_count} merges. Skipping...")
                            continue
                        
                        # Build model with matching vocab size and embedding dim
                        from task3 import NeuralBigramModel
                        vocab_size = len(bpe.vocab)
                        model = NeuralBigramModel(vocab_size, emb_dim)
                        
                        # Load checkpoint safely
                        checkpoint = torch.load(model_file, map_location='cpu')
                        model.load_state_dict(checkpoint)
                        model.eval()
                        
                        generated = generate_text_neural_bigram(model, bpe, args.context, args.max_tokens, args.temperature)
                        print(f"Generated: {generated}")
                        
                    except Exception as e:
                        print(f"ERROR loading {model_file}: {e}")
                        continue
            else:
                print("\n--- TASK 3: No Neural Bigram models found ---")
            
            # Task 4: GPT models
            gpt_files = glob.glob("gpt_model_*.pt")
            if gpt_files:
                print(f"\n--- TASK 4: GPT Models ({len(gpt_files)} found) ---")
                for model_file in sorted(gpt_files):
                    print(f"\nLoading: {model_file}")
                    try:
                        # Load checkpoint first to get model configuration
                        checkpoint = torch.load(model_file, map_location='cpu')
                        
                        # Extract configuration from checkpoint
                        config = checkpoint.get('model_config', checkpoint.get('config'))
                        vocab_size = checkpoint.get('vocab_size')
                        
                        if not config or not vocab_size:
                            print(f"ERROR: Incomplete checkpoint data in {model_file}. Skipping...")
                            continue
                        
                        # Extract merge count from filename
                        merge_count = 2000  # default
                        if 'merge' in model_file:
                            parts = model_file.split('merge')
                            if len(parts) >= 2 and parts[1].split('_')[0].isdigit():
                                merge_count = int(parts[1].split('_')[0])
                        
                        print(f"Extracted: merge_count={merge_count}")
                        
                        # Load BPE with the correct merge count
                        bpe = load_cached_bpe(merge_count, "lower_nopunct")
                        if not bpe:
                            print(f"ERROR: No BPE model found for {merge_count} merges. Skipping...")
                            continue
                        
                        # Create model with checkpoint configuration
                        from task4 import GPTModel
                        model = GPTModel(
                            vocab_size=vocab_size,  # Use checkpoint vocab size
                            n_embd=config['n_embd'],
                            n_head=config['n_head'],
                            n_layer=config['n_layer'],
                            chunk_size=config['chunk_size']
                        )
                        
                        # Try to load the checkpoint
                        try:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            print(f"Successfully loaded GPT model with vocab_size={vocab_size}")
                        except RuntimeError as e:
                            print(f"ERROR: Checkpoint size mismatch: {e}")
                            continue
                        
                        model.eval()
                        
                        generated = generate_text_gpt(model, bpe, args.context, args.max_tokens, args.temperature)
                        print(f"Generated: {generated}")
                        
                    except Exception as e:
                        print(f"ERROR loading {model_file}: {e}")
                        continue
            else:
                print("\n--- TASK 4: No GPT models found ---")
            
            print("\n" + "=" * 60)
            print("Completed generation from all available models!")
            
        elif args.task == 2:
            # Find ALL n-gram model files (filter out results files)
            all_files = glob.glob("task2_*.pkl")
            model_files = [f for f in all_files if not f.endswith('_results.pkl') and not f.endswith('_final.pkl')]
            
            if not model_files:
                print("ERROR: No task2 model files found")
                return
            
            print(f"Found {len(model_files)} Task 2 models. Generating from each one...")
            print("=" * 60)
            
            for i, model_file in enumerate(sorted(model_files), 1):
                print(f"\n--- Model {i}/{len(model_files)}: {model_file} ---")
                
                try:
                    # Extract merge count from filename (e.g., task2_1000_...)
                    merge_count = 2000  # default
                    if '_' in model_file:
                        parts = model_file.split('_')
                        if len(parts) >= 2 and parts[1].isdigit():
                            merge_count = int(parts[1])
                    
                    print(f"Extracted: merge_count={merge_count}")
                    
                    # Load BPE with the correct merge count
                    bpe = load_cached_bpe(merge_count, "lower_nopunct")
                    if not bpe:
                        print(f"ERROR: No BPE model found for {merge_count} merges. Skipping...")
                        continue
                    
                    # Load model
                    from task2 import NGramLanguageModel
                    with open(model_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    n_order = int(model_file.split('_')[-1].replace('.pkl', ''))
                    model = NGramLanguageModel(n_order, 1.0)
                    model.ngram_counts = data['ngram_counts']
                    model.context_counts = data['context_counts']
                    model.interpolation_weights = data['interpolation_weights']
                    model.vocab_size = data['vocab_size']
                    
                    generated = generate_text_ngram(model, bpe, args.context, args.max_tokens, args.temperature)
                    print(f"Generated: {generated}")
                    
                except Exception as e:
                    print(f"ERROR loading {model_file}: {e}")
                    continue
            
            print(f"\n" + "=" * 60)
            print(f"Completed generation from {len(model_files)} Task 2 models!")
            
        elif args.task == 3:
            # Find ALL neural bigram model files (only final models)
            model_files = glob.glob("task3_*_final.pt")
            if not model_files:
                print("ERROR: No task3 model files found")
                return

            print(f"Found {len(model_files)} Task 3 models. Generating from each one...")
            print("=" * 60)
            
            for i, model_file in enumerate(sorted(model_files), 1):
                print(f"\n--- Model {i}/{len(model_files)}: {model_file} ---")
                
                try:
                    # Parse merge_count and emb_dim from filename
                    import re
                    m = re.search(r"task3_(\d+)_emb_dim=(\d+)_", model_file)
                    if not m:
                        print(f"ERROR: Could not parse merge_count/emb_dim from {model_file}. Skipping...")
                        continue
                    
                    merge_count = int(m.group(1))
                    emb_dim = int(m.group(2))
                    
                    print(f"Extracted: merge_count={merge_count}, emb_dim={emb_dim}")

                    # Load the matching BPE cache
                    bpe = load_cached_bpe(merge_count, "lower_nopunct")
                    if not bpe:
                        print(f"ERROR: No BPE cache for {merge_count} merges. Skipping...")
                        continue

                    # Build model with matching vocab size and embedding dim
                    from task3 import NeuralBigramModel
                    vocab_size = len(bpe.vocab)
                    model = NeuralBigramModel(vocab_size, emb_dim)

                    # Load checkpoint safely
                    checkpoint = torch.load(model_file, map_location='cpu')
                    model.load_state_dict(checkpoint)
                    model.eval()

                    generated = generate_text_neural_bigram(model, bpe, args.context, args.max_tokens, args.temperature)
                    print(f"Generated: {generated}")
                    
                except Exception as e:
                    print(f"ERROR loading {model_file}: {e}")
                    continue
            
            print(f"\n" + "=" * 60)
            print(f"Completed generation from {len(model_files)} Task 3 models!")
            
        elif args.task == 4:
            # Find ALL GPT model files
            model_files = glob.glob("gpt_model_*.pt")
            if not model_files:
                print("ERROR: No task4 model files found")
                return
            
            print(f"Found {len(model_files)} Task 4 models. Generating from each one...")
            print("=" * 60)
            
            for i, model_file in enumerate(sorted(model_files), 1):
                print(f"\n--- Model {i}/{len(model_files)}: {model_file} ---")
                
                try:
                    # Load checkpoint first to get model configuration
                    checkpoint = torch.load(model_file, map_location='cpu')
                    
                    # Extract configuration from checkpoint
                    config = checkpoint.get('model_config', checkpoint.get('config'))
                    vocab_size = checkpoint.get('vocab_size')
                    
                    if not config or not vocab_size:
                        print(f"ERROR: Incomplete checkpoint data in {model_file}. Skipping...")
                        continue
                    
                    # Extract merge count from filename (e.g., gpt_model_merge1000_...)
                    merge_count = 2000  # default
                    if 'merge' in model_file:
                        parts = model_file.split('merge')
                        if len(parts) >= 2 and parts[1].split('_')[0].isdigit():
                            merge_count = int(parts[1].split('_')[0])
                    
                    print(f"Extracted: merge_count={merge_count}")
                    
                    # Load BPE with the correct merge count
                    bpe = load_cached_bpe(merge_count, "lower_nopunct")
                    if not bpe:
                        print(f"ERROR: No BPE model found for {merge_count} merges. Skipping...")
                        continue
                    
                    # Create model with checkpoint configuration
                    from task4 import GPTModel
                    model = GPTModel(
                        vocab_size=vocab_size,  # Use checkpoint vocab size
                        n_embd=config['n_embd'],
                        n_head=config['n_head'],
                        n_layer=config['n_layer'],
                        chunk_size=config['chunk_size']
                    )
                    
                    # Try to load the checkpoint
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"Successfully loaded GPT model with vocab_size={vocab_size}")
                    except RuntimeError as e:
                        print(f"ERROR: Checkpoint size mismatch: {e}")
                        continue
                    
                    model.eval()
                    
                    generated = generate_text_gpt(model, bpe, args.context, args.max_tokens, args.temperature)
                    print(f"Generated: {generated}")
                    
                except Exception as e:
                    print(f"ERROR loading {model_file}: {e}")
                    continue
            
            print(f"\n" + "=" * 60)
            print(f"Completed generation from {len(model_files)} Task 4 models!")
            
        else:
            print(f"ERROR: Task {args.task} not supported")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()