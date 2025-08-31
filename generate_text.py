#!/usr/bin/env python3
"""
Complete Text Generation Script - Fixed for All Models
Supports: N-gram models, Neural Bigram models, and GPT models
Usage: 
  python generate_text.py --model task2_ngram_2000 --context "text"
  python generate_text.py --model task3_neural_bigram_2000 --context "text"  
  python generate_text.py --model task4_gpt_2000 --context "text"
  python generate_text.py --compare --context "text"
"""

import argparse
import pickle
import torch
import numpy as np
import os
import json
from pathlib import Path

from utils import load_cached_bpe

def import_models():
    """Import models only when needed"""
    try:
        from task2 import NGramLanguageModel
    except ImportError:
        NGramLanguageModel = None
    
    try:
        from task3 import NeuralBigramModel  
    except ImportError:
        NeuralBigramModel = None
    
    try:
        from task4 import GPTModel
    except ImportError:
        GPTModel = None
    
    return NGramLanguageModel, NeuralBigramModel, GPTModel

def load_ngram_model(merge_count):
    """Load N-gram model from task2 results"""
    try:
        with open('task2_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        if merge_count not in results:
            return None, None, None
        
        ngram_results = results[merge_count]['ngram_results']
        best_key = min(ngram_results.items(), key=lambda x: x[1].get('val_perplexity', float('inf')))[0]
        best_n = int(best_key.split('=')[1]) if '=' in best_key else 3
        
        model_path = ngram_results[best_key].get('model_path')
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, best_n, ngram_results[best_key].get('val_perplexity')
        
    except Exception as e:
        print(f"Could not load N-gram model: {e}")
    
    return None, None, None

def load_neural_bigram_model(merge_count):
    """Load Neural Bigram model from task3 results"""
    try:
        with open('task3_fixed_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        if merge_count not in results:
            return None, None, None
        
        data = results[merge_count]
        best_config = data['best_config']
        best_info = data['hyperparameter_results'][best_config]
        
        bpe = load_cached_bpe(merge_count, "lower_nopunct")
        if bpe is None:
            return None, None, None
        
        vocab_size = len(bpe.vocab)
        emb_dim = best_info['embedding_dim']
        
        _, NeuralBigramModel, _ = import_models()
        if NeuralBigramModel is None:
            return None, None, None
        
        model = NeuralBigramModel(vocab_size, emb_dim)
        
        checkpoint_path = best_info.get('final_checkpoint_path') or best_info.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            return model, best_config, best_info.get('val_perplexity')
        
    except Exception as e:
        print(f"Could not load Neural Bigram model: {e}")
    
    return None, None, None

def load_gpt_model(merge_count, config_name=None):
    """Load GPT model from task4 results or direct files"""
    print(f"Loading GPT model with {merge_count} merges...")
    
    # First, try to load from task4 results to get model path
    try:
        with open('task4_gpt_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        if merge_count in results and 'gpt_results' in results[merge_count]:
            gpt_results = results[merge_count]['gpt_results']
            print(f"Available GPT configs: {list(gpt_results.keys())}")
            
            if config_name:
                target_configs = [k for k in gpt_results.keys() if config_name.lower() in k.lower()]
            else:
                target_configs = list(gpt_results.keys())
            
            if target_configs:
                best_config = target_configs[0]
                model_info = gpt_results[best_config]
                model_path = model_info.get('model_path')
                
                if model_path and os.path.exists(model_path):
                    print(f"Found model path in results: {model_path}")
                    
                    checkpoint = torch.load(model_path, map_location='cpu')
                    config = checkpoint.get('model_config', checkpoint.get('config'))
                    vocab_size = checkpoint.get('vocab_size')
                    
                    if not config or not vocab_size:
                        print("ERROR: Incomplete checkpoint data")
                        return None, None, None
                    
                    try:
                        from task4 import GPTModel
                    except ImportError:
                        print("ERROR: Cannot import GPTModel from task4.py")
                        return None, None, None
                    
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
                    
                    perplexity = model_info.get('final_val_perplexity')
                    print(f"GPT model loaded successfully")
                    if perplexity:
                        print(f"Validation perplexity: {perplexity:.2f}")
                    
                    return model, config, perplexity
                else:
                    print(f"Model file not found: {model_path}")
            else:
                print(f"No matching config found for: {config_name}")
    
    except FileNotFoundError:
        print("task4_gpt_results.pkl not found, trying direct file search...")
    except Exception as e:
        print(f"Error reading task4 results: {e}")
    
    # Fallback: Direct file search
    possible_files = [
        f'gpt_model_merge{merge_count}_gpt_small.pt',
        f'gpt_model_merge{merge_count}_gpt_medium.pt',
        f'gpt_model_merge{merge_count}_gpt_large.pt',
        f'gpt_model_{merge_count}_gpt_small.pt',
        f'gpt_model_{merge_count}_gpt_medium.pt',
    ]
    
    if config_name:
        specific_file = f'gpt_model_merge{merge_count}_{config_name.lower().replace("-", "_")}.pt'
        possible_files.insert(0, specific_file)
    
    for model_file in possible_files:
        if os.path.exists(model_file):
            print(f"Found model file: {model_file}")
            
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                config = checkpoint.get('model_config', checkpoint.get('config'))
                vocab_size = checkpoint.get('vocab_size')
                
                if config and vocab_size:
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
                    
                    training_info = checkpoint.get('training_info', {})
                    perplexity = training_info.get('final_val_perplexity')
                    
                    print(f"GPT model loaded from direct file")
                    return model, config, perplexity
                    
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                continue
    
    print("Could not load any GPT model")
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

def compare_all_models(context, max_tokens=30, temperature=0.8, merge_count=2000):
    """Compare generation from all available models"""
    print("MODEL COMPARISON")
    print(f"Context: '{context}'")
    print(f"Settings: max_tokens={max_tokens}, temperature={temperature}, BPE_merges={merge_count}")
    print("=" * 80)
    
    bpe = load_cached_bpe(merge_count, "lower_nopunct")
    if bpe is None:
        print("ERROR: No BPE model found. Run task1.py first.")
        return
    
    results = {}
    
    # Test N-gram model
    print("1. N-GRAM MODEL")
    ngram_model, ngram_n, ngram_perplexity = load_ngram_model(merge_count)
    if ngram_model:
        try:
            generated = generate_text_ngram(ngram_model, bpe, context, max_tokens, temperature)
            results['ngram'] = generated
            print(f"SUCCESS: N-gram (n={ngram_n}): {generated}")
            if ngram_perplexity:
                print(f"         Validation perplexity: {ngram_perplexity:.2f}")
        except Exception as e:
            print(f"FAILED: N-gram generation failed: {e}")
            results['ngram'] = f"Error: {e}"
    else:
        print("FAILED: N-gram model not available")
        results['ngram'] = "Not available"
    
    print()
    
    # Test Neural Bigram model  
    print("2. NEURAL BIGRAM MODEL")
    neural_model, neural_config, neural_perplexity = load_neural_bigram_model(merge_count)
    if neural_model:
        try:
            generated = generate_text_neural_bigram(neural_model, bpe, context, max_tokens, temperature)
            results['neural_bigram'] = generated
            print(f"SUCCESS: Neural Bigram: {generated}")
            if neural_perplexity:
                print(f"         Validation perplexity: {neural_perplexity:.2f}")
            print(f"         Config: {neural_config}")
        except Exception as e:
            print(f"FAILED: Neural Bigram generation failed: {e}")
            results['neural_bigram'] = f"Error: {e}"
    else:
        print("FAILED: Neural Bigram model not available")
        results['neural_bigram'] = "Not available"
    
    print()
    
    # Test GPT model
    print("3. GPT MODEL")
    gpt_model, gpt_config, gpt_perplexity = load_gpt_model(merge_count)
    if gpt_model:
        try:
            generated = generate_text_gpt(gpt_model, bpe, context, max_tokens, temperature)
            results['gpt'] = generated
            print(f"SUCCESS: GPT: {generated}")
            if gpt_perplexity:
                print(f"         Validation perplexity: {gpt_perplexity:.2f}")
            if gpt_config:
                print(f"         Config: {gpt_config['n_layer']} layers, {gpt_config['n_embd']} dim")
        except Exception as e:
            print(f"FAILED: GPT generation failed: {e}")
            results['gpt'] = f"Error: {e}"
    else:
        print("FAILED: GPT model not available")
        results['gpt'] = "Not available"
    
    print()
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for model_name, result in results.items():
        model_display = model_name.replace('_', ' ').title()
        if "Error" in str(result) or "Not available" in str(result):
            print(f"{model_display:15}: FAILED - {result}")
        else:
            word_count = len(result.split())
            print(f"{model_display:15}: {result}")
            print(f"{' ' * 15}  ({word_count} words)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Text generation with model comparison')
    parser.add_argument('--model', help='Model: task2_ngram_2000, task3_neural_bigram_2000, task4_gpt_2000')
    parser.add_argument('--context', default="to be or not to be", help='Context text')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (GPT only)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling (GPT only)')
    parser.add_argument('--compare', action='store_true', help='Compare all available models')
    parser.add_argument('--merge_count', type=int, default=2000, help='BPE merge count to use')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_all_models(
            args.context, args.max_tokens, args.temperature, args.merge_count
        )
        return
    
    if not args.model:
        print("ERROR: Must specify --model or use --compare")
        print("Examples:")
        print("  python generate_text.py --model task2_ngram_2000 --context 'hello world'")
        print("  python generate_text.py --model task3_neural_bigram_2000 --context 'hello world'")
        print("  python generate_text.py --model task4_gpt_2000 --context 'hello world'")
        print("  python generate_text.py --compare --context 'hello world'")
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
                if perplexity:
                    print(f"Model perplexity: {perplexity:.2f}")
            else:
                print("ERROR: Could not load N-gram model")
        
        elif task_num == 3:
            model, config, perplexity = load_neural_bigram_model(merge_count)
            if model:
                generated = generate_text_neural_bigram(model, bpe, args.context, args.max_tokens, args.temperature)
                print(f"Generated: {generated}")
                if perplexity:
                    print(f"Model perplexity: {perplexity:.2f}")
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
                if perplexity:
                    print(f"Model perplexity: {perplexity:.2f}")
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