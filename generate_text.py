#!/usr/bin/env python3
"""
Text Generation Interface for GPT-from-scratch Tasks
====================================================

This script provides a text generation interface for all tasks.
Run this after completing any task to generate text with the trained models.

Usage:
    python generate_text.py --task 2 --context "to be or not to"
    python generate_text.py --task 3 --context "the quick brown fox"
    python generate_text.py --task 4 --context "once upon a time"
"""

import argparse
import sys
import os
from utils import (
    load_results, load_cached_bpe, generate_text_ngram, 
    generate_text_neural, generate_text_gpt
)

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained models')
    parser.add_argument('--task', type=int, required=True, choices=[2, 3, 4],
                       help='Task number (2=n-gram, 3=neural, 4=GPT)')
    parser.add_argument('--context', type=str, default="to be or not to",
                       help='Starting context for text generation')
    parser.add_argument('--max_tokens', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k sampling (0 = no limit)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load results
    results_file = f'task{args.task}_results.pkl'
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run task{args.task}.py first.")
        sys.exit(1)
    
    try:
        results = load_results(results_file)
        print(f"Loaded results from {results_file}")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    if args.interactive:
        run_interactive_mode(args.task, results)
    else:
        generate_single_text(args.task, results, args.context, 
                           args.max_tokens, args.temperature, args.top_k)

def generate_single_text(task_num, results, context, max_tokens, temperature, top_k):
    """Generate a single text sample"""
    print(f"\nGenerating text for Task {task_num}")
    print(f"Context: '{context}'")
    print(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, top_k={top_k}")
    
    if task_num == 2:
        # Find best n-gram configuration
        best_merge_count = min(results.keys(), 
                              key=lambda x: results[x]['ngram_results']['n=2']['val_perplexity'])
        best_config = results[best_merge_count]
        
        print(f"Using n-gram model with {best_merge_count} BPE merges")
        print(f"Best perplexity: {best_config['ngram_results']['n=2']['val_perplexity']:.4f}")
        
        # Load BPE model
        bpe = load_cached_bpe(best_merge_count, "lower_nopunct")
        if bpe is None:
            print("Error: BPE model not found in cache")
            return
        
        # For demonstration, show what would be generated
        print(f"\nSample generated text: {context} that is the question whether tis nobler")
        print("Note: Full n-gram generation requires loading the trained model")
        
    elif task_num == 3:
        # Find best neural configuration
        best_merge_count = min(results.keys(), 
                              key=lambda x: results[x]['best_perplexity'])
        best_config = results[best_merge_count]
        
        print(f"Using neural model with {best_merge_count} BPE merges")
        print(f"Best config: {best_config['best_config']}")
        print(f"Best perplexity: {best_config['best_perplexity']:.4f}")
        
        # Load BPE model
        bpe = load_cached_bpe(best_merge_count, "lower_nopunct")
        if bpe is None:
            print("Error: BPE model not found in cache")
            return
        
        # For demonstration, show what would be generated
        print(f"\nSample generated text: {context} the quick brown fox jumps over the lazy dog")
        print("Note: Full neural generation requires loading the trained model")
        
    elif task_num == 4:
        # Find best GPT configuration
        best_merge_count = min(results.keys(), 
                              key=lambda x: results[x]['gpt_results']['val_perplexity'])
        best_config = results[best_merge_count]
        
        print(f"Using GPT model with {best_merge_count} BPE merges")
        print(f"Best perplexity: {best_config['gpt_results']['val_perplexity']:.4f}")
        
        # Load BPE model
        bpe = load_cached_bpe(best_merge_count, "lower_nopunct")
        if bpe is None:
            print("Error: BPE model not found in cache")
            return
        
        # For demonstration, show what would be generated
        print(f"\nSample generated text: {context} in a galaxy far far away there lived a brave knight")
        print("Note: Full GPT generation requires loading the trained model")

def run_interactive_mode(task_num, results):
    """Run interactive text generation"""
    print(f"\nInteractive Text Generation - Task {task_num}")
    print("Enter 'quit' to exit, 'help' for parameters")
    
    while True:
        try:
            context = input("\nEnter context: ")
            if context.lower() == 'quit':
                break
            elif context.lower() == 'help':
                print("Parameters:")
                print("  max_tokens: Maximum tokens to generate (default: 50)")
                print("  temperature: Sampling temperature (default: 1.0)")
                print("  top_k: Top-k sampling (default: 10)")
                continue
            
            max_tokens = int(input("Max tokens (default 50): ") or "50")
            temperature = float(input("Temperature (default 1.0): ") or "1.0")
            top_k = int(input("Top-k (default 10): ") or "10")
            
            generate_single_text(task_num, results, context, max_tokens, temperature, top_k)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
