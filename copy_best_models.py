import shutil

def copy_best_models():
    """Copy only the best performing models to results directory"""
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Best Task 2 models (4 best performing)
    task2_best = [
        "ngram_backoff_max4_alpha0.4_flatten_1000merges.pkl",  # 10.40 PPL
        "ngram_backoff_max4_alpha0.4_flatten_2000merges.pkl",   # ~14.66 PPL
        "ngram_backoff_max4_alpha0.4_flatten_3000merges.pkl",  # ~15.19 PPL
        "ngram_backoff_max4_alpha0.4_minimal_2000merges.pkl",   # ~17.22 PPL
    ]
    
    # BPE files that the Task 2 models actually need
    bpe_files = [
        "bpe_cache_1000_flatten.pkl",    # For flatten_1000merges model
        "bpe_cache_2000_flatten.pkl",   # For flatten_2000merges model  
        "bpe_cache_3000_flatten.pkl",  # For flatten_3000merges model
        "bpe_cache_2000_minimal.pkl",   # For minimal_2000merges model
    ]


     # Best Task 3 models (4-gram models are best performing)
    task3_best = [
        "neural_4gram_flatten_1000merges.pt",  # Best: 12.51 PPL
        "neural_4gram_flatten_2000merges.pt",  # Second best
        "neural_4gram_flatten_3000merges.pt",  # Third best
        "neural_4gram_minimal_2000merges.pt",  # Minimal strategy
    ]


    # Best Task 4 models (GPT models with best performance)
    task4_best = [
        "gpt_flatten_1000merges.pt",  # Best: 13.08 PPL
        "gpt_flatten_2000merges.pt",  # Second best
        "gpt_flatten_3000merges.pt",  # Third best
        "gpt_minimal_2000merges.pt",  # Minimal strategy
    ]
    
    
    print("Copying BPE files needed by Task 2 models...")
    for bpe in bpe_files:
        if os.path.exists(bpe):
            shutil.copy2(bpe, f"results/{bpe}")
            print(f"✓ {bpe}")
        else:
            print(f"✗ {bpe} not found")
    
    print("\nCopying best Task 2 models...")
    for model in task2_best:
        if os.path.exists(f"task2/{model}"):
            shutil.copy2(f"task2/{model}", f"results/{model}")
            print(f"✓ {model}")
        else:
            print(f"✗ {model} not found")
    
   
    print("\nCopying best Task 3 models...")
    for model in task3_best:
        if os.path.exists(f"task3/{model}"):
            shutil.copy2(f"task3/{model}", f"results/{model}")
            print(f"✓ {model}")
        else:
            print(f"✗ {model} not found")
    
    
    print("\nCopying best Task 4 models...")
    for model in task4_best:
        if os.path.exists(f"task4/{model}"):
            shutil.copy2(f"task4/{model}", f"results/{model}")
            print(f"✓ {model}")
        else:
            print(f"✗ {model} not found")
    
    print("\n✓ Done! Best models copied to results/")

if __name__ == "__main__":
    copy_best_models()
