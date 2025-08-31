# Task 1: Tokenization-Only (BPE)
# - Split Shakespeare into train/test/valid
# - Train BPE with different merges (k) and normalization techniques
# - Measure average tokens per word (TPW) + reconstruction check
# - Print results and save to task1_tok_results.json
# - Uses caching to avoid retraining BPE models

import json
from utils import (
    load_and_slice_data, BPE, evaluate_bpe_configuration, 
    print_configuration_summary, create_result_entry, save_results,
    load_cached_bpe, save_cached_bpe
)

# Configuration
PERCENTAGE = 0.02                       # 0.01=1%, 0.1=10%, 1.0=full - Using 2% for better training
MERGE_COUNTS = [1000, 2000]  # Only 1000 and 2000 merge counts
NORMALIZATION_TECHNIQUES = ["lower_nopunct", "aggressive"]  # Simplified as requested

def main():
    """
    What it does: Main function to run Task 1
    Args:
        None
    Returns:
        None
    """
    print("Task 1: BPE Tokenization")
    print("=" * 50)
    
    # Load and slice data
    train_text, valid_text, test_text = load_and_slice_data(PERCENTAGE)

    # Results storage
    results = []
    
    # Test different BPE configurations
    for normalization_technique in NORMALIZATION_TECHNIQUES:
        for merge_count in MERGE_COUNTS:
            print(f"\nTesting BPE with {merge_count} merges and {normalization_technique} normalization...")
            
            # Try to load cached BPE model first
            bpe = load_cached_bpe(merge_count, normalization_technique)
            
            if bpe is None:
                # Train new BPE model
                bpe = BPE()
                bpe.fit(train_text, merge_count, normalization_technique)
                
                # Cache the trained model
                save_cached_bpe(bpe, merge_count, normalization_technique)
            
            # Evaluate configuration
            evaluation_results = evaluate_bpe_configuration(
                bpe, train_text, valid_text, test_text, normalization_technique
            )
            
            # Print summary
            print_configuration_summary(normalization_technique, merge_count, bpe, evaluation_results)
            
            # Store results
            result_entry = create_result_entry(
                normalization_technique, merge_count, bpe, evaluation_results
            )
            results.append(result_entry)

    # Save results
    save_results(results, 'task1_results.pkl')
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['evaluation']['valid']['avg_tokens_per_word'])
    print(f"\nBest configuration:")
    print(f"  Normalization: {best_config['normalization_technique']}")
    print(f"  Merge count: {best_config['merge_count']}")
    print(f"  Validation avg tokens/word: {best_config['evaluation']['valid']['avg_tokens_per_word']:.4f}")
    
    print("\nTask 1 completed!")

if __name__ == "__main__":
    main()
