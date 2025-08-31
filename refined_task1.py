is_task1_done = False

def task1_with_cleaning():
    print("=" * 60)
    print("Task 1: BPE Training with Data Export")
    print("=" * 60)
    
    # Load and validate input
    input_file = "Shakespeare_clean_full.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    print(f"Original text: {len(full_text):,} characters")
    
    # Configuration
    cleaning_strategies = {
        "minimal": normalize_text_minimal,
        "lowercase": normalize_text_lowercase,
        "flatten": normalize_text_flatten
    }
    merge_counts = [1000, 2000, 3000]
    results = {}
    
    for strategy_name, clean_func in cleaning_strategies.items():
        print(f"\nTesting {strategy_name.upper()} cleaning strategy")
        print("-" * 40)
        
        # Clean text and prepare splits
        cleaned_text = clean_func(full_text)
        train_end = int(len(cleaned_text) * 0.99)
        bpe_train_text = cleaned_text[:train_end]
        bpe_test_text = cleaned_text[train_end:]
        
        print(f"Cleaned: {len(cleaned_text):,} chars")
        print(f"Train: {len(bpe_train_text):,} chars ({len(bpe_train_text)/len(cleaned_text)*100:.1f}%)")
        print(f"Test: {len(bpe_test_text):,} chars ({len(bpe_test_text)/len(cleaned_text)*100:.1f}%)")
        
        # Normalize existing data files
        print("Normalizing data files...")
        for input_file, output_suffix in [
            ('Shakespeare_clean_train.txt', 'train'),
            ('Shakespeare_clean_valid.txt', 'valid'),
            ('Shakespeare_clean_test.txt', 'test')
        ]:
            if os.path.exists(input_file):
                with open(input_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                normalized_text = clean_func(raw_text)
                output_file = f'shakespeare_{strategy_name}_{output_suffix}.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(normalized_text)
                print(f"  Saved: {output_file} ({len(normalized_text):,} chars)")
            else:
                print(f"  Warning: {input_file} not found")
        
        # Test BPE with different merge counts
        strategy_results = {}
        for merges in merge_counts:
            print(f"\nTesting {merges} merges...")
            
            cache_path = f"bpe_cache_{merges}_{strategy_name}.pkl"
            tokenizer = BPETokenizerSimple()
            
            tokenizer.train_or_load(
                bpe_train_text,
                vocab_size=merges + 300,
                allowed_special={"<|endoftext|>"},
                cache_path=cache_path
            )
            
            # Quick validation
            sample = "To be or not to be, that is the question."
            tokens = tokenizer.encode(sample)
            decoded = tokenizer.decode(sample)
            
            print(f"  Sample: '{sample}' -> {len(tokens)} tokens")
            print(f"  Vocab: {len(tokenizer.vocab)} | Merges: {len(tokenizer.bpe_merges)}")
            
            strategy_results[merges] = {
                'vocab_size': len(tokenizer.vocab),
                'merges': len(tokenizer.bpe_merges),
                'tokens_per_sample': len(tokens)
            }
        
        results[strategy_name] = strategy_results
    
    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    for strategy_name, strategy_results in results.items():
        print(f"\n{strategy_name.upper()} Cleaning:")
        print(f"{'Merges':<8} {'Vocab Size':<12} {'Actual Merges':<15} {'Tokens/Sample':<12}")
        print("-" * 50)
        
        for merges, result in strategy_results.items():
            print(f"{merges:<8} {result['vocab_size']:<12} {result['merges']:<15} {result['tokens_per_sample']:<12}")
    
    print("\nAnalysis:")
    print("  - Minimal: Preserves Shakespeare's structure")
    print("  - Lowercase: Good balance for training")
    print("  - Flatten: Fastest tokenization, loses structure")
    print("  - More merges -> Larger vocab -> Fewer tokens per text")
    print("=" * 60)
    
    global is_task1_done
    is_task1_done = True

if not is_task1_done:    
    task1_with_cleaning()
