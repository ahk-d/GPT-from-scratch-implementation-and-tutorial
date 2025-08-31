import gradio as gr
import torch
import pickle
import os
import json
import math
import random
import glob
import zipfile
import tempfile
from collections import Counter, defaultdict
import torch.nn as nn
import torch.nn.functional as F

class BPETokenizerSimple:
    """
    A simple BPE (Byte Pair Encoding). This implementation follows Sebastian Raschka's production-ready approach.
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb

    """

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """Train the BPE tokenizer from scratch."""
        # Preprocess: Replace spaces with "Ġ"
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text))
            if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text, allowed_special=None, norm=None):
        """Encode the input text into a list of token IDs."""
        import re

        # Apply normalization if specified
        if norm == 'lower_nopunct':
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        elif norm == 'minimal_clean':
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        token_ids = []

        # Handle special tokens if enabled
        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = (
                "(" + "|".join(
                    re.escape(tok)
                    for tok in sorted(allowed_special, key=len, reverse=True)
                ) + ")"
            )

            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None, norm=None))

                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()

            text = text[last_index:]

        # Handle text with potential newlines
        tokens = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")

            words = line.split()
            for j, word in enumerate(words):
                if i == 0 and j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    def tokenize_with_bpe(self, token):
        """Tokenize a single token using BPE merges."""
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        i += 2
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        symbols = [self.vocab[id_num] for id_num in token_ids]

        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            if bigram is None or bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        """Decode a list of token IDs back into a string."""
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_to_cache(self, cache_path):
        """Save the trained tokenizer to cache file."""
        cache_data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'bpe_merges': self.bpe_merges,
            'bpe_ranks': self.bpe_ranks
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✓ Saved tokenizer to cache: {cache_path}")

    def load_from_cache(self, cache_path):
        """Load the trained tokenizer from cache file."""
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.vocab = cache_data['vocab']
            self.inverse_vocab = cache_data['inverse_vocab']
            self.bpe_merges = cache_data['bpe_merges']
            self.bpe_ranks = cache_data['bpe_ranks']
            print(f"✓ Loaded tokenizer from cache: {cache_path}")
            return True
        return False

    def train_or_load(self, text, vocab_size, allowed_special={"<|endoftext|>"}, cache_path=None):
        """Train tokenizer or load from cache if available."""
        if cache_path and os.path.exists(cache_path):
            if self.load_from_cache(cache_path):
                return True

        print("Training new tokenizer...")
        self.train(text, vocab_size, allowed_special)

        if cache_path:
            self.save_to_cache(cache_path)

        return False

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        """Find the most or least frequent pair."""
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Mode must be 'most' or 'least'")

    def replace_pair(self, token_ids, pair, new_id):
        """Replace all occurrences of a pair with a new token ID."""
        new_token_ids = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                new_token_ids.append(new_id)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        return new_token_ids

# Hugging Face Spaces utilities
def extract_results_zip():
    """Extract results.zip if it exists for HF Spaces deployment"""
    if os.path.exists("results.zip"):
        print("Extracting results.zip for Hugging Face Spaces...")
        with zipfile.ZipFile("results.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✓ Extracted results.zip")
        return True
    return False

# Load BPE and model utilities
def find_bpe_file():
    """Recursively search for BPE cache file"""
    # First try to extract from results.zip
    extract_results_zip()
    
    # Exact BPE files we have
    bpe_files = [
        "bpe_cache_1000_flatten.pkl",
        "bpe_cache_2000_flatten.pkl", 
        "bpe_cache_3000_flatten.pkl",
        "bpe_cache_2000_minimal.pkl"
    ]
    
    # Check results directory first, then root
    for bpe_file in bpe_files:
        if os.path.exists(f"results/{bpe_file}"):
            return f"results/{bpe_file}"
        elif os.path.exists(bpe_file):
            return bpe_file
    
    # Fallback patterns
    patterns = [
        "bpe_cache_*_lower_nopunct.pkl",
        "bpe_cache_*.pkl", 
        "*bpe*.pkl"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            print(f"Found BPE file: {files[0]}")
            return files[0]
        
        # Search in subdirectories
        files = glob.glob(f"**/{pattern}", recursive=True)
        if files:
            print(f"Found BPE file: {files[0]}")
            return files[0]
    
    return None

def load_cached_bpe_from_path(filepath):
    """Load BPE model from specific file path"""
    try:
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Create BPETokenizerSimple instance and load the cached data
        bpe = BPETokenizerSimple()
        bpe.vocab = cache_data['vocab']
        bpe.inverse_vocab = cache_data['inverse_vocab']
        bpe.bpe_merges = cache_data['bpe_merges']
        bpe.bpe_ranks = cache_data['bpe_ranks']
        
        print(f"Loaded BPE from: {filepath}")
        return bpe
    except Exception as e:
        print(f"Failed to load BPE from {filepath}: {e}")
        return None

def normalize_text(text, normalization_type):
    """Normalize text according to specified strategy"""
    import re
    if normalization_type == "minimal_clean":
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    elif normalization_type == "lower_nopunct":
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    return text

# Classical N-gram model for Task 2 cached models
class BackoffNGram:
    """N-gram model with stupid backoff."""

    def __init__(self, max_n, tokenizer, alpha=0.4):
        """
        Initialize backoff model.

        Args:
            max_n: Maximum n-gram order
            tokenizer: BPE tokenizer
            alpha: Backoff discount factor
        """
        self.max_n = max_n
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.model = NGramModel(max_n, tokenizer, k=0.01)  # Small k for backoff

    def train(self, text):
        """Train underlying n-gram model."""
        print(f"Training backoff model (max_n={self.max_n}, alpha={self.alpha})...")
        self.model.train(text)

    def get_backoff_probability(self, ngram, n):
        """
        Get probability with stupid backoff.

        If count(ngram) > 0: use MLE
        Else: backoff to (n-1)-gram with discount alpha
        """
        if n == 1:
            # Base case: use smoothed unigram
            return self.model.get_probability(ngram, 1)

        count = self.model.ngram_counts[n][ngram]

        if count > 0:
            # Use MLE (relative frequency)
            context = ngram[:-1]
            context_count = self.model.context_counts[n][context]
            return count / context_count if context_count > 0 else 0
        else:
            # Backoff to lower order
            lower_ngram = ngram[1:]  # Remove first token
            return self.alpha * self.get_backoff_probability(lower_ngram, n - 1)

    def calculate_perplexity(self, text):
        """Calculate perplexity using backoff."""
        tokens = self.tokenizer.encode(text)
        log_prob_sum = 0.0

        for i in range(len(tokens)):
            # Use highest order possible at each position
            for n in range(self.max_n, 0, -1):
                if i >= n - 1:
                    if n == 1:
                        ngram = (tokens[i],)
                    else:
                        ngram = tuple(tokens[i - n + 1:i + 1])

                    prob = self.get_backoff_probability(ngram, n)
                    if prob > 0:
                        log_prob_sum += math.log(prob)
                    break

        perplexity = math.exp(-log_prob_sum / len(tokens))
        return perplexity

    def generate_text(self, max_length=50, context="", method='sampling'):
        """Generate text using highest-order model."""
        return self.model.generate_text(self.max_n, max_length, context, method)

class NGramModel:
    def __init__(self, bpe_model, normalization='lower_nopunct'):
        self.bpe_model = bpe_model
        self.normalization = normalization
        self.models = {}
        self.vocab = set()
        self.START, self.END = '<START>', '<END>'
        self._gen_vocab = None
        self.interpolation_weights = {}

    def _addk(self, ngram, n, k=1.0):
        m = self.models[n]
        c = m['ng'].get(ngram, 0)
        if n == 1:
            N = sum(m['ng'].values())
            return (c + k) / (N + k * len(self._gen_vocab))
        C = m['ctx'].get(ngram[:-1], 0)
        return (c + k) / (C + k * len(self._gen_vocab))

    def _backoff(self, ngram, n):
        for order in range(n, 0, -1):
            if order in self.models and len(ngram) >= order:
                sub = ngram[-order:]
                m = self.models[order]
                if m['ng'].get(sub, 0) > 0 or order == 1:
                    return self._addk(sub, order)
        return 1.0 / len(self._gen_vocab)

    def _candidates(self, ctx_gram, n):
        if n > 1 and ctx_gram in self.models[n]['ctx']:
            ng = self.models[n]['ng']
            toks = [g[-1] for g in ng if g[:-1] == ctx_gram and g[-1] != self.START]
            if toks:
                return toks
        return list(self._gen_vocab)

    def _is_word_boundary(self, token):
        if token == self.END:
            return True
        s = self.bpe_model.decode([token])
        return bool(s) and (s[-1].isspace() or s[0].isspace() or s[-1] in '.,!?;:-—')

    def generate(self, context, n=3, max_words=25, method='argmax', temperature=1.0):
        ctx = self.bpe_model.encode(context, norm=self.normalization)
        hist = (ctx[-(n-1):] if len(ctx) >= n-1 else [self.START]*(n-1-len(ctx)) + ctx)
        words = 0
        out = []
        recent = []

        while words < max_words:
            gram = tuple(hist[-(n-1):]) if n > 1 else tuple()
            cand = self._candidates(gram, n)
            
            if not cand:
                toks = list(self._gen_vocab)
                scores = [self._addk((t,), 1) for t in toks]
                t = toks[scores.index(max(scores))]
                if t == self.END:
                    break
                out.append(t)
                hist.append(t)
                recent.append(t)
                if self._is_word_boundary(t):
                    words += 1
                continue

            probs = []
            for t in cand:
                if n > 1:
                    seq = (hist[-(n-1):] + [t])[-n:]
                    ng = tuple(seq)
                else:
                    ng = (t,)
                probs.append(max(self._backoff(ng, n), 1e-12))

            penalties = [1.3**recent[-5:].count(t) for t in cand]
            logits = [math.log(p/pen) for p, pen in zip(probs, penalties)]

            if method == 'argmax':
                t = cand[max(range(len(logits)), key=lambda i: logits[i])]
            else:
                zt = max(1e-6, float(temperature))
                logits = [x/zt for x in logits]
                m = max(logits); exps = [math.exp(x-m) for x in logits]; Z = sum(exps)
                w = [e/Z for e in exps]
                t = random.choices(cand, weights=w, k=1)[0]

            if t == self.END:
                break
            out.append(t)
            hist.append(t)
            recent.append(t)
            if self._is_word_boundary(t):
                words += 1

        text = ' '.join(self.bpe_model.decode(out).split()).strip()
        return text

    @classmethod
    def load_model(cls, filepath, bpe_model):
        """Load a cached classical n-gram model from Task 2"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        instance = cls(bpe_model, model_data['normalization'])
        instance.models = model_data['models']
        instance.vocab = set(model_data['vocab'])
        instance.interpolation_weights = model_data['interpolation_weights']
        instance._gen_vocab = set(model_data['generation_vocab'])
        instance.START = model_data['start_end_tokens']['START']
        instance.END = model_data['start_end_tokens']['END']
        
        return instance

# Neural N-gram model architecture (Task 3)
class NeuralNgramModel(nn.Module):
    def __init__(self, vocab_size, n, n_embd=256, n_hidden=512, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n = n
        self.n_embd = n_embd
        
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        if n == 1:
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(n_embd, vocab_size)
        else:
            inp = n_embd * (n - 1)
            self.fc1 = nn.Linear(inp, n_hidden)
            self.drop1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
            self.drop2 = nn.Dropout(dropout)
            self.out = nn.Linear(n_hidden // 2, vocab_size)

    def forward(self, ctx_ids):
        if self.n == 1:
            B = ctx_ids.size(0)
            x = self.embedding.weight.mean(dim=0, keepdim=True).expand(B, -1)
            x = self.drop(x)
            logits = self.out(x)
        else:
            emb = self.embedding(ctx_ids)
            x = emb.view(emb.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.drop1(x)
            x = F.relu(self.fc2(x))
            x = self.drop2(x)
            logits = self.out(x)
        return logits

# GPT model architecture (Task 4) - Simplified for inference
class CausalSelfAttention(nn.Module):
    """Causal self-attention with dropout."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            )
        )

    def forward(self, x):
        B, T, C = x.size()

        # Calculate Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """MLP block."""

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    """GPT Language Model."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        """Generate text."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class ModelManager:
    def __init__(self):
        self.models = {}
        self.bpe_tokenizers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_all_models()

    def load_all_models(self):
        """Load all models from directory"""
        print("Loading all models from directory...")
        
        # Load BPE tokenizers
        bpe_files = [
            "bpe_cache_1000_flatten.pkl",
            "bpe_cache_2000_flatten.pkl", 
            "bpe_cache_3000_flatten.pkl",
            "bpe_cache_2000_minimal.pkl"
        ]
        
        for bpe_file in bpe_files:
            if os.path.exists(bpe_file):
                bpe = load_cached_bpe_from_path(bpe_file)
                if bpe:
                    self.bpe_tokenizers[bpe_file] = bpe
                    print(f"Loaded BPE: {bpe_file}")
        
        # Load all model files
        all_files = [
            "ngram_backoff_max4_alpha0.4_flatten_1000merges.pkl",
            "ngram_backoff_max4_alpha0.4_flatten_2000merges.pkl", 
            "ngram_backoff_max4_alpha0.4_flatten_3000merges.pkl",
            "ngram_backoff_max4_alpha0.4_minimal_2000merges.pkl",
            "neural_4gram_flatten_1000merges.pt",
            "neural_4gram_flatten_2000merges.pt",
            "neural_4gram_flatten_3000merges.pt", 
            "neural_4gram_minimal_2000merges.pt",
            "gpt_flatten_1000merges.pt",
            "gpt_flatten_2000merges.pt",
            "gpt_flatten_3000merges.pt",
            "gpt_minimal_2000merges.pt"
        ]
        
        for model_file in all_files:
            if os.path.exists(model_file):
                try:
                    self.load_single_model(model_file)
                    print(f"Loaded model: {model_file}")
                except Exception as e:
                    print(f"Failed to load {model_file}: {e}")
        
        print(f"Total models loaded: {len(self.models)}")

    def load_single_model(self, filepath):
        """Load a single model file"""
        if filepath.endswith('.pkl'):
            # Classical model
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Get appropriate BPE
            bpe = self.get_bpe_for_model(filepath)
            if not bpe:
                print(f"No BPE found for {filepath}")
                return
            
            # Create model with proper initialization
            model = NGramModel(bpe, 'lower_nopunct')
            
            # Handle different data structures
            if hasattr(model_data, 'get'):
                # Dictionary format
                model.models = model_data['models']
                model.vocab = set(model_data['vocab'])
                model.interpolation_weights = model_data['interpolation_weights']
                model._gen_vocab = set(model_data['generation_vocab'])
                model.START = model_data['start_end_tokens']['START']
                model.END = model_data['start_end_tokens']['END']
            else:
                # Object format - extract attributes
                model.models = getattr(model_data, 'models', {})
                model.vocab = set(getattr(model_data, 'vocab', []))
                model.interpolation_weights = getattr(model_data, 'interpolation_weights', {})
                model._gen_vocab = set(getattr(model_data, 'generation_vocab', []))
                start_end_tokens = getattr(model_data, 'start_end_tokens', {'START': '<START>', 'END': '<END>'})
                model.START = start_end_tokens.get('START', '<START>')
                model.END = start_end_tokens.get('END', '<END>')
            
            # Ensure the model has the required attributes
            if not hasattr(model, 'models') or not model.models:
                print(f"Warning: Model {filepath} has no models data")
                return
            
            self.models[filepath] = {'type': 'classical', 'model': model}
            
        elif filepath.endswith('.pt'):
            # Neural or GPT model
            checkpoint = torch.load(filepath, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            if 'neural' in filepath:
                # Neural model
                vocab_size = state_dict['embedding.weight'].shape[0]
                n_embd = state_dict['embedding.weight'].shape[1]
                n_hidden = state_dict.get('fc1.weight', torch.zeros(256, 1)).shape[0]
                
                model = NeuralNgramModel(vocab_size=vocab_size, n=4, n_embd=n_embd, n_hidden=n_hidden)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                self.models[filepath] = {'type': 'neural', 'model': model}
            else:
                # GPT model
                vocab_size, n_embd = state_dict['wte.weight'].shape
                n_head = 4
                n_layer = 2
                block_size = 64
                
                class Config:
                    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
                        self.vocab_size = vocab_size
                        self.n_embd = n_embd
                        self.n_head = n_head
                        self.n_layer = n_layer
                        self.block_size = block_size
                        self.dropout = dropout
                
                cfg = Config(vocab_size, n_embd, n_head, n_layer, block_size, 0.1)
                model = GPTModel(cfg)
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)
                model.eval()
                
                self.models[filepath] = {'type': 'gpt', 'model': model}

    def get_bpe_for_model(self, filepath):
        """Get appropriate BPE tokenizer for model"""
        if '1000merges' in filepath:
            return self.bpe_tokenizers.get("bpe_cache_1000_flatten.pkl")
        elif '2000merges' in filepath and 'minimal' in filepath:
            return self.bpe_tokenizers.get("bpe_cache_2000_minimal.pkl")
        elif '2000merges' in filepath:
            return self.bpe_tokenizers.get("bpe_cache_2000_flatten.pkl")
        elif '3000merges' in filepath:
            return self.bpe_tokenizers.get("bpe_cache_3000_flatten.pkl")
        else:
            return list(self.bpe_tokenizers.values())[0] if self.bpe_tokenizers else None

    def parse_neural_filename(self, filename):
        """Extract n-gram order and configuration from Task 3 neural model filename"""
        basename = os.path.basename(filename).lower()
        
        # Extract n-gram order
        n = None
        if 'n1_' in basename or '_1gram' in basename:
            n = 1
        elif 'n2_' in basename or '_2gram' in basename:
            n = 2
        elif 'n3_' in basename or '_3gram' in basename:
            n = 3
        elif 'n4_' in basename or '_4gram' in basename:
            n = 4
        
        # Extract configuration
        config = None
        if '1000merges' in basename:
            config = '1000merges'
        elif '2000merges' in basename and 'minimal' in basename:
            config = 'minimal'
        elif '2000merges' in basename:
            config = '2000merges'
        elif '3000merges' in basename:
            config = '3000merges'
        
        return n, config

    def parse_gpt_filename(self, filename):
        """Extract GPT model size from Task 4 filename"""
        basename = os.path.basename(filename).lower()
        if '1000merges' in basename:
            return '1000merges'
        elif '2000merges' in basename:
            return '2000merges'
        elif '3000merges' in basename:
            return '3000merges'
        elif 'minimal' in basename:
            return 'minimal'
        return 'unknown'

    def parse_classical_filename(self, filename):
        """Extract n-gram order from Task 2 classical model filename"""
        basename = os.path.basename(filename).lower()
        if '1gram' in basename:
            return 1
        elif '2gram' in basename:
            return 2
        elif '3gram' in basename:
            return 3
        elif '4gram' in basename:
            return 4
        return None

    def load_models(self):
        """Load all available models from filesystem"""
        model_files = self.find_model_files()
        
        # Load BPE tokenizers for each configuration
        self.bpe_tokenizers = {}
        for config, bpe_file in model_files['bpe'].items():
            bpe = load_cached_bpe_from_path(bpe_file)
            if bpe:
                self.bpe_tokenizers[config] = bpe
                print(f"Loaded BPE tokenizer for {config}")
        
        # Use the first available BPE as default
        if self.bpe_tokenizers:
            self.bpe = list(self.bpe_tokenizers.values())[0]
        else:
            print("WARNING: No BPE model found. Creating minimal demo BPE.")
            class DemoBPE:
                def __init__(self):
                    self.vocab = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'be', 'thou'])
                def encode(self, text, norm=None):
                    return text.lower().split()[:10]
                def decode(self, tokens):
                    return ' '.join(str(t) for t in tokens)
            self.bpe = DemoBPE()
        
        # Build vocabulary from BPE model
        if hasattr(self.bpe, 'vocab') and self.bpe.vocab:
            base_vocab = sorted(list(self.bpe.vocab))
        else:
            # Fallback vocabulary
            base_vocab = ['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'be', 'thou', 'shall', 'will', 'have', 'with', 'as', 'for', 'this', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use']
        
        specials = ['<START>', '<END>', '<UNK>']
        self.vocab = base_vocab + [s for s in specials if s not in base_vocab]
        self.v2i = {t: i for i, t in enumerate(self.vocab)}
        self.i2v = {i: t for t, i in self.v2i.items()}
        
        # Load models by type
        self.load_classical_models(model_files['classical'])
        self.load_neural_models(model_files['neural'])
        self.load_gpt_models(model_files['gpt'])

    def load_classical_models(self, file_list):
        """Load Task 2 classical model checkpoints"""
        for filepath in file_list:
            try:
                # Determine which BPE tokenizer to use based on filename
                bpe_config = None
                if '1000merges' in filepath:
                    bpe_config = '1000merges'
                elif '2000merges' in filepath and 'minimal' in filepath:
                    bpe_config = 'minimal'
                elif '2000merges' in filepath:
                    bpe_config = '2000merges'
                elif '3000merges' in filepath:
                    bpe_config = '3000merges'
                
                # Get the appropriate BPE tokenizer
                if bpe_config and bpe_config in self.bpe_tokenizers:
                    bpe = self.bpe_tokenizers[bpe_config]
                else:
                    bpe = self.bpe  # fallback to default
                
                # Load the classical model data
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Handle different data structures
                if hasattr(model_data, 'get'):
                    # Dictionary format
                    normalization = model_data.get('normalization', 'lower_nopunct')
                    models = model_data['models']
                    vocab = set(model_data['vocab'])
                    interpolation_weights = model_data['interpolation_weights']
                    generation_vocab = set(model_data['generation_vocab'])
                    start_end_tokens = model_data['start_end_tokens']
                else:
                    # Object format - extract attributes
                    normalization = getattr(model_data, 'normalization', 'lower_nopunct')
                    models = getattr(model_data, 'models', {})
                    vocab = set(getattr(model_data, 'vocab', []))
                    interpolation_weights = getattr(model_data, 'interpolation_weights', {})
                    generation_vocab = set(getattr(model_data, 'generation_vocab', []))
                    start_end_tokens = getattr(model_data, 'start_end_tokens', {'START': '<START>', 'END': '<END>'})
                
                # Create a new NGramModel instance
                model = NGramModel(bpe, normalization)
                model.models = models
                model.vocab = vocab
                model.interpolation_weights = interpolation_weights
                model._gen_vocab = generation_vocab
                model.START = start_end_tokens['START']
                model.END = start_end_tokens['END']
                
                n = self.parse_classical_filename(filepath)
                if n is not None:
                    # Use unique key with configuration
                    model_key = f"{n}gram_{bpe_config}"
                    self.classical_models[model_key] = model
                    print(f"Loaded classical {n}-gram {bpe_config} from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Failed to load classical model {filepath}: {e}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")

    def load_neural_models(self, file_list):
        """Load Task 3 neural model checkpoints"""
        for filepath in file_list:
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                
                # Handle Task 3 checkpoint format
                state_dict = checkpoint.get('state', checkpoint)
                cfg = checkpoint.get('cfg', {})
                
                n, config = self.parse_neural_filename(filepath)
                if n is None or config is None:
                    continue
                
                # Infer architecture from state dict
                vocab_size = state_dict['embedding.weight'].shape[0]
                n_embd = state_dict['embedding.weight'].shape[1]
                
                # Infer hidden size from fc1 layer
                if 'fc1.weight' in state_dict:
                    n_hidden = state_dict['fc1.weight'].shape[0]
                else:
                    n_hidden = 256
                
                model = NeuralNgramModel(
                    vocab_size=vocab_size,
                    n=n,
                    n_embd=n_embd,
                    n_hidden=n_hidden,
                    dropout=0.1  # Low for inference
                )
                
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                # Use unique key with configuration
                model_key = f"{n}gram_{config}"
                self.neural_models[model_key] = model
                print(f"Loaded neural {n}-gram {config} from {os.path.basename(filepath)}")
                
            except Exception as e:
                print(f"Failed to load neural model {filepath}: {e}")

    def load_gpt_models(self, file_list):
        """Load Task 4 GPT model checkpoints"""
        for filepath in file_list:
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                
                # Handle Task 4 checkpoint format
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                size = self.parse_gpt_filename(filepath)
                
                # Infer architecture from state dict
                wte_size = state_dict['wte.weight'].shape
                vocab_size, n_embd = wte_size
                
                # Infer other parameters
                n_head = 4  # default
                if 'h.0.attn.c_attn.weight' in state_dict:
                    attn_weight = state_dict['h.0.attn.c_attn.weight']
                    n_head = attn_weight.shape[0] // (3 * n_embd)
                
                # Count layers
                n_layer = 0
                for key in state_dict.keys():
                    if key.startswith('h.') and '.attn.c_attn.weight' in key:
                        layer_num = int(key.split('.')[1])
                        n_layer = max(n_layer, layer_num + 1)
                if n_layer == 0:
                    n_layer = 2  # Based on the error messages, seems like 2 layers
                
                # Infer block size
                block_size = 64
                if 'wpe.weight' in state_dict:
                    block_size = state_dict['wpe.weight'].shape[0]
                
                # Create config object
                class Config:
                    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
                        self.vocab_size = vocab_size
                        self.n_embd = n_embd
                        self.n_head = n_head
                        self.n_layer = n_layer
                        self.block_size = block_size
                        self.dropout = dropout
                
                cfg = Config(vocab_size, n_embd, n_head, n_layer, block_size, 0.1)
                
                model = GPTModel(cfg)
                
                # Load with strict=False to handle naming differences
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)
                model.eval()
                
                model_key = size
                if model_key not in self.gpt_models:
                    self.gpt_models[model_key] = model
                    print(f"Loaded GPT {size} from {os.path.basename(filepath)}")
                
            except Exception as e:
                print(f"Failed to load GPT model {filepath}: {e}")

    def generate_text(self, model_type, model_name, context, max_length=50, temperature=0.8):
        """Generate text using specified model"""
        try:
            if model_type == "Classical N-gram":
                if model_name in self.classical_models:
                    n = int(model_name[0])
                    return self.classical_models[model_name].generate(
                        context, n=n, max_words=max_length//3, temperature=temperature
                    )
                else:
                    return "Classical model not available"
                    
            elif model_type == "Neural N-gram":
                if model_name in self.neural_models:
                    return self.neural_generate(model_name, context, max_length, temperature)
                else:
                    return "Neural model not available"
                    
            elif model_type == "GPT":
                if model_name in self.gpt_models:
                    return self.gpt_generate(model_name, context, max_length, temperature)
                else:
                    return "GPT model not available"
                    
        except Exception as e:
            return f"Generation failed: {str(e)}"

    def neural_generate(self, model_name, context, max_length, temperature):
        """Generate using Task 3 neural n-gram model"""
        model = self.neural_models[model_name]
        n = model.n
        
        # Use BPE vocabulary directly for neural models
        bpe_vocab = sorted(list(self.bpe.vocab)) if hasattr(self.bpe, 'vocab') else []
        specials = ['<START>', '<END>', '<UNK>']
        full_vocab = bpe_vocab + [s for s in specials if s not in bpe_vocab]
        v2i = {t: i for i, t in enumerate(full_vocab)}
        i2v = {i: t for t, i in v2i.items()}
        
        ctx_tokens = self.bpe.encode(context, norm='lower_nopunct')
        if len(ctx_tokens) < n - 1:
            ctx_tokens = ['<START>'] * (n - 1 - len(ctx_tokens)) + ctx_tokens
        
        out = list(ctx_tokens)
        
        with torch.no_grad():
            for _ in range(max_length):
                if n == 1:
                    ctx_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                else:
                    ctx_ids = torch.tensor([[v2i.get(t, v2i['<UNK>']) for t in out[-(n-1):]]],
                                         device=self.device)
                
                logits = model(ctx_ids) / max(1e-6, float(temperature))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                next_token = i2v[next_id]
                
                if next_token == '<END>':
                    break
                out.append(next_token)
        
        clean = [t for t in out if t not in ('<START>', '<END>', '<UNK>')]
        return self.bpe.decode(clean)

    def gpt_generate(self, model_name, context, max_length, temperature):
        """Generate using Task 4 GPT model"""
        model = self.gpt_models[model_name]
        
        ctx_tokens = self.bpe.encode(context, norm='lower_nopunct')
        ctx_ids = torch.tensor([[self.v2i.get(t, self.v2i['<UNK>']) for t in ctx_tokens]], 
                              device=self.device)
        
        with torch.no_grad():
            generated = model.generate(ctx_ids, max_new_tokens=max_length, temperature=temperature)
            tokens = [self.i2v.get(i, '<UNK>') for i in generated[0].tolist()]
            return self.bpe.decode(tokens)

# Initialize model manager
print("Initializing model manager...")
model_manager = ModelManager()

def generate_text_simple(selected_model, context, max_length, temperature):
    """Simple text generation"""
    if not context.strip():
        return "❌ Please enter some context text to generate from."
    
    if selected_model not in model_manager.models:
        return "❌ Model not found."
    
    try:
        model_info = model_manager.models[selected_model]
        model = model_info['model']
        model_type = model_info['type']
        
        if model_type == 'classical':
            try:
                # Debug: Check model attributes
                if not hasattr(model, 'models') or not model.models:
                    return "❌ Classical model has no models data"
                if not hasattr(model, 'bpe_model') or not model.bpe_model:
                    return "❌ Classical model has no BPE tokenizer"
                
                # Try generation
                result = model.generate(context, n=4, max_words=max_length//3, temperature=temperature)
                if not result or result.strip() == "":
                    return "⚠️ Classical model generated empty text"
                return result
            except Exception as e:
                import traceback
                return f"❌ Classical model generation failed: {str(e)}\n\nFull error: {traceback.format_exc()}"
        elif model_type == 'neural':
            # Get BPE for neural model
            bpe = model_manager.get_bpe_for_model(selected_model)
            if bpe:
                # Use BPE vocabulary for neural models
                bpe_vocab = sorted(list(bpe.vocab)) if hasattr(bpe, 'vocab') else []
                specials = ['<START>', '<END>', '<UNK>']
                full_vocab = bpe_vocab + [s for s in specials if s not in bpe_vocab]
                v2i = {t: i for i, t in enumerate(full_vocab)}
                i2v = {i: t for t, i in v2i.items()}
                
                ctx_tokens = bpe.encode(context, norm='lower_nopunct')
                if len(ctx_tokens) < 3:
                    ctx_tokens = ['<START>'] * (3 - len(ctx_tokens)) + ctx_tokens
                
                out = list(ctx_tokens)
                
                with torch.no_grad():
                    for _ in range(max_length):
                        ctx_ids = torch.tensor([[v2i.get(t, v2i['<UNK>']) for t in out[-3:]]], device=model_manager.device)
                        logits = model(ctx_ids) / max(1e-6, float(temperature))
                        probs = F.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, 1).item()
                        next_token = i2v[next_id]
                        
                        if next_token == '<END>':
                            break
                        out.append(next_token)
                
                clean = [t for t in out if t not in ('<START>', '<END>', '<UNK>')]
                result = bpe.decode(clean)
            else:
                result = "BPE tokenizer not found for neural model"
        elif model_type == 'gpt':
            # Get BPE for GPT model
            bpe = model_manager.get_bpe_for_model(selected_model)
            if bpe:
                ctx_tokens = bpe.encode(context, norm='lower_nopunct')
                ctx_ids = torch.tensor([ctx_tokens], device=model_manager.device)
                generated = model.generate(ctx_ids, max_new_tokens=max_length, temperature=temperature)
                result = bpe.decode(generated[0].tolist())
            else:
                result = "BPE tokenizer not found for GPT model"
        else:
            result = "Unknown model type"
        
        return result if result else "⚠️ Model generated empty text."
        
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"

# Create simple Gradio interface
with gr.Blocks(title="Shakespeare Language Models") as demo:
    gr.Markdown("# 🎭 Shakespeare Language Model Generator")
    
    # Get available models
    available_models = list(model_manager.models.keys())
    
    with gr.Row():
        with gr.Column():
            selected_model = gr.Dropdown(
                choices=available_models,
                label="Select Model",
                value=available_models[0] if available_models else None
            )
            
            context = gr.Textbox(
                label="Context/Prompt",
                placeholder="to be or not to be",
                lines=3
            )
            
            max_length = gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=5,
                label="Max Length"
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="Temperature"
            )
            
            generate_btn = gr.Button("Generate Text", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=12,
                show_copy_button=True
            )
    
    generate_btn.click(
        fn=generate_text_simple,
        inputs=[selected_model, context, max_length, temperature],
        outputs=[output]
    )

if __name__ == "__main__":
    # Launch with Hugging Face Spaces configuration
    demo.launch(
        server_name="0.0.0.0",  # Required for HF Spaces
        server_port=7860,        # Default HF Spaces port
        share=False,            # Don't create public link
        show_error=True,         # Show errors in UI
        quiet=False,            # Show startup messages
        debug=False             # Disable debug mode for production
    )
