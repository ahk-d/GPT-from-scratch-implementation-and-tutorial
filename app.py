import gradio as gr
import torch
import pickle
import os
import json
import math
import random
import glob
from collections import Counter, defaultdict
import torch.nn as nn
import torch.nn.functional as F

# Load BPE and model utilities
def find_bpe_file():
    """Recursively search for BPE cache file"""
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
            bpe = pickle.load(f)
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
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd=96, n_head=4, n_layer=3, block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([GPTBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, float(temperature))
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

class ModelManager:
    def __init__(self):
        self.bpe = None
        self.vocab = None
        self.v2i = None
        self.i2v = None
        self.classical_models = {}
        self.neural_models = {}
        self.gpt_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()

    def find_model_files(self):
        """Scan for model files with patterns matching Task 2/3/4 outputs"""
        model_files = {
            'classical': [],
            'neural': [],
            'gpt': [],
            'bpe': None
        }
        
        # Find BPE file
        model_files['bpe'] = find_bpe_file()
        
        # Task 2: Classical models (*.pkl files with 'classical' in name)
        classical_patterns = ["*classical*.pkl", "task2/*.pkl"]
        for pattern in classical_patterns:
            files = glob.glob(pattern, recursive=True)
            model_files['classical'].extend(files)
        
        # Task 3: Neural models (*.pt files with 'neural' in name)
        neural_patterns = ["*neural*.pt", "task3/*neural*.pt", "*n[1-4]_lr*.pt"]
        for pattern in neural_patterns:
            files = glob.glob(pattern, recursive=True)
            model_files['neural'].extend(files)
        
        # Task 4: GPT models (*.pt files with 'gpt' in name)
        gpt_patterns = ["*gpt*.pt", "task4/*gpt*.pt", "*gpt*merges*.pt"]
        for pattern in gpt_patterns:
            files = glob.glob(pattern, recursive=True)
            model_files['gpt'].extend(files)
        
        # Remove duplicates
        for key in ['classical', 'neural', 'gpt']:
            model_files[key] = list(set(model_files[key]))
        
        print(f"Found {len(model_files['classical'])} classical model files")
        print(f"Found {len(model_files['neural'])} neural model files")
        print(f"Found {len(model_files['gpt'])} GPT model files")
        print(f"BPE file: {model_files['bpe']}")
        
        return model_files

    def parse_neural_filename(self, filename):
        """Extract n-gram order from Task 3 neural model filename"""
        basename = os.path.basename(filename).lower()
        if 'n1_' in basename or '_1gram' in basename:
            return 1
        elif 'n2_' in basename or '_2gram' in basename:
            return 2
        elif 'n3_' in basename or '_3gram' in basename:
            return 3
        elif 'n4_' in basename or '_4gram' in basename:
            return 4
        return None

    def parse_gpt_filename(self, filename):
        """Extract GPT model size from Task 4 filename"""
        basename = os.path.basename(filename).lower()
        if 'tiny' in basename:
            return 'tiny'
        elif 'small' in basename:
            return 'small'
        elif 'medium' in basename:
            return 'medium'
        elif 'large' in basename:
            return 'large'
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
        
        # Load BPE
        if model_files['bpe']:
            self.bpe = load_cached_bpe_from_path(model_files['bpe'])
        
        if self.bpe is None:
            print("WARNING: No BPE model found. Creating minimal demo BPE.")
            class DemoBPE:
                def __init__(self):
                    self.vocab = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'be', 'thou'])
                def encode(self, text, norm=None):
                    return text.lower().split()[:10]
                def decode(self, tokens):
                    return ' '.join(str(t) for t in tokens)
            self.bpe = DemoBPE()
        
        # Build vocabulary
        base_vocab = sorted(list(self.bpe.vocab)) if hasattr(self.bpe, 'vocab') else ['the', 'and', 'to']
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
                model = NGramModel.load_model(filepath, self.bpe)
                n = self.parse_classical_filename(filepath)
                if n is not None:
                    model_key = f"{n}gram"
                    if model_key not in self.classical_models:
                        self.classical_models[model_key] = model
                        print(f"Loaded classical {n}-gram from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Failed to load classical model {filepath}: {e}")

    def load_neural_models(self, file_list):
        """Load Task 3 neural model checkpoints"""
        for filepath in file_list:
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                
                # Handle Task 3 checkpoint format
                state_dict = checkpoint.get('state', checkpoint)
                cfg = checkpoint.get('cfg', {})
                
                n = self.parse_neural_filename(filepath)
                if n is None:
                    continue
                
                model = NeuralNgramModel(
                    vocab_size=len(self.vocab),
                    n=n,
                    n_embd=cfg.get('n_embd', 256),
                    n_hidden=512,
                    dropout=0.1  # Low for inference
                )
                
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                model_key = f"{n}gram"
                if model_key not in self.neural_models:
                    self.neural_models[model_key] = model
                    print(f"Loaded neural {n}-gram from {os.path.basename(filepath)}")
                
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
                    n_layer = 3
                
                # Infer block size
                block_size = 64
                if 'wpe.weight' in state_dict:
                    block_size = state_dict['wpe.weight'].shape[0]
                
                model = GPTModel(
                    vocab_size=vocab_size,
                    n_embd=n_embd,
                    n_head=n_head,
                    n_layer=n_layer,
                    block_size=block_size,
                    dropout=0.1
                )
                
                model.load_state_dict(state_dict)
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
        
        ctx_tokens = self.bpe.encode(context, norm='lower_nopunct')
        if len(ctx_tokens) < n - 1:
            ctx_tokens = ['<START>'] * (n - 1 - len(ctx_tokens)) + ctx_tokens
        
        out = list(ctx_tokens)
        
        with torch.no_grad():
            for _ in range(max_length):
                if n == 1:
                    ctx_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                else:
                    ctx_ids = torch.tensor([[self.v2i.get(t, self.v2i['<UNK>']) for t in out[-(n-1):]]],
                                         device=self.device)
                
                logits = model(ctx_ids) / max(1e-6, float(temperature))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                next_token = self.i2v[next_id]
                
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

def generate_text_interface(model_type, model_name, context, max_length, temperature):
    """Interface function for Gradio"""
    if not context.strip():
        return "Please enter some context text."
    
    return model_manager.generate_text(model_type, model_name, context, max_length, temperature)

def update_model_choices(model_type):
    """Update model choices based on selected type"""
    if model_type == "Classical N-gram":
        choices = list(model_manager.classical_models.keys()) if model_manager.classical_models else ["No models available"]
        default = "3gram" if "3gram" in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=default)
    elif model_type == "Neural N-gram":
        choices = list(model_manager.neural_models.keys()) if model_manager.neural_models else ["No models available"]
        default = "3gram" if "3gram" in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=default)
    elif model_type == "GPT":
        choices = list(model_manager.gpt_models.keys()) if model_manager.gpt_models else ["No models available"]
        default = "medium" if "medium" in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=default)

# Create Gradio interface
with gr.Blocks(title="Shakespeare Language Models") as demo:
    gr.Markdown("# Shakespeare Language Model Generator")
    gr.Markdown("Generate Shakespearean text using classical n-grams, neural networks, or GPT models!")
    
    # Display loaded models info
    gr.Markdown(f"""
    **Available Models:**
    - **Classical N-grams** (Task 2): {len(model_manager.classical_models)} models
    - **Neural N-grams** (Task 3): {len(model_manager.neural_models)} models  
    - **GPT Models** (Task 4): {len(model_manager.gpt_models)} models
    
    *Models are automatically loaded from checkpoint files in the current directory.*
    """)
    
    with gr.Row():
        with gr.Column():
            model_type = gr.Dropdown(
                choices=["Classical N-gram", "Neural N-gram", "GPT"],
                value="Classical N-gram",
                label="Model Type",
                info="Choose the type of language model"
            )
            
            model_name = gr.Dropdown(
                choices=["No models available"],
                value=None,
                label="Specific Model",
                info="Select a specific model variant"
            )
            
            context = gr.Textbox(
                label="Context/Prompt",
                placeholder="to be or not to be",
                lines=2,
                info="Enter starting text for generation"
            )
            
            with gr.Row():
                max_length = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Max Length",
                    info="Maximum tokens to generate"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Randomness (higher = more creative)"
                )
            
            generate_btn = gr.Button("Generate Text", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=10,
                max_lines=15,
                info="Shakespeare-style text generated by the selected model"
            )
    
    # Update model choices when type changes
    model_type.change(
        fn=update_model_choices,
        inputs=[model_type],
        outputs=[model_name]
    )
    
    # Generate text when button is clicked
    generate_btn.click(
        fn=generate_text_interface,
        inputs=[model_type, model_name, context, max_length, temperature],
        outputs=[output]
    )
    
    # Example prompts for different model types
    gr.Examples(
        examples=[
            ["Classical N-gram", "3gram", "to be or not to be", 50, 0.8],
            ["Neural N-gram", "3gram", "fair is foul and foul is fair", 40, 0.9],
            ["GPT", "medium", "wherefore art thou romeo", 60, 0.7],
            ["Classical N-gram", "2gram", "shall I compare thee", 45, 0.6],
            ["GPT", "small", "now is the winter", 55, 0.8],
        ],
        inputs=[model_type, model_name, context, max_length, temperature],
    )
    
    # Footer with model info
    gr.Markdown("""
    ---
    **Model Information:**
    
    - **Classical N-grams**: Statistical models using Byte-Pair Encoding with add-one smoothing and backoff
    - **Neural N-grams**: Embedding-based neural networks trained on Shakespeare with early stopping
    - **GPT Models**: Transformer-based autoregressive models with causal self-attention
    
    All models are trained on Shakespeare's complete works and use consistent BPE tokenization.
    """)

if __name__ == "__main__":
    demo.launch()