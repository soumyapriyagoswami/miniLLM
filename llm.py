"""
Mini LLM Chatbot — PyTorch Edition
====================================
Character-level Transformer trained from scratch with PyTorch.
Includes:
  - Richer dataset (dataset.json)
  - CharTokenizer
  - Transformer decoder (multi-head attention, FFN, LayerNorm, residuals)
  - Training loop with Adam + LR scheduler
  - Model save / load
  - Interactive chatbot REPL
  - Flask API server for the web UI  (run with --serve)

Usage:
    python llm.py                  # train + interactive REPL
    python llm.py --serve          # train + start web server
    python llm.py --load --serve   # load saved model + start web server
"""

import json, math, random, argparse, os, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH  = "mini_llm.pt"
DATASET_PATH = "dataset.json"

# ─────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────
def load_corpus(path: str) -> str:
    with open(path) as f:
        data = json.load(f)
    corpus = " ".join(data["texts"])
    print(f"[Dataset] {len(data['texts'])} texts | {len(corpus)} chars")
    return corpus

# ─────────────────────────────────────────
# 2. Tokenizer
# ─────────────────────────────────────────
class CharTokenizer:
    def __init__(self, corpus: str):
        self.vocab   = sorted(set(corpus))
        self.stoi    = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos    = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)
        print(f"[Tokenizer] Vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos.get(i, "?") for i in ids)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            obj = json.load(f)
        tok = cls.__new__(cls)
        tok.vocab = obj["vocab"]
        tok.stoi  = {ch: i for i, ch in enumerate(tok.vocab)}
        tok.itos  = {i: ch for ch, i in tok.stoi.items()}
        tok.vocab_size = len(tok.vocab)
        return tok

# ─────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        QKV = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        QKV = QKV.permute(2, 0, 3, 1, 4)   # (3, B, H, T, dh)
        Q, K, V = QKV[0], QKV[1], QKV[2]

        scale  = math.sqrt(self.d_head)
        scores = (Q @ K.transpose(-2, -1)) / scale   # (B, H, T, T)

        # causal mask
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = self.drop(F.softmax(scores, dim=-1))
        ctx  = (attn @ V).transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        return self.out(ctx)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 256, n_layers: int = 3, max_seq_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))       # (B, T, V)

    @torch.no_grad()
    def generate(self, tokenizer: CharTokenizer, prompt: str,
                 max_new: int = 150, temperature: float = 0.8,
                 top_k: int = 40) -> str:
        self.eval()
        tokens = tokenizer.encode(prompt)
        idx    = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        for _ in range(max_new):
            idx_c  = idx[:, -self.max_seq_len:]
            logits = self(idx_c)[:, -1, :]        # (1, V)
            logits = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, next_t], dim=1)
        return tokenizer.decode(idx[0].tolist())

# ─────────────────────────────────────────
# 4. Training helpers
# ─────────────────────────────────────────
def get_batch(data: torch.Tensor, seq_len: int, batch_size: int = 16):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([data[i:i+seq_len]   for i in ix])
    y  = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def train(model: MiniLLM, tokenizer: CharTokenizer, data: torch.Tensor,
          steps: int = 2000, seq_len: int = 64, batch_size: int = 16,
          lr: float = 3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)
    model.train()
    losses = []
    print(f"\n[Training] steps={steps} seq_len={seq_len} batch={batch_size} device={DEVICE}")
    for step in range(1, steps + 1):
        x, y = get_batch(data, seq_len, batch_size)
        logits = model(x)                          # (B, T, V)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % 200 == 0 or step == 1:
            avg = sum(losses[-200:]) / len(losses[-200:])
            print(f"  Step {step:5d}/{steps} | loss={loss.item():.4f} | avg={avg:.4f} | lr={scheduler.get_last_lr()[0]:.5f}")
    print(f"[Training] Done. Final avg loss: {sum(losses[-200:]) / len(losses[-200:]):.4f}")
    return losses


def save_model(model: MiniLLM, tokenizer: CharTokenizer, cfg: dict):
    torch.save({"model": model.state_dict(), "cfg": cfg}, MODEL_PATH)
    tokenizer.save("tokenizer.json")
    print(f"[Save] Model → {MODEL_PATH}  |  Tokenizer → tokenizer.json")


def load_model(device: str = DEVICE):
    ckpt      = torch.load(MODEL_PATH, map_location=device)
    cfg       = ckpt["cfg"]
    tokenizer = CharTokenizer.load("tokenizer.json")
    model     = MiniLLM(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Load] Model loaded from {MODEL_PATH}")
    return model, tokenizer

# ─────────────────────────────────────────
# 5. REPL chatbot
# ─────────────────────────────────────────
def chat_repl(model: MiniLLM, tokenizer: CharTokenizer):
    print("\n" + "="*55)
    print("  Mini LLM Chatbot — type 'quit' to exit")
    print("="*55)
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not prompt or prompt.lower() in ("quit", "exit", "bye"):
            print("Bye!"); break
        response = model.generate(tokenizer, prompt + " ", max_new=150, temperature=0.75, top_k=40)
        # strip the prompt from the output for cleaner display
        reply = response[len(prompt):].strip()
        print(f"Bot: {reply}")

# ─────────────────────────────────────────
# 6. Flask web server (optional)
# ─────────────────────────────────────────
def start_server(model: MiniLLM, tokenizer: CharTokenizer, port: int = 5000):
    try:
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("[Server] Installing flask and flask-cors...")
        os.system("pip install flask flask-cors --break-system-packages -q")
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS

    app = Flask(__name__, static_folder=".")
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(".", "chatbot_ui.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        data   = request.json or {}
        prompt = data.get("message", "").strip()
        temp   = float(data.get("temperature", 0.75))
        top_k  = int(data.get("top_k", 40))
        max_t  = int(data.get("max_tokens", 150))
        if not prompt:
            return jsonify({"error": "empty message"}), 400
        response = model.generate(tokenizer, prompt + " ", max_new=max_t,
                                  temperature=temp, top_k=top_k)
        reply = response[len(prompt):].strip()
        return jsonify({"response": reply, "full": response})

    @app.route("/status")
    def status():
        total = sum(p.numel() for p in model.parameters())
        return jsonify({"status": "ok", "vocab_size": tokenizer.vocab_size,
                        "parameters": total, "device": DEVICE})

    print(f"\n[Server] Starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

# ─────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Start Flask web server")
    parser.add_argument("--load",  action="store_true", help="Load saved model instead of training")
    parser.add_argument("--port",  type=int, default=5000)
    args = parser.parse_args()

    if args.load and Path(MODEL_PATH).exists():
        model, tokenizer = load_model()
    else:
        corpus    = load_corpus(DATASET_PATH)
        tokenizer = CharTokenizer(corpus)
        data      = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)

        cfg = dict(vocab_size=tokenizer.vocab_size, d_model=128, n_heads=4,
                   d_ff=256, n_layers=3, max_seq_len=128, dropout=0.1)
        model = MiniLLM(**cfg).to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Model] Parameters: {total:,}  |  Device: {DEVICE}")

        train(model, tokenizer, data, steps=2000, seq_len=64, batch_size=16, lr=3e-3)
        save_model(model, tokenizer, cfg)
        model.eval()

    # Quick generation test
    print("\n── Generation samples ──")
    for prompt in ["What is attention", "Hello how are", "The transformer"]:
        out = model.generate(tokenizer, prompt, max_new=80, temperature=0.7)
        print(f"  [{prompt}] → {out[len(prompt):][:80]}")

    if args.serve:
        start_server(model, tokenizer, port=args.port)
    else:
        chat_repl(model, tokenizer)