<div align="center">

<br/>

```
╔╦╗╦╔╗╔╦  ╦  ╦  ╔╦╗
║║║║║║║║  ║  ║  ║║║
╩ ╩╩╝╚╝╩  ╩═╝╩═╝╩ ╩
```

# Mini LLM

**A character-level Transformer built from scratch with PyTorch — trained, chatted, and served in one file.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

*Train a conversational Transformer in minutes. Chat via terminal or browser.*

<br/>

</div>

---

## What is this?

Mini LLM is a fully self-contained, character-level language model implemented in a single Python file. It trains a GPT-style decoder Transformer from scratch on a curated multi-domain dataset, then exposes the model either as an interactive REPL or a Flask-powered web chatbot — no external API keys, no giant dependencies, no magic.

It's designed to be a **readable, hackable, educational baseline** — the kind of project you read to understand how transformers actually work, then extend however you like.

---

## Architecture

```
Input text
    │
    ▼
┌─────────────────────────┐
│   CharTokenizer         │  character-level vocabulary
│   vocab_size ≈ 87       │  encode / decode / save / load
└───────────┬─────────────┘
            │  token + position embeddings
            ▼
┌─────────────────────────┐
│  Embedding Layer        │  d_model = 128
│  tok_emb + pos_emb      │  max_seq_len = 128
└───────────┬─────────────┘
            │
    ┌───────┴──────┐  × 3 layers
    │              │
    ▼              │
┌──────────────┐   │
│ LayerNorm    │   │
│              │   │
│ CausalSelf   │   │  4 heads, d_head = 32
│  Attention   │   │  causal mask + top-k sampling
│              │   │
│ + residual ──┘   │
└──────┬───────────┘
       │
    ┌──┴──────────────┐
    │ LayerNorm        │
    │ FFN (GELU)       │  d_ff = 256
    │ + residual       │
    └──────┬───────────┘
           │
           ▼
    ┌─────────────┐
    │  LayerNorm  │
    │  LM Head    │  weight-tied with tok_emb
    └─────────────┘
           │
           ▼
    next-token logits
```

**Key design choices:**

- Weight tying between the token embedding and the output head (reduces parameters, improves generalisation)
- Pre-norm (`LN → Attn → residual`) for more stable training
- Cosine annealing LR scheduler with `eta_min = lr × 0.05`
- Gradient clipping at `1.0` to prevent exploding gradients
- `top-k` + temperature sampling at inference

---

## Dataset

The model trains on **`dataset.json`** — a hand-curated v3 corpus of 116 conversational and instructional texts spanning 8 domains:

| Domain | Examples |
|---|---|
| `conversation` | Greetings, casual dialogue, small talk |
| `machine_learning` | Transformers, attention, training loops |
| `coding` | Python snippets, debugging, explanations |
| `science` | Physics, biology, chemistry Q&A |
| `mathematics` | Arithmetic, algebra, proofs |
| `reasoning` | Logic puzzles, step-by-step deductions |
| `storytelling` | Short narratives and creative prompts |
| `instruction_following` | Task completion, instruction chains |

All texts follow a `System / User / Assistant` dialogue format, matching how the model is prompted at inference.

---

## Project Structure

```
my_llm/
├── llm.py                         # Everything: model, training, REPL, Flask server
├── dataset.json                   # Training corpus (116 texts, 8 domains)
├── tokenizer.json                 # Saved character vocabulary
├── mini_llm.pt                    # Pretrained checkpoint
├── chatbot_ui.html                # Web UI for the Flask server
└── mini_llm_interactive_explainer.html   # Visual explainer of the architecture
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch flask flask-cors
```

### 2. Train from scratch + open REPL

```bash
python llm.py
```

This will:
1. Load and tokenize `dataset.json`
2. Train for 2 000 steps (~1–2 min on CPU, seconds on GPU)
3. Save `mini_llm.pt` and `tokenizer.json`
4. Run a few generation samples
5. Drop you into an interactive chat REPL

### 3. Load a saved model + start the web UI

```bash
python llm.py --load --serve
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--load` | `False` | Load `mini_llm.pt` instead of training |
| `--serve` | `False` | Start Flask web server after setup |
| `--port` | `5000` | Port for the web server |

---

## API Reference

When running with `--serve`, two endpoints are available:

### `POST /chat`

Send a message and receive a generated reply.

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is attention?", "temperature": 0.75, "top_k": 40, "max_tokens": 150}'
```

```json
{
  "response": "Attention is a mechanism that allows the model to ...",
  "full": "What is attention? Attention is a mechanism that allows ..."
}
```

**Request body parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `message` | `string` | required | The user's prompt |
| `temperature` | `float` | `0.75` | Sampling temperature (higher = more creative) |
| `top_k` | `int` | `40` | Top-k token filtering |
| `max_tokens` | `int` | `150` | Maximum new characters to generate |

### `GET /status`

Returns model health and stats.

```json
{
  "status": "ok",
  "vocab_size": 87,
  "parameters": 531456,
  "device": "cpu"
}
```

---

## Model Configuration

Defaults used for the included checkpoint:

```python
cfg = dict(
    vocab_size   = 87,      # characters in vocabulary
    d_model      = 128,     # embedding / hidden dimension
    n_heads      = 4,       # attention heads
    d_ff         = 256,     # feed-forward inner dimension
    n_layers     = 3,       # transformer blocks
    max_seq_len  = 128,     # context window
    dropout      = 0.1,
)
# Total parameters: ~531 K
```

Training hyperparameters:

```python
steps      = 2000
seq_len    = 64
batch_size = 16
lr         = 3e-3       # cosine annealed → lr × 0.05
optimizer  = Adam(betas=(0.9, 0.999), weight_decay=1e-4)
```

---

## Extending the Project

**Larger model** — bump `d_model`, `n_heads`, `n_layers`, `d_ff` in `cfg`. A 256-dim / 4-layer model trains well on a single GPU.

**Better tokenizer** — swap `CharTokenizer` for a BPE tokenizer (e.g. `tiktoken`) to dramatically reduce sequence length and improve coherence.

**Bigger dataset** — add more texts to `dataset.json` under the `"texts"` key. The corpus is plain strings; formatting doesn't need to change.

**Streaming responses** — replace the Flask `jsonify` return with a `StreamingResponse` or SSE endpoint and update `chatbot_ui.html` to consume the stream.

**CUDA training** — set `DEVICE = "cuda"` (auto-detected if available) and increase `batch_size` to 64–128 for faster iteration.

---

## Requirements

- Python ≥ 3.10
- `torch` ≥ 2.0
- `flask` + `flask-cors` (only needed for `--serve`)

No other dependencies.

---

## License

MIT — do whatever you want with it. A star is appreciated if this helped you learn something. ⭐

---

<div align="center">

*Built to understand. Made to be broken and reassembled.*

</div>