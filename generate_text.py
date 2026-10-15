
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "input.txt"
VOCAB_PATH = PROJECT_DIR / "data" / "vocab.json"

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "models"
    / "baseline_model.pth"
)

OUTPUT_DIR = PROJECT_DIR / "generated_output"
OUTPUT_PATH = OUTPUT_DIR / "generated_baseline.txt"

BLOCK_SIZE = 64
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
PROMPT = "ROMEO:"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Vocabulary
# --------------------------------------------------


with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

if isinstance(vocab_data, list):
    # List where each position is a character's token ID
    itos = vocab_data
    stoi = {char: index for index, char in enumerate(itos)}

elif "itos" in vocab_data and "stoi" in vocab_data:
    itos = vocab_data["itos"]
    stoi = vocab_data["stoi"]

else:
    # Dictionary mapping characters to token IDs
    stoi = vocab_data
    itos = {int(index): char for char, index in stoi.items()}

vocab_size = len(stoi)



# --------------------------------------------------
# Model
# --------------------------------------------------

class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(64, head_size, bias=False)
        self.query = nn.Linear(64, head_size, bias=False)
        self.value = nn.Linear(64, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, 64)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd=64, n_head=4):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=64,
        block_size=64,
        n_head=4,
        n_layer=2,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.final_ln(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# --------------------------------------------------
# Generate and save baseline sample
# --------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=64,
        block_size=BLOCK_SIZE,
        n_head=4,
        n_layer=2,
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # Supports either a raw state_dict or a wrapped checkpoint.
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    # Encode the prompt
    prompt_ids = [stoi[char] for char in PROMPT]
    context = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    # Fixed seed makes the sample reproducible on the same setup.
    torch.manual_seed(42)

    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )[0].tolist()

    generated_text = "".join(itos[token_id] for token_id in generated_ids)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Prompt: {PROMPT!r}")
    print(f"New tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Saved generated text to: {OUTPUT_PATH}")
    print("\n--- Generated sample ---\n")
    print(generated_text)


if __name__ == "__main__":
    main()
