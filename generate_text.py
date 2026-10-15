from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1. Configuration
# -------------------------
device = "cpu"

batch_size = 16
block_size = 64
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1

# -------------------------
# 2. Load vocabulary
# -------------------------
text = Path("data/input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join(itos[i] for i in tokens)

# -------------------------
# 3. Transformer components
# -------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_ln(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=500):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# -------------------------
# 4. Load trained weights
# -------------------------
model = TinyGPT().to(device)
model.load_state_dict(
    torch.load("tiny_gpt.pth", map_location=device)
)
model.eval()

print("Trained model loaded successfully!")

# -------------------------
# 5. Generate Shakespeare-like text
# -------------------------
prompt = "ROMEO:"
context = torch.tensor(
    [encode(prompt)], dtype=torch.long, device=device
)

generated = model.generate(context, max_new_tokens=500)
output = decode(generated[0].tolist())

print("\n--- Generated Text ---\n")
print(output)

Path("generated_output.txt").write_text(
    output, encoding="utf-8"
)
print("\nSaved generated text to generated_output.txt")