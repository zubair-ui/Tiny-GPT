import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------
# 1. Configuration
# -------------------------
torch.manual_seed(1337)

device = "cpu"

batch_size = 16
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
eval_iters = 20

n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1

# -------------------------
# 2. Load and encode data
# -------------------------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# -------------------------
# 3. Batch generator
# -------------------------
def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


# -------------------------
# 4. Evaluate loss
# -------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results = {}

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()
    return results


# -------------------------
# 5. Self-attention head
# -------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1)
        weights = weights * (k.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        output = weights @ v

        return output


# -------------------------
# 6. Multi-head attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.dropout(self.proj(output))


# -------------------------
# 7. Feed-forward network
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# 8. Transformer block
# -------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
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


# -------------------------
# 9. Tiny GPT language model
# -------------------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.final_ln(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits.view(B, T, vocab_size), loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx


# -------------------------
# 10. Train the model
# -------------------------
model = TinyGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("Vocabulary size:", vocab_size)
print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)

# Evaluate initial loss
initial_losses = estimate_loss(model)
print("Initial losses:", initial_losses)

# Training loop
for step in range(max_iters):

    # Evaluate periodically
    if step % eval_interval == 0:
        losses = estimate_loss(model)

        print(
            f"Step {step:4d}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    # Sample a training batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Reset old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

# Final evaluation
final_losses = estimate_loss(model)
print("\nFinal losses:", final_losses)

# Save the trained model
torch.save(model.state_dict(), "tiny_gpt.pth")
print("Model saved to tiny_gpt.pth")

