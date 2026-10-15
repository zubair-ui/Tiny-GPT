
from pathlib import Path
import json
import torch

# 1. Load the dataset
data_path = Path("data/input.txt")
text = data_path.read_text(encoding="utf-8")

# 2. Build a character vocabulary
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# 3. Encode the text into integer IDs
encoded_text = [stoi[ch] for ch in text]
data = torch.tensor(encoded_text, dtype=torch.long)

# 4. Split into training and validation sets
split_index = int(0.90 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# 5. Save vocabulary for later use
vocab_path = Path("data/vocab.json")
vocab_path.write_text(
    json.dumps(chars, ensure_ascii=False),
    encoding="utf-8"
)

# 6. Display a summary
print(f"Total characters: {len(text):,}")
print(f"Vocabulary size: {vocab_size}")
print(f"Training tokens: {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")
print(f"\nOriginal text: {text[:100]!r}")
print(f"Encoded IDs:   {data[:30].tolist()}")
print(f"Decoded text:  {''.join(itos[i] for i in data[:100].tolist())!r}")
print(f"\nVocabulary saved to: {vocab_path}")
