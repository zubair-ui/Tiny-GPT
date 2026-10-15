from pathlib import Path
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

output_path = data_dir / "input.txt"

response = requests.get(DATA_URL, timeout=30)
response.raise_for_status()

text = response.text
output_path.write_text(text, encoding="utf-8")

print(f"Dataset saved to: {output_path}")
print(f"Characters: {len(text):,}")
print("\nFirst 500 characters:\n")
print(text[:500])