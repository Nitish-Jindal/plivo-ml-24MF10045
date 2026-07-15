import json
import os

print("Scanning corpus for unique characters...")
text = open("../data/train_corpus.txt", encoding="utf-8").read()
chars = sorted(list(set(text)))

vocab = {}
idx = 256  # 0-255 are reserved for raw byte fallbacks

for c in chars:
    b = c.encode("utf-8")
    if len(b) > 1: # It's a multi-byte character (e.g., Devanagari)
        vocab[c] = idx
        idx += 1

vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f)

print(f"Added {len(vocab)} multi-byte characters to vocab.")
print(f"New total vocab size: {256 + len(vocab)}")