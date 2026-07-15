import json
import os

class CharByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.char_to_id = {}
        self.id_to_bytes = {}
        
        vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.char_to_id = json.load(f)
            
            self.vocab_size = 256 + len(self.char_to_id)
            for char, idx in self.char_to_id.items():
                self.id_to_bytes[idx] = char.encode("utf-8")

    def encode(self, text):
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                # Fallback to pure bytes for unseen characters (lossless guarantee)
                ids.extend(list(char.encode("utf-8")))
        return ids

    def decode(self, ids):
        b_array = bytearray()
        for i in ids:
            if i < 256:
                b_array.append(i)
            else:
                b_array.extend(self.id_to_bytes[i])
        return bytes(b_array).decode("utf-8", errors="replace")

def load(path=None):
    return CharByteTokenizer()