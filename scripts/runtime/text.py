"""Small runtime text helpers; importing these never loads PyTorch."""
VOCAB = ["<blank>"] + list(" abcdefghijklmnopqrstuvwxyz'")


def decode_greedy(indices):
    previous = 0
    output = []
    for raw in indices:
        index = int(raw)
        if index != 0 and index != previous:
            output.append(VOCAB[index])
        previous = index
    return "".join(output).strip()

import re
import unicodedata

def normalize_text(text):
    text = text.lower().replace("’", "'")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", text)).strip()
