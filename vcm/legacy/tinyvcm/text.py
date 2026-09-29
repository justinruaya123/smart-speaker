"""Small runtime text helpers; importing these never loads PyTorch."""



import re
import unicodedata

def normalize_text(text):
    text = text.lower().replace("’", "'")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", text)).strip()
