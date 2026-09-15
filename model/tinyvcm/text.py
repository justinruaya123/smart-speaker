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
