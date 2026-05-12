class CharTokenizer:
    def __init__(self, corpus: str):
        # Step 1: find every unique character that appears in the corpus.
        # `sorted` makes the mapping deterministic — without it, the IDs would
        # change every time we create a new tokenizer from the same text.
        unique_chars = sorted(set(corpus))

        # Step 2: build two lookup tables.
        # `stoi` (string-to-integer): given a character, return its ID.
        # `itos` (integer-to-string): given an ID, return its character.
        # These are just Python dicts — very fast O(1) lookups.
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Step 3: record the vocabulary size.
        # The embedding matrix we build later will need one row per token,
        # so it needs to know how many unique tokens exist.
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> list[int]:
        # Convert each character to its integer ID using `stoi`.
        # The model will receive this list of integers — never the raw text.
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        # Reverse: convert each integer ID back to its character using `itos`,
        # then join the characters into a string.
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)

    # --- Vocabulary ---
    print("=== Step 1: Building the vocabulary ===")
    print(f"Input sentence: {sentence!r}")
    print(f"Unique characters (sorted): {sorted(tok.stoi.keys())}")
    print(f"Vocabulary size: {tok.vocab_size}")
    print()

    # --- stoi: character -> integer ---
    print("=== Step 2: stoi — character to integer ID ===")
    print("Each unique character gets a unique integer label (sorted alphabetically):")
    for ch, idx in tok.stoi.items():
        print(f"  {ch!r:4s} -> {idx}")
    print()

    # --- itos: integer -> character ---
    print("=== Step 3: itos — integer ID back to character ===")
    print("The reverse mapping lets us decode predictions back to text:")
    for idx, ch in tok.itos.items():
        print(f"  {idx} -> {ch!r}")
    print()

    # --- encode ---
    print("=== Step 4: encode — text to list of integers ===")
    ids = tok.encode(sentence)
    print(f"encode({sentence!r})")
    print(f"  -> {ids}")
    print("Notice: the model will only ever see these integers, never the raw characters.")
    print()

    # --- decode ---
    print("=== Step 5: decode — integers back to text ===")
    recovered = tok.decode(ids)
    print(f"decode({ids})")
    print(f"  -> {recovered!r}")
    print(f"Round-trip check (encode then decode == original): {recovered == sentence}")
