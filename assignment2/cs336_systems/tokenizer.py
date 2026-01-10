import regex as re
from typing import Iterator, Iterable
import json
from transformers import GPT2Tokenizer
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]  # remove empty strings

def split_to_words(text):
    "Split text into words."
    return PAT.findall(text)

def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)

def apply_merges(word_bytes, merges_set, vocab_to_id):
    word_bytes = list(word_bytes)
    
    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        # Apply best merge
        word_bytes = (
            word_bytes[:best_pair_idx]
            + [merged]
            + word_bytes[best_pair_idx + 2:]
        )

    return tuple(word_bytes)

def encode_merged(text,merges,vocab_to_id):
    word_list = split_to_words(text)
    tokens=[]
    for word in word_list:
        word_bytes=word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes,merges,vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens

class Tokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer with support for special tokens.
    This tokenizer converts text into sequences of token IDs using a vocabulary and merge rules.
    It supports special tokens that should not be merged during encoding.
    Attributes:
        vocab (dict[int, bytes]): Mapping from token IDs to byte sequences.
        merges (set[tuple[bytes, bytes]]): Set of valid merge pairs for BPE.
        special_tokens (list[str]): List of special token strings (e.g., ["<|endoftext|>"]).
        special_tokens_bytes (list[bytes]): UTF-8 encoded versions of special tokens.
        vocab_to_id (dict[bytes, int]): Reverse mapping from byte sequences to token IDs.
    Args:
        vocab (dict[int, bytes]): Dictionary mapping token IDs to byte sequences.
        merges (list[tuple[bytes, bytes]]): List of byte pair merge rules.
        special_tokens (list[str], optional): List of special tokens to preserve during tokenization.
    Methods:
        from_files(vocab_filepath, merges_filepath, special_tokens=None):
            Class method to create a Tokenizer from vocabulary and merges files.
            Args:
                vocab_filepath (str): Path to vocabulary file (tab-separated: id, byte representation).
                merges_filepath (str): Path to merges file (tab-separated: left bytes, right bytes).
                special_tokens (list[str], optional): List of special tokens.
            Returns:
                Tokenizer: A new Tokenizer instance.
        encode(text):
            Encode a string into a list of token IDs.
            Args:
                text (str): Input text to tokenize.
            Returns:
                list[int]: List of token IDs.
        encode_iterable(iterable):
            Lazily encode an iterable of strings into token IDs.
            Args:
                iterable (Iterable[str]): An iterable of text chunks (e.g., file handle).
            Yields:
                int: Token IDs one at a time.
        decode(ids):
            Decode a sequence of token IDs back into text.
            Args:
                ids (list[int]): List of token IDs.
            Returns:
                str: Decoded text (invalid UTF-8 sequences replaced with � character).
    """ 
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode('utf-8') for i in self.special_tokens]
        

        self.vocab_to_id={v:k for k,v in vocab.items()}

        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id

    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Loads from the clean Hex text format"""
        vocab = {}
        with open(f"{vocab_filepath}", 'r') as f:
            for line in f:
                idx, hx = line.strip().split('\t')
                vocab[int(idx)] = bytes.fromhex(hx)
        
        merges = []
        with open(f"{merges_filepath}", 'r') as f:
            for line in f:
                p1, p2 = line.strip().split('\t')
                merges.append((bytes.fromhex(p1), bytes.fromhex(p2)))
                
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')
    
if __name__ == "__main__":
    # Example usage
    tokenizer = Tokenizer.from_files('train/tinystories_bpe_vocab.txt', 'train/tinystories_bpe_merges.txt', special_tokens=["<|endoftext|>"])
    text = "Hello, world!Fuck you!<|endoftext|>"
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)
    # 
    eos_token_id = tokenizer.vocab_to_id[b"<|endoftext|>"]
    print("EOS Token ID:", eos_token_id)