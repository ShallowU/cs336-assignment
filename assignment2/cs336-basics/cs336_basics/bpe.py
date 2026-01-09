import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from itertools import pairwise
import yaml
import time
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_by_special(text, special_tokens, drop_special=True):
    """
    Split text by special tokens.

    Args:
        text (str): The input text to split.
        special_tokens (list): A list of special token strings to split by.
        drop_special (bool, optional): If True, special tokens are removed from the result.
            If False, special tokens are kept as separate chunks. Defaults to True.

    Returns:
        list: A list of text chunks. If drop_special is True, only the text between
            special tokens is returned. If False, both text chunks and special tokens
            are included in the result.

    Examples:
        >>> split_by_special("Hello<|end|>World", ["<|end|>"], drop_special=True)
        ['Hello', 'World']
        
        >>> split_by_special("Hello<|end|>World", ["<|end|>"], drop_special=False)
        ['Hello', '<|end|>', 'World']
    """
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]  # remove empty strings

def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)

def count_word(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes)>=2:
            word_cnt[word_bytes]+=1
    return word_cnt

def merge_dicts(dicts):
    """
    Merge multiple dictionaries by summing values for common keys.

    Args:
        dicts: An iterable of dictionaries with integer values.

    Returns:
        A defaultdict(int) containing all keys from input dictionaries,
        with values being the sum of values across all dictionaries for each key.

    Example:
        >>> dict1 = {'a': 1, 'b': 2}
        >>> dict2 = {'b': 3, 'c': 4}
        >>> result = merge_dicts([dict1, dict2])
        >>> dict(result)
        {'a': 1, 'b': 5, 'c': 4}
    """
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged

def count_pair(word_cnt):
    """
    Count the frequency of consecutive byte pairs in a collection of words.

    Args:
        word_cnt: A dictionary mapping word representations (as tuples/sequences of bytes) 
                 to their occurrence counts.

    Returns:
        A dictionary mapping byte pairs (tuples of two consecutive bytes) to their 
        total frequency across all words, weighted by each word's count.

    Example:
        >>> word_cnt = {(1, 2, 3): 2, (2, 3, 4): 1}
        >>> result = count_pair(word_cnt)
        >>> result[(1, 2)]
        2
        >>> result[(2, 3)]
        3
    """
    pair_cnt = defaultdict(int)
    for word_bytes,cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1],word_bytes[1:]):
            pair_cnt[pair]+=cnt
    return pair_cnt

def get_max_pair(pair_cnt):
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))  # lexicographic tie-breaker
    return max_pair


def get_basic_vocab(special_tokens):
    vocab={token:bytes([token]) for token in range(256)}

    for i,token in enumerate(special_tokens):
        token_id = 256+i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def apply_merge(word_bytes,merge):
    merged = merge[0]+merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        # Check for match
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i+1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

def update_cnt(word_cnt,pair_cnt, merge_pair):

    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt) # copy with defaultdict

    for word_bytes,cnt in word_cnt.items():

        #----------for word cnt ---------------

        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Keep the original count if the merge not appear in the key
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes]+=cnt
            continue

        # Use updated key if merge appear
        new_word = apply_merge(word_bytes,merge_pair)
        new_word_cnt[new_word]+=cnt

        #--------for pair cnt ----------------

        # Decrease all old pair counts
        for pair in old_pairs:
            new_pair_cnt[pair]-=cnt
            if new_pair_cnt[pair] ==0:
                del new_pair_cnt[pair]

        # Count new pairs in the new word
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt,new_pair_cnt

def update_cnt_fast(word_cnt, pair_cnt, merge_pair):
    a, b = merge_pair
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt)  # copy

    for wbytes, cnt in word_cnt.items():
        # cheap presence check (no list/zip)
        has = False
        i, n = 0, len(wbytes) - 1
        while i < n:
            if wbytes[i] == a and wbytes[i+1] == b:
                has = True
                break
            i += 1
        if not has:
            new_word_cnt[wbytes] += cnt
            continue

        # decrement old pairs (iterator, no list)
        for p in pairwise(wbytes):
            v = new_pair_cnt[p] - cnt
            if v: new_pair_cnt[p] = v
            else: new_pair_cnt.pop(p, None)

        # merge & add new pairs
        new_w = apply_merge(wbytes, merge_pair)
        new_word_cnt[new_w] += cnt
        for p in pairwise(new_w):
            new_pair_cnt[p] += cnt

    return new_word_cnt, new_pair_cnt

def train_bpe(input_path,vocab_size,special_tokens):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the input text.

    This function implements the BPE algorithm to learn a vocabulary of a specified size
    by iteratively merging the most frequent pairs of tokens in the training data.

    Args:
        input_path (str): Path to the input text file to train on.
        vocab_size (int): The desired final vocabulary size, including special tokens.
        special_tokens (list): List of special tokens to be preserved during tokenization
            (e.g., ['<PAD>', '<UNK>', '<EOS>']).

    Returns:
        tuple: A tuple containing:
            - vocab (dict): A dictionary mapping token IDs to token strings. The first
              entries are basic vocabulary (bytes and special tokens), followed by
              merged tokens.
            - merges (list): A list of tuples representing the merge operations in order,
              where each tuple contains the pair of tokens that were merged.

    Process:
        1. Reads text from the input file
        2. Splits text by special tokens to preserve them
        3. Counts word frequencies in the corpus
        4. Counts pairs of adjacent tokens
        5. Iteratively merges the most frequent pair until vocab_size is reached
        6. Returns the final vocabulary and merge operations

    Note:
        - Uses parallel processing for large chunk counts (>= 4) to improve performance
        - The base vocabulary size is determined by basic tokens and special tokens
        - Number of merges = vocab_size - base_vocab_size
    """

    text = read_text(input_path)
    chunks = split_by_special(text,special_tokens)

    # Only parallelize if chunk count is big enough
    if len(chunks) < 4: word_dicts = list(map(count_word, chunks))
    else: word_dicts = process_map(count_word, chunks, chunksize=100)

    word_cnt = merge_dicts(word_dicts)
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges=vocab_size-base_vocab_size

    merges = []
    for i in range(n_merges):
        max_pair = get_max_pair(pair_cnt)
        vocab[base_vocab_size+i] = max_pair[0]+max_pair[1]
        merges.append(max_pair)
        word_cnt, pair_cnt = update_cnt_fast(word_cnt,pair_cnt,max_pair)
    return vocab, merges

def save_tokenizer_yaml(vocab, merges, fname):
    "Save vocab and merges to a YAML file with UTF-8 decoding for readability."
    # Convert bytes → string for readability
    vocab_serializable = {}
    for k, v in vocab.items():
        if isinstance(v, bytes):
            vocab_serializable[k] = v.decode("utf-8", errors="replace")
        else:
            vocab_serializable[k] = v
    
    merges_serializable = []
    for a, b in merges:
        a_str = a.decode("utf-8", errors="replace") if isinstance(a, bytes) else a
        b_str = b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        merges_serializable.append([a_str, b_str])  # 用列表而非元组
    
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(
            {"vocab": vocab_serializable, "merges": merges_serializable},
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )

def load_tokenizer_yaml(fname):
    "Load vocab and merges from a YAML file, converting strings back to bytes."
    with open(fname, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    vocab_loaded = {
        int(k): v.encode("utf-8") if isinstance(v, str) else v
        for k, v in data["vocab"].items()
    }
    merges_loaded = [
        (a.encode("utf-8") if isinstance(a, str) else a, 
         b.encode("utf-8") if isinstance(b, str) else b)
        for a, b in data["merges"]
    ]
    return vocab_loaded, merges_loaded

if __name__ == "__main__":
    time_start = time.time()
    vocab, merges = train_bpe(
            # input_path='../data/TinyStoriesV2-GPT4-valid.txt',
            input_path="../data/owt_valid.txt",
            vocab_size=10_000,
            special_tokens=["<|endoftext|>"],
        )
    time_end = time.time()
    print(f"Training BPE took {time_end - time_start:.2f} seconds.")
    # save_tokenizer_yaml(vocab, merges, 'TinyStories_train_tokenizer.yaml')
    with open("owt_valid_vocab.txt", "w", encoding="utf-8") as f:
        for i, tok in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open("owt_valid_merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left}\t{right}\n")