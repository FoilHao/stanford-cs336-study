import os
import regex as re
from typing import BinaryIO
from collections import defaultdict
import multiprocessing as mp

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """ 
    Chunk the file into parts based on the split_special_token.
    Returns a list of chunk boundaries.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundaries
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # Adjust chunk boundaries to respect special tokens
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":  # End of file
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def str_to_bytes_tuple(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in word.encode("utf-8"))

def pre_tokenization_chunk(chunk: str) -> dict[tuple[bytes, ...], int]: 
    pre_token_cnts = defaultdict(int)
    for match in re.finditer(PAT, chunk):
        word = match[0]
        pre_token_cnts[str_to_bytes_tuple(word)] += 1

    return pre_token_cnts

def merge_pre_token_cnts(cnts_list: list[dict[tuple[bytes], int]]) -> dict[tuple[bytes], int]:
    merged_cnts = defaultdict(int)
    for cnts in cnts_list:
        for pair, cnt in cnts.items():
            merged_cnts[pair] += cnt
    return merged_cnts

def pre_tokenization_parallel(
    input_path: str | os.PathLike,
    split_special_token: bytes,
    desired_num_chunks: int,
) -> dict[tuple[bytes], int]:
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)

        with mp.Pool(processes=min(desired_num_chunks, mp.cpu_count())) as pool:
            results = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # split chunk with split_special_token to ensure no token in chunk 
                sub_chunks = re.split(re.escape(split_special_token.decode("utf-8")), chunk)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) > 0:  # 跳过空字符串
                        result = pool.apply_async(pre_tokenization_chunk, args=(sub_chunk,))
                        results.append(result)
            
            # Collect results after processing chunks
            results = [r.get() for r in results]

    pre_token_cnts = merge_pre_token_cnts(results)
    return pre_token_cnts

def bpe_tokenization(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """ Trains a BPE tokenizer and returns the vocabulary and merges """

    # Initialize vocab with 0~256 bytes and special tokens    
    vocab = {i : bytes([i]) for i in range(256)}
    token_id = max(vocab.keys()) + 1
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab.values():
            vocab[token_id] = token_bytes
            token_id += 1
    
    # Pre-tokenization
    split_special_token = "<|endoftext|>".encode("utf-8")
    desired_num_chunks = 64
    pre_token_cnts = pre_tokenization_parallel(input_path, split_special_token, desired_num_chunks)
    
    # Merge iteratively to create BPE merges
    merges = []

    pair_cnts = defaultdict(int)
    for pre_token, cnt in pre_token_cnts.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])
            pair_cnts[pair] += cnt

    while len(vocab) < vocab_size:        
        top_pair = max(pair_cnts, key=lambda k: (pair_cnts[k], k))
        
        merges.append(top_pair)
        new_token = top_pair[0] + top_pair[1]
        vocab[token_id] = new_token
        token_id += 1
        
        # Update pre_token_cnts and pair_cnts with new merged token
        affected_pairs = set()
        new_pair_cnts = defaultdict(int)
        updated_pre_token_cnts = []
        for pre_token, cnt in pre_token_cnts.items():
            indices = [i for i in range(len(pre_token) - 1) if pre_token[i:i + 2] == top_pair]
            if not indices:
                continue
            # update pre_token_cnts
            new_pre_token = []
            i = 0
            while i < len(pre_token):
                if i in indices:
                    new_pre_token.append(new_token)
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    i += 1
            updated_pre_token_cnts.append((pre_token, tuple(new_pre_token), cnt))

            # update pair_cnts
            for i in range(len(pre_token) - 1):
                old_pair = (pre_token[i], pre_token[i + 1])
                affected_pairs.add(old_pair)
                pair_cnts[old_pair] -= cnt
            
            for i in range(len(new_pre_token) - 1):
                new_pair = (new_pre_token[i], new_pre_token[i+1])
                new_pair_cnts[new_pair] += cnt

        # update affected pre_token_cnts only
        for pre_token, new_pre_token, cnt in updated_pre_token_cnts:
            if pre_token != new_pre_token:
                pre_token_cnts[new_pre_token] = pre_token_cnts.get(new_pre_token, 0) + cnt
                del pre_token_cnts[pre_token]
        
        # update affected pair_cnts only
        for old_pair in affected_pairs:
            if pair_cnts[old_pair] <= 0:
                del pair_cnts[old_pair]
        for pair, cnt in new_pair_cnts.items():
            pair_cnts[pair] += cnt

    return vocab, merges