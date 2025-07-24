import os
import time
import regex as re
from typing import BinaryIO, Union, List, Tuple, Dict
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

    mini_chunk_size = 128  # Read ahead by 4k bytes at a time

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
        word = match.group()
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
                    if len(sub_chunk.strip()) > 0:
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
        if not pair_cnts:
            break     
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

def profile_bpe_tokenization(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str],
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer and return vocabulary/merges with detailed timing metrics
    """
    timings = {}
    def record_time(step_name):
        nonlocal timings
        current_time = time.time()
        if 'start_time' not in timings:
            timings['start_time'] = current_time
        if 'last_time' not in timings:
            timings['last_time'] = current_time
        
        timings[step_name] = {
            'start': timings['last_time'],
            'end': current_time,
            'duration': current_time - timings['last_time']
        }
        timings['last_time'] = current_time

    record_time('total_start')

    record_time('vocab_initialization_start')
    vocab = {i: bytes([i]) for i in range(256)}
    token_id = max(vocab.keys()) + 1
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab.values():
            vocab[token_id] = token_bytes
            token_id += 1
    record_time('vocab_initialization_end')

    record_time('pre_tokenization_start')
    split_special_token = "<|endoftext|>".encode("utf-8")
    desired_num_chunks = 1024
    pre_token_cnts = pre_tokenization_parallel(input_path, split_special_token, desired_num_chunks)
    record_time('pre_tokenization_end')

    record_time('pair_counts_initialization_start')
    pair_cnts = defaultdict(int)
    for pre_token, cnt in pre_token_cnts.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pair_cnts[pair] += cnt
    record_time('pair_counts_initialization_end')

    record_time('merge_iteration_start')
    merges = []
    merge_timings = {
        'top_pair_selection': 0,
        'pre_token_update': 0,
        'pair_counts_update': 0,
        'dict_operations': 0
    }
    
    while len(vocab) < vocab_size:
        if not pair_cnts:
            break
        
        start = time.time()
        top_pair = max(pair_cnts, key=lambda k: (pair_cnts[k], k))
        merge_timings['top_pair_selection'] += time.time() - start

        merges.append(top_pair)
        new_token = top_pair[0] + top_pair[1]
        vocab[token_id] = new_token
        token_id += 1

        start = time.time()
        affected_pairs = set()
        new_pair_cnts = defaultdict(int)
        updated_pre_token_cnts = []
        for pre_token, cnt in pre_token_cnts.items():
            indices = [i for i in range(len(pre_token) - 1) if pre_token[i:i + 2] == top_pair]
            if not indices:
                continue
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
        merge_timings['pre_token_update'] += time.time() - start

        start = time.time()

        for pre_token, new_pre_token, cnt in updated_pre_token_cnts:
            for i in range(len(pre_token) - 1):
                old_pair = (pre_token[i], pre_token[i + 1])
                affected_pairs.add(old_pair)
                pair_cnts[old_pair] -= cnt
            
            for i in range(len(new_pre_token) - 1):
                new_pair = (new_pre_token[i], new_pre_token[i + 1])
                new_pair_cnts[new_pair] += cnt
        merge_timings['pair_counts_update'] += time.time() - start

        start = time.time()
        for pre_token, new_pre_token, cnt in updated_pre_token_cnts:
            if pre_token != new_pre_token:
                pre_token_cnts[new_pre_token] = pre_token_cnts.get(new_pre_token, 0) + cnt
                del pre_token_cnts[pre_token]

        for old_pair in affected_pairs:
            if pair_cnts[old_pair] <= 0:
                del pair_cnts[old_pair]
        for pair, cnt in new_pair_cnts.items():
            pair_cnts[pair] += cnt
        merge_timings['dict_operations'] += time.time() - start

    record_time('merge_iteration_end')
    record_time('total_end')

    def format_time(seconds):
        return f"{seconds:.4f}s"
    
    print("\n===== BPE Tokenization Timings =====")
    print(f"Total time: {format_time(timings['total_end']['duration'])}")
    print("\n1. Vocabulary Initialization:")
    print(f"   - Time: {format_time(timings['vocab_initialization_end']['duration'])}")
    print("\n2. Pre-tokenization:")
    print(f"   - Time: {format_time(timings['pre_tokenization_end']['duration'])}")
    print("\n3. Pair Counts Initialization:")
    print(f"   - Time: {format_time(timings['pair_counts_initialization_end']['duration'])}")
    print("\n4. Merge Iteration (Total):")
    print(f"   - Time: {format_time(timings['merge_iteration_end']['duration'])}")
    print("   - Breakdown:")
    print(f"     * Top Pair Selection: {format_time(merge_timings['top_pair_selection'])}")
    print(f"     * Pre-token Update: {format_time(merge_timings['pre_token_update'])}")
    print(f"     * Pair Counts Update: {format_time(merge_timings['pair_counts_update'])}")
    print(f"     * Dictionary Operations: {format_time(merge_timings['dict_operations'])}")
    print("\n=====================================")

    return vocab, merges

def save_vocab_and_merges(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    save_dir: Union[str, os.PathLike],
    vocab_filename: str = "vocab.txt",
    merges_filename: str = "merges.txt",
    readable: bool = True,
) -> None:
    """
    Save vocabulary and merges to files with optional human-readable formatting
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Helper function to convert bytes to readable string
    def bytes_to_readable(b: bytes) -> str:
        """Convert bytes to string with special handling for non-printable characters"""
        try:
            # Try regular UTF-8 decoding first
            s = b.decode('utf-8')
            # Replace non-printable characters with hex representations
            return ''.join(c if c.isprintable() else f'<0x{c.encode("utf-8").hex().upper()}>' for c in s)
        except UnicodeDecodeError:
            # If decoding fails, use hex representation for the entire byte sequence
            return f'<0x{b.hex().upper()}>'
    
    # Save vocabulary
    vocab_filepath = os.path.join(save_dir, vocab_filename)
    with open(vocab_filepath, 'w', encoding='utf-8') as f:
        for token_id, token_bytes in vocab.items():
            if readable:
                token_str = bytes_to_readable(token_bytes)
            else:
                token_str = token_bytes.decode('utf-8', errors='replace')
            f.write(f"{token_id}\t{token_str}\n")
    
    # Save merges
    merges_filepath = os.path.join(save_dir, merges_filename)
    with open(merges_filepath, 'w', encoding='utf-8') as f:
        for step, (token1, token2) in enumerate(merges, start=1):
            if readable:
                token1_str = bytes_to_readable(token1)
                token2_str = bytes_to_readable(token2)
            else:
                token1_str = token1.decode('utf-8', errors='replace')
                token2_str = token2.decode('utf-8', errors='replace')
            f.write(f"{step}\t{token1_str} + {token2_str}\n")
    
    print(f"Vocabulary saved to: {vocab_filepath}")
    print(f"Merges saved to: {merges_filepath}")
    
if __name__ == "__main__":
    input_path = '../data/owt_train.txt'
    vocab_size = 32000
    special_tokens = ['<|endoftext|>']
    vocab, merges = profile_bpe_tokenization(input_path, vocab_size, special_tokens)
    save_vocab_and_merges(vocab, merges, "../data/tokenizer_output/owt_train")
