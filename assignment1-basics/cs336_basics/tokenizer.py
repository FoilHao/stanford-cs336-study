import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges

        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # handle user-defined special_tokens
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode('utf-8') for token in self.special_tokens]
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.vocab.values():
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
        
        self.bytes_to_token_id = {v: k for k, v in self.vocab.items()}
        
    
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        pass
    
    def _str_to_bytes_tuple(self, text: str):
        return tuple(bytes([b]) for b in text.encode("utf-8"))
    
    def encode_regular_tokens(self, text: str) -> list[int]:
        # pre-tokenization
        pre_tokens = []
        for match in re.finditer(self.PAT, text):
            if not match:
                continue
            pre_tokens.append(match.group(0))
        # merge pre_tokens according to merges
        token_ids = []
        for pre_token in pre_tokens:
            pre_token_bytes = list(self._str_to_bytes_tuple(pre_token))
            # merge successively. eg: merge pre_token with 1st merged_token, merge the result with 2nd merged_token, ...
            for merge_token in self.merges:
                i = 0
                while i < len(pre_token_bytes):
                    if i < len(pre_token_bytes) -1 and (pre_token_bytes[i], pre_token_bytes[i + 1]) == merge_token:
                        merged_token = merge_token[0] + merge_token[1]
                        pre_token_bytes = pre_token_bytes[:i] + [merged_token] + pre_token_bytes[i + 2:]
                    else:
                        i += 1
            token_ids.extend([self.bytes_to_token_id[b] for b in pre_token_bytes])
        return token_ids     
            
        
    def encode(self, text: str) -> list[int]:
        tokens = []
        if not text:
            return tokens
        
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]
        
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.bytes_to_token_id[part.encode('utf-8')])
            else:
                tokens.extend(self.encode_regular_tokens(part))
                
        return tokens
                
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        return full_bytes.decode('utf-8', errors='replace')