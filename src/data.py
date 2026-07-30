from pathlib import Path

import torch
from torch.utils.data import Dataset

def load_text(
    file_path: str
) -> str:
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    text = path.read_text(encoding="utf-8")
    
    if not text:
        raise ValueError(f"File is empty: {file_path}")
    
    return text

def build_vocab(
    text: str
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    
    characters = sorted(set(text))
    
    char_to_idx = {
        char: idx 
        for idx, char in enumerate(characters)
    }
    
    idx_to_char = {
        idx: char
        for idx, char in enumerate(characters)
    }
    
    return characters, char_to_idx, idx_to_char

def encode_text(
    text: str,
    char_to_idx: dict[str, int]
) -> torch.Tensor:
    
    encoded = [char_to_idx[char]
               for char in text]
    
    return torch.tensor(encoded, dtype=torch.long)

def decode_text(
    encoded: torch.Tensor,
    idx_to_char: dict[int, str]
) -> str:
    
    decoded = "".join(idx_to_char[idx] for idx in encoded)
    
    return decoded

class CharacterDataset(Dataset):
    def __init__(
        self,
        encoded_text: torch.Tensor,
        block_size: int
    ):
        
        if encoded_text.ndim != 1:
            raise ValueError("Encoded text must be a 1D tensor.")
        
        if block_size <= 0:
            raise ValueError("Block size must be a positive integer.")
        
        if len(encoded_text) < block_size:
            raise ValueError("Encoded text length must be greater than block size.")
        
        
        self.encoded_text = encoded_text
        self.block_size = block_size
        
    def __len__(self) -> int:
        
        length = len(self.encoded_text) - self.block_size
        
        return length if length > 0 else 0
    
    def __getitem__(
        self,
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        block = self.encoded_text[idx:idx + self.block_size]
        
        input_seq = block[:-1]
        output_seq = block[1:]
        
        return input_seq, output_seq
