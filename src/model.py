import torch
from torch import nn

import math

class CausalSelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.key = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )
        
        self.query = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )
        
        self.value = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )
        
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        _, seq_length, _ = x.size()
        
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        attention_scores = torch.matmul(
            queries,
            keys.transpose(-2, -1)
        ) / math.sqrt(queries.size(-1))
        
        mask = self.causal_mask[:seq_length, :seq_length]
        
        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float('-inf')
        )
        
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )
        
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, values)
        
        return output

class MultiHeadCasualSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of heads."
            )
        
        self.head_dim = embedding_dim // num_heads
        
        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(
                embedding_dim,
                self.head_dim,
                block_size,
                dropout
            )
            for _ in range(num_heads)
        ])
        
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        heads = [
            head(x)
            for head in self.heads
        ]
        
        heads_concat = torch.cat(heads, dim=-1)
        
        output = self.output_projection(heads_concat)
        output = self.dropout(output)
        
        return output

class CharacterGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )
        
    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, seq_length = token_ids.size()
        
        if seq_length > self.block_size:
            raise ValueError(
                f"Sequence length {seq_length} exceeds block size {self.block_size}."
            )
            
        token_embeddings = self.token_embedding(token_ids)
        
        positions = torch.arange(
            seq_length,
            device=token_ids.device
        )
        
        position_embeddings = self.position_embedding(positions)
        
        embeddings = token_embeddings + position_embeddings
        
        return embeddings