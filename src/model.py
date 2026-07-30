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

class MultiHeadCausalSelfAttention(nn.Module):
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
    
class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        hidden_dim = 4 * embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        return self.net(x)
    
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim,
            num_heads,
            block_size,
            dropout
        )
        
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        
        return x

class CharacterGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.1,
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
        
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_dim,
                num_heads,
                block_size,
                dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        
        self.language_model_head = nn.Linear(
            embedding_dim,
            vocab_size
        )
        
    def forward(
        self,
        token_ids: torch.Tensor,
        target: torch.Tensor | None = None,
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
        
        x = self.transformer_blocks(embeddings)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)
        loss = None
        
        if target is not None:
            batch_size, seq_length, vocab_size = logits.size()
            logits_flat = logits.reshape(batch_size * seq_length, vocab_size)
            target_flat = target.reshape(batch_size * seq_length)
            
            loss = nn.functional.cross_entropy(logits_flat, target_flat)
        
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        
        self.eval()
        
        for _ in range(max_new_tokens):
            
            context = token_ids[:, -self.block_size:]
            
            logits, _ = self(context)
            
            logits = logits[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        
        return token_ids