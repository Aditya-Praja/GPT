import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, split

from src.data import (
    CharacterDataset,
    load_text,
    build_vocab,
    encode_text,
)

from src.model import CharacterGPT

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a character-level GPT model."
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/input.txt"),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/character_gpt.pth"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )

    return parser.parse_args()

def choose_device() -> torch.device:
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else: device = torch.device("cpu")
    
    return device

def create_dataloaders(
    encoded_text: torch.Tensor,
    block_size: int,
    batch_size: int,
    validation_fraction: float
) -> tuple[DataLoader, DataLoader]:
    
    split_idx = int(len(encoded_text) * (1 - validation_fraction))
    
    train_data = encoded_text[:split_idx]
    validation_data = encoded_text[split_idx:]
    
    train_dataset = CharacterDataset(
        train_data,
        block_size
    )
    
    validation_dataset = CharacterDataset(
        validation_data,
        block_size
    )
    
    training_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return training_dataloader, validation_dataloader

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    
    model.train()
    
    total_loss = 0.0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        _, loss = (inputs, targets)
        
        if loss is None:
            raise RuntimeError(
                "The model did not return a training loss."
            )
            
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss / len(dataloader)

def evaluate(
    model: CharacterGPT,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        
        for inputs, targets in dataloader:
            
            _, loss = model(inputs, targets)
            
            if loss is None:
                raise RuntimeError(
                    "The model did not return a evalutation loss."
                )      
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main() -> None:
    
    args = parse_arguments()
    
    if args.embedding_dim % args.num_heads != 0:
        raise ValueError(
            "Embedding dimension must be divisible by the number of heads."
        )
    
    device = choose_device()
    print(f"Using device: {device}")
    
    text = load_text(args.data_path)
    chars, char_to_idx, idx_to_char = build_vocab(text)
    encoded_text = encode_text(
        text,
        char_to_idx
    )
    
    encoded_text = torch.tensor(encoded_text, dtype=torch.long).to(device)
    
    training_dataloader, validation_dataloader = create_dataloaders(
        encoded_text,
        args.block_size,
        args.batch_size,
        args.validation_fraction
    )
    
    model = CharacterGPT(
        vocab_size=len(chars),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        block_size=args.block_size,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    
    best_validation_loss = float("inf")
    
    args.model_path.parent.mkdir(
        parents=True, 
        exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        
        train_loss = train_one_epoch(
            model,
            training_dataloader,
            optimizer,
            device
        )
        
        validation_loss = evaluate(
            model,
            validation_dataloader,
            device
        )
        
        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {validation_loss:.4f}"
        )
        
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            
            torch.save(
                {
                "model_state_dict": model.state_dict(),
                "characters": chars,
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "config": {
                        "vocab_size": len(chars),
                        "embedding_dim": args.embedding_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "block_size": args.block_size,
                        "dropout": args.dropout
                    },
                    "validation_loss": validation_loss,
                },
                args.model_path
            )
            
            print(f"Saved Improved Model to: {args.model_path}")
            
if __name__ == "__main__":
    main()
        
    