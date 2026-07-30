import argparse
from pathlib import Path

import torch

from src.data import decode_text, encode_text
from src.model import CharacterGPT

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate text using a trained "
            "character-level GPT model."
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/character_gpt.pth"),
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="Starting text used to begin generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Number of new characters to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )

    return parser.parse_args()

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_checkpoint(
    model_path: Path,
    device: torch.device
) -> dict:
    
    checkpoint = torch.load(
        model_path, 
        map_location=device, 
        weights_only=False
        )
    
    return checkpoint

def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device
) -> CharacterGPT:
    
    model = CharacterGPT(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        block_size=checkpoint["block_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"]
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

def encode_prompt(
    prompt: str,
    char_to_idx: dict[str, int],
    device: torch.device
) -> torch.Tensor:
    
    unknown_chars = sorted(
        {
            char
            for char in prompt
            if char not in char_to_idx
        }
    )
    
    if unknown_chars:
        raise ValueError(
            f"Prompt contains unknown characters: {unknown_chars}"
        )
    
    encoded_prompt = encode_text(prompt, char_to_idx)
    
    prompt_tensor = torch.tensor(
        encoded_prompt, 
        dtype=torch.long, 
        device=device
    )
    
    return prompt_tensor.unsequeeze(0)

def main() -> None:
    
    args = parse_arguments()
    
    device = choose_device()
    print(f"Using device: {device}")
    
    checkpoint = load_checkpoint(
        args.model_path,
        device
    )
    
    model = build_model_from_checkpoint(
        checkpoint,
        device
    )
    
    prompt_tensor = encode_prompt(
        args.prompt,
        checkpoint["char_to_idx"],
        device
    )
    
    generated_tensor = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    generated_text_list = generated_tensor[0].detach().cpu().tolist()
    
    generated_text = decode_text(
        generated_text_list,
        checkpoint["idx_to_char"]
    )
    
    print("Generated Text:")
    print(generated_text)
    
if __name__ == "__main__":
    main()