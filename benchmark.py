#!/usr/bin/env python
"""
Benchmarking Script for End-to-End Throughput, I/O Overhead, Disk Overhead, and GPU Compute

This script loads the WikiText-103 dataset (downloads it if not found)
from a specified path, wraps it into a PyTorch Dataset/DataLoader,
and runs a simple language model training loop.

It prints:
  - Overall epoch training time (which correlates with throughput)
  - Average I/O (data transfer) time per batch
  - Average forward and backward pass times per batch

Usage example:
  python benchmark.py --data_path ~/wikitext-103 --batch_size 32 --epochs 3
  python benchmark.py --data_path ./data/wikitext-103 --cpu # Run on CPU, save data in ./data/wikitext-103
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, load_dataset
import os

##############################
# Minimal Tokenizer & Dataset
##############################

class DummyTokenizer:
    """
    A very simple tokenizer that splits on whitespace.
    It builds a vocabulary on the fly.
    """
    def __init__(self, vocab=None, unk_token="<unk>"):
        if vocab is None:
            self.vocab = {"<pad>": 0, "<unk>": 1}
            self.next_index = 2
        else:
            self.vocab = vocab
            self.next_index = max(vocab.values()) + 1
        self.unk_token = unk_token
        self.pad_token_id = self.vocab["<pad>"]

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_index
                self.next_index += 1
            ids.append(self.vocab[token])
        return ids

    def __call__(self, text, truncation, padding, max_length, return_tensors):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        if len(ids) > max_length:
            if truncation:
                ids = ids[:max_length]
            else:
                raise ValueError("Text is too long")
        # Pad sequence if necessary
        if len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        # Create a simple attention mask: 1 for token, 0 for padding
        attention_mask = [1 if i < len(tokens) else 0 for i in range(max_length)]
        # Return as tensors (simulate a batch dimension)
        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

class WikiTextDataset(Dataset):
    """
    PyTorch Dataset wrapper for a Hugging Face WikiText-103 dataset split.
    Each example is assumed to be a dict with a "text" field.
    """
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(tokenizer.vocab) # Capture initial vocab size

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        text = self.hf_dataset[idx]['text']
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove the batch dimension
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

##############################
# Simple Language Model
##############################

class SimpleLanguageModel(nn.Module):
    """
    A simple language model: Embedding -> LSTM -> Linear.
    """
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size # Store vocab_size in the model

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_length)
        # Check if input_ids are within vocab size - for debugging CUDA error
        max_input_id = torch.max(input_ids).item() if input_ids.numel() > 0 else 0
        if max_input_id >= self.vocab_size:
            print(f"Error: Input ID {max_input_id} exceeds vocab size {self.vocab_size}")
        embedded = self.embedding(input_ids)              # (batch, seq, embed)
        lstm_out, _ = self.lstm(embedded)                   # (batch, seq, hidden)
        logits = self.linear(lstm_out)                      # (batch, seq, vocab)
        return logits

##############################
# Training Loop with Timing
##############################

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_io_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    num_batches = len(dataloader)
    
    try:
        from tqdm import tqdm
        progress_bar = tqdm(dataloader, desc="Training", leave=True)
        iterator = progress_bar
    except ImportError:
        print(f"Tip: Install tqdm (pip install tqdm) for a progress bar")
        # Fall back to simple counter if tqdm not available
        iterator = dataloader
        print(f"Starting training on {num_batches} batches...")
        # Print a dot every 10 batches as minimal progress indicator
        progress_counter = 0

    for batch in iterator:
        # Simple fallback progress if tqdm not available
        if 'tqdm' not in locals() and progress_counter % 10 == 0:
            print(".", end="", flush=True)
        progress_counter = progress_counter + 1 if 'tqdm' not in locals() else 0
            
        # --- I/O Timing: Moving batch to GPU (simulate data transfer overhead) ---
        start_io = time.time()
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        io_time = time.time() - start_io

        # --- Forward Pass Timing ---
        start_forward = time.time()
        logits = model(input_ids)
        forward_time = time.time() - start_forward

        # Compute loss. Using input_ids as a dummy target.
        # Shift logits and input_ids for sequence-to-sequence loss calculation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = criterion(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        
        # Update progress bar if tqdm is available
        if 'tqdm' in locals():
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Backward Pass Timing ---
        start_backward = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - start_backward

        total_loss += loss.item()
        total_io_time += io_time
        total_forward_time += forward_time
        total_backward_time += backward_time
    
    # Add newline if using simple progress indicator
    if 'tqdm' not in locals():
        print()  # New line after dots

    avg_loss = total_loss / num_batches
    avg_io_time = total_io_time / num_batches
    avg_forward_time = total_forward_time / num_batches
    avg_backward_time = total_backward_time / num_batches
    return avg_loss, avg_io_time, avg_forward_time, avg_backward_time

##############################
# Main Benchmarking Function
##############################

def main():
    parser = argparse.ArgumentParser(description="PyTorch Benchmarking for Data Pipeline and Model Performance")
    parser.add_argument("--data_path", type=str, default="~/wikitext-103", # Default path in home directory
                        help="Path to the WikiText-103 dataset. Dataset will be downloaded here if not found.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--prefetch_factor", type=int, default=1, help="Prefetch factor for DataLoader (set low to reveal I/O overhead)")
    parser.add_argument("--cpu", action='store_true', help="Force CPU execution even if CUDA is available") # Flag to force CPU
    args = parser.parse_args()

    # Determine device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = args.data_path # Get data path from arguments

    # -----------------------------
    # 1. Load the Dataset from Disk (or Download)
    # -----------------------------
    print("Loading WikiText-103 dataset...")
    if not os.path.exists(os.path.expanduser(data_path)): # Check if dataset exists at path
        print(f"Dataset not found at {data_path}. Downloading...")
        # Download and save the dataset
        wikitext_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir=os.path.expanduser(data_path))
        wikitext_dataset.save_to_disk(os.path.expanduser(data_path)) # Save to disk in Arrow format
        print(f"Dataset downloaded and saved to {data_path}")
    else:
        print(f"Dataset found at {data_path}. Loading from disk...")

    # Load from disk in either case (downloaded or already present)
    wikitext_dataset = load_from_disk(os.path.expanduser(data_path))
    train_split = wikitext_dataset["train"]
    print(f"Dataset loaded: {len(train_split)} training examples.")

    # -----------------------------
    # 2. Initialize the Tokenizer and Dataset Wrapper
    # -----------------------------
    tokenizer = DummyTokenizer()  # Replace with a proper tokenizer if desired
    
    # Build vocabulary by sampling some examples from the dataset
    num_samples = min(1000, len(train_split))  # Sample up to 1000 examples
    print(f"Building initial vocabulary by sampling {num_samples} examples...")
    for i in range(num_samples):
        sample_idx = i  # Sequential sampling for determinism
        text = train_split[sample_idx]['text']
        # Just tokenize to build vocabulary
        tokens = tokenizer.tokenize(text)
        tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"Initial vocabulary size after sampling: {len(tokenizer.vocab)}")
    
    train_dataset = WikiTextDataset(train_split, tokenizer, max_length=args.max_length)

    # -----------------------------
    # 3. Set up the DataLoader with minimal prefetch
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,              # Adjust based on your machine
        prefetch_factor=args.prefetch_factor,
        pin_memory=False if args.cpu else True # Disable pin_memory if CPU is forced
    )

    # -----------------------------
    # 4. Initialize the Model, Loss, and Optimizer
    # -----------------------------
    # Use a larger fixed vocab size to accommodate all tokens
    vocab_size = max(len(tokenizer.vocab) * 10, 30000)  # Either 10x current vocab or at least 30000
    print(f"Setting model vocabulary size to {vocab_size}")
    
    model = SimpleLanguageModel(vocab_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # 5. Run the Training Loop and Time It
    # -----------------------------
    print("Starting training benchmark...")
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.time()
        avg_loss, avg_io, avg_forward, avg_backward = train_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - start_epoch

        print(f"Epoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Epoch Training Time: {epoch_time:.2f} seconds")
        print(f"  Avg I/O (data transfer) Time per Batch: {avg_io:.4f} seconds")
        print(f"  Avg Forward Pass Time per Batch: {avg_forward:.4f} seconds")
        print(f"  Avg Backward Pass Time per Batch: {avg_backward:.4f} seconds")
    print("Benchmarking complete.")

if __name__ == "__main__":
    main()