"""
dataset.py — PyTorch Dataset class for the intent classification task.

The IntentDataset wraps our FAQ DataFrame and converts text+label rows into
the tensor format that PyTorch's DataLoader feeds into the model during training.

Key design decision: we tokenize ALL texts upfront in __init__ rather than
tokenizing one-by-one inside __getitem__. This is faster because tokenization
happens once at startup rather than on every batch during training.
"""

import os
import sys
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class IntentDataset(Dataset):
    """
    A PyTorch Dataset that wraps the FAQ pandas DataFrame.

    Each item returned by __getitem__ is a dict with three keys:
        - 'input_ids'      : Token IDs for the text, shape (MAX_LENGTH,)
        - 'attention_mask' : 1 where there is a real token, 0 for padding, shape (MAX_LENGTH,)
        - 'labels'         : Integer class index for the intent, scalar tensor

    Example usage:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        df = load_faq_csv(split="train")
        dataset = IntentDataset(df, tokenizer, max_length=64)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
    """

    def __init__(self, df, tokenizer, max_length: int = config.MAX_LENGTH):
        """
        Args:
            df        : pandas DataFrame with columns 'text' and 'intent'.
            tokenizer : A HuggingFace tokenizer (e.g. DistilBertTokenizer).
            max_length: Maximum token length. Texts longer than this are truncated;
                        shorter texts are padded to this length.
        """
        self.labels = [config.LABEL2ID[intent] for intent in df["intent"].tolist()]

        # Tokenize all texts at once — faster than doing it one-by-one in __getitem__
        # padding="max_length"  → pads short sentences to max_length with zeros
        # truncation=True       → cuts long sentences to max_length
        # return_tensors="pt"   → returns PyTorch tensors directly
        encodings = tokenizer(
            df["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Store the tensors — shape is (num_examples, max_length)
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self) -> int:
        """Returns the total number of examples in this dataset split."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one training example as a dict of tensors.

        Args:
            idx: Integer index of the example to retrieve.

        Returns:
            dict with keys: 'input_ids', 'attention_mask', 'labels'
        """
        return {
            "input_ids":      self.input_ids[idx],       # shape: (MAX_LENGTH,)
            "attention_mask": self.attention_mask[idx],   # shape: (MAX_LENGTH,)
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),  # scalar
        }
