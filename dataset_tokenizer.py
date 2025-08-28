import re
import pandas as pd
from collections import Counter
from typing import Dict, List, Union, Optional
import torch

class DatasetTokenizer:
    """
    A tokenizer that builds its vocabulary from the dataset text.
    Converts low-frequency tokens to UNK based on a configurable threshold.
    """
    
    def __init__(self, csv_file: str, text_column: str = 'text', split: str = 'train', 
                 max_length: int = 512, min_freq: int = 2, max_vocab_size: int = 50000):
        """
        Initialize tokenizer from dataset.
        
        Args:
            csv_file: Path to CSV file containing text data
            text_column: Name of the column containing text
            split: Which split to use for building vocabulary (should be 'train')
            max_length: Maximum sequence length
            min_freq: Minimum frequency for a token to be included in vocabulary
            max_vocab_size: Maximum vocabulary size (hard limit)
        """
        self.max_length = max_length
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.eos_token = "[EOS]"
        
        # Token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.eos_token_id = 5
        
        # Build vocabulary from dataset (only train split)
        self._build_vocabulary(csv_file, text_column, split)
        
        print(f"Built vocabulary with {len(self.word_to_id)} tokens from {split} split")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _build_vocabulary(self, csv_file: str, text_column: str, split: str):
        """Build vocabulary from dataset text (only from specified split)"""
        # Read dataset
        try:
            df = pd.read_csv(csv_file)
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV file")
            if 'split' not in df.columns:
                raise ValueError("Column 'split' not found in CSV file")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            # Fallback to basic vocabulary
            self._build_basic_vocabulary()
            return
        
        # Filter by split (only use train split for vocabulary building)
        df_split = df[df['split'] == split]
        if df_split.empty:
            print(f"No data found for split '{split}', using basic vocabulary")
            self._build_basic_vocabulary()
            return
        
        # Extract text from the specified split only
        all_text = []
        for text in df_split[text_column].dropna():
            if isinstance(text, str):
                all_text.append(text.lower())
        
        if not all_text:
            print(f"No valid text found in {split} split, using basic vocabulary")
            self._build_basic_vocabulary()
            return
        
        # Tokenize all text
        all_tokens = []
        for text in all_text:
            tokens = self._tokenize_text(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        filtered_tokens = {token: count for token, count in token_counts.items() 
                          if count >= self.min_freq}
        
        # Sort by frequency (most frequent first)
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        self.word_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id,
            self.eos_token: self.eos_token_id,
        }
        
        # Add dataset tokens (respecting max_vocab_size)
        for token, _ in sorted_tokens:
            if len(self.word_to_id) < self.max_vocab_size:
                self.word_to_id[token] = len(self.word_to_id)
            else:
                break
        
        # Create reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.next_id = len(self.word_to_id)

        self.vocab_size = len(self.word_to_id)
        
        # Print vocabulary statistics
        print(f"Dataset tokens: {len(all_tokens)}")
        print(f"Unique tokens: {len(token_counts)}")
        print(f"Tokens with freq >= {self.min_freq}: {len(filtered_tokens)}")
        print(f"Final vocabulary size: {self.vocab_size}")
    
    def _build_basic_vocabulary(self):
        """Fallback to basic vocabulary if dataset loading fails"""
        # Common English words
        common_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "shall",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "good", "bad", "big", "small", "new", "old", "young", "hot", "cold", "warm", "cool",
            "red", "blue", "green", "yellow", "black", "white", "brown", "pink", "purple", "orange",
            "cat", "dog", "bird", "fish", "car", "house", "book", "tree", "water", "food", "time",
            "baby", "child", "kid", "boy", "girl", "man", "woman", "person", "people", "family"
        ]
        
        self.word_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id,
            self.eos_token: self.eos_token_id,
        }
        
        for word in common_words:
            self.word_to_id[word] = len(self.word_to_id)
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.next_id = len(self.word_to_id)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Convert text to list of tokens (words)"""
        if not isinstance(text, str):
            return []
        
        # Simple word tokenization: split on whitespace and punctuation
        # Convert to lowercase and remove empty tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.unk_token_id)
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id_to_word.get(id, self.unk_token) for id in ids]
    
    def encode(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs and attention mask"""
        # Add special tokens
        tokens = [self.cls_token] + self._tokenize_text(text) + [self.sep_token]
        
        # Convert to IDs
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Truncate or pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            input_ids[-1] = self.sep_token_id  # Ensure SEP token at end
        else:
            # Pad with PAD tokens
            input_ids.extend([self.pad_token_id] * (self.max_length - len(input_ids)))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        attention_mask.extend([0] * (self.max_length - len(tokens)))
        
        # Create special tokens mask (1 for special tokens, 0 for regular words)
        special_tokens_mask = [1 if token in [self.cls_token, self.sep_token, self.pad_token, self.mask_token, self.eos_token] else 0 for token in tokens]
        special_tokens_mask.extend([0] * (self.max_length - len(tokens)))
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask]),
            'special_tokens_mask': torch.tensor([special_tokens_mask])
        }
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Main interface method - handles both single text and batch"""
        if isinstance(text, str):
            return self.encode(text, **kwargs)
        else:
            # Handle batch of texts
            batch_input_ids = []
            batch_attention_mask = []
            batch_special_tokens_mask = []
            
            for t in text:
                encoded = self.encode(t, **kwargs)
                batch_input_ids.append(encoded['input_ids'].squeeze(0))
                batch_attention_mask.append(encoded['attention_mask'].squeeze(0))
                batch_special_tokens_mask.append(encoded['special_tokens_mask'].squeeze(0))
            
            return {
                'input_ids': torch.stack(batch_input_ids),
                'attention_mask': torch.stack(batch_attention_mask),
                'special_tokens_mask': torch.stack(batch_special_tokens_mask)
            }
    
    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)
    
    @property
    def pad_token_id(self) -> int:
        return self.pad_token_id
    
    @property
    def unk_token_id(self) -> int:
        return self.unk_token_id
    
    @property
    def cls_token_id(self) -> int:
        return self.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        return self.sep_token_id
    
    @property
    def mask_token_id(self) -> int:
        return self.mask_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.eos_token_id
