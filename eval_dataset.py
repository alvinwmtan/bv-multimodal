import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Center crop transform for evaluation (always deterministic)
centre_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

class EvalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, image_preprocess=centre_crop, tokenizer=None):
        """
        Args:
            csv_file (str): Path to the CSV manifest file with columns for images and texts
            img_dir (str): Directory containing the images
            transform: Optional transform to be applied to the image (e.g., CLIP preprocessor)
            image_preprocess: Basic image preprocessing pipeline (default: centre_crop for evaluation)
            tokenizer: Tokenizer for processing text
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_preprocess = image_preprocess
        self.tokenizer = tokenizer
        
        self.image_cols = [col for col in self.data.columns if col.startswith('image')]
        self.text_cols = [col for col in self.data.columns if col.startswith('text')]
        
        self.vocab = None
        self.vocab_tokens = None
        if 'text1' in self.data.columns:
            self.vocab = sorted(list(self.data['text1'].unique()))
            if self.tokenizer:
                self.vocab_tokens = self.tokenizer(
                    self.vocab,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )

    def __len__(self):
        return len(self.data)

    def process_image(self, img_path):
        """Helper function to process a single image"""
        image = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        
        if self.image_preprocess:
            image = self.image_preprocess(image)

        if self.transform:
            image = self.transform(image, return_tensors='pt', do_rescale=not self.image_preprocess)['pixel_values'].squeeze(0)
            
        return image

    def process_text(self, text):
        """Helper function to process a single text"""
        if self.tokenizer:
            text_tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': text_tokens['input_ids'].squeeze(0),
                'attention_mask': text_tokens['attention_mask'].squeeze(0)
            }
        # Return empty tensors if no tokenizer
        return {
            'input_ids': torch.zeros(1, dtype=torch.long),
            'attention_mask': torch.zeros(1, dtype=torch.long)
        }

    def get_vocab_data(self):
        """Return vocabulary and its tokenized form"""
        return {
            'vocab': self.vocab,
            'vocab_tokens': self.vocab_tokens
        }

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        images = {
            col: self.process_image(row[col])
            for col in self.image_cols
        }
        
        texts = {
            col: {
                'original': row[col],
                'processed': self.process_text(row[col])
            }
            for col in self.text_cols
        }

        return {
            'images': images,
            'texts': texts
        } 