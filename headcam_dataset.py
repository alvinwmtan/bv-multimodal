import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DataCollatorForLanguageModeling

# Center crop transform for validation and test
centre_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Training augmentation transform
train_augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(1.0, 1.0)),  # Random square crop with resize
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.ToTensor()
])

class HeadcamDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, image_preprocess=None, 
                 tokenizer=None, text_loss_type='mlm', split='train', mlm_probability=0.3):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        
        # Set image preprocessing based on split
        if image_preprocess is not None:
            self.image_preprocess = image_preprocess
        else:
            if split == 'train':
                self.image_preprocess = train_augment
            else:
                self.image_preprocess = centre_crop
        
        self.tokenizer = tokenizer
        self.text_loss_type = text_loss_type
        self.mlm_probability = mlm_probability
        self.max_length = 512  # Default max length for text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        text = self.data.iloc[idx]['text']

        # Apply appropriate preprocessing based on split
        if self.image_preprocess:
            # Use the specified preprocessing (augmentation for train, center crop for val/test)
            image = self.image_preprocess(image)
        else:
            # Fallback to center crop if no preprocessing specified
            image = centre_crop(image)

        if self.transform:
            image = self.transform(image, return_tensors='pt', do_rescale=not self.image_preprocess)['pixel_values'].squeeze(0)

        # Initialize default tensors
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        special_tokens_mask = torch.zeros(self.max_length, dtype=torch.long)
        next_token_labels = torch.zeros(self.max_length, dtype=torch.long)
        labels = torch.zeros(self.max_length, dtype=torch.long)
        masked_input_ids = torch.zeros(self.max_length, dtype=torch.long)

        if self.tokenizer:
            if not isinstance(text, str): # some nan in text; restoring as space
                text = " "
            text_tokens = self.tokenizer(text, padding='max_length', truncation=True, 
                                         return_special_tokens_mask=True, return_tensors='pt')
            input_ids = text_tokens['input_ids'].squeeze(0)
            attention_mask = text_tokens['attention_mask'].squeeze(0)
            special_tokens_mask = text_tokens['special_tokens_mask'].squeeze(0)

            if self.text_loss_type == 'ntp':
                next_token_labels = input_ids.clone()
                next_token_labels[:-1] = input_ids[1:]
                next_token_labels[-1] = self.tokenizer.eos_token_id

            if self.text_loss_type == 'mlm':
                labels = input_ids.clone()
                probability_matrix = torch.full(labels.shape, self.mlm_probability)
                probability_matrix = probability_matrix * attention_mask.bool()
                probability_matrix = probability_matrix * (1-special_tokens_mask).bool()
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = -100  # only compute loss on masked tokens
                masked_input_ids = input_ids.clone()
                masked_input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'special_tokens_mask': special_tokens_mask,
            'next_token_labels': next_token_labels,
            'labels': labels,
            'masked_input_ids': masked_input_ids
        }
    
    # TODO: visualise data examples and model preds during training

# class HeadcamDataCollator(DataCollatorForLanguageModeling):
#     def __call__(self, features):
#         batch = {}
#         batch["original_input_ids"] = torch.stack([f["input_ids"] for f in features])
#         batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
#         batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        
#         mlm_batch = super().__call__(features)
#         batch["input_ids"] = mlm_batch["input_ids"]
#         batch["labels"] = mlm_batch["labels"]
        
#         return batch