import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

class HeadcamDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, tokenizer=None, split='train', mlm_probability=0.3):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        text = self.data.iloc[idx]['text']

        if self.transform:
            image = self.transform(image, return_tensors='pt')['pixel_values'].squeeze(0)

        if self.tokenizer:
            text_tokens = self.tokenizer(text, padding='max_length', truncation=True, 
                                         return_special_tokens_mask=True, return_tensors='pt')
            input_ids = text_tokens['input_ids'].squeeze(0)
            attention_mask = text_tokens['attention_mask'].squeeze(0)
            special_tokens_mask = text_tokens['special_tokens_mask'].squeeze(0)

            # ## NTP loss
            # # if input_ids.size(0) == 0:
            # #     input_ids = None
            # #     attention_mask = None
            # #     next_token_labels = None
            # # else:
            # next_token_labels = input_ids.clone()
            # next_token_labels[:-1] = input_ids[1:]
            # next_token_labels[-1] = self.tokenizer.eos_token_id

            ## MLM loss
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
            # 'next_token_labels': next_token_labels
            'labels': labels,
            'masked_input_ids': masked_input_ids
        }

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