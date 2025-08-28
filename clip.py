import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import lightning as L

### LOSSES ################################
# contrastive loss definition
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
    
    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
        loss_img = F.cross_entropy(logits_per_image / self.temp, labels)
        loss_txt = F.cross_entropy(logits_per_text / self.temp, labels)
        return (loss_img + loss_txt) / 2

### MODEL ################################
class CLIPModel(L.LightningModule):
    def __init__(self, vision_encoder, text_encoder, config, eval_loaders=None):
        super().__init__()
        self.config = config
        self.vision_model = vision_encoder
        self.text_model = text_encoder
        
        # Projection layers
        self.visual_projection = nn.Linear(vision_encoder.config.hidden_size, self.config['projection_dim'])
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, self.config['projection_dim'])
        # TODO: check CLIP projection_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config['logit_scale_init_value'])
        self.eval_loaders = eval_loaders

        if self.config['freeze']:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # contrastive loss function
        self.contrastive_loss_fn = ContrastiveLoss(temp=self.config['contrastive_temp'])

        self.save_hyperparameters(ignore=['vision_encoder', 'text_encoder'])

    def forward(self, input_ids, pixel_values, attention_mask=None, position_ids=None):
        text_embeds = self.get_text_features(input_ids, attention_mask)
        image_embeds = self.get_image_features(pixel_values)
        
        # Calculate match similarity for flat embeddings
        match = image_embeds @ text_embeds.T
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_image = match * logit_scale
        logits_per_text = match.t() * logit_scale
        
        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "text_embeds": text_embeds,
            "image_embeds": image_embeds
        }
    
    def get_text_features(self, input_ids, attention_mask=None):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        seq_lengths = attention_mask.sum(dim=1).tolist()
        start_indices = [0]
        for length in seq_lengths[:-1]:
            start_indices.append(start_indices[-1] + length)

        last_hidden_state = outputs['hidden_states'][-1]

        if self.config['text_loss_type'] == 'mlm':
            cls_output = torch.stack([last_hidden_state[idx, :] for idx in start_indices])
            text_embeds = self.text_projection(cls_output)
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            pooled_output = last_hidden_state[batch_indices, sequence_lengths, :]
            text_embeds = self.text_projection(pooled_output)
        
        return text_embeds

    def get_image_features(self, pixel_values):
        outputs = self.vision_model(pixel_values)
        pooled_output = outputs.pooler_output
        image_embeds = self.visual_projection(pooled_output)
        
        return image_embeds
    
    def configure_optimizers(self):
        vision_params = {"params": self.vision_model.parameters(), "lr": self.config['lr']['vision']}
        text_params = {"params": self.text_model.parameters(), "lr": self.config['lr']['text']}
        proj_params = {"params": list(self.visual_projection.parameters()) + 
                                list(self.text_projection.parameters()), 
                    "lr": self.config['lr']['projection']}
        other_params = {"params": [p for n, p in self.named_parameters() 
                                  if not any(x in n for x in ['vision_model', 'text_model', 
                                                              'visual_projection', 'text_projection'])],
                    "lr": self.config['lr']['main']}
        
        optimizer = AdamW([vision_params, text_params, proj_params, other_params])
        return optimizer
    
    ## Steps and evaluations
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def _shared_step(self, batch, batch_idx, stage):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch['pixel_values']
        
        # contrastive loss
        outputs = self(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        logits_per_image = outputs["logits_per_image"]
        logits_per_text = outputs["logits_per_text"]
        contrastive_loss = self.contrastive_loss_fn(logits_per_image, logits_per_text)

        # log losses to wandb
        log_step = stage == "train"
        log_epoch = stage in ["val", "test"]
        self.log(f"{stage}/contrastive_loss", contrastive_loss, on_step=log_step, on_epoch=log_epoch, sync_dist=True)

        return contrastive_loss

    def _shared_eval_step(self, stage):
        """Shared evaluation step for validation and testing"""
        self._run_eval("lab-s", "label", stage)
        self._run_eval("lab-s", "image", stage)
        self._run_eval("lab-say", "label", stage)
        self._run_eval("lab-say", "image", stage)
        self._run_eval("toybox", "label", stage)
        self._run_eval("toybox", "image", stage)

    def on_validation_epoch_end(self):
        """Run evaluation at the end of validation epoch"""
        self._shared_eval_step("val")

    def on_test_epoch_end(self):
        """Run evaluation at the end of test epoch"""
        self._shared_eval_step("test")

    @torch.no_grad()
    def _run_eval(self, eval_name, eval_type, stage):
        """
        Run evaluation on the model using different evaluation types.
        Args:
            eval_name: Name of the evaluation dataset to use ('lab-s', 'lab-say', or 'toybox')
            eval_type: One of ["label", "image", "text"]
            stage: Stage name for logging
        Returns:
            accuracy: Accuracy score for the evaluation
        """
        device = next(self.parameters()).device
        correct = 0
        total = 0
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        eval_loader = self.eval_loaders[eval_name]

        if eval_type == "label":
            vocab_data = eval_loader.dataset.get_vocab_data()
            vocab = vocab_data['vocab']
            vocab_tokens = {k: v.to(device) for k, v in vocab_data['vocab_tokens'].items()}
            vocab_embeds = self.get_text_features(vocab_tokens['input_ids'], vocab_tokens['attention_mask'])
            vocab_embeds = F.normalize(vocab_embeds, p=2, dim=-1)
            
        for batch in eval_loader:
            # Move batch to device - handle nested structure
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(v) for v in obj]
                else:
                    return obj
            
            batch = move_to_device(batch)
            
            if eval_type == "label":
                # get image1 embedding
                image1_embeds = self.get_image_features(batch['images']['image1'])
                image1_embeds = F.normalize(image1_embeds, p=2, dim=-1)
                
                # calculate logits against vocabulary
                logits = torch.matmul(image1_embeds, vocab_embeds.t()) * logit_scale
                pred_idx = logits.argmax(dim=1)
                correct_idx = vocab.index(batch['texts']['text1']['original'][0])
                correct += (pred_idx == correct_idx).sum().item()
                total += 1
                
            elif eval_type == "image":
                # get text1 embedding
                text1_embeds = self.get_text_features(
                    batch['texts']['text1']['processed']['input_ids'],
                    batch['texts']['text1']['processed']['attention_mask']
                )
                text1_embeds = F.normalize(text1_embeds, p=2, dim=-1)
                
                # get all image embeddings
                image_embeds = []
                for img_key in batch['images'].keys():
                    img_embed = self.get_image_features(batch['images'][img_key])
                    img_embed = F.normalize(img_embed, p=2, dim=-1)
                    image_embeds.append(img_embed)
                image_embeds = torch.cat(image_embeds, dim=0)
                
                # calculate logits
                logits = torch.matmul(text1_embeds, image_embeds.t()) * logit_scale
                pred_idx = logits.argmax(dim=1)
                correct += (pred_idx == 0).sum().item()  # image1 is at index 0
                total += 1
            
            elif eval_type == "text":
                # get image1 embedding
                image1_embeds = self.get_image_features(batch['images']['image1'])
                image1_embeds = F.normalize(image1_embeds, p=2, dim=-1)
                
                # get all text embeddings
                text_embeds = []
                for txt_key in batch['texts'].keys():
                    txt_embed = self.get_text_features(
                        batch['texts'][txt_key]['processed']['input_ids'],
                        batch['texts'][txt_key]['processed']['attention_mask']
                    )
                    txt_embed = F.normalize(txt_embed, p=2, dim=-1)
                    text_embeds.append(txt_embed)
                text_embeds = torch.cat(text_embeds, dim=0)
                
                # calculate logits
                logits = torch.matmul(image1_embeds, text_embeds.t()) * logit_scale
                pred_idx = logits.argmax(dim=1)
                correct += (pred_idx == 0).sum().item()  # text1 is at index 0
                total += 1
            
            else:
                raise ValueError(f"Evaluation type {eval_type} not implemented.")

        accuracy = correct / total if total > 0 else 0.0
        self.log(f"{stage}/eval_{eval_name}_{eval_type}_accuracy", accuracy, on_epoch=True, sync_dist=True)
        
        return accuracy

    def on_fit_start(self):
        """Run validation at the start of training (epoch 0)"""
        if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders is not None:
            self.trainer.validate(self, self.trainer.val_dataloaders, verbose=False)
