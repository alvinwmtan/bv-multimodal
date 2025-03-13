import copy
import sys
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from transformers import ViTImageProcessor, ViTConfig, ViTModel
# from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import ModernBertConfig, ModernBertForMaskedLM, AutoTokenizer

from headcam_dataset import HeadcamDataset
from mem_bank import MemoryBankContrastive

cfg = OmegaConf.load('config.yaml')
args = OmegaConf.from_dotlist(sys.argv[1:])
cfg.merge_with(args)

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

# DINO loss definition
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.07, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # Apply softmax with temperature
        teacher_output = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        student_output = F.log_softmax(student_output / self.student_temp, dim=-1)
        
        # Compute DINO loss
        loss = torch.mean(torch.sum(-teacher_output * student_output, dim=-1))

        # Update the center with moving average
        self.update_center(teacher_output)
        
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

### MODEL ################################
class MultimodalModel(L.LightningModule):
    def __init__(self, vision_encoder, text_encoder, config):
        super().__init__()
        self.config = config
        self.vision_model = vision_encoder
        self.text_model = text_encoder
        self.visual_projection = nn.Linear(vision_encoder.config.hidden_size, self.config['projection_dim'])
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, self.config['projection_dim'])
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config['logit_scale_init_value'])

        if self.config['freeze']:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # loss functions
        # self.ntp_loss_fn = nn.CrossEntropyLoss()
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        # self.contrastive_loss_fn = ContrastiveLoss(temp=self.config['contrastive_temp'])
        self.contrastive_loss_fn = MemoryBankContrastive(dim=self.config['projection_dim'], temp=self.config['contrastive_temp'])
        self.dino_loss_fn = DINOLoss(
            out_dim=self.config['projection_dim'],
            teacher_temp=self.config['dino_teacher_temp'],
            student_temp=self.config['dino_student_temp'],
            center_momentum=self.config['dino_center_momentum']
        )
        
        # EMA teacher model for DINO
        self.teacher_model = copy.deepcopy(self.vision_model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters(ignore=['vision_encoder', 'text_encoder'])

    def forward(self, input_ids, pixel_values, attention_mask=None, position_ids=None):
        text_embeds = self.get_text_features(input_ids, attention_mask)#, position_ids)
        image_embeds = self.get_image_features(pixel_values)
        
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_image.t()
        
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
        cls_output = torch.stack([last_hidden_state[idx, :] for idx in start_indices])
        text_embeds = self.text_projection(cls_output)

        # sequence_lengths = attention_mask.sum(dim=1) - 1
        # batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)

        # pooled_output = last_hidden_state[batch_indices, sequence_lengths, :]
        # text_embeds = self.text_projection(pooled_output)
        return text_embeds

    def get_image_features(self, pixel_values):
        outputs = self.vision_model(pixel_values)
        pooled_output = outputs.pooler_output
        image_embeds = self.visual_projection(pooled_output)
        return image_embeds
    
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
        labels = batch['labels']
        masked_input_ids = batch['masked_input_ids']
        
        # DINO
        dino_loss = 0.0
        if (self.config['dino_weight'] != 0.0):
            student_outputs = self.get_image_features(pixel_values)
            with torch.no_grad():
                teacher_outputs = self.visual_projection(self.teacher_model(pixel_values).pooler_output)
            dino_loss = self.dino_loss_fn(student_outputs, teacher_outputs)

        # # next token prediction
        # ntp_loss = 0.0
        # if (self.config['ntp_weight'] != 0.0):
        #     text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        #     ntp_loss = text_model_outputs['loss']

        # masked language modelling
        mlm_loss = 0.0
        if (self.config['mlm_weight'] != 0.0):
            mlm_outputs = self.text_model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels)
            # mlm_loss = self.mlm_loss_fn(mlm_outputs.logits.view(-1, self.text_model.config.vocab_size), labels.view(-1))
            mlm_loss = mlm_outputs.loss

        # contrastive
        contrastive_loss = 0.0
        if (self.config['contrastive_weight'] != 0.0):
            outputs = self(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            # logits_per_image = outputs["logits_per_image"]
            # logits_per_text = outputs["logits_per_text"]
            # contrastive_loss = self.contrastive_loss_fn(logits_per_image, logits_per_text)
            contrastive_loss = self.contrastive_loss_fn(outputs["image_embeds", "text_embeds"])

        # total loss
        total_loss = self.config['contrastive_weight'] * contrastive_loss + \
                     self.config['mlm_weight'] * mlm_loss + \
                     self.config['dino_weight'] * dino_loss
        
        # log losses to wandb
        log_step = stage == "train"
        log_epoch = stage in ["val", "test"]
        self.log(f"{stage}/total_loss", total_loss, on_step=log_step, on_epoch=log_epoch, sync_dist=True)
        self.log(f"{stage}/dino_loss", dino_loss, on_step=log_step, on_epoch=log_epoch, sync_dist=True)
        # self.log(f"{stage}/ntp_loss", ntp_loss, on_step=log_step, on_epoch=log_epoch, sync_dist=True)
        self.log(f"{stage}/mlm_loss", mlm_loss, on_step=log_step, on_epoch=log_epoch, sync_dist=True)
        self.log(f"{stage}/contrastive_loss", contrastive_loss, on_step=log_step, on_epoch=log_epoch, sync_dist=True)
        
        return total_loss

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

    @torch.no_grad()
    def on_after_backward(self):
        # EMA update for teacher model after backward pass
        for student_param, teacher_param in zip(self.vision_model.parameters(), self.teacher_model.parameters()):
            teacher_param.data = self.config['dino_teacher_momentum'] * teacher_param.data + (1. - self.config['dino_teacher_momentum']) * student_param.data

### MAIN ################################
# init data
torch.set_float32_matmul_precision("high")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

train_ds = HeadcamDataset(
    csv_file=cfg.ds_file,
    img_dir=cfg.img_dir,
    transform=image_processor,
    tokenizer=tokenizer,
    split='train'
)

val_ds = HeadcamDataset(
    csv_file=cfg.ds_file,
    img_dir=cfg.img_dir,
    transform=image_processor,
    tokenizer=tokenizer,
    split='val'
)

test_ds = HeadcamDataset(
    csv_file=cfg.ds_file,
    img_dir=cfg.img_dir,
    transform=image_processor,
    tokenizer=tokenizer,
    split='test'
)

train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

# init model
if (cfg.pretrained):
    vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
    # text_encoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    text_encoder = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
else:
    vit_config = ViTConfig()
    vision_encoder = ViTModel(vit_config)

    # gpt2_config = GPT2Config()
    # text_encoder = GPT2LMHeadModel(gpt2_config)

    modernbert_config = ModernBertConfig()
    text_encoder = ModernBertForMaskedLM(modernbert_config)

contrastive_model = MultimodalModel(vision_encoder, text_encoder, cfg.model_config)

wandb_logger = WandbLogger(project="bv-multimodal", name=cfg.wandb_name)

# trainer
seed_everything(42, workers=True)
trainer = Trainer(
    logger=wandb_logger, 
    max_epochs=10, 
    accelerator="auto",
    devices=cfg.devices,
    precision=16,
    gradient_clip_val=1.0,
    accumulate_grad_batches=cfg.accumulate_grad_batches,
    deterministic=True,
    strategy='ddp_find_unused_parameters_true',
)

trainer.fit(contrastive_model, train_dl, val_dl)

test_result = trainer.test(contrastive_model, test_dl)
