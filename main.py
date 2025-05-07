import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer parallelism to avoid fork issues

import sys
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from transformers import ViTImageProcessor, ViTConfig, ViTModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import ModernBertConfig, ModernBertForMaskedLM, AutoTokenizer

from headcam_dataset import HeadcamDataset
from eval_dataset import EvalDataset
from multimodal import ContrastiveLoss, DINOLoss, MultimodalModel

cfg = OmegaConf.load('config.yaml')
args = OmegaConf.from_dotlist(sys.argv[1:])
cfg.merge_with(args)

### MAIN ################################
# init data
torch.set_float32_matmul_precision("high")
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
if cfg.model_config.text_loss_type == 'mlm':
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
else:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

train_ds = HeadcamDataset(
    csv_file=cfg.ds_file,
    img_dir=cfg.img_dir,
    transform=image_processor,
    tokenizer=tokenizer,
    text_loss_type=cfg.model_config.text_loss_type,
    split='train'
)

val_ds = HeadcamDataset(
    csv_file=cfg.ds_file,
    img_dir=cfg.img_dir,
    transform=image_processor,
    tokenizer=tokenizer,
    text_loss_type=cfg.model_config.text_loss_type,
    split='val'
)

test_ds = HeadcamDataset(
    csv_file=cfg.ds_file,
    img_dir=cfg.img_dir,
    transform=image_processor,
    tokenizer=tokenizer,
    text_loss_type=cfg.model_config.text_loss_type,
    split='test'
)

train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

eval_datasets = {
    'lab-s': EvalDataset(
        csv_file='manifest_sc_4afc.csv',
        img_dir=cfg.img_dir,
        transform=image_processor,
        tokenizer=tokenizer
    ),
    'lab-say': EvalDataset(
        csv_file='manifest_sc_full.csv',
        img_dir=cfg.img_dir,
        transform=image_processor,
        tokenizer=tokenizer
    ),
    'toybox': EvalDataset(
        csv_file='manifest_toybox.csv',
        img_dir=cfg.img_dir,
        transform=image_processor,
        tokenizer=tokenizer
    )
}

eval_loaders = {
    name: DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for name, dataset in eval_datasets.items()
}

# init model
if (cfg.pretrained):
    vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
    if cfg.model_config.text_loss_type == 'mlm':
        text_encoder = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
    else:
        text_encoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    
else:
    vit_config = ViTConfig()
    vision_encoder = ViTModel(vit_config)

    if cfg.model_config.text_loss_type == 'mlm':
        modernbert_config = ModernBertConfig()
        text_encoder = ModernBertForMaskedLM(modernbert_config)
    else:
        gpt2_config = GPT2Config()
        text_encoder = GPT2LMHeadModel(gpt2_config)

contrastive_model = MultimodalModel(vision_encoder, text_encoder, cfg.model_config, eval_loaders)

wandb_logger = WandbLogger(project="bv-multimodal", name=cfg.wandb_name)

# trainer
seed_everything(42, workers=True)
trainer = Trainer(
    logger=wandb_logger, 
    max_epochs=10, 
    accelerator="auto",
    devices=cfg.devices,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    accumulate_grad_batches=cfg.accumulate_grad_batches,
    deterministic=True,
    strategy='ddp_find_unused_parameters_true',
)

trainer.fit(contrastive_model, train_dl, val_dl)

test_result = trainer.test(contrastive_model, test_dl)
