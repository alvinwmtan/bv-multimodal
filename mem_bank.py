import torch
from torch import nn
import torch.nn.functional as F

class MemoryBankContrastive(nn.Module):
    def __init__(self, dim=512, size=4096, temp=0.07):
        super().__init__()
        self.size = size
        self.temp = temp
        self.register_buffer("queue_img", torch.randn(dim, size))
        self.register_buffer("queue_txt", torch.randn(dim, size))
        self.queue_img = F.normalize(self.queue_img, dim=0)
        self.queue_txt = F.normalize(self.queue_txt, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_embeds, txt_embeds):
        batch_size = img_embeds.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace keys
        if ptr + batch_size <= self.size:
            self.queue_img[:, ptr:ptr + batch_size] = img_embeds.T
            self.queue_txt[:, ptr:ptr + batch_size] = txt_embeds.T
        else:
            # wraparound
            remaining = self.size - ptr
            self.queue_img[:, ptr:] = img_embeds[:remaining].T
            self.queue_txt[:, ptr:] = txt_embeds[:remaining].T
            self.queue_img[:, :batch_size-remaining] = img_embeds[remaining:].T
            self.queue_txt[:, :batch_size-remaining] = txt_embeds[remaining:].T
            
        self.queue_ptr[0] = (ptr + batch_size) % self.size
        
    def forward(self, img_embeds, txt_embeds):       
        # get logits for batch and mem 
        logits_per_img_batch = torch.matmul(img_embeds, txt_embeds.t()) / self.temp
        logits_per_txt_batch = logits_per_img_batch.t()
        
        logits_per_img_memory = torch.matmul(img_embeds, self.queue_txt) / self.temp
        logits_per_txt_memory = torch.matmul(txt_embeds, self.queue_img) / self.temp
        
        labels = torch.arange(img_embeds.size(0)).to(img_embeds.device)
        
        # calculate loss
        loss_img = F.cross_entropy(
            torch.cat([logits_per_img_batch, logits_per_img_memory], dim=1), 
            labels
        )
        loss_txt = F.cross_entropy(
            torch.cat([logits_per_txt_batch, logits_per_txt_memory], dim=1), 
            labels
        )
        
        # # calculate accuracy (not returned for now)
        # image_pred = torch.argmax(logits_per_img_batch, dim=-1)
        # text_pred = torch.argmax(logits_per_txt_batch, dim=-1)
        # image_accuracy = (image_pred == labels).sum() / labels.size(0)
        # text_accuracy = (text_pred == labels).sum() / labels.size(0)

        # update
        self._dequeue_and_enqueue(img_embeds, txt_embeds)
        
        return (loss_img + loss_txt) / 2