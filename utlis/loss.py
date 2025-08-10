from core import torch 
from data.dataloader import get_batch_mmap
from config import eval_iters

def estimate_loss(model , data_loader,config):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch_mmap(split)
                if X is None:  # Handle case when get_batch returns None
                    continue
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out 