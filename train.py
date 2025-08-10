from config import *
from model import GPTModel
from data.dataloader import get_batch_mmap 
from utlis.loss import estimate_loss
from utlis.metrics import calculate_perplexity
from utlis.save_load import save_model, load_model
import torch, time

model = GPTModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
counter = 0


for epoch in range(max_epochs):
    for iter in range(2200000):
        xb, yb = get_batch_mmap('train')
        if xb is None: continue

        counter += 1
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if counter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Iter {counter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
            print(model.generate(torch.zeros((1,1),dtype=torch.long,device=device),max_size=256))

        if counter % 5000 == 0:
            save_model(model, counter)

perplexity = calculate_perplexity(model)
print(f"Final Perplexity: {perplexity}")
save_model(model, counter)
