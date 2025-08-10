from core import torch 
from torch import dropout
import torch.nn.functional as F
from config import context_win , n_embd,device,vocab_size,n_head,n_layer,dropout

from torch import nn


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size)
        self.query = nn.Linear(n_embd,head_size)
        self.value = nn.Linear(n_embd,head_size)
        self.register_buffer('tril',torch.tril(torch.ones(context_win,context_win)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    




class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding = nn.Embedding(context_win,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0 , std =0.02)
            if module.bias is not None : 
                torch.nn.init.zeros_(module.bias)
        elif  isinstance(module , nn.Embedding):
            torch.nn.init.normal_(module.weight ,mean = 0.0 , std = 0.02 )

    def forward(self,index,target=None):
        B,T = index.shape
        
        token = self.token_embedding_table(index)
        pos_embd = self.position_embedding(torch.arange(T,device=device))
        x = token + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if target == None:
            loss = None
        else:    
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
            
        return logits,loss  
    

    def generate(self, index , max_size ) :
        for _ in range(max_size): 
            index_cropped = index [:,-context_win:]
            logits , loss = self(index_cropped)
            logits = logits[: , -1 , :] 
            prob = F.softmax(logits, dim =-1)
            next_word = torch.multinomial(prob , num_samples =1 )
            index = torch.cat((index,next_word),dim = 1)
        return index 
    