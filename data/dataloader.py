import torch , os 
from config import train_file , val_file , context_win , batch_size , device
import tiktoken
import mmap
import logging 

enc = tiktoken.get_encoding("cl100k_base")
logging.basicConfig(filename='data_loading.log', level=logging.WARNING) 
current_position =0

def get_batch_mmap(split):
    global current_position 
    filename = train_file if split == 'train' else val_file
    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mmapped_file.seek(current_position)

            data_chunk = mmapped_file.read(context_win * batch_size).decode('utf-8', errors='replace')
            if not data_chunk:
                current_position = 0
                return None

            data = torch.tensor(enc.encode(data_chunk), dtype=torch.long)
            current_position += len(data_chunk)

            ix = torch.randint(len(data) - context_win, (batch_size,))
            x = torch.stack([data[i:i+context_win] for i in ix])
            y = torch.stack([data[i+1:i+context_win+1] for i in ix])

            return x.to(device), y.to(device)

    except UnicodeDecodeError:
            print("UnicodeDecodeError")
            pass    
    logging.error(f"Failed to decode file {filename} with any supported encoding")
    return None