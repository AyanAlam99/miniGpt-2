import torch 

def calculate_perplexity(model, data_loader, config):
   
    # Set the model to evaluation mode
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
   
    num_batches_per_validation = config.eval_iters 

    with torch.no_grad():
        for _ in range(num_batches_per_validation):
            try:
                # Get a batch of validation data
                X, Y = data_loader.get_batch('val') 
                
                if X is None:
                    continue  # Skip if the batch is empty or invalid
                
                
                # 1. Get the loss for THIS SPECIFIC BATCH directly from the model
                logits, loss = model(X, Y)
                
                # 2. Accumulate the total loss, weighting it by the number of tokens
                total_loss += loss.item() * Y.numel()
                total_tokens += Y.numel()
            
            except Exception as e:
                print(f"An error occurred during perplexity calculation: {e}")
                continue
    
    # Avoid division by zero if no tokens were processed
    if total_tokens == 0:
        return float('inf') 

    # Calculate the average loss over all processed tokens
    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss))
    
    
    model.train()
    
    return perplexity