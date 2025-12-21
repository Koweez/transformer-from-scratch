import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from models.transformer import Transformer
from transformers import AutoTokenizer
from config import config
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            logits = output.permute(0, 2, 1)
            targets = tgt[:, 1:]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model

def collate_fn(batch, tokenizer, src_lang='de', tgt_lang='en', max_length=128):
    src_texts = [item['translation'][src_lang] for item in batch]
    tgt_texts = [item['translation'][tgt_lang] for item in batch]
    
    src_encodings = tokenizer(src_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    tgt_encodings = tokenizer(tgt_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    
    return src_encodings['input_ids'], tgt_encodings['input_ids']

tokenizer = AutoTokenizer.from_pretrained('t5-small')
print('Tokenizer vocabulary size:', tokenizer.vocab_size)
model = Transformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    d_model=config['d_model'],
    num_heads=config['num_heads'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    d_ff=config['d_ff'],
    dropout=config['dropout'],
)

# print(model)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1e-3)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (config['d_model'] ** -0.5) * min((step + 1) ** -0.5, (step + 1) * (config['warming_up_steps'] ** -1.5)))


dataset = load_dataset('wmt14', 'de-en', split='train[:1%]', cache_dir='data/')
print(f'Dataset loaded with {len(dataset)} samples.')

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))

trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=10, device=device)
torch.save(trained_model.state_dict(), 'transformer_model.pth')

print('Model training complete and saved to transformer_model.pth')
