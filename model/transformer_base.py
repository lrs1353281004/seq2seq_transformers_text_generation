import torch.nn as nn
import torch
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class transformer_base(nn.Module):
    def __init__(self,vocab_size,dim=512,nhead=16,load_pretrain=''):
        super().__init__()
        self.vocab_size=vocab_size
        if load_pretrain:
            print('load pretrained embedding in:{} \n'.format(load_pretrain))
            self.embedding=nn.Embedding.from_pretrained(torch.load(load_pretrain))
            self.embedding.requires_grad_=True
        else:
            self.embedding=nn.Embedding(self.vocab_size,dim)
        self.PE=PositionalEncoding(dim)
        self.transformer=nn.Transformer(d_model=dim ,nhead=nhead, num_encoder_layers=12)
        self.lr_2_vocab=nn.Linear(dim,self.vocab_size)
    def forward(self,src_ids,tgt_ids,src_pad_mask=None,tgt_pad_mask=None,tgt_mask=None):
        src = self.embedding(src_ids)
        tgt = self.embedding(tgt_ids)
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        src = self.PE(src)
        tgt = self.PE(tgt)
        out = self.transformer(src=src,tgt=tgt,src_key_padding_mask=src_pad_mask,tgt_key_padding_mask=tgt_pad_mask, \
                                memory_key_padding_mask=src_pad_mask,tgt_mask=tgt_mask)
        out = self.lr_2_vocab(out)
        return out
if __name__=="__main__":
    import os
    CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(os.path.join(CURRENT_PATH,'../'))
    from utils.tokenizer import basic_tokenizer
    from utils.dialogue_dataset import dialogue_dataset,collate_func
    tokenizer= basic_tokenizer(os.path.join(CURRENT_PATH,'../data/basic_vocab.txt'))
    dataset = dialogue_dataset(os.path.join(CURRENT_PATH,'../data/LCCC_base/test'),tokenizer)
    batch_size=10
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True,collate_fn=collate_func)
    for batch_idx, batch in tqdm(enumerate(train_loader),total=int(len(train_loader.dataset) / batch_size) + 1):
        src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch=batch['src_ids'],batch['tgt_ids'],batch['src_pad_mask'],batch['tgt_pad_mask'],batch['tgt_mask']
        break
    ##
    model=transformer_base(tokenizer.vocab_size,300,15,os.path.join(CURRENT_PATH,'../data/pretrained_embed.pkl')) #,os.path.join(CURRENT_PATH,'../data/pretrained_embed.pkl')
    out=model(src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch)
    print(out.shape)
