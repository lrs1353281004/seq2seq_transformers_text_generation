import torch.nn as nn
import torch
class transformer_base(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(self.vocab_size,512)
        self.transformer=nn.Transformer(nhead=16, num_encoder_layers=12)
        self.lr_2_vocab=nn.Linear(512,self.vocab_size)
    def forward(self,src_ids,tgt_ids,src_pad_mask=None,tgt_pad_mask=None,tgt_mask=None):
        src = self.embedding(src_ids)
        tgt = self.embedding(tgt_ids)
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
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
    model=transformer_base(tokenizer.vocab_size)
    out=model(src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch)
    print(out.shape)
